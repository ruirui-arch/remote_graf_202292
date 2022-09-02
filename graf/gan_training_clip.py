import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

from submodules.GAN_stability.gan_training.train import toggle_grad, Trainer as TrainerBase
from submodules.GAN_stability.gan_training.train import compute_grad2
from submodules.GAN_stability.gan_training.eval import Evaluator as EvaluatorBase
from submodules.GAN_stability.gan_training.metrics import FIDEvaluator, KIDEvaluator

from .utils import save_video, color_depth_map



class CLIPTrainer(TrainerBase):
    def __init__(self, *args, clip_loss, clip_weight, use_amp=False, **kwargs):
        super(CLIPTrainer, self).__init__(*args, **kwargs)
        self.clip_loss = clip_loss
        self.clip_weight = clip_weight
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler

    def generator_trainstep_withCLIP(self, y, z, feats, text_inputs):
        """
        CLIP loss guided stage II training (finetune)
        :param y: sampled discriminator codes
        :param z: sampled generator codes (appearance and style codes)
        :param feats: CLIP encoded features (image features or text features)
        :return:
        """
        assert (y.size(0) == z.size(0))
        if not self.use_amp:
            return self.generator_trainstep_withCLIP_noAMP(y, z, feats, text_inputs)
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            x_fake = self.generator(z, y, clip_feat=feats)
            d_fake = self.discriminator(x_fake, y)
            gloss = self.compute_gloss(d_fake, 1)
            clip_loss = self.compute_clipLoss(x_fake, text_inputs)
        gloss += clip_loss
        self.scaler.scale(gloss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        return gloss.item(), clip_loss.item()

    def generator_trainstep_withCLIP_noAMP(self, y, z, feats, text_inputs):
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y, clip_feat=feats)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_gloss(d_fake, 1)
        clip_loss = self.compute_clipLoss(x_fake, text_inputs)
        gloss += clip_loss
        gloss.backward()
        self.g_optimizer.step()
        return gloss.item(), clip_loss.item()

    def compute_gloss(self, d_outs, target):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = 0
        for d_out in d_outs:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            if self.gan_type == 'standard':
                loss += F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.gan_type == 'wgan':
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError
        return loss / len(d_outs)

    def compute_clipLoss(self, fake, feat):
        fake = [fake] if not isinstance(fake, list) else fake
        clip_loss = 0
        for f in fake:
            clip_loss += self.clip_weight * self.clip_loss(f, feat)
        return clip_loss / len(fake)

    def discriminator_trainstep_clipLoss(self, x_real, y, z, clip_feat):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()  # 这个更像是调整模式而不是调用函数
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)  # assign 1 to real logits

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()  # regularizer: real sample -> discriminator -> reg grad2
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y, clip_feat=clip_feat)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()


class Evaluator(EvaluatorBase):
    def __init__(self, eval_fid_kid, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        if eval_fid_kid:
            self.inception_eval = FIDEvaluator(
                device=self.device,
                batch_size=self.batch_size,
                resize=True,
                n_samples=20000,
                n_samples_fake=1000,
            )

    def get_rays(self, pose):
        return self.generator.val_ray_sampler(self.generator.H, self.generator.W,
                                              self.generator.focal, pose)[0]

    def create_samples(self, z, poses=None):
        self.generator.eval()

        N_samples = len(z)
        device = self.generator.device
        z = z.to(device).split(self.batch_size)
        if poses is None:
            rays = [None] * len(z)
        else:
            rays = torch.stack([self.get_rays(poses[i].to(device)) for i in range(N_samples)])
            rays = rays.split(self.batch_size)

        rgb, disp, acc = [], [], []
        with torch.no_grad():
            for z_i, rays_i in tqdm(zip(z, rays), total=len(z), desc='Create samples...'):
                bs = len(z_i)
                if rays_i is not None:
                    rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)  # Bx2x(HxW)xC -> 2x(BxHxW)x3
                rgb_i, disp_i, acc_i, _ = self.generator(z_i, rays=rays_i)

                reshape = lambda x: x.view(bs, self.generator.H, self.generator.W, x.shape[1]).permute(0, 3, 1,
                                                                                                       2)  # (NxHxW)xC -> NxCxHxW
                rgb.append(reshape(rgb_i).cpu())
                disp.append(reshape(disp_i).cpu())
                acc.append(reshape(acc_i).cpu())

        rgb = torch.cat(rgb)
        disp = torch.cat(disp)
        acc = torch.cat(acc)

        depth = self.disp_to_cdepth(disp)

        return rgb, depth, acc

    def make_video(self, basename, z, poses, as_gif=True):
        """ Generate images and save them as video.
        z (N_samples, zdim): latent codes
        poses (N_frames, 3 x 4): camera poses for all frames of video
        """
        N_samples, N_frames = len(z), len(poses)

        # reshape inputs
        z = z.unsqueeze(1).expand(-1, N_frames, -1).flatten(0, 1)  # (N_samples x N_frames) x z_dim
        poses = poses.unsqueeze(0) \
            .expand(N_samples, -1, -1, -1).flatten(0, 1)  # (N_samples x N_frames) x 3 x 4

        rgbs, depths, accs = self.create_samples(z, poses=poses)

        reshape = lambda x: x.view(N_samples, N_frames, *x.shape[1:])
        rgbs = reshape(rgbs)
        depths = reshape(depths)
        print('Done, saving', rgbs.shape)

        fps = min(int(N_frames / 2.), 25)  # aim for at least 2 second video
        for i in range(N_samples):
            save_video(rgbs[i], basename + '{:04d}_rgb.mp4'.format(i), as_gif=as_gif, fps=fps)
            save_video(depths[i], basename + '{:04d}_depth.mp4'.format(i), as_gif=as_gif, fps=fps)

    def disp_to_cdepth(self, disps):
        """Convert depth to color values"""
        if (disps == 2e10).all():  # no values predicted
            return torch.ones_like(disps)

        near, far = self.generator.render_kwargs_test['near'], self.generator.render_kwargs_test['far']

        disps = disps / 2 + 0.5  # [-1, 1] -> [0, 1]

        depth = 1. / torch.max(1e-10 * torch.ones_like(disps), disps)  # disparity -> depth
        depth[disps == 1e10] = far  # set undefined values to far plane

        # scale between near, far plane for better visualization
        depth = (depth - near) / (far - near)

        depth = np.stack([color_depth_map(d) for d in depth[:, 0].detach().cpu().numpy()])  # convert to color
        depth = (torch.from_numpy(depth).permute(0, 3, 1, 2) / 255.) * 2 - 1  # [0, 255] -> [-1, 1]

        return depth

    def compute_fid_kid(self, sample_generator=None):
        if sample_generator is None:
            def sample():
                while True:
                    z = self.zdist.sample((self.batch_size,))
                    rgb, _, _ = self.create_samples(z)
                    # convert to uint8 and back to get correct binning
                    rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8).to(torch.float) / 255. * 2 - 1
                    yield rgb.cpu()

            sample_generator = sample()

        fid, (kids, vars) = self.inception_eval.get_fid_kid(sample_generator)
        kid = np.mean(kids)
        return fid, kid


