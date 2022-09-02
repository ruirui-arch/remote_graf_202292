import torch
import clip
from PIL import Image
from math import sqrt


class CLIPLoss(torch.nn.Module):

    def __init__(self, N_samples):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda",
                                                download_root='/ghome/zhangkd/newJob/nerf/nerf-pytorch-master/clip_model')
        self.N_samples_sqrt = int(sqrt(N_samples))
        # self.upsample = torch.nn.Upsample(scale_factor=7)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def encode_image(self, image):
        image = self.preprocess(Image.open(image)).unsqueeze(0).to(device='cuda')
        image_feats = self.model.encode_image(image)
        return image_feats

    def encode_text(self, description):
        text_inputs = torch.cat([clip.tokenize(description)]).to(device="cuda")
        with torch.no_grad():
            text_feats = self.model.encode_text(text_inputs)
        return text_feats

    def forward(self, image, text):
        image = image.reshape(self.N_samples_sqrt, self.N_samples_sqrt, 3).unsqueeze(0)
        image = image.permute(0, 3, 1, 2).contiguous()
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        # image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity.mean()  # .mean()
