# Based on the description from CLIPNeRF paper

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class StyleMLP(nn.Module):
    def __init__(self, style_dim=8, embed_dim=128, style_depth=1):
        super().__init__()
        self.activation = F.relu

        lin_block = nn.Linear
        first_block = nn.Linear(style_dim, embed_dim)
        self.mlp = nn.ModuleList([first_block] + [lin_block(embed_dim, embed_dim) for _ in range(style_depth - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.mlp):
            x = self.activation(layer(x))
        return x


class CLIPNeRF(nn.Module):
    def __init__(self, W_nerf=256, D_nerf=8, W_deform=256, D_deform=4, W_sigma=256,
                 D_sigma=1, W_rgb=128, D_rgb=2, W_bottleneck=8, input_ch=3, input_ch_views=3, output_ch=4,
                 shape_dim=128, appearance_dim=128, embed_dim=128, skip=[4], style_depth=1, style_dim=128,
                 separate_codes=True,
                 use_viewdirs=True, finetune_codes=False, W_CLIP=128, W_mapper=256, D_mapper=2, pre_add=False, **kwargs):
        """

        :param W_nerf: The middle channel dim of the NeRF body
        :param D_nerf: The layer num of the NeRF body
        :param W_deform: The middle channel dim of the deformation network
        :param D_deform: The layer num of the deformation network
        :param W_sigma: The middle channel dim of the sigma prediction network (If we use multiple FC layers to predict the sigma network)
        :param D_sigma: The layer num of sigma prediction network
        :param W_rgb: Middle channel of the rgb prediction network
        :param D_rgb: Layer num of the rgb prediction network
        :param W_bottleneck: NeRF body -> feature -> rgb prediction (Feature channel num)
        :param input_ch: Input dim of pts
        :param input_ch_views: Input dim of view direction
        :param output_ch: output dim (if not adopt the view direction)
        :param shape_dim: Dim of shape code
        :param appearance_dim: Dim of appearance code
        :param embed_dim: If > 0, adopt another MLP to reformulate shape and appearance codes
        :param skip: None / List, if List -> indicates which layer of NeRF body should be reinjected the input pts
        :param style_depth: If embed dim > 0, it indicates how many FCs are imposed to reformulate the latent codes
        :param separate_codes: If True, separate the codes from a unified sampled latent code group
        :param use_viewdirs: Whether to use view direction
        :param finetune_codes: Only valid in stage II, finetune the shape/ppearance mappers but fix all the other layers
        :param W_CLIP: The channel num of CLIP (output from CLIP encoder)
        :param W_mapper: The middle dim of the mappers
        :param D_mapper: The num of the layers in the mappers
        :param pre_add: If True: add the delta codes before the code reformulation process
        :param kwargs: Just given a place
        """
        super(CLIPNeRF, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
        self.shape_dim = shape_dim
        self.appearance_dim = appearance_dim
        self.separate_codes = separate_codes
        self.get_cached = None  # Updated by render_path to get cache
        self.activation = F.relu
        self.embed_dim = embed_dim
        self.pre_add = pre_add
        self.style_dim = style_dim
        if skip is None:
            skip = []
        self.skip = skip

        if self.embed_dim > 0:
            self.shape_linear1 = StyleMLP(shape_dim, embed_dim, style_depth)
            self.shape_linear2 = StyleMLP(shape_dim, embed_dim, style_depth)
            self.shape_fusion = nn.Linear(self.input_ch + embed_dim, input_ch)
            self.appearance_linear = StyleMLP(appearance_dim, embed_dim, style_depth)
            shape_dim, appearance_dim = embed_dim, embed_dim

        deform_inChannel = self.input_ch + shape_dim
        self.deformation_network = nn.ModuleList(
            [nn.Linear(deform_inChannel, W_deform)] + [nn.Linear(W_deform, W_deform) for _ in range(D_deform - 2)] + [
                nn.Linear(W_deform, self.input_ch)]
        )
        self.pts_linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W_nerf)] + [
                nn.Linear(W_nerf, W_nerf) if i not in skip else nn.Linear(W_nerf + self.input_ch, W_nerf) for i in range(D_nerf - 1)]
        )
        if use_viewdirs:
            # Not the original NeRF setting, because there is only one MLP after viewpoint
            if D_sigma > 1:
                self.sigma_linear = nn.Sequential(
                    *[nn.Sequential(nn.Linear(W_nerf, W_sigma), nn.ReLU())],
                    *[nn.Sequential(nn.Linear(W_sigma, W_sigma), nn.ReLU()) for _ in range(D_sigma - 2)],
                    *[nn.Linear(W_sigma, 1)]
                )
            else:
                self.sigma_linear = nn.Linear(W_nerf, 1)
            self.feature_linear = nn.Linear(W_nerf, W_bottleneck)
            # concatenation of [view direction, feature from Base NeRF network, appearance code]
            rgb_inChannel = self.input_ch_views + W_bottleneck + appearance_dim
            self.rgb_network = nn.Sequential(
                *[nn.Sequential(nn.Linear(rgb_inChannel, W_rgb), nn.ReLU())],
                *[nn.Sequential(nn.Linear(W_rgb, W_rgb), nn.ReLU()) for _ in range(D_rgb - 2)]
            )
            self.rgb_output = nn.Linear(W_rgb, 3)
        else:
            self.output_linear = nn.Linear(W_nerf, output_ch)

        self.finetune_codes = finetune_codes
        if finetune_codes:
            self.shape_mapper = nn.Sequential(
                *[nn.Sequential(nn.Linear(W_CLIP, W_mapper), nn.ReLU())],
                *[nn.Sequential(nn.Linear(W_mapper, W_mapper), nn.ReLU())],
                *[nn.Sequential(nn.Linear(W_mapper, shape_dim), nn.ReLU())],
            )
            self.appearance_mapper = nn.Sequential(
                *[nn.Sequential(nn.Linear(W_CLIP, W_mapper), nn.ReLU())],
                *[nn.Sequential(nn.Linear(W_mapper, W_mapper), nn.ReLU())],
                *[nn.Sequential(nn.Linear(W_mapper, appearance_dim), nn.ReLU())],
            )

    def forward(self, x, styles, feat=None):
        """image_features: Encoded by CLIP (governed in the outer space)
        text_features: Encoded by CLIP (governed in the outer space)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        if self.separate_codes:
            shape_code, appearance_code = styles[:, :self.style_dim], styles[:, self.style_dim:]
        else:
            shape_code, appearance_code = styles, styles

        if self.finetune_codes:
            assert feat is not None
            delta_zs = self.shape_mapper(feat)
            delta_za = self.appearance_mapper(feat)
            if self.pre_add or self.embed_dim <= 0:
                shape_code = shape_code + delta_zs
                appearance_code = appearance_code + delta_za

        # deformation network
        if self.embed_dim > 0:
            shape_code1 = self.shape_linear1(shape_code)
            if self.finetune_codes and (not self.pre_add):
                shape_code1 = shape_code1 + delta_zs
            h = torch.cat((input_pts, shape_code1), dim=-1)
            for i, l in enumerate(self.deformation_network):
                h = self.deformation_network[i](h)
                h = self.activation(h)
            shape_code2 = self.shape_linear2(shape_code)
            if self.finetune_codes and (not self.pre_add):
                shape_code2 = shape_code2 + delta_zs
            h = torch.cat((h, shape_code2), dim=-1)
            h = self.shape_fusion(h)
        else:
            h = torch.cat((input_pts, shape_code), dim=-1)
            for i, l in enumerate(self.deformation_network):
                h = self.deformation_network[i](h)
                if i != len(self.deformation_network) - 1:
                    h = self.activation(h)
        h = input_pts + torch.tanh(h)  # formula (3)

        # Basic NeRF Network
        for i, l in enumerate(self.pts_linear):
            h = self.pts_linear[i](h)
            h = self.activation(h)
            if i in self.skip:
                h = torch.cat((input_pts, h), dim=-1)

        if self.use_viewdirs:
            # sigma generation (based on features)
            sigma = self.sigma_linear(h)
            # feature generation (for rgb synthesis)
            feature = self.feature_linear(h)
            # concatenation of [view direction, feature, appearance code]
            if self.embed_dim > 0:
                appearance_code = self.appearance_linear(appearance_code)
                if self.finetune_codes and (not self.pre_add):
                    appearance_code = appearance_code + delta_za
            h = torch.cat((input_views, feature, appearance_code), dim=-1)
            h = self.rgb_network(h)
            rgb = self.rgb_output(h)
            outputs = torch.cat([rgb, sigma], -1)
        else:
            outputs = self.output_linear(h)
        return outputs


if __name__ == '__main__':
    input_names = ['x', 'styles']
    output_names = ['outputs']

    # define models
    params = {'W_nerf': 256, 'D_nerf': 8, 'W_deform': 256, 'D_deform': 4,
              'W_sigma': 256, 'D_sigma': 1, 'W_rgb': 128, 'D_rgb': 2,
              'W_bottleneck': 8, 'input_ch': 63, 'input_ch_views': 27,
              'output_ch': 4, 'shape_dim': 128, 'appearance_dim': 128,
              'embed_dim': 128, 'skip': None, 'style_depth': 1,
              'separate_codes': False, 'use_viewdirs': True, 'finetune_codes': True,
              'W_CLIP': 128, 'W_mapper': 256, 'D_mapper': 2, 'pre_add': False}

    model = CLIPNeRF(
        W_nerf= params['W_nerf'], D_nerf=params['D_nerf'], W_deform=params['W_deform'],
        D_deform=params['D_deform'], W_sigma=params['W_sigma'], D_sigma=params['D_sigma'],
        W_rgb=params['W_rgb'], D_rgb=params['D_rgb'], W_bottleneck=params['W_bottleneck'],
        input_ch=params['input_ch'], input_ch_views=params['input_ch_views'], output_ch=params['output_ch'],
        shape_dim=params['shape_dim'], appearance_dim=params['appearance_dim'], embed_dim=params['embed_dim'],
        skip=params['skip'], style_depth=params['style_depth'], separate_codes=params['separate_codes'],
        use_viewdirs=params['use_viewdirs'], finetune_codes=params['finetune_codes'], W_CLIP=params['W_CLIP'],
        W_mapper=params['W_mapper'], D_mapper=params['D_mapper'], pre_add=params['pre_add']
    )

    print(model)

    for k, v in model.named_parameters():
        print(f'k: {k}')

    print(list(model.parameters())[0][1])

    x = torch.randn((65536, 90))
    y = torch.randn((65536, 128))


    # torch.onnx.export(model, (x, y), 'CLIPNeRF_embed.onnx', input_names=input_names,
    #                   output_names=output_names, verbose=True)