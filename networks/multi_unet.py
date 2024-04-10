import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        # Assuming the input from each block is concatenated along the channel dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiPathResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilation1=2, dilation2=3, use_dilate_conv=True):
        super(BiPathResBlock, self).__init__()
        # Define two ResBlocks and two DilatedResBlocks in sequence for each path
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            ResBlock(mid_channels, mid_channels)
        )
        self.dilated_resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            DilatedResBlock(mid_channels, mid_channels, dilation=dilation1)
        )
        self.dilated_resblock2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            DilatedResBlock(mid_channels, mid_channels, dilation=dilation2)
        )
        # Define the Fusion Block
        self.fusionblock = FusionBlock(3 * mid_channels, out_channels)
        self.use_dilate_conv = use_dilate_conv

    def forward(self, x):
        res_out = self.resblock(x)
        dilated_res_out = self.dilated_resblock(x)
        dilated_res_out2 = self.dilated_resblock2(x)
        if self.use_dilate_conv:
            x = self.fusionblock(res_out, dilated_res_out, dilated_res_out2)
        else:
            x = self.fusionblock(res_out, res_out, res_out)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, use_dilate_conv=True):
        super(CNNEncoder, self).__init__()

        # Define channel transitions from the input to the deepest layer
        channels = [3, 64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList()

        for idx in range(1, len(channels)):
            self.layers.append(BiPathResBlock(channels[idx - 1], channels[idx], channels[idx], use_dilate_conv=use_dilate_conv))
            if idx != len(channels) - 1:
                self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, BiPathResBlock):  # Conditionally append feature maps following DoubleResBlock layers
                features.append(x)
        # # Include the final feature map post application of MaxPool2d layer for completeness of the hierarchical representations
        # features.append(x)

        return features


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=None):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


class SpatialCrossScaleIntegrator(nn.Module):
    def __init__(self, channel_sizes, target_dim=256, nhead=8, depth=2):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, target_dim, 1) for in_channels in channel_sizes
        ])
        # Positional encodings are not used for simplicity.
        # Implement positional encoding if necessary.
        self.transformer = SpatialTransformerEncoder(d_model=target_dim, nhead=nhead, num_layers=depth)
        self.reproj_layers = nn.ModuleList([
            nn.Conv2d(target_dim, out_channels, 1) for out_channels in channel_sizes
        ])
        self.h_w = [(56, 56), (28, 28), (14, 14)]  # Hardcoded sizes for outputs
        self.target_dim = target_dim

    def forward(self, inputs):
        # Project and flatten
        flatten_features = []
        for i, x in enumerate(inputs):
            x_proj = self.proj_layers[i](x)
            flatten_features.append(rearrange(x_proj, 'b c h w -> b (h w) c'))

        # Combine & Transform
        combined = torch.cat(flatten_features, dim=1)
        split_sizes = [h * w for h, w in self.h_w]
        transformed = self.transformer(combined)

        hs_ws_splits = torch.split(transformed, split_sizes, dim=1)
        outputs = []
        for i, split in enumerate(hs_ws_splits):
            split = rearrange(split, f'b (h w) c -> b c h w', h=self.h_w[i][0], w=self.h_w[i][1])
            out = self.reproj_layers[i](split)
            outputs.append(out)
        return outputs


class Encoder(nn.Module):
    def __init__(self, use_scale_integrator=True, use_dilate_conv=True):
        super(Encoder, self).__init__()
        self.Encoder = CNNEncoder(use_dilate_conv=use_dilate_conv)
        self.num_module = 3
        self.fusion_list = [256, 512, 1024]
        self.use_scale_integrator = use_scale_integrator
        self.scale_integrator = SpatialCrossScaleIntegrator(self.fusion_list) if use_scale_integrator else None

        self.fuser = nn.ModuleList()
        for i in range(self.num_module):
            self.fuser.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder(x)
        feature_trans = features[-3:]

        if self.scale_integrator is not None:
            feature_trans = self.scale_integrator(feature_trans)
        skips.extend(features[:2])
        for i in range(self.num_module):
            skip = self.fuser[i](torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips


class Decoder(nn.Module):
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=1, padding=1)
            self.conv2 = ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)
            self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
            self.in_channels = in_channels

        def forward(self, x, skip=None):
            x = self.upscale(x)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    class LastDecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=1, padding=1)
            self.conv2 = ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)
            self.in_channels = in_channels

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    def __init__(self):
        super().__init__()
        self.encoder_channels = [512, 256, 128, 64]
        self.first_block = self.LastDecoderBlock(1024, self.encoder_channels[0])
        self.blocks = nn.ModuleList([
            self.DecoderBlock(2 * in_ch, out_ch)
            for in_ch, out_ch in zip(self.encoder_channels[:-1], self.encoder_channels[1:])
        ])
        self.last_block = self.DecoderBlock(self.encoder_channels[-1], self.encoder_channels[-1])

    def forward(self, encoder_skips):
        x = encoder_skips[-1]  # Start from the deepest feature map
        x = self.first_block(x)
        for skip, block in zip(reversed(encoder_skips[:-1]), self.blocks):
            x = block(x, skip)
        x = self.last_block(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class MultiUnet(nn.Module):
    def __init__(self, num_classes, use_scale_integrator=True, use_dilate_conv=True):
        super().__init__()
        self.encoder = Encoder(use_scale_integrator=use_scale_integrator, use_dilate_conv=use_dilate_conv)
        self.decoder = Decoder()
        self.segmentation_head = SegmentationHead(in_channels=64, out_channels=num_classes, kernel_size=3,)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips = self.encoder(x)
        out = self.decoder(encoder_skips)
        logits = self.segmentation_head(out)
        return logits


if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 224, 224)  # Batch size 1, 3 channels, 64x64 image
    model = MultiUnet(num_classes=9, use_dilate_conv=False)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters()) / 1000000)
