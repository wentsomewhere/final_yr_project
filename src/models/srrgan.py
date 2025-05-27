import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class TextAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(TextAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        att = self.attention(x)
        return residual + x * att

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + 4 * growth_rate, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attention = TextAttentionBlock(nf)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.attention(x5 * 0.2 + x)

class RRDB(nn.Module):
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf)
        self.rdb2 = ResidualDenseBlock(nf)
        self.rdb3 = ResidualDenseBlock(nf)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class Upsampler(nn.Module):
    def __init__(self, nf, scale):
        super(Upsampler, self).__init__()
        self.conv = nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=23):
        super(Generator, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(
            *[ResidualDenseBlock(nf) for _ in range(nb)]
        )
        self.conv_up = nn.Sequential(
            Upsampler(nf, 2),
            Upsampler(nf, 2)
        )
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(feat)
        up = self.conv_up(body)
        out = self.conv_last(up)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_nc=1, nf=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(nf, nf*2, 3, padding=1)
        self.conv4 = nn.Conv2d(nf*2, nf*2, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(nf*2, nf*4, 3, padding=1)
        self.conv6 = nn.Conv2d(nf*4, nf*4, 4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(nf*4, nf*8, 3, padding=1)
        self.conv8 = nn.Conv2d(nf*8, nf*8, 4, stride=2, padding=1)
        self.conv9 = nn.Conv2d(nf*8, nf*8, 3, padding=1)
        self.conv10 = nn.Conv2d(nf*8, nf*8, 3, padding=1)
        self.conv11 = nn.Conv2d(nf*8, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = self.lrelu(self.conv10(x))
        x = self.conv11(x)
        return x

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Convert grayscale to RGB by repeating the channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)

class SRRGAN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=23, scale=4):
        super(SRRGAN, self).__init__()
        self.generator = Generator(in_nc, out_nc, nf, nb)
        self.discriminator = Discriminator(in_nc)
        self.feature_extractor = VGGFeatureExtractor()
        
        # Loss weights
        self.lambda_perceptual = 1.0
        self.lambda_ocr = 1.0
        self.lambda_pixel = 0.01
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.generator(x)

    def get_discriminator_loss(self, real, fake):
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake.detach())
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        return d_loss_real + d_loss_fake

    def get_generator_loss(self, real, fake):
        # Adversarial loss
        fake_pred = self.discriminator(fake)
        g_loss_adv = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        
        # Perceptual loss
        real_feat = self.feature_extractor(real)
        fake_feat = self.feature_extractor(fake)
        g_loss_percep = F.mse_loss(fake_feat, real_feat)
        
        # Pixel loss
        g_loss_pixel = F.mse_loss(fake, real)
        
        return g_loss_adv + self.lambda_perceptual * g_loss_percep + self.lambda_pixel * g_loss_pixel 