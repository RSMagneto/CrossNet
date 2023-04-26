import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from thop import profile

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, inter_size=32, output_size=128, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, inter_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(inter_size, inter_size*2, kernel_size, stride, padding=0, bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(inter_size*2, output_size, kernel_size, stride, padding=0, bias=True)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return out1, out2

class Encoder(torch.nn.Module):
    def __init__(self, input_size=256, kernel_size=3, stride=1):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Awaremodel(torch.nn.Module):
    def __init__(self, input_size=256, kernel_size=1, stride=1):
        super(Awaremodel, self).__init__()

        self.conv_gamma = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_gamma_kernel1 = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_gamma_kernel3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, 3, stride, padding=0, bias=True)
        )
        self.conv_gamma_tail = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
        )

        self.conv_beta = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_beta_kernel1 = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_beta_kernel3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, 3, stride, padding=0, bias=True)
        )
        self.conv_beta_tail = nn.Sequential(
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        gamma = self.conv_gamma(x)
        gamma_kernel1 = self.conv_gamma_kernel1(gamma)
        gamma_kernel3 = self.conv_gamma_kernel3(gamma)
        gamma_out = self.conv_gamma_tail(gamma_kernel1*gamma_kernel3+gamma)

        beta = self.conv_beta(x)
        beta_kernel1 = self.conv_beta_kernel1(beta)
        beta_kernel3 = self.conv_beta_kernel3(beta)
        beta_out = self.conv_beta_tail(beta_kernel1*beta_kernel3+beta)
        return gamma_out, beta_out


class Decoder(torch.nn.Module):
    def __init__(self, input_size=256, kernel_size=3, stride=1):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True)
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size*2, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True)
        )
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(input_size, input_size//2, kernel_size, stride, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1,x2],1))
        out = self.conv4(x3)
        return out

class UpBlock(nn.Module):
    def __init__(self, channels=256):
        super(UpBlock, self).__init__()
        self.conv_1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        x1 = self.conv_1(x1)
        x2 = F.interpolate(x1, scale_factor=2, mode='bicubic', align_corners=True)
        x2 = self.conv_2(x2)
        return x2

class DownBlock(nn.Module):
    def __init__(self, channels=256):
        super(DownBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(128),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        )
        self.conv_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(64),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        return x2

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.encoder_ms_full = ConvBlock(4)
        self.encoder_pan_full = ConvBlock(1)
        self.encoder_full = Encoder()
        self.aware_full = Awaremodel()
        self.decoder_full = Decoder()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 64, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 32, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 4, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, full_pan, full_ms):
        full_ms = F.interpolate(full_ms, size=(full_pan.shape[2], full_pan.shape[3]), mode='bilinear')
        ms_inter, ms_f = self.encoder_ms_full(full_ms)
        pan_inter, pan_f = self.encoder_pan_full(full_pan)
        out_f1 = torch.cat([ms_f, pan_f], 1)
        en_full = self.encoder_full(out_f1)
        gamma_f, beta_f = self.aware_full(out_f1)
        # print((en_full * gamma_f + beta_f).shape) # 2, 256, 256, 256
        FF_HRMS = self.decoder_full(en_full * gamma_f + beta_f)
        FF_HRMS = self.conv(FF_HRMS)
        return FF_HRMS, gamma_f, beta_f, en_full

class ReduceModel(nn.Module):
    def __init__(self):
        super(ReduceModel, self).__init__()
        self.encoder_ms_reduce = ConvBlock(4)
        self.encoder_pan_reduce = ConvBlock(1)
        self.encoder_reduce = Encoder()
        self.aware_reduce = Awaremodel()
        self.decoder_reduce = Decoder()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 64, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 32, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 4, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, redu_pan, redu_ms):
        redu_ms = F.interpolate(redu_ms, size=(redu_pan.shape[2], redu_pan.shape[3]), mode='bilinear')
        ms_inter, ms_r = self.encoder_ms_reduce(redu_ms)
        pan_inter, pan_r = self.encoder_pan_reduce(redu_pan)
        out_s1 = torch.cat([ms_r, pan_r], 1)
        en_reduce = self.encoder_reduce(out_s1)
        gamma_s, beta_s = self.aware_reduce(out_s1)
        SF_HRMS = self.decoder_reduce(en_reduce * gamma_s + beta_s)
        SF_HRMS = self.conv(SF_HRMS)
        return SF_HRMS, gamma_s, beta_s, en_reduce


class CrossModel(nn.Module):
    def __init__(self):
        super(CrossModel, self).__init__()
        self.decoder_cross_full = Decoder()
        self.decoder_cross_reduce = Decoder()
        self.upblock_gamma = UpBlock()
        self.upblock_beta = UpBlock()
        self.downblock_gamma = DownBlock()
        self.downblock_beta = DownBlock()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 64, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 32, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 4, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 64, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 32, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 4, 3, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, fulllist, reducelist):
        gamma_f, beta_f, en_full = fulllist[1], fulllist[2], fulllist[3]
        gamma_r, beta_r, en_reduce = reducelist[1], reducelist[2], reducelist[3]
        gamma_f_c = self.upblock_gamma(gamma_r)
        beta_f_c = self.upblock_beta(beta_r)
        gamma_r_c = self.downblock_gamma(gamma_f)
        beta_r_c = self.downblock_beta(beta_f)
        CF_HRMS = self.decoder_cross_full(en_full * gamma_f_c + beta_f_c)
        CR_HRMS = self.decoder_cross_reduce(en_reduce * gamma_r_c + beta_r_c)
        CF_HRMS = self.conv1(CF_HRMS)
        CR_HRMS = self.conv2(CR_HRMS)
        return CF_HRMS, CR_HRMS

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fullmodel = FullModel()
        self.reducemodel = ReduceModel()
        self.crossmodel = CrossModel()
        self.pool = nn.AdaptiveAvgPool2d(16)
        self.domain_classifier_full = nn.Sequential(
            nn.Linear(16*16*256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier_reduce = nn.Sequential(
            nn.Linear(16*16*256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, full_pan, full_ms, redu_pan, redu_ms, alpha):
        fulllist = self.fullmodel(full_pan, full_ms)
        reducelist = self.reducemodel(redu_pan, redu_ms)
        CF_HRMS, CR_HRMS = self.crossmodel(fulllist, reducelist)
        FF_HRMS = fulllist[0]
        FR_HRMS = reducelist[0]
        full_domain_output = self.domain_classifier_full(ReverseLayerF.apply(self.pool(fulllist[-1]).view(-1, 16*16*256), alpha))
        redu_domain_output = self.domain_classifier_reduce(ReverseLayerF.apply(self.pool(reducelist[-1]).view(-1, 16*16*256), alpha))

        return FF_HRMS, FR_HRMS, CF_HRMS, CR_HRMS, full_domain_output, redu_domain_output

if __name__ == '__main__':


    tnt = Model()
    reducedmodel = ReduceModel()
    full_ms = torch.randn(2, 4, 64, 64)
    full_pan = torch.randn(2, 1, 256, 256)
    reduce_ms = torch.randn(2, 4, 16, 16)
    reduce_pan = torch.randn(2, 1, 64, 64)
    FF_HRMS, FR_HRMS, CF_HRMS, CR_HRMS, full_domain_output, redu_domain_output = tnt(full_pan,full_ms,reduce_pan,reduce_ms, 0)
    print('ok')
