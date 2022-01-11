import torch
import torch.nn as nn
from fastai.layers import ConvLayer, PixelShuffle_ICNR, ResBlock
from torch import Tensor

arch_name = "cloudnet_plus"


class ContractionBlock(nn.Module):
    def __init__(self, ni, residual=False):
        super().__init__()

        if residual:
            self.conv1 = ResBlock(ni, ks=3, stride=1, padding=1, bias=True)
            self.conv2 = ConvLayer(ni, 2 * ni, ks=1, stride=1, padding=0, bias=True)
            self.conv3 = ResBlock(2 * ni, ks=3, stride=1, padding=1, bias=True)

            self.conv4 = ResBlock(ni, ks=1, stride=1, padding=0, bias=True)
        else:
            self.conv1 = ConvLayer(ni, ni, ks=3, stride=1, padding=1, bias=True)
            self.conv2 = ConvLayer(ni, 2 * ni, ks=1, stride=1, padding=0, bias=True)
            self.conv3 = ConvLayer(2 * ni, 2 * ni, ks=3, stride=1, padding=1, bias=True)

            self.conv4 = ConvLayer(ni, ni, ks=1, stride=1, padding=0, bias=True)

        # TODO test replacing maxpool in original cloudNet with convolution downsampling
        # self.conv5 = ResBlock(2 * ni)
        self.layer5 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        x1 = self.conv1(inp)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        x2 = self.conv4(inp)
        x2 = torch.cat([x2, inp], dim=1)
        x = x1 + x2
        # x = self.conv5(x)
        return self.layer5(x)


#
#
#
class FeedforwardBlock(nn.Module):
    def __init__(self, ni, n, residual=False):
        super().__init__()

        self.poolings = []

        for i in range(n):

            s = 2 ** (n - i)

            layer = nn.MaxPool2d(3, stride=s, padding=1)

            # TODO test replacing maxpool in original cloudNet with convolution downsampling
            # self.layer5 = ConvLayer(2*ni, 2*ni, ks=3, stride=2, padding=1, bias=True)

            self.poolings.append(layer)
            if residual:
                self.conv = ResBlock(ni, ks=3, stride=1, padding=1, bias=True)
            else:
                self.conv = ConvLayer(ni, ni, ks=3, stride=1, padding=1, bias=True)

    def forward(self, inp: Tensor) -> Tensor:

        n = len(self.poolings)

        output = inp[n]

        for i in reversed(range(n)):
            s = n - i

            x = inp[i]
            concat_list = []
            for p in range(2 ** s):
                concat_list.append(x)

            x = torch.cat(concat_list, dim=1)
            #            x = self.relu(x)
            x = self.poolings[i](x)

            output = output + x

        output = self.conv(output)
        return output


class ExpandingBlock(nn.Module):
    def __init__(self, ni, residual=False):
        super().__init__()

        # TODO test replacing transposed convolution in original cloudNet with subpixel convolution:
        #        self.upsmpl = nn.ConvTranspose2d(2 * ni, ni, 3, stride=2, padding=1, output_padding=1)
        self.upsmpl = PixelShuffle_ICNR(2 * ni, ni, scale=2)

        self.conv1 = ConvLayer(2 * ni, ni, ks=3, stride=1, padding=1, bias=True)

        if residual:
            self.conv2 = ResBlock(ni, ks=3, stride=1, padding=1, bias=True)
            self.conv3 = ResBlock(ni, ks=3, stride=1, padding=1, bias=True)
        else:
            self.conv2 = ConvLayer(ni, ni, ks=3, stride=1, padding=1, bias=True)
            self.conv3 = ConvLayer(ni, ni, ks=3, stride=1, padding=1, bias=True)

    def forward(self, *inp):
        (pE_input, FF_input, C_input) = inp

        x = y = FF_input

        if pE_input is not None:
            x = y = self.upsmpl(pE_input)

        x = torch.cat([FF_input, x], axis=1)

        x = self.conv2(self.conv1(x))

        return self.conv3(C_input + x + y)


class UpsamplingBlock(nn.Module):
    def __init__(self, ni, nout, n, residual=False):
        super().__init__()

        self.n = n

        self.upsmpl = nn.Upsample(scale_factor=(2 ** (n + 2), 2 ** (n + 2)))
        # self.upsmpl = PixelShuffle_ICNR(ni, ni, scale=2 ** (n + 2))

        #        if residual:
        #            self.conv2 = ConvLayer(nout, nout, ks=1, stride=1, padding=0)
        self.conv1 = ConvLayer(ni, nout, ks=1, stride=1, padding=0)

    def forward(self, inp: Tensor) -> Tensor:
        x = self.upsmpl(inp)
        x = self.conv1(x)
        return x


class CloudNetPlus(nn.Module):
    def __init__(self, input_channels=4, n_classes=1, inception_depth=6, residual=False):

        super().__init__()

        self.kindred = []

        # ich - channels in initial input
        # is - width of initial input
        #
        # example input: 4@198x198

        # contraction blocks
        # INPUT of Nth block
        #   sch*2**n @ is*2**-n X is*2**-n
        #   i.e for n = 2, 16 @ 48x48
        # output of Nth block
        #   sch*2**(n+1) @ is*2**-(n+1) X is*2**-(n+1)
        #   i.e for n = 2, 32 @ 24x24
        self.c_blocks = []

        for i in range(0, inception_depth):
            self.c_blocks.append(ContractionBlock(input_channels * 2 ** i, residual=residual))

        # feedforward blocks
        # INPUT of Nth block
        #   sch*2**n @ is*2**-n X is*2**-n
        #   i.e for n = 2, 16 @ 48x48
        # output of Nth block
        #   sch*2**(n+1) @ is*2**-(n+1) X is*2**-(n+1)
        #   i.e for n = 2, 32 @ 24x24
        self.ff_blocks = []
        for i in range(1, inception_depth):
            self.ff_blocks.append(FeedforwardBlock(input_channels * 2 ** (i + 1), i, residual=residual))

        self.e_blocks = []
        for i in range(1, inception_depth):
            self.e_blocks.append(ExpandingBlock(input_channels * 2 ** (i + 1), residual=residual))

        self.u_blocks = []
        for i in range(0, inception_depth - 1):
            self.u_blocks.append(UpsamplingBlock(input_channels * 2 ** (i + 2), n_classes, i))
        self.u_blocks.append(
            UpsamplingBlock(input_channels * 2 ** (inception_depth), n_classes, inception_depth - 2, residual=residual)
        )

        self.conv = ConvLayer(n_classes, n_classes, ks=3, stride=1, padding=1, act_cls=None)

        self.kindred.extend(self.c_blocks)
        self.kindred.extend(self.ff_blocks)
        self.kindred.extend(self.e_blocks)
        self.kindred.extend(self.u_blocks)

        self.kindred.append(self.conv)

    def forward(self, inp: Tensor) -> Tensor:

        c_acts = []

        for i, c_block in enumerate(self.c_blocks):
            if i == 0:
                c_act = c_block(inp)
            else:
                c_act = c_block(c_act)
            c_acts.append(c_act)

        ff_acts = []
        ff_input = [c_acts[0]]
        for i, ff_block in enumerate(self.ff_blocks):
            ff_input.append(c_acts[i + 1])

            ff_act = ff_block(ff_input.copy())
            ff_acts.append(ff_act)

        e_acts: list = []
        e_act = None  # sibling E activation
        for i in reversed(range(len(self.e_blocks))):
            e_block = self.e_blocks[i]
            c_act = c_acts[i + 1]
            ff_act = ff_acts[i]

            e_act = e_block(e_act, ff_act, c_act)
            e_acts.insert(0, e_act)

        n = len(self.u_blocks)

        u_sum = self.u_blocks[n - 1](c_acts[n - 1])

        for i in range(0, len(self.u_blocks) - 1):
            u_sum = u_sum + self.u_blocks[i](e_acts[i])

        return self.conv(u_sum)

    def children(self):
        r"""Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for module in self.kindred:
            yield module
