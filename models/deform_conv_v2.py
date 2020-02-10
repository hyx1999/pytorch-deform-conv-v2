import torch
import torch.nn as nn


class DeformConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 modulation=True):
        super(DeformConv2d, self).__init__()

        assert groups > 0
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert isinstance(kernel_size, int)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.modulation = modulation

        self.zero_padding = nn.ZeroPad2d(padding)

        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=(1, 1, kernel_size[0] * kernel_size[1]),
                              groups=groups,
                              bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        # nn.init.constant_(self.conv.weight, 1)

        self.offset_conv = nn.Conv2d(in_channels=in_channels,
                                     out_channels=kernel_size[0] * kernel_size[1] * 2 * groups,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)

        if modulation:
            self.m_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=kernel_size[0] * kernel_size[1] * groups,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=True)
            nn.init.constant_(self.m_conv.weight, 0)
            if self.m_conv.bias is not None:
                nn.init.constant_(self.m_conv.bias, 0)

    def calc_p0(self, batch, n, h, w, x_h, x_w, dtype):
        h_st = ((self.kernel_size[0] - 1) // 2) * self.dilation
        h_ed = x_h - (self.kernel_size[0] // 2) * self.dilation
        w_st = ((self.kernel_size[1] - 1) // 2) * self.dilation
        w_ed = x_w - (self.kernel_size[1] // 2) * self.dilation

        p0_x, p0_y = torch.meshgrid(
            torch.arange(h_st, h_ed, self.stride),
            torch.arange(w_st, w_ed, self.stride)
        )
        p0_x = p0_x.view(1, 1, h, w).repeat(batch, n, 1, 1)
        p0_y = p0_y.view(1, 1, h, w).repeat(batch, n, 1, 1)
        p0 = torch.cat((p0_x, p0_y), dim=1).repeat(1, self.groups, 1, 1).type(dtype)
        return p0

    def calc_pn(self, batch, n, h, w, dtype):
        h_st = -((self.kernel_size[0] - 1) // 2) * self.dilation
        h_ed = (self.kernel_size[0] // 2) * self.dilation + 1
        w_st = -((self.kernel_size[1] - 1) // 2) * self.dilation
        w_ed = (self.kernel_size[1] // 2) * self.dilation + 1

        pn_x, pn_y = torch.meshgrid(
            torch.arange(h_st, h_ed, self.dilation),
            torch.arange(w_st, w_ed, self.dilation)
        )
        pn_x = torch.flatten(pn_x).view(1, n, 1, 1)
        pn_y = torch.flatten(pn_y).view(1, n, 1, 1)
        pn = torch.cat((pn_x, pn_y), dim=1).repeat(batch, self.groups, h, w).type(dtype)
        return pn

    def calc_x_q(self, x, q, batch, n, h, w):
        channel = x.size(1)
        x_w = x.size(3)
        x = x.contiguous().view(batch, channel, -1)

        idx = q[..., :n] * x_w + q[..., n:]
        idx = idx.contiguous().view(batch, 1, h, w, n).repeat(1, channel, 1, 1, 1)
        idx = idx.contiguous().view(batch, channel, -1)

        return x.gather(dim=-1, index=idx).contiguous().view(batch, channel, h, w, n)

    '''
    def reshape(self, x, batch, channel, n, h, w):
        k = self.kernel_size[0]
        x_slices = [x[..., s:s+k].contiguous().view(batch, channel, h, w * k)
                    for s in range(0, n, k)]
        x = torch.cat(x_slices, dim=-1)
        x = x.contiguous().view(batch, channel, h * k, w * k)
        return x
    '''

    def forward(self, x: torch.Tensor):
        offset = self.offset_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
            m = m.contiguous().permute(0, 2, 3, 1)
        if self.padding > 0:
            x = self.zero_padding(x)

        dtype = x.data.type()
        batch = x.size(0)
        channel = x.size(1)

        n = self.kernel_size[0] * self.kernel_size[1]
        h = offset.size(2)
        w = offset.size(3)

        p0 = self.calc_p0(batch, n, h, w, x.size(2), x.size(3), dtype)
        pn = self.calc_pn(batch, n, h, w, dtype)
        p = offset + p0 + pn  # [batch, 2 * n * groups, h, w]

        p = p.contiguous().permute(0, 2, 3, 1)  # [batch, h, w, 2 * n * groups]

        all_q_tl = p.detach().floor()
        all_q_br = all_q_tl + 1

        delta_channel = channel // self.groups

        x_p = []
        for s in range(0, 2 * n * self.groups, 2 * n):
            q_tl = all_q_tl[..., s:s+2*n]
            q_br = all_q_br[..., s:s+2*n]
            q_tl = torch.cat((torch.clamp(q_tl[..., :n], 0, x.size(2) - 1),
                              torch.clamp(q_tl[..., n:], 0, x.size(3) - 1)), dim=3).long()
            q_br = torch.cat((torch.clamp(q_br[..., :n], 0, x.size(2) - 1),
                              torch.clamp(q_br[..., n:], 0, x.size(3) - 1)), dim=3).long()
            q_tr = torch.cat((q_tl[..., :n], q_br[..., n:]), dim=3)
            q_bl = torch.cat((q_br[..., :n], q_tl[..., n:]), dim=3)

            p_slice = p[..., s:s+2*n]
            p_slice = torch.cat((torch.clamp(p_slice[..., :n], 0, x.size(2) - 1),
                                 torch.clamp(p_slice[..., n:], 0, x.size(3) - 1)), dim=3)

            c = s // (2 * n)
            x_slice = x[:, c * delta_channel: (c + 1) * delta_channel, :, :]

            x_q_tl = self.calc_x_q(x_slice, q_tl, batch, n, h, w)
            x_q_tr = self.calc_x_q(x_slice, q_tr, batch, n, h, w)
            x_q_bl = self.calc_x_q(x_slice, q_bl, batch, n, h, w)
            x_q_br = self.calc_x_q(x_slice, q_br, batch, n, h, w)

            g_tl = (1 + (q_tl[..., :n].type_as(p) - p_slice[..., :n])) * \
                   (1 + (q_tl[..., n:].type_as(p) - p_slice[..., n:]))
            g_tr = (1 + (q_tr[..., :n].type_as(p) - p_slice[..., :n])) * \
                   (1 - (q_tr[..., n:].type_as(p) - p_slice[..., n:]))
            g_bl = (1 - (q_bl[..., :n].type_as(p) - p_slice[..., :n])) * \
                   (1 + (q_bl[..., n:].type_as(p) - p_slice[..., n:]))
            g_br = (1 - (q_br[..., :n].type_as(p) - p_slice[..., :n])) * \
                   (1 - (q_br[..., n:].type_as(p) - p_slice[..., n:]))

            x_p_slice = g_tl.unsqueeze(dim=1) * x_q_tl + \
                        g_tr.unsqueeze(dim=1) * x_q_tr + \
                        g_bl.unsqueeze(dim=1) * x_q_bl + \
                        g_br.unsqueeze(dim=1) * x_q_br

            if self.modulation:
                s2 = s // 2
                m_slice = m[..., s2:s2 + n]
                x_p_slice *= m_slice.unsqueeze(dim=1)

            x_p.append(x_p_slice)

        x_p = torch.cat(x_p, dim=1)
        output = self.conv(x_p).squeeze(dim=-1)
        return output
