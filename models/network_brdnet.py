
import torch.nn as nn
import models.basicblock as B
import torch


class BRDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='NR'):

        super(BRDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        ''' 
               在后面加了res
    
        '''
        # 上层
        m_head = B.conv(in_nc, nc, mode='C' + act_mode, bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(15)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.upNet = B.sequential(m_head, *m_body, m_tail)

        # 下层
        m_head_l = B.conv(in_nc, nc, mode='C' + act_mode, bias=bias)
        m_body_d_l = []
        for i in range(7):
            # m_body_d_l.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=dilation))
            m_body_d_l.append(B.conv(nc, nc, mode='D' + act_mode[-1], padding=2, bias=bias, dilation=2))
        # m_body_d_l.append(B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode, bias=bias))
        m_body_d_l.append(B.conv(nc, nc, mode='C' + act_mode, bias=bias))
        for i in range(6):
            m_body_d_l.append(B.conv(nc, nc, mode='D' + act_mode[-1], padding=2, bias=bias, dilation=2))
        m_body_d_l.append(B.conv(nc, nc, mode='C' + act_mode, bias=bias))
        m_tail_l = B.conv(nc, out_nc, mode='C', bias=bias)
        self.downNet = B.sequential(m_head_l, *m_body_d_l, m_tail_l)

        self.conv = B.conv(out_nc * 2, out_nc, mode='C', bias=bias)

    def forward(self, x):
        out1 = self.upNet(x)
        out2 = self.downNet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        return x - out
'''
        在前面加res
        
        # 加了3个res块和一个卷积
        self.conv1 = B.conv(in_nc, nc, mode='C', bias=bias)
        res = []
        for i in range(1):
            res.append(B.ResBlock_CDC(in_channels=nc, out_channels=nc))
            res.append(B.conv(nc, nc, mode='R', bias=bias))  # 改了之后变为残差块后面加R
        self.res = B.sequential(*res)
        # ---------
        # 上层
        m_head = B.conv(nc, nc, mode='C' + act_mode, bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(15)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.upNet = B.sequential(m_head, *m_body, m_tail)

        # 下层
        m_head_l = B.conv(nc, nc, mode='C' + act_mode, bias=bias)
        m_body_d_l = []
        for i in range(7):
            # m_body_d_l.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=dilation))
            m_body_d_l.append(B.conv(nc, nc, mode='D' + act_mode[-1], padding=2, bias=bias, dilation=2))
        # m_body_d_l.append(B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode, bias=bias))
        m_body_d_l.append(B.conv(nc, nc, mode='C' + act_mode, bias=bias))
        for i in range(6):
            m_body_d_l.append(B.conv(nc, nc, mode='D' + act_mode[-1], padding=2, bias=bias, dilation=2))
        m_body_d_l.append(B.conv(nc, nc, mode='C' + act_mode, bias=bias))
        m_tail_l = B.conv(nc, out_nc, mode='C', bias=bias)
        self.downNet = B.sequential(m_head_l, *m_body_d_l, m_tail_l)

        self.conv2 = B.conv(out_nc * 2, out_nc, mode='C', bias=bias)
'''


if __name__ == '__main__':
    res = ResBRD()
    print(res)
