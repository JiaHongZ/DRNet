
import torch.nn as nn
import models.basicblock as B



# --------------------------------------------
# DRNet
# --------------------------------------------
class DRNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        super(DRNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = []
        for i in range(nb-2):
            m_body.append(B.conv(nc,nc,mode='C'+act_mode,bias=bias))
            m_body.append(B.ResBlock_CDC(in_channels=nc, out_channels=nc))
            m_body.append(B.conv(nc,nc,mode='R',bias=bias))  #改了之后变为残差块后面加R
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.model = B.sequential(m_head, *m_body, m_tail)



    def forward(self, x):
        n = self.model(x)
        return x-n

if __name__ == '__main__':
    res = DRNet()
    print(res)
