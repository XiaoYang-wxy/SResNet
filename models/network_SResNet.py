import torch.nn as nn
import models.residual_block as B


"""
# --------------------------------------------
# SResNet
# --------------------------------------------
"""


class SResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, nr=4):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of basic blocks
        nr: total number of residual calculations
        # ------------------------------------
        """
        super(SResNet, self).__init__()
        # if interval != nb:
        #     body_number = int(nb/interval-2)
        #     head = B.residual_block(in_nc, nc, nc, interval)
        #     body = [B.residual_block(nc, nc, nc, interval)
        #             for _ in range(body_number)]
        #     tail = B.residual_block(nc, out_nc, nc, interval)
        #     self.model = B.sequential(head, *body, tail)

        self.nr = nr
        if nr == 0:
            head = B.basic_block(in_nc, nc)
            body = [B.basic_block(nc, nc) for _ in range(nb-2)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model = B.sequential(head, *body, tail)
        elif nr == 1:
            head = B.basic_block(in_nc, nc)
            body = [B.basic_block(nc, nc) for _ in range(nb-2)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model = B.sequential(head, *body, tail)
        elif nr == 2:
            interval = int(nb/nr)
            head = B.basic_block(in_nc, nc)
            body1 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            body2 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model1 = B.sequential(head, *body1)
            self.conv = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model2 = B.sequential(*body2, tail)
        elif nr == 4:
            interval = int(nb/nr)
            head = B.basic_block(in_nc, nc)
            body1 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            body2 = [B.basic_block(nc, nc) for _ in range(interval)]
            body3 = [B.basic_block(nc, nc) for _ in range(interval)]
            body4 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model1 = B.sequential(head, *body1)
            self.conv1 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model2 = B.sequential(*body2)
            self.conv2 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model3 = B.sequential(*body3)
            self.conv3 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model4 = B.sequential(*body4, tail)
        elif nr == 8:
            interval = int(nb/nr)
            head = B.basic_block(in_nc, nc)
            body1 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            body2 = [B.basic_block(nc, nc) for _ in range(interval)]
            body3 = [B.basic_block(nc, nc) for _ in range(interval)]
            body4 = [B.basic_block(nc, nc) for _ in range(interval)]
            body5 = [B.basic_block(nc, nc) for _ in range(interval)]
            body6 = [B.basic_block(nc, nc) for _ in range(interval)]
            body7 = [B.basic_block(nc, nc) for _ in range(interval)]
            body8 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model1 = B.sequential(head, *body1)
            self.conv1 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model2 = B.sequential(*body2)
            self.conv2 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model3 = B.sequential(*body3)
            self.conv3 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model4 = B.sequential(*body4)
            self.conv4 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model5 = B.sequential(*body5)
            self.conv5 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model6 = B.sequential(*body6)
            self.conv6 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model7 = B.sequential(*body7)
            self.conv7 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model8 = B.sequential(*body8, tail)
        elif nr == 16:
            interval = int(nb/nr)
            head = B.basic_block(in_nc, nc)
            body1 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            body2 = [B.basic_block(nc, nc) for _ in range(interval)]
            body3 = [B.basic_block(nc, nc) for _ in range(interval)]
            body4 = [B.basic_block(nc, nc) for _ in range(interval)]
            body5 = [B.basic_block(nc, nc) for _ in range(interval)]
            body6 = [B.basic_block(nc, nc) for _ in range(interval)]
            body7 = [B.basic_block(nc, nc) for _ in range(interval)]
            body8 = [B.basic_block(nc, nc) for _ in range(interval)]
            body9 = [B.basic_block(nc, nc) for _ in range(interval)]
            body10 = [B.basic_block(nc, nc) for _ in range(interval)]
            body11 = [B.basic_block(nc, nc) for _ in range(interval)]
            body12 = [B.basic_block(nc, nc) for _ in range(interval)]
            body13 = [B.basic_block(nc, nc) for _ in range(interval)]
            body14 = [B.basic_block(nc, nc) for _ in range(interval)]
            body15 = [B.basic_block(nc, nc) for _ in range(interval)]
            body16 = [B.basic_block(nc, nc) for _ in range(interval-1)]
            tail = nn.Conv2d(nc, out_nc, kernel_size=3, padding=1)
            self.model1 = B.sequential(head, *body1)
            self.conv1 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model2 = B.sequential(*body2)
            self.conv2 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model3 = B.sequential(*body3)
            self.conv3 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model4 = B.sequential(*body4)
            self.conv4 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model5 = B.sequential(*body5)
            self.conv5 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model6 = B.sequential(*body6)
            self.conv6 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model7 = B.sequential(*body7)
            self.conv7 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model8 = B.sequential(*body8)
            self.conv8 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model9 = B.sequential(*body9)
            self.conv9 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model10 = B.sequential(*body10)
            self.conv10 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model11 = B.sequential(*body11)
            self.conv11 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model12 = B.sequential(*body12)
            self.conv12 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model13 = B.sequential(*body13)
            self.conv13 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model14 = B.sequential(*body14)
            self.conv14 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model15 = B.sequential(*body15)
            self.conv15 = nn.Conv2d(in_nc, nc, kernel_size=1)
            self.model16 = B.sequential(*body16, tail)

    def forward(self, x):
        if self.nr == 0:
            y = self.model(x)
            return y
        elif self.nr == 1:
            y = self.model(x)
            return x-y
        elif self.nr == 2:
            y1 = self.model1(x)
            z = self.conv(x)
            y2 = self.model2(z-y1)
            return x-y2
        elif self.nr == 4:
            y1 = self.model1(x)
            z = self.conv1(x)
            y2 = self.model2(z-y1)
            z = self.conv2(x)
            y1 = self.model3(z-y2)
            z = self.conv3(x)
            y2 = self.model4(z-y1)
            return x-y2
        elif self.nr == 8:
            y1 = self.model1(x)
            z = self.conv1(x)
            y2 = self.model2(z-y1)
            z = self.conv2(x)
            y1 = self.model3(z-y2)
            z = self.conv3(x)
            y2 = self.model4(z-y1)
            z = self.conv4(x)
            y1 = self.model5(z-y2)
            z = self.conv5(x)
            y2 = self.model6(z-y1)
            z = self.conv6(x)
            y1 = self.model7(z-y2)
            z = self.conv7(x)
            y2 = self.model8(z-y1)
            return x-y2
        elif self.nr == 16:
            y1 = self.model1(x)
            z = self.conv1(x)
            y2 = self.model2(z-y1)
            z = self.conv2(x)
            y1 = self.model3(z-y2)
            z = self.conv3(x)
            y2 = self.model4(z-y1)
            z = self.conv4(x)
            y1 = self.model5(z-y2)
            z = self.conv5(x)
            y2 = self.model6(z-y1)
            z = self.conv6(x)
            y1 = self.model7(z-y2)
            z = self.conv7(x)
            y2 = self.model8(z-y1)
            z = self.conv8(x)
            y1 = self.model9(z-y2)
            z = self.conv9(x)
            y2 = self.model10(z-y1)
            z = self.conv10(x)
            y1 = self.model11(z-y2)
            z = self.conv11(x)
            y2 = self.model12(z-y1)
            z = self.conv12(x)
            y1 = self.model13(z-y2)
            z = self.conv13(x)
            y2 = self.model14(z-y1)
            z = self.conv14(x)
            y1 = self.model15(z-y2)
            z = self.conv15(x)
            y2 = self.model16(z-y1)
            return x-y2
