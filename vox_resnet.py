import torch.nn as nn



class VoxRex(nn.Module):
    def __init__(self, in_channels):
        super(VoxRex, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + x


class VoxResNet(nn.Module):
    ''' base backend '''

    def __init__(self, in_channels, num_classes, ftrlen=[32, 64, 64, 64]):  # number of filters
        super(VoxResNet, self).__init__()
        ftr1, ftr2, ftr3, ftr4 = ftrlen
        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels, ftr1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr1, kernel_size=3, padding=1, bias=False)
        )
        # stage 2
        self.conv1_2 = nn.Sequential(
            nn.BatchNorm3d(ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)  # stride: (D, H, W)
        )
        self.voxres2 = VoxRex(ftr2)
        self.voxres3 = VoxRex(ftr2)
        # stage 3
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(ftr2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr2, ftr3, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        )
        self.voxres5 = VoxRex(ftr3)
        self.voxres6 = VoxRex(ftr3)
        # stage 4
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(ftr3),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr3, ftr4, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        )

        self.voxres8 = VoxRex(ftr4)
        self.voxres9 = VoxRex(ftr4)

    def forward(self, x):
        h1 = self.foward_stage1(x)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)  # input h4???
        return h4

    def foward_stage1(self, x):  # output of each stage goes to classifier
        h = self.conv1_1(x)
        return h

    def foward_stage2(self, x):
        h = self.conv1_2(x)
        h = self.voxres2(h)
        h = self.voxres3(h)
        return h

    def foward_stage3(self, x):
        h = self.conv4(x)
        h = self.voxres5(h)
        h = self.voxres6(h)
        return h

    def foward_stage4(self, x):
        h = self.conv7(x)
        h = self.voxres8(h)
        h = self.voxres9(h)
        return h