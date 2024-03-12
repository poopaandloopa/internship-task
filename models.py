from torch import nn


class ConvModel(nn.Module):
    def __init__(self, n_classes):
        super(ConvModel, self).__init__()
        self.model = nn.Sequential(nn.ConstantPad1d(padding=8,
                                                    value=0),
                                   nn.Conv1d(in_channels=1,
                                             out_channels=32,
                                             kernel_size=16,
                                             stride=2,
                                             bias=True),  # 32x1x1751 -> 32x32x876
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=876),
                                   nn.Dropout1d(p=0.2),

                                   nn.ConstantPad1d(padding=8,
                                                    value=0),
                                   nn.Conv1d(in_channels=32,
                                             out_channels=64,
                                             kernel_size=16,
                                             stride=2,
                                             bias=True),  # 32x32x876 -> 32x64x439
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=439),
                                   nn.Dropout1d(p=0.2),

                                   nn.ConstantPad1d(padding=8,
                                                    value=0),
                                   nn.Conv1d(in_channels=64,
                                             out_channels=128,
                                             kernel_size=16,
                                             stride=2,
                                             bias=True),  # 32x64x439 -> 32x128x220
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=220),
                                   nn.Dropout1d(p=0.2),

                                   nn.ConstantPad1d(padding=8,
                                                    value=0),
                                   nn.Conv1d(in_channels=128,
                                             out_channels=256,
                                             kernel_size=16,
                                             stride=2,
                                             bias=True),  # 32x128x220 -> 32x256x111
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=111),
                                   nn.Dropout1d(p=0.2),

                                   nn.ConstantPad1d(padding=8,
                                                    value=0),
                                   nn.Conv1d(in_channels=256,
                                             out_channels=512,
                                             kernel_size=16,
                                             stride=2,
                                             bias=True),  # 32x256x111 -> 32x512x56
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=56),
                                   nn.Dropout1d(p=0.2),

                                   nn.Conv1d(in_channels=512,
                                             out_channels=n_classes,
                                             kernel_size=56,
                                             stride=1,
                                             bias=True),
                                   nn.Flatten())

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], 1, x.shape[1]))


class ConvModelResNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvModelResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                             out_channels=64,
                                             kernel_size=21,
                                             stride=2,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=866))  # 32x1x1751 -> 32x64x866
        self.conv_block1 = nn.Sequential(nn.Conv1d(in_channels=64,
                                                   out_channels=64,
                                                   kernel_size=21,
                                                   stride=1,
                                                   padding=10,
                                                   bias=True),
                                         nn.ReLU(),
                                         nn.LayerNorm(normalized_shape=866))
        self.conv_blocks1 = nn.ModuleList([self.conv_block1 for _ in range(2)])

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64,
                                             out_channels=128,
                                             kernel_size=21,
                                             stride=2,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=423))
        self.conv_block2 = nn.Sequential(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=21,
                                                   stride=1,
                                                   padding=10,
                                                   bias=True),
                                         nn.ReLU(),
                                         nn.LayerNorm(normalized_shape=423))
        self.conv_blocks2 = nn.ModuleList([self.conv_block2 for _ in range(2)])

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128,
                                             out_channels=256,
                                             kernel_size=21,
                                             stride=2,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=202))
        self.conv_block3 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                   out_channels=256,
                                                   kernel_size=21,
                                                   stride=1,
                                                   padding=10,
                                                   bias=True),
                                         nn.ReLU(),
                                         nn.LayerNorm(normalized_shape=202))
        self.conv_blocks3 = nn.ModuleList([self.conv_block3 for _ in range(2)])

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=256,
                                             out_channels=512,
                                             kernel_size=21,
                                             stride=2,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.LayerNorm(normalized_shape=91))
        self.conv_block4 = nn.Sequential(nn.Conv1d(in_channels=512,
                                                   out_channels=512,
                                                   kernel_size=21,
                                                   stride=1,
                                                   padding=10,
                                                   bias=True),
                                         nn.ReLU(),
                                         nn.LayerNorm(normalized_shape=91))
        self.conv_blocks4 = nn.ModuleList([self.conv_block4 for _ in range(4)])

        self.conv_final = nn.Conv1d(in_channels=512,
                                    out_channels=n_classes,
                                    kernel_size=91,
                                    stride=1,
                                    padding=0,
                                    bias=True)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])

        x = self.conv1(x)
        cache = [x]
        for i, block in enumerate(self.conv_blocks1, 1):
            x = block(x)
            cache.append(x)
            if i % 2 == 0:
                x = cache[i - 2] + x

        x = self.conv2(x)
        cache = [x]
        for i, block in enumerate(self.conv_blocks2, 1):
            x = block(x)
            cache.append(x)
            if i % 2 == 0:
                x = cache[i - 2] + x

        x = self.conv3(x)
        cache = [x]
        for i, block in enumerate(self.conv_blocks3, 1):
            x = block(x)
            cache.append(x)
            if i % 2 == 0:
                x = cache[i - 2] + x

        x = self.conv4(x)
        cache = [x]
        for i, block in enumerate(self.conv_blocks4, 1):
            x = block(x)
            cache.append(x)
            if i % 2 == 0:
                x = cache[i - 2] + x

        return self.conv_final(x).squeeze()
