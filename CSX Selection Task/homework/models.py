import torch
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:  # For downsampling or projection shortcut
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x


class CNNClassifier(torch.nn.Module):
    def __init__(
        self,
        layers=[16, 32, 64, 128],
        n_input_channels=3,
        n_output_channels=6,
        kernel_size=7,
    ):
        super(CNNClassifier, self).__init__()

        L = []
        self.c = n_input_channels
        L.append(
            torch.nn.Conv2d(
                self.c, 8, kernel_size=kernel_size, stride=2, padding=3, bias=False
            )
        )
        L.append(torch.nn.BatchNorm2d(8))
        L.append(torch.nn.ReLU(inplace=True))
        L.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.c = 8

        for l in layers:
            # Two residual blocks per layer
            L.append(ResidualBlock(self.c, l, kernel_size=3))
            L.append(ResidualBlock(l, l, kernel_size=3))
            L.append(torch.nn.Dropout2d(p=0.2))  # Dropout layer
            self.c = l

        self.network = torch.nn.Sequential(*L)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.c, n_output_channels, bias=True)

    def forward(self, x):
        # Apply input normalization
        x = F.normalize(x, p=2.0, dim=1, eps=1e-12)
        out = self.network(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class Tripleblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tripleblock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FCN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=5, channels=[32, 64, 128, 256]):
        super(FCN, self).__init__()
        self.down = torch.nn.ModuleList()
        self.up = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Downward path
        for channel in channels:
            self.down.append(Tripleblock(in_channels, channel))
            in_channels = channel

        # Upward path
        for channel in channels[::-1]:
            self.up.append(torch.nn.ConvTranspose2d(channel * 2, channel, 2, 2))
            self.up.append(Tripleblock(channel * 2, channel))

        # Latent part
        self.latent = Tripleblock(channels[-1], channels[-1] * 2)

        # Last layer
        self.lastlayer = torch.nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        # Apply input normalization
        x = F.normalize(x, p=2.0, dim=1, eps=1e-12)

        skips = []
        b, _, h, w = x.shape
        x = F.interpolate(x, size=(256, 256))

        for down in self.down:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.latent(x)
        skips = skips[::-1]

        for i in range(0, len(self.up), 2):
            x = self.up[i](x)
            skip_con = skips[i // 2]
            if x.shape != skip_con.shape:
                x = F.resize(x, size=skip_con.shape[2:])
            concat_skip = torch.cat((skip_con, x), 1)
            x = self.up[i + 1](concat_skip)

        x = self.lastlayer(x)
        x = F.interpolate(x, size=(h, w))

        return x


model_factory = {
    "cnn": CNNClassifier,
    "fcn": FCN,
}


def save_model(model):
    from os import path

    from torch import save

    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(
                model.state_dict(),
                path.join(path.dirname(path.abspath(__file__)), f"{n}.th"),
            )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from os import path

    from torch import load

    r = model_factory[model]()
    r.load_state_dict(
        load(
            path.join(path.dirname(path.abspath(__file__)), "%s.th" % model),
            map_location="cpu",
        )
    )
    return r
