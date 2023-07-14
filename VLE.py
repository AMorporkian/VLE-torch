import torch.nn as nn
import torch.nn.functional as F
import torch
from conv_lstm import ConvLSTM

class Encoder(nn.Module):
    def __init__(self, in_channels=6, depth=4, width=16):
        super(Encoder, self).__init__()

        self.depth = depth
        self.width = width
        self.encoder = nn.ModuleList()

        for i in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else width * 2**i,
                        width * 2 ** (i + 1),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        width * 2 ** (i + 1),
                        width * 2 ** (i + 1),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )

    def forward(self, x):
        for encoder in self.encoder:
            x = encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3, depth=4, width=16):
        super(Decoder, self).__init__()

        self.depth = depth
        self.width = width
        self.decoder = nn.ModuleList()

        for i in range(depth, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        width * 2**i, width * 2 ** (i - 1), kernel_size=2, stride=2
                    ),
                    nn.Conv2d(
                        width * 2 ** (i - 1),
                        width * 2 ** (i - 1),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        width * 2 ** (i - 1),
                        width * 2 ** (i - 1),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(width, out_channels, kernel_size=3, stride=1, padding=0),
                nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=0,),
                #nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=0),
                #nn.Softmax2d(),
            )
        )

    def forward(self, x):
        for decoder in self.decoder:
            x = decoder(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=hidden_size,
                                kernel_size=3,
                                padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x + residual
    
class LSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, num_layers):
        super(LSTMBlock, self).__init__()
        self.convlstm = ConvLSTM(input_dim=in_channels,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 num_layers=num_layers,
                                 batch_first=True,)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Add an extra dimension for the sequence length
        mask, _ = self.convlstm(x.unsqueeze(1))
        mask = mask[0].squeeze(1)
        mask = self.sigmoid(mask)
        #mask = (mask > 0.5).float()
        return x, mask

    
class VLE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, max_tokens=5, depth=4, width=16, image_size=512):
        super(VLE, self).__init__()
        self.encoder = Encoder(in_channels, depth, width)
        self.decoder = Decoder(out_channels, depth, width)
        self.residual_block = ResidualBlock(out_channels, out_channels, 64)
        self.lstm_block = LSTMBlock(out_channels, [64, 128, 1], (3,3), 3)
    
    def forward(self, residual):
        # Pass the residual and the current mask through the residual block
        residual = self.residual_block(residual)
        residual, mask = self.lstm_block(residual)
        masked_residual = residual * mask
        # Encode the masked residual into a token, conditioned on the mask
        token = self.encoder(masked_residual)

        # Decode the token into a partial reconstruction
        partial_reconstruction = self.decoder(token)

        return partial_reconstruction, mask