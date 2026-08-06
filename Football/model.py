import torch
import torch.nn as nn

from config import FEATURE_DIM, NUM_CLASSES, HIDDEN, DILATIONS, DROPOUT, WINDOW

class ResidualBlock(nn.Module):

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation= dilation
        )

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.project = nn.Conv1d(channels, channels, kernel_size= 1)

    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.project(h)
        return x + h


class Spotter(nn.Module):
    def __init__(
            self,
            feature_dim: int = FEATURE_DIM,
            hidden: int = HIDDEN,
            num_classes: int = NUM_CLASSES,
            dilations =  DILATIONS,
            dropout: float = DROPOUT
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, d ,dropout) for d in dilations]
        )
        self.head = nn.Conv1d(hidden, num_classes, kernel_size=1)

        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        x = x.transpose(1, 2)

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        logits = self.head(h)

        return logits.transpose(1,2)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(DILATIONS)



if __name__ == "__main__":
    model = Spotter()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(4, WINDOW, FEATURE_DIM)
    y = model(x)
    print(f"params:          {n_params:,}")
    print(f"receptive field: {model.receptive_field} steps "
          f"({model.receptive_field / 2:.0f}s at 2fps)")
    print(f"input:           {tuple(x.shape)}")
    print(f"output:          {tuple(y.shape)}")

