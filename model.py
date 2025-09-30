import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """CSRNet model for crowd counting.

    Supports two frontends:
    - 'vgg16_bn' (default): uses torchvision.models.vgg16_bn with frontend cut at index 33
    - 'vgg16': uses torchvision.models.vgg16 (no batch norm) with frontend cut at index 23
    """

    def __init__(self, load_weights: bool = True, variant: str = "vgg16_bn"):
        super().__init__()
        assert variant in {"vgg16_bn", "vgg16"}, "variant must be 'vgg16_bn' or 'vgg16'"

        # Handle torchvision API changes gracefully
        vgg = None
        if variant == "vgg16_bn":
            if load_weights:
                try:
                    # Newer torchvision API (>=0.13)
                    from torchvision.models import VGG16_BN_Weights
                    vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
                except Exception:
                    vgg = models.vgg16_bn(pretrained=True)
            else:
                try:
                    vgg = models.vgg16_bn(weights=None)
                except Exception:
                    vgg = models.vgg16_bn(pretrained=False)
            frontend_cut = 33
        else:  # variant == 'vgg16'
            if load_weights:
                try:
                    from torchvision.models import VGG16_Weights
                    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                except Exception:
                    vgg = models.vgg16(pretrained=True)
            else:
                try:
                    vgg = models.vgg16(weights=None)
                except Exception:
                    vgg = models.vgg16(pretrained=False)
            # Common CSRNet cutoff for plain VGG16
            frontend_cut = 23

        self.frontend = nn.Sequential(*list(vgg.features.children())[:frontend_cut])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def load_trained_model(checkpoint_path: str, device: torch.device, variant: str = "vgg16_bn") -> CSRNet:
    """Load CSRNet and its weights from a checkpoint path, mapped to the given device."""
    model = CSRNet(load_weights=False, variant=variant)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
