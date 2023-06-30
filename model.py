import torch.nn as nn
import segmentation_models_pytorch as smp

class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg
        if cfg.model_name == 'Unet':
            model_struct = smp.Unet
        elif cfg.model_name == 'Linknet':
            model_struct = smp.Linknet
        elif cfg.model_name == 'FPN':
            model_struct = smp.FPN
        elif cfg.model_name == 'PSPNet':
            model_struct = smp.PSPNet
        elif cfg.model_name == 'PAN':
            model_struct = smp.PAN
        else:
            model_struct = smp.Unet
            print("Invalid model name! Using default model (Unet).")

        self.encoder = model_struct(
            encoder_name=cfg.backbone,
            encoder_weights=weight,
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        return output


def build_model(cfg, weight="imagenet"):
    print('model_name:', cfg.model_name)
    print('backbone:', cfg.backbone)

    model = CustomModel(cfg, weight)

    return model
