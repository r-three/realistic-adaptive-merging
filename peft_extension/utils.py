import torch.nn as nn


def init_module_weights(
    module: nn.Module,
    init_type: str = "normal",
):
    if init_type is None:
        return

    for param in module.parameters():
        if len(param.shape) >= 2:
            if init_type == "normal":
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif init_type == "constant":
                nn.init.constant_(param, 0.0)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
        else:
            nn.init.constant_(param, 0.0)
