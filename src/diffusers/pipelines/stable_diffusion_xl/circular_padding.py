import torch.nn.functional as F
from torch import Tensor
import torch
from diffusers.models.lora import LoRACompatibleConv
from typing import Optional

def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def replacement_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    xmode = 'circular' if self.circular_padding_mode in ['both', 'x'] else 'constant'
    ymode = 'circular' if self.circular_padding_mode in ['both', 'y'] else 'constant'

    working = F.pad(x, (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0), mode=xmode)
    working = F.pad(working, (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3]), mode=ymode)

    return F.conv2d(working, weight, bias, self.stride, (0,0), self.dilation, self.groups)


def replacement_lora_forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    xmode = 'circular' if self.circular_padding_mode in ['both', 'x'] else 'constant'
    ymode = 'circular' if self.circular_padding_mode in ['both', 'y'] else 'constant'

    working = F.pad(hidden_states, (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0), mode=xmode)
    working = F.pad(working, (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3]), mode=ymode)
    original_outputs =  F.conv2d(working, self.weight, self.bias, self.stride, (0,0), self.dilation, self.groups)

    if self.lora_layer is None:
        return original_outputs
    else:
        return original_outputs + (scale * self.lora_layer(hidden_states))

class CircularPaddingMixin:
    def set_circular_padding_mode(self, mode='both'):
        unet_convs = [layer for layer in flatten(self.unet) if (isinstance(layer, torch.nn.Conv2d))]
        vae_convs = [layer for layer in flatten(self.vae) if (isinstance(layer, torch.nn.Conv2d))]

        for layer in unet_convs + vae_convs:
            layer.circular_padding_mode = mode

            if isinstance(layer, LoRACompatibleConv):
                layer.forward = replacement_lora_forward.__get__(layer, LoRACompatibleConv)
            else:
                layer._conv_forward = replacement_forward.__get__(layer, torch.nn.Conv2d)

    def disable_circular_padding_mode(self):
        unet_convs = [layer for layer in flatten(self.unet) if (isinstance(layer, torch.nn.Conv2d))]
        vae_convs = [layer for layer in flatten(self.vae) if (isinstance(layer, torch.nn.Conv2d))]

        for layer in unet_convs + vae_convs:
            if isinstance(layer, LoRACompatibleConv):
                layer.forward = LoRACompatibleConv.forward.__get__(layer, LoRACompatibleConv)
            else:
                layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)
