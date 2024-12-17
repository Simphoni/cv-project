import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusers import (
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import (
    StableDiffusionLoraLoaderMixin
)
from .utils import (
    get_diffusers_state_dict,
    unwrap_model,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.training_utils import _set_state_dict_into_text_encoder
from peft import set_peft_model_state_dict

from .utils import convert_trainable_dtype

def get_save_model_hook(is_main_process: bool):
    def save_model_hook(models, weights, output_dir):
        if not is_main_process:
            return
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None


        for model in models:
            if isinstance(unwrap_model(model), UNet2DConditionModel):
                unet_lora_layers_to_save = get_diffusers_state_dict(model)
            elif isinstance(unwrap_model(model), CLIPTextModel):
                text_encoder_one_lora_layers_to_save = get_diffusers_state_dict(model)
            elif isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                text_encoder_two_lora_layers_to_save = get_diffusers_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            if weights:
                weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )
    
    return save_model_hook
    
def get_load_model_hook(is_main_process: bool, train_text_encoder: bool, mixed_precision: str):
    def load_model_hook(models, input_dir):
        if not is_main_process:
            return
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, UNet2DConditionModel):
                unet_ = model
            elif isinstance(model, CLIPTextModel):
                text_encoder_one_ = model
            elif isinstance(model, CLIPTextModelWithProjection):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {}
        for k, v in lora_state_dict.items():
            if k.startswith("unet."):
                new_key = k[5:]
                unet_state_dict[new_key] = v
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_)

        if mixed_precision == "fp16":
            models_ = [unet_]
            if train_text_encoder:
                models_.append(text_encoder_one_)
                models_.append(text_encoder_two_)
            convert_trainable_dtype(models_, dtype=torch.float32)

    return load_model_hook