import ast
import base64
import os
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Union
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL


class HfModel(str, Enum):
    """
    A class that holds the HuggingFace Hub model IDs.
    """

    SD_V2_1 = "stabilityai/stable-diffusion-2-1-base"
    SD_VAE = "stabilityai/sd-vae-ft-mse"


class SchedulerConfig(Enum):
    """
    A class that holds scheduler configuration values for inference.
    """

    DDIM = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": True,
        "steps_offset": 1,
    }
    EULER_DISCRETE = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "use_karras_sigmas": True,
        "steps_offset": 1,
    }


def model_fn(model_dir: str) -> Any:
    """
    A function for SageMaker endpoint that loads the model from the model directory.
    Args:
        model_dir: The directory where the model is stored.
    Returns:
        The dictionary of model component names and their instances.
    """
    scheduler_type = os.getenv("SCHEDULER_TYPE", "DDIM")

    if scheduler_type.upper() == "DDIM":
        scheduler = DDIMScheduler(**SchedulerConfig.DDIM.value)
    elif scheduler_type.upper() == "EULERDISCRETE":
        scheduler = EulerDiscreteScheduler(
            **SchedulerConfig.EULER_DISCRETE.value,
        )
    else:
        scheduler = None
        ValueError("The 'scheduler_type' must be one of 'DDIM' or 'EulerDiscrete'.")

    pretrained_model_name_or_path = os.environ.get(
        "PRETRAINED_MODEL_NAME_OR_PATH", HfModel.SD_V2_1.value
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipeline.load_lora_weights(model_dir)

    if ast.literal_eval(os.environ.get("USE_FT_VAE", "False")):
        pipeline.vae = AutoencoderKL.from_pretrained(
            HfModel.SD_VAE.value, torch_dtype=torch.float16
        ).to("cuda")

    return {"pipeline": pipeline}


def predict_fn(
    data: Dict[str, Union[int, float, str]], model_components: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    A function for SageMaker endpoint to generate images.
    Args:
        data: The input data.
        model_components: The model components.
    Returns:
        The dictionary of generated images in base64 encoding format.
    """
    prompt = data.pop("prompt", data)
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", None)
    cross_attention_scale = data.pop("cross_attention_scale", 1.0)

    pipeline = model_components["pipeline"]

    negative_prompt = (
        None
        if negative_prompt is None or len(negative_prompt) == 0
        else negative_prompt
    )
    generator = (
        None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    )

    generated_images = pipeline(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        cross_attention_kwargs={"scale": cross_attention_scale},
    )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"images": encoded_images}
