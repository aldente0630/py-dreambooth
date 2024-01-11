import ast
import base64
import os
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, EulerDiscreteScheduler
from safetensors.torch import load_file


class HfModel(str, Enum):
    """
    A class that holds the HuggingFace Hub model IDs.
    """

    SDXL_V1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_REFINER_V1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"


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


def model_fn(model_dir: str) -> Dict[str, Any]:
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
        "PRETRAINED_MODEL_NAME_OR_PATH", HfModel.SDXL_V1_0.value
    )
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    if ast.literal_eval(os.environ.get("TRAIN_TEXT_ENCODER_TI", "True")):
        state_dict = load_file(model_dir)
        pipeline.load_textual_inversion(
            state_dict["clip_l"],
            token=["<s0>", "<s1>"],
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
        )
        pipeline.load_textual_inversion(
            state_dict["clip_g"],
            token=["<s0>", "<s1>"],
            text_encoder=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer_2,
        )
    pipeline.load_lora_weights(model_dir)

    if ast.literal_eval(os.environ.get("USE_REFINER", "False")):
        refiner = DiffusionPipeline.from_pretrained(
            HfModel.SDXL_REFINER_V1_0.value,
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
    else:
        refiner = None

    return {"pipeline": pipeline, "refiner": refiner}


def predict_fn(
    data: Dict[str, Any], model_components: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    A function for SageMaker endpoint to generate images.
    Args:
        data: The input data.
        model_components: The model components.
    Returns:
        The dictionary of generated images in base64 encoding format.
    """
    prompt = data.pop("prompt", "")
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", 42)
    high_noise_frac = data.pop("high_noise_frac", 0.7)
    cross_attention_scale = data.pop("cross_attention_scale", 1.0)

    pipeline, refiner = model_components["model"], model_components["refiner"]

    negative_prompt = (
        None
        if negative_prompt is None or len(negative_prompt) == 0
        else negative_prompt
    )
    generator = (
        None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    )

    if refiner:
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            denoising_end=high_noise_frac,
            generator=generator,
            output_type="latent",
            cross_attention_kwargs={"scale": cross_attention_scale},
        )["images"]
        generated_images = refiner(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
        )["images"]

    else:
        generated_images = pipeline(
            prompt=prompt,
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
