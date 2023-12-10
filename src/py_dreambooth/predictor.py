import logging
from abc import ABCMeta, abstractmethod
from typing import Final, List, Optional
import boto3
import sagemaker
import torch
from PIL import Image
from sagemaker.huggingface.estimator import HuggingFaceModel
from .model import BaseModel, SdDreamboothLoraModel, SdxlDreamboothLoraModel
from .utils.aws_helpers import create_role_if_not_exists
from .utils.image_helpers import decode_base64_image
from .utils.misc import log_or_print

DEFAULT_INSTANCE_TYPE: Final = "ml.g4dn.xlarge"

PYTORCH_VERSION: Final = "2.0.0"
TRANSFORMER_VERSION: Final = "4.28.1"
PY_VERSION: Final = "py310"


class BasePredictor(metaclass=ABCMeta):
    """
    An abstract class to represent the predictor
    Args:
        model: The base model instance to use for prediction
        logger: The logger to use for logging messages
    """

    def __init__(self, model: BaseModel, logger: Optional[logging.Logger]) -> None:
        self.model = model
        self.logger = logger

    @abstractmethod
    def predict(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        num_images_per_prompt: int,
        seed: Optional[int],
        high_noise_frac: Optional[float],
        cross_attention_scale: Optional[float],
    ) -> List[Image.Image]:
        """
        Generate images given a prompt
        Args:
            prompt: The prompt to use for generating images
            height: The height of the generated images
            width: The width of the generated images
            num_inference_steps: The number of inference steps to use for generating images
            guidance_scale: The guidance scale to use for generating images
            negative_prompt: The negative prompt to use for generating images
            num_images_per_prompt: The number of images to generate per prompt
            seed: The seed to use for generating random numbers
            high_noise_frac: The fraction of the noise to use for denoising
            cross_attention_scale: The scale to use for cross-attention
        Returns:
            A list of PIL images representing the generated images
        """

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate the prompt
        Args:
            prompt: The prompt to validate
        Returns:
            Whether the prompt is valid or not
        """
        return not (
            self.model.subject_name in prompt.lower()
            and self.model.class_name in prompt.lower()
        )


class LocalPredictor(BasePredictor):
    """
    A class to represent the local predictor
    Args:
        model: The base model instance to use for inference
        output_dir: The output directory
        logger: The logger to use for logging messages
    """

    def __init__(
        self, model: BaseModel, output_dir: str, logger: Optional[logging.Logger] = None
    ):
        super().__init__(model, logger)

        model_components = self.model.load_model(output_dir)
        self.pipeline = model_components["pipeline"]
        self.refiner = model_components.get("refiner")

        log_or_print(
            f"The model has loaded from the directory, '{output_dir}'.", self.logger
        )

    def predict(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 4,
        seed: Optional[int] = None,
        high_noise_frac: float = 0.7,
        cross_attention_scale: float = 1.0,
    ) -> List[Image.Image]:
        """
        Generate images given a prompt
        Args:
            prompt: The prompt to use for generating images
            height: The height of the generated images
            width: The width of the generated images
            num_inference_steps: The number of inference steps to use for generating images
            guidance_scale: The guidance scale to use for generating images
            negative_prompt: The negative prompt to use for generating images
            num_images_per_prompt: The number of images to generate per prompt
            seed: seed: The seed to use for generating random numbers
            high_noise_frac: The fraction of the noise to use for denoising
            cross_attention_scale: The scale to use for cross-attention
        Returns:
            A list of PIL images representing the generated images
        """
        if self.validate_prompt(prompt):
            log_or_print(
                "Warning: the subject and class names are not included in the prompt.",
                self.logger,
            )

        generator = (
            None
            if seed is None
            else torch.Generator(device=self.model.device).manual_seed(seed)
        )

        if isinstance(self.model, (SdDreamboothLoraModel, SdxlDreamboothLoraModel)):
            kwargs = {"cross_attention_kwargs": {"scale": cross_attention_scale}}
        else:
            kwargs = {}

        if self.refiner:
            image = self.pipeline(
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
                **kwargs,
            )["images"]
            generated_images = self.refiner(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                denoising_start=high_noise_frac,
            )["images"]

        else:
            generated_images = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                **kwargs,
            )["images"]

        return generated_images


class AWSPredictor(BasePredictor):
    """
    A class to represent the AWS predictor
    Args:
        model: The base model instance to use for inference
        s3_model_uri: The S3 URI of the model
        boto_session: The boto session to use for AWS interactions
        iam_role_name: The name of the IAM role to use
        sm_infer_instance_type: The SageMaker instance type to use for inference
        sm_endpoint_name: The name of the SageMaker endpoint to use for inference
        logger: The logger to use for logging messages
    """

    def __init__(
        self,
        model: BaseModel,
        s3_model_uri: str,
        boto_session: boto3.Session,
        iam_role_name: Optional[str] = None,
        sm_infer_instance_type: Optional[str] = None,
        sm_endpoint_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(model, logger)

        role_name = (
            create_role_if_not_exists(
                boto_session,
                boto_session.region_name,
                logger=self.logger,
            )
            if iam_role_name is None
            else iam_role_name
        )
        infer_instance_type = (
            DEFAULT_INSTANCE_TYPE
            if sm_infer_instance_type is None
            else sm_infer_instance_type
        )
        self.endpoint_name = (
            "py-dreambooth" if sm_endpoint_name is None else sm_endpoint_name
        )

        env = {
            "PRETRAINED_MODEL_NAME_OR_PATH": model.pretrained_model_name_or_path,
            "SCHEDULER_TYPE": model.scheduler_type,
        }
        if hasattr(model, "use_ft_vae") and model.use_ft_vae:
            env.update({"USE_FT_VAE": "True"})
        if hasattr(model, "use_refiner") and model.use_refiner:
            env.update({"USE_REFINER": "True"})

        sm_session = sagemaker.session.Session(boto_session=boto_session)

        hf_model = HuggingFaceModel(
            role=role_name,
            model_data=s3_model_uri,
            entry_point="inference.py",
            transformers_version=TRANSFORMER_VERSION,
            pytorch_version=PYTORCH_VERSION,
            py_version=PY_VERSION,
            source_dir=model.infer_source_dir,
            env=None if len(env) == 0 else env,
            sagemaker_session=sm_session,
        )

        self.predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=infer_instance_type,
            endpoint_name=self.endpoint_name,
        )

        log_or_print(
            f"The model has deployed to the endpoint, '{self.endpoint_name}'.",
            self.logger,
        )

    def delete_endpoint(self) -> None:
        self.predictor.delete_endpoint()
        log_or_print(
            f"The endpoint, '{self.endpoint_name}', has been deleted.", self.logger
        )

    def predict(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 4,
        seed: Optional[int] = None,
        high_noise_frac: float = 0.7,
        cross_attention_scale: float = 1.0,
    ) -> List[Image.Image]:
        """
        Generate images given a prompt
        Args:
            prompt: The prompt to use for generating images
            height: The height of the generated images
            width: The width of the generated images
            num_inference_steps: The number of inference steps to use for generating images
            guidance_scale: The guidance scale to use for generating images
            negative_prompt: The negative prompt to use for generating images
            num_images_per_prompt: The number of images to generate per prompt
            seed: seed: The seed to use for generating random numbers
            high_noise_frac: The fraction of the noise to use for denoising
            cross_attention_scale: The scale to use for cross-attention
        Returns:
            A list of PIL images representing the generated images
        """
        if self.validate_prompt(prompt):
            log_or_print(
                "Warning: the subject and class names are not included in the prompt.",
                self.logger,
            )

        data = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "high_noise_frac": high_noise_frac,
            "cross_attention_scale": cross_attention_scale,
        }

        if negative_prompt:
            data.update(**{"negative_prompt": negative_prompt})

        if seed:
            data.update(**{"seed": seed})

        generated_images = self.predictor.predict(data)
        generated_images = [
            decode_base64_image(image) for image in generated_images["images"]
        ]

        return generated_images
