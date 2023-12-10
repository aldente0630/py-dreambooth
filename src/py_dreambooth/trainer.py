import logging
import os
import shlex
import subprocess
from abc import ABCMeta, abstractmethod
from typing import Final, Optional, Union
import sagemaker
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.utils import unique_name_from_base
from .dataset import AWSDataset, LocalDataset
from .model import BaseModel, SdDreamboothLoraModel, SdxlDreamboothLoraModel
from .predictor import AWSPredictor, BasePredictor, LocalPredictor
from .utils.aws_helpers import create_role_if_not_exists, make_s3_uri
from .utils.misc import decompress_file, log_or_print

DEFAULT_INSTANCE_TYPE: Final = "ml.g4dn.xlarge"

PYTORCH_VERSION: Final = "2.0.0"
TRANSFORMER_VERSION: Final = "4.28.1"
PY_VERSION: Final = "py310"

STEP_MULTIPLIER: Final = 100

BASE_DIR: Final = "/opt/ml/processing"


class BaseTrainer(metaclass=ABCMeta):
    """
    An abstract class to represent the trainer
    Args:
        config_path: The path to the Accelerate config file
        report_to: The solution to report results and log
        wandb_api_key: The API key to use for logging to WandB
        logger: The logger to use for logging messages
    """

    def __init__(
        self,
        config_path: Optional[str],
        report_to: Optional[str],
        wandb_api_key: Optional[str],
        logger: Optional[logging.Logger],
    ) -> None:
        self.config_path = config_path
        self.report_to = "tensorboard" if report_to is None else report_to
        self.wandb_api_key = wandb_api_key
        self.logger = logger

    @abstractmethod
    def fit(self, model: BaseModel, dataset: LocalDataset) -> BasePredictor:
        """
        Fit a model to a dataset
        Args:
            model: The base model instance to be fitted
            dataset: The local dataset instance to fit the model
        Returns:
            The base predictor instance of the fitted model
        """


class LocalTrainer(BaseTrainer):
    """
    A class to represent the local trainer
    Args:
        config_path: The path to the Accelerate config file
        output_dir: The directory to store the model
        report_to: The solution to report results and log
        wandb_api_key: The API key to use for logging to WandB
        logger: The logger to use for logging messages
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        report_to: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(config_path, report_to, wandb_api_key, logger)

        self.output_dir = "output" if output_dir is None else output_dir
        os.makedirs(os.path.join(os.getcwd(), self.output_dir), exist_ok=True)

    def fit(self, model: BaseModel, dataset: LocalDataset) -> LocalPredictor:
        """
        Fit a model to a dataset
        Args:
            model: The base model instance to be fitted
            dataset: The local dataset instance to fit the model
        Returns:
            The local predictor instance of the fitted model
        """
        kwargs = {
            "data_dir": dataset.preproc_data_dir,
            "output_dir": self.output_dir,
            "report_to": self.report_to,
            "compress_output": "False",
        }

        if model.max_train_steps is None:
            if isinstance(model, (SdDreamboothLoraModel, SdxlDreamboothLoraModel)):
                max_train_steps = round(STEP_MULTIPLIER * len(dataset))
            else:
                max_train_steps = round(0.75 * STEP_MULTIPLIER * len(dataset))

            kwargs["max_train_steps"] = max_train_steps

        else:
            max_train_steps = model.max_train_steps

        model = model.set_members(**kwargs)

        if self.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

        command = model.make_command(self.config_path)

        log_or_print(
            f"The model training has begun.\n'max_train_steps' is set to {max_train_steps}.",
            self.logger,
        )
        _ = subprocess.run(shlex.split(command), check=True)
        log_or_print("The model training has ended.", self.logger)

        predictor = LocalPredictor(model, self.output_dir, self.logger)

        return predictor


class AWSTrainer(BaseTrainer):
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        report_to: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        deploy_sm_endpoint: bool = True,
        iam_role_name: Optional[str] = None,
        sm_train_instance_type: Optional[str] = None,
        sm_infer_instance_type: Optional[str] = None,
        sm_endpoint_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(config_path, report_to, wandb_api_key, logger)

        self.output_dir = None
        self.deploy_sm_endpoint = deploy_sm_endpoint
        self.role_name = iam_role_name
        self.train_instance_type = (
            DEFAULT_INSTANCE_TYPE
            if sm_train_instance_type is None
            else sm_train_instance_type
        )
        self.infer_instance_type = None
        self.endpoint_name = None

        if self.deploy_sm_endpoint:
            self.infer_instance_type = (
                DEFAULT_INSTANCE_TYPE
                if sm_infer_instance_type is None
                else sm_infer_instance_type
            )
            self.endpoint_name = (
                "py-dreambooth" if sm_endpoint_name is None else sm_endpoint_name
            )

        else:
            self.output_dir = "output" if output_dir is None else output_dir
            os.makedirs(os.path.join(os.getcwd(), self.output_dir), exist_ok=True)

        self.sm_session = None

    def fit(
        self, model: BaseModel, dataset: AWSDataset
    ) -> Union[AWSPredictor, LocalPredictor]:
        """
        Fit a model to a dataset
        Args:
            model: The base model instance to be fitted
            dataset: The AWS dataset instance to fit the model
        Returns:
            The AWS or local predictor instance of the fitted model
        """
        self.sm_session = sagemaker.session.Session(boto_session=dataset.boto_session)

        self.role_name = (
            create_role_if_not_exists(
                dataset.boto_session,
                dataset.boto_session.region_name,
                logger=self.logger,
            )
            if self.role_name is None
            else self.role_name
        )

        config_uri = self.sm_session.upload_data(
            os.path.join(os.path.dirname(model.train_code_path), "requirements.txt"),
            bucket=dataset.bucket_name,
            key_prefix=f"{dataset.project_name}/config",
        )
        config_uri = "/".join(config_uri.split("/")[:-1])

        log_or_print(
            f"The requirements file has been uploaded to '{config_uri}'.",
            self.logger,
        )

        dataset_uri = make_s3_uri(dataset.bucket_name, dataset.dataset_prefix)

        kwargs = {
            "data_dir": f"{BASE_DIR}/dataset",
            "output_dir": f"{BASE_DIR}/model",
            "report_to": self.report_to,
            "compress_output": "True",
        }

        if model.max_train_steps is None:
            if isinstance(model, (SdDreamboothLoraModel, SdxlDreamboothLoraModel)):
                max_train_steps = round(len(dataset) * STEP_MULTIPLIER)
            else:
                max_train_steps = round(0.75 * len(dataset) * STEP_MULTIPLIER)

            kwargs["max_train_steps"] = max_train_steps

        else:
            max_train_steps = model.max_train_steps

        model = model.set_members(**kwargs)
        arguments = model.get_arguments()
        command = [
            "accelerate",
            "launch",
        ]

        if self.config_path:
            _ = self.sm_session.upload_data(
                self.config_path,
                bucket=dataset.bucket_name,
                key_prefix=f"{dataset.project_name}/config",
            )

            log_or_print(
                f"The Accelerate config file has been uploaded to '{config_uri}'.",
                self.logger,
            )

            config_filename = os.path.basename(self.config_path)
            command += ["--config_file", f"{BASE_DIR}/config/{config_filename}"]

        model_prefix = f"{dataset.project_name}/model"
        model_uri = make_s3_uri(dataset.bucket_name, model_prefix)
        code_uri = make_s3_uri(dataset.bucket_name, f"{dataset.project_name}/code")

        processor = HuggingFaceProcessor(
            role=self.role_name,
            instance_count=1,
            instance_type=self.train_instance_type,
            transformers_version=TRANSFORMER_VERSION,
            pytorch_version=PYTORCH_VERSION,
            py_version=PY_VERSION,
            command=command,
            code_location=code_uri,
            base_job_name=dataset.project_name,
            sagemaker_session=self.sm_session,
            env={"WANDB_API_KEY": self.wandb_api_key} if self.wandb_api_key else None,
        )

        log_or_print(
            f"The model training has begun.\n'max_train_steps' is set to {max_train_steps}.",
            self.logger,
        )

        processor.run(
            inputs=[
                ProcessingInput(
                    source=config_uri,
                    destination=f"{BASE_DIR}/config",
                    input_name="config",
                ),
                ProcessingInput(
                    source=dataset_uri,
                    destination=f"{BASE_DIR}/dataset",
                    input_name="dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source=f"{BASE_DIR}/model",
                    destination=model_uri,
                    output_name="model",
                )
            ],
            code=model.train_code_path,
            arguments=arguments,
            logs=True,
            job_name=unique_name_from_base(dataset.project_name),
        )

        log_or_print("The model training has ended.", self.logger)

        if self.deploy_sm_endpoint:
            predictor = AWSPredictor(
                model,
                f"{model_uri}/model.tar.gz",
                dataset.boto_session,
                self.role_name,
                self.infer_instance_type,
                self.endpoint_name,
                self.logger,
            )

        else:
            _ = self.sm_session.download_data(
                self.output_dir,
                bucket=dataset.bucket_name,
                key_prefix=model_prefix,
            )

            decompress_file(
                os.path.join(self.output_dir, "model.tar.gz"), compression="tar"
            )

            predictor = LocalPredictor(model, self.output_dir, self.logger)

        return predictor
