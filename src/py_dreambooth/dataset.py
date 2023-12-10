import logging
import os
import shutil
from enum import Enum
from typing import Optional
import boto3
from huggingface_hub import snapshot_download
from tqdm import tqdm
from .utils.aws_helpers import (
    create_bucket_if_not_exists,
    delete_files_in_s3,
    make_s3_uri,
    upload_dir_to_s3,
)
from .utils.image_helpers import (
    detect_face_and_resize_image,
    get_image_paths,
    resize_and_center_crop_image,
    validate_dir,
)
from .utils.misc import log_or_print


class HfRepoId(str, Enum):
    """
    A class that holds the HuggingFace Hub repository ID
    """

    DOG_EXAMPLE = "diffusers/dog-example"


class LocalDataset:
    """
    A class that represents a local dataset
    Args:
        data_dir: The name of the local directory containing the images
            you want the model to train on
        logger: The logger to use for logging messages
    """

    def __init__(self, data_dir: str, logger: Optional[logging.Logger] = None) -> None:
        self.raw_data_dir = data_dir
        self.preproc_data_dir = data_dir
        self.resolution = None
        self.logger = logger

        os.makedirs(self.raw_data_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(get_image_paths(self.preproc_data_dir))

    def download_examples(self, repo_id: Optional[str] = None) -> "LocalDataset":
        """
        Download a dataset from the HuggingFace Hub to the local directory
        Args:
            repo_id: The ID of the HuggingFace Hub repository to download
        Returns:
            The local dataset instance
        """
        shutil.rmtree(self.raw_data_dir)

        if repo_id is None:
            repo_id = HfRepoId.DOG_EXAMPLE.value

        snapshot_download(
            repo_id,
            local_dir=self.raw_data_dir,
            repo_type="dataset",
            ignore_patterns=".gitattributes",
        )

        msg = f"The dataset '{repo_id}' has been downloaded from the HuggingFace Hub to '{self.raw_data_dir}'."
        log_or_print(msg, self.logger)

        return self

    def preprocess_images(
        self, resolution: int = 1024, detect_face: bool = False
    ) -> "LocalDataset":
        """
        Preprocess (crop and resize) images in a local directory
        Args:
            resolution: The resolution to resize the images to
            detect_face: Whether to detect faces and crop around them. If not, crop by center
        Returns:
            The local dataset instance
        """
        validate_dir(self.raw_data_dir)

        self.resolution = resolution
        self.preproc_data_dir = "_".join([self.raw_data_dir, "preproc"])

        if os.path.exists(self.preproc_data_dir):
            shutil.rmtree(self.preproc_data_dir)

        os.makedirs(self.preproc_data_dir)

        preprocess_func = (
            detect_face_and_resize_image
            if detect_face
            else resize_and_center_crop_image
        )

        raw_image_paths = get_image_paths(self.raw_data_dir)
        msg = f"A total of {len(raw_image_paths)} images were found."
        log_or_print(msg, self.logger)

        for image_path in tqdm(raw_image_paths):
            try:
                preproc_image = preprocess_func(
                    image_path, self.resolution, self.resolution
                )
                preproc_image.save(
                    os.path.join(
                        self.preproc_data_dir, image_path.split(os.path.sep)[-1]
                    )
                )

            except RuntimeError as error:
                log_or_print(str(error), self.logger)
                continue

        num_preproc_images = len(self)
        if num_preproc_images == 0:
            raise RuntimeError("There are no preprocessed images.")

        msg = f"A total of {num_preproc_images} images were preprocessed and stored in the path '{self.preproc_data_dir}'."
        log_or_print(msg, self.logger)

        return self


class AWSDataset(LocalDataset):
    """
    A class that represents an AWS dataset
    Args:
        data_dir: The name of the local directory containing the images
            you want the model to train on
        boto_session: The Boto session to use for AWS interactions
        project_name: The name of the project,
            which will be used for task names, save locations, etc.
        s3_bucket_name: The name of the S3 bucket to use for the dataset
        logger: The logger to use for logging messages
    """

    def __init__(
        self,
        data_dir: str,
        boto_session: boto3.Session,
        project_name: Optional[str] = None,
        s3_bucket_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(data_dir, logger)

        self.boto_session = boto_session
        self.project_name = "py-dreambooth" if project_name is None else project_name
        self.dataset_prefix = f"{self.project_name}/dataset"

        self.bucket_name = (
            create_bucket_if_not_exists(
                self.boto_session, self.boto_session.region_name, logger=self.logger
            )
            if s3_bucket_name is None
            else s3_bucket_name
        )

    def get_s3_model_uri(self) -> str:
        """
        Get the S3 URI for the model artifact
        Returns:
            The S3 URI
        """
        return make_s3_uri(
            self.bucket_name, f"{self.project_name}/model", "model.tar.gz"
        )

    def upload_images(self) -> "AWSDataset":
        """
        Upload images to the S3 bucket
        Returns:
            The AWS dataset instance
        """
        delete_files_in_s3(
            self.boto_session,
            self.bucket_name,
            self.dataset_prefix,
            logger=self.logger,
        )

        validate_dir(self.preproc_data_dir)

        upload_dir_to_s3(
            self.boto_session,
            self.preproc_data_dir,
            self.bucket_name,
            self.dataset_prefix,
            logger=self.logger,
        )

        return self
