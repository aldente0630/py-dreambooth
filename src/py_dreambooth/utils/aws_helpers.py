import json
import logging
import os
from typing import Optional, List
import boto3
from .misc import log_or_print


def create_bucket_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Create a S3 bucket if it does not exist.
    Args:
        boto_session: The boto session to use for AWS interactions.
        region_name: AWS region name.
        logger: The logger to use for logging messages.
    Returns:
        The name of the S3 bucket created.
    """
    s3_client = boto_session.client("s3")
    sts_client = boto_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    bucket_name = f"sagemaker-{region_name}-{account_id}"
    if (
        s3_client.head_bucket(Bucket=bucket_name)["ResponseMetadata"]["HTTPStatusCode"]
        == 404
    ):
        s3_client.create_bucket(Bucket=bucket_name)
        msg = f"The following S3 bucket was created: {bucket_name}"
        log_or_print(msg, logger)

    else:
        msg = f"The following S3 bucket was found: {bucket_name}"
        log_or_print(msg, logger)

    return bucket_name


def create_role_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Create an IAM role if it does not exist.
    Args:
        boto_session: The boto session to use for AWS interactions.
        region_name: AWS region name.
        logger: The logger to use for logging messages.
    Returns:
        The name of the IAM role created.
    """
    iam_client = boto_session.client("iam")

    role_name = f"AmazonSageMaker-ExecutionRole-{region_name}"
    try:
        role = iam_client.get_role(RoleName=role_name)
        msg = f"The following IAM role was found: {role['Role']['Arn']}"

    except iam_client.exceptions.NoSuchEntityException:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="SageMaker Execution Role",
        )
        policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        msg = f"The following IAM role was created: {role['Role']['Arn']}"

    log_or_print(msg, logger)
    return role_name


def delete_files_in_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Delete files in a S3 bucket.
    Args:
        boto_session: The boto session to use for AWS interactions.
        bucket_name: The name of the S3 bucket.
        prefix: The S3 prefix of the files to delete.
        logger: The logger to use for logging messages.
    """
    s3_resource = boto_session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        s3_resource.Object(bucket_name, obj.key).delete()
        msg = f"The 's3://{bucket_name}/{obj.key}' file has been deleted."
        log_or_print(msg, logger)


def make_s3_uri(bucket: str, prefix: str, filename: Optional[str] = None) -> str:
    """
    Make a S3 URI.
    Args:
        bucket: The S3 bucket name.
        prefix: The S3 prefix.
        filename: The filename.
    Returns:
        The S3 URI.
    """
    prefix = prefix if filename is None else os.path.join(prefix, filename)
    return f"s3://{bucket}/{prefix}"


def upload_dir_to_s3(
    boto_session: boto3.Session,
    local_dir: str,
    bucket_name: str,
    prefix: str,
    file_ext_to_excl: Optional[List[str]] = None,
    public_readable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload a directory to a S3 bucket.
    Args:
        boto_session: The boto session to use for AWS interactions.
        local_dir: The local directory to upload.
        bucket_name: The name of the S3 bucket.
        prefix: The S3 prefix.
        file_ext_to_excl: The file extensions to exclude from the upload.
        public_readable: Whether the files should be public readable.
        logger: The logger to use for logging messages.
    """
    s3_client = boto_session.client("s3")
    file_ext_to_excl = [] if file_ext_to_excl is None else file_ext_to_excl
    extra_args = {"ACL": "public-read"} if public_readable else {}

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.split(".")[-1] not in file_ext_to_excl:
                s3_client.upload_file(
                    os.path.join(root, file),
                    bucket_name,
                    f"{prefix}/{file}",
                    ExtraArgs=extra_args,
                )
                msg = f"The '{file}' file has been uploaded to 's3://{bucket_name}/{prefix}/{file}'."
                log_or_print(msg, logger)
