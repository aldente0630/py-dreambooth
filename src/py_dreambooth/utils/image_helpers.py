import base64
import os
from glob import glob
from io import BytesIO
from itertools import chain
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
from autocrop import Cropper


def decode_base64_image(image_string: str) -> Image.Image:
    """
    Decodes a base64 encoded image string and returns an Image object.
    Args:
        image_string: The base64 encoded image string.
    Returns:
        The Image object.
    """
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def encode_base64_image(file_name: str) -> str:
    """
    Encodes an image file to a base64 encoded string.
    Args:
        file_name: The name of the image file.
    Returns:
        The base64 encoded string.
    """
    with open(file_name, "rb") as image:
        image_string = base64.b64encode(bytearray(image.read())).decode()
    return image_string


def detect_face_and_resize_image(
    image_path: str, tgt_width: int, tgt_height: int
) -> Image:
    """
    Detects faces in an image and resizes it to the specified resolution.
    Args:
        image_path: The path to the image.
        tgt_width: The target width of the image.
        tgt_height: The target height of the image.
    Returns:
        The resized Image object.
    """
    cropper = Cropper(width=tgt_width, height=tgt_height)
    cropped_array = cropper.crop(image_path)

    if cropped_array is None:
        msg = f"No faces detected in the image '{image_path.split(os.path.sep)[-1]}'."
        raise RuntimeError(msg)

    return Image.fromarray(cropped_array)


def display_images(
    images: List[Image.Image],
    n_columns: int = 3,
    fig_size: int = 20,
) -> None:
    """
    Displays a grid of images.
    Args:
        images: The list of images to display.
        n_columns: The number of columns in the grid.
        fig_size: The size of the figure (width).
    """
    n_columns = min(len(images), n_columns)
    quotient, remainder = divmod(len(images), n_columns)
    if remainder > 0:
        quotient += 1
    width, height = images[0].size
    plt.figure(figsize=(fig_size, fig_size / n_columns * quotient * height / width))

    for i, image in enumerate(images):
        plt.subplot(quotient, n_columns, i + 1)
        plt.axis("off")
        plt.imshow(image, aspect="auto")

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()


def get_image_grid(images: List[Image.Image], n_columns: int = 3) -> Image.Image:
    """
    Get a grid of images.
    Args:
        images: The list of images to display.
        n_columns: The number of columns in the grid.
    Returns:
        The grid of images as an Image object.
    """
    n_columns = min(len(images), n_columns)
    quotient, remainder = divmod(len(images), n_columns)
    if remainder > 0:
        quotient += 1
    width, height = images[0].size
    grid = Image.new(
        "RGB",
        size=(
            n_columns * width,
            quotient * height,
        ),
    )

    for i, image in enumerate(images):
        grid.paste(image, box=(i % n_columns * width, i // n_columns * height))

    return grid


def get_image_paths(images_dir: str) -> List[str]:
    """
    Get a list of image paths from a directory.
    Args:
        images_dir: The directory containing the images.
    Returns:
        A list of image paths.
    """
    return list(
        chain(
            glob(os.path.join(images_dir, "*.[jJ][pP]*[Gg]")),
            glob(os.path.join(images_dir, "*.[Pp][Nn][Gg]")),
        )
    )


def resize_and_center_crop_image(
    image_path: str, tgt_width: int, tgt_height: int
) -> Image:
    """
    Resize and center crop an image.
    Args:
        image_path: The path to the image.
        tgt_width: The target width of the image.
        tgt_height: The target height of the image.
    Returns:
        The resized and centered cropped Image object.
    """
    image = Image.open(image_path).convert("RGB")
    src_width, src_height = image.size

    if src_width > src_height:
        left = (src_width - src_height) / 2
        top = 0
        right = (src_width + src_height) / 2
        bottom = src_height
    else:
        top = (src_height - src_width) / 2
        left = 0
        bottom = (src_height + src_width) / 2
        right = src_width

    image = image.crop((left, top, right, bottom))
    image = image.resize((tgt_width, tgt_height))

    return image


def validate_dir(tgt_dir: str) -> None:
    """
    Validates that a directory exists and is not empty.
    Args:
        tgt_dir: The directory to validate.
    """
    if not os.path.exists(tgt_dir) and len(get_image_paths(tgt_dir)) == 0:
        raise ValueError(f"The directory '{tgt_dir}' does not exist or is empty.")
