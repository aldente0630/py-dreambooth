import logging
import os
import shutil
from typing import Optional


def delete_dir_with_name(
    root_dir: str, dir_name: str, logger: Optional[logging.Logger] = None
) -> None:
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for name in dirs:
            if name == dir_name:
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path)
                msg = f"The deleted directory is '{dir_path}'."
                if logger is None:
                    print(msg)
                else:
                    logger.info(msg)
