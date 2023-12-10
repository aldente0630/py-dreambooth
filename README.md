# Py-Dreambooth
- - -
![Samples](assets/asset-002.png)
**Py-Dreambooth** is a Python package that makes it easy to create AI avatar images from photos of you, your family, friends, or pets!
1. Tasks are pre-configured with the most efficient defaults, which greatly streamlines the workload. A number of helper functions are also provided.
2. This is designed to be modular and extensible to many different models. Currently supported models are the *Stable Diffusion Dreambooth*, *Stable Diffusion Dreambooth LoRA*, and *Stable Diffusion XL Dreambooth LoRA*.
3. This is designed to give you the flexibility to choose local or cloud resources to train your model and generate images.

## ⚙️ How to Install
- - -
```shell
pip install py-dreambooth
```

 ## 🚀 Quick Start
- - -
* Prepare about 10-20 high-quality solo selfie photos (jpg or png) and put them in a specific directory.
* Please run on a machine with a GPU of 16GB or more. (If you're fine-tuning *SDXL*, you'll need 24GB of VRAM.)
```python
from py_dreambooth.dataset import LocalDataset
from py_dreambooth.model import SdDreamboothModel
from py_dreambooth.trainer import LocalTrainer
from py_dreambooth.utils.image_helpers import display_images
from py_dreambooth.utils.prompt_helpers import make_prompt

DATA_DIR = "data"  # The directory where you put your prepared photos
OUTPUT_DIR = "models"  

dataset = LocalDataset(DATA_DIR)
dataset = dataset.preprocess_images(detect_face=True)

SUBJECT_NAME = "<YOUR-NAME>"  
CLASS_NAME = "person"

model = SdDreamboothModel(subject_name=SUBJECT_NAME, class_name=CLASS_NAME)
trainer = LocalTrainer(output_dir=OUTPUT_DIR)

predictor = trainer.fit(model, dataset)

# Use the prompt helper to create an awesome AI avatar!
prompt = next(make_prompt(SUBJECT_NAME, CLASS_NAME))

images = predictor.predict(
    prompt, height=768, width=512, num_images_per_prompt=2,
)

display_images(images, fig_size=10)
```

## 🏃‍♀️ Tutorials  
- - -
* Take a look at the [01-local-tutorial.ipynb](ipynb/01-local-tutorial.ipynb) file to learn how to get it running on your local *Jupyter Notebook*.
* If you're interested in running it with AWS cloud resources, take a look at the [02-aws-tutorial.ipynb](ipynb/02-aws-tutorial.ipynb) file.
* Or, get started right away with the [*Google Colab Notebook*](https://colab.research.google.com/drive/1jIv8210dOFLWXAL8gP3SpMQcLHmSHlVS?usp=sharing) here!

## 📚 Documentation
- - -
* Full documentation can be found here: https://py-dreambooth.readthedocs.io.

### References
- - -
* [*DreamBooth*: Fine-Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (Paper)](https://arxiv.org/abs/2208.12242)
* [*LoRA*: Low-Rank Adaptation of Large Language Models (Paper)](https://arxiv.org/abs/2106.09685)
* [Fine-Tune Text-to-Image *Stable Diffusion* Models with *Amazon SageMaker JumpStart* (Blog)](https://aws.amazon.com/blogs/machine-learning/fine-tune-text-to-image-stable-diffusion-models-with-amazon-sagemaker-jumpstart/)
* [Training *Stable Diffusion* with *Dreambooth* Using 🧨 *Diffusers* (Blog)](https://huggingface.co/blog/dreambooth)
* [*Diffusers*: *DreamBooth* Training Example](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README.md#dreambooth-training-example)
* [*Diffusers*: *DreamBooth* Training Example for *Stable Diffusion XL* (*SDXL*)](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md)