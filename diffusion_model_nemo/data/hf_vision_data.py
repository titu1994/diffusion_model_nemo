from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from typing import Optional, Dict

from datasets import load_dataset
from torchvision import transforms

from nemo.core import Dataset, typecheck
from nemo.core.neural_types import NeuralType

# define image transformations (e.g. using torchvision)
transform = Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
)


# define function
def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


def get_transform(image_size: int):
    transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    return transform


def get_reverse_transform():
    reverse_transform = Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )
    return reverse_transform


class HFVisionDataset(Dataset):
    def __init__(self, name: str, split: str):
        super().__init__()

        dataset = load_dataset(name, split=split)
        self.dataset = dataset.with_transform(transforms).remove_columns("label")

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
