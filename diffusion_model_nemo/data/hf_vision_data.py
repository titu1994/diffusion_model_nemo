from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from typing import Optional, Dict

from datasets import load_dataset, Dataset as HFDataset
from torchvision import transforms
from huggingface_hub.hf_api import HfFolder

from nemo.core import Dataset, typecheck
from nemo.core.neural_types import NeuralType

# define image transformations (e.g. using torchvision)
transform = Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
)

infer_transform = Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


# define function

def get_image_key(examples: dict) -> str:
    # resolve the key for the data loader
    possible_keys = ['image', 'img']
    image_key = None
    for key in possible_keys:
        if key in examples:
            image_key = key
            break

    if image_key is None:
        raise ValueError(f"Could not retrieve image from the dataset. Tried {possible_keys}, which "
                         f"did not match dataset columns : {list(examples.keys())}")

    return image_key


def train_transforms(examples: dict):
    image_key = get_image_key(examples)

    examples["pixel_values"] = [transform(image) for image in examples[image_key]]
    del examples[image_key]

    return examples


def test_transforms(examples: dict):
    image_key = get_image_key(examples)

    examples["pixel_values"] = [infer_transform(image) for image in examples[image_key]]
    del examples[image_key]

    return examples


def get_transform(image_size: int, scale: bool = True, center_crop: bool = False):
    transfm = [
        Resize(image_size),
    ]

    if center_crop:
        transfm.append(CenterCrop(image_size))

    # turn into Numpy array of shape HWC, divide by 255
    transfm.append(ToTensor())

    if scale:
        transfm.append(Lambda(lambda t: (t * 2) - 1))

    transform = Compose(transfm)
    return transform


def get_reverse_transform(inverse_scale=True, uint=False):
    transfm = []

    if inverse_scale:
        transfm.append(Lambda(lambda t: (t + 1) / 2))

    transfm.extend(
        [
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy()),
        ]
    )

    if uint:
        transfm.append(Lambda(lambda t: t.astype(np.uint8)))

    transfm.append(ToPILImage())

    reverse_transform = Compose(transfm)
    return reverse_transform


class HFVisionDataset(Dataset):
    def __init__(self, name: str, split: str, cache_dir: str = None, mode='train'):
        super().__init__()

        assert mode in ['train', 'test'], "Mode must be in train or test"

        has_auth_token = HfFolder.get_token() is not None
        dataset = load_dataset(name, split=split, cache_dir=cache_dir, use_auth_token=has_auth_token)

        if mode == 'train':
            self.dataset = dataset.with_transform(train_transforms)
        elif mode == 'test':
            self.dataset = dataset.with_transform(test_transforms)
        else:
            raise ValueError("Wrong mode")

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
