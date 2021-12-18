"""Custom faces dataset."""
import os
import enum
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LABELS(enum.Enum):
    REAL = 0
    FAKE = 1


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """

    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

        self.real_image_paths = [os.path.join(os.path.realpath(self.root_path), 'real', name) for name in
                                 self.real_image_names]
        self.fake_image_paths = [os.path.join(os.path.realpath(self.root_path), 'fake', name) for name in
                                 self.fake_image_names]

        self.ds_paths = self.real_image_paths + self.fake_image_paths
        self.ds_labels = [LABELS.REAL] * len(self.real_image_names) + [LABELS.FAKE] * len(self.fake_image_names)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        img = Image.open(self.ds_paths[index])

        if self.transform:
            img = self.transform(img)

        # check if img is tensor, if not, transform to tensor
        if not torch.is_tensor(img):
            img = transforms.ToTensor()(img)

        return img, self.ds_labels[index].value

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.ds_paths)
