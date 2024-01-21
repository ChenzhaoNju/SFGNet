from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize
from dataset import DatasetFromFolderEval, DatasetFromFolder


def transform():
    return Compose([
        ToTensor(),
    ])


def transform2():
    return Compose([
        # Resize((256, 256)),
        ToTensor(),
    ])


def get_training_set(data_dir, label_dir, patch_size, data_augmentation):
    return DatasetFromFolder(data_dir, label_dir, patch_size, data_augmentation, transform=transform())


def get_eval_set(data_dir, label_dir):
    return DatasetFromFolderEval(data_dir, label_dir, transform=transform2())

