import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config

# Training transforms
def get_train_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return train_transform
# Validation transforms
def get_valid_transform(IMAGE_SIZE):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform


def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        config.ROOT_DIR, 
        transform=(get_train_transform(config.IMAGE_SIZE))
    )
    dataset_test = datasets.ImageFolder(
        config.ROOT_DIR, 
        transform=(get_valid_transform(config.IMAGE_SIZE))
    )
    dataset_size = len(dataset)
    # Calculate the validation dataset size.
    valid_size = int(config.VALID_SPLIT*dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset.classes

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=config.NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=config.BATCH_SIZE, 
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    return train_loader, valid_loader 