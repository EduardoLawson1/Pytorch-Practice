"""
Contains functionality for creating Pytorch DataLoaders's for
image classification data
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders
  Takes in a training firectory and testing directory path and turns them into
  Pytorch Datasets and then into Pytorch DataLoaders

  ARGS:
  train_dir: Path to training directory
  test_dir: Path to testing directory
  transforms: torchvision transforms to perform on data
  batch_size: Number of samples per batch in each DataLoader
  num_workers: An integer for number of workers per DataLoader

  Returns:
  A tuple of (train_dattaloader, test_dataloader, class_names)
  Where class_names is a list of the target classes
  eXAMPLE USAGE:
  train_datloader, test_dataloader, class_names = create_dataçoaders(train_dir="path",
  test_dir="path",
  transform=some_transform,
  batch_size=32,
  num_workers=4)
  """
  # Use Image Folder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn our images into Pytorch DataLoaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )


  return train_dataloader, test_dataloader, class_names
