import os
from torchvision import datasets, transforms
import torch

# Define MNIST transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def split_and_save_data(dataset, num_clients):
    if dataset == 'mnist':
        trainset = datasets.MNIST(root='./data/MNIST', train=True, transform=transform, download=True)
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform, download=True)
    data_len = len(trainset)
    for i in range(num_clients):
        # Determine start and end indices
        start_idx = int(i * data_len / num_clients)
        end_idx = int((i + 1) * data_len / num_clients)
        
        subset = torch.utils.data.Subset(trainset, range(start_idx, end_idx))
        folder_name = f'./data/{dataset}/client_{i}'
        os.makedirs(folder_name, exist_ok=True)
        
        for idx, (image, label) in enumerate(subset):
            image_folder = os.path.join(folder_name, str(label))
            os.makedirs(image_folder, exist_ok=True)
            image_path = os.path.join(image_folder, f"{idx}.png")
            transforms.ToPILImage()(image).save(image_path)

# Split data into 10 parts and save to folders
num_clients = 5
split_and_save_data(dataset='mnist', mum_clients=num_clients)
