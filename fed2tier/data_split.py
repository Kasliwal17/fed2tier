import os
from torchvision import datasets, transforms
import torch

# Define MNIST transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

def split_and_save_data(dataset, num_clients):
    data_len = len(dataset)
    for i in range(num_clients):
        # Determine start and end indices
        start_idx = int(i * data_len / num_clients)
        end_idx = int((i + 1) * data_len / num_clients)
        
        subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        folder_name = f'./data/client_{i}'
        os.makedirs(folder_name, exist_ok=True)
        
        for idx, (image, label) in enumerate(subset):
            image_folder = os.path.join(folder_name, str(label))
            os.makedirs(image_folder, exist_ok=True)
            image_path = os.path.join(image_folder, f"{idx}.png")
            transforms.ToPILImage()(image).save(image_path)

# Split data into 10 parts and save to folders
num_clients = 5
split_and_save_data(mnist_dataset, num_clients)
