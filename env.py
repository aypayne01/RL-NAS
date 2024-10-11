import torch
import torchvision
import torchvision.transforms as transforms

class TaskEnvironment:
    def __init__(self, dataset, task_type):
        self.dataset = dataset
        self.task_type = task_type
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset == 'CIFAR-10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def evaluate(self, architecture):
        # This is a stub, replace with actual training and evaluation code
        print(f"Evaluating architecture: {architecture}")
        return torch.rand(1).item()  # Random reward for example purposes