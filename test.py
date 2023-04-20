from torchvision import datasets

train_dataset = datasets.MNIST(root='./My_Datasets/study', train=True, download=True)
test_dataset = datasets.MNIST(root='./My_Datasets/study', train=False, download=True)