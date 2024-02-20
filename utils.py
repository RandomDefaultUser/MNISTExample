import torch
import matplotlib.pyplot as plt
import torchvision

mnist_raw = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())


def test_single_image(network, example_number):
    with torch.no_grad():
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        image, label = mnist_raw[example_number]
        ax.imshow(image[0], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        image = image.reshape(-1, 28*28)
        output = network(image)
        ax.set_title("Korrekt: {}, Vorhersage: {}".format(label, torch.max(output.data,1).indices[0]))
        plt.show()


def draw_single_digit(number):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    image, label = mnist_raw[number]
    ax.imshow(image[0], cmap='gray', interpolation='none')
    ax.set_title("Label: {}".format(label))
    ax.set_xticks([])
    ax.set_yticks([])
