import torch
from torchvision import transforms, datasets

from VGAN import VanillaGAN
from arguments import get_args


def mnist_data():
    DATA_FOLDER = './torch_data/VGAN/MNIST'
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = mnist_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

args = get_args()

gan = VanillaGAN(args, data_loader=data_loader)
gan.train(num_epochs=args.num_epochs)

