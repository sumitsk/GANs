import torch
from torchvision import transforms, datasets

from VGAN import VanillaGAN
from DCGAN import DCGAN
from arguments import get_args


def mnist_data(conv=False):
    DATA_FOLDER = './torch_data/VGAN/MNIST'
    # data transformations 
    trs = [transforms.Resize(64)] if conv else []
    trs = trs + [transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    compose = transforms.Compose(trs)
        
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

args = get_args()

# vanilla GAN
# data = mnist_data()
# data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# gan = VanillaGAN(args, data_loader=data_loader)
# gan.train(num_epochs=args.num_epochs)

# DCGAN
data = mnist_data(conv=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
gan = DCGAN(args, data_loader=data_loader)
gan.train(num_epochs=args.num_epochs)
