import torch
import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
# from IPython import display
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# TensorBoard Data will be stored in './runs' path
import ipdb


class Logger:
    def __init__(self, model_name, data_name, log_dir, save_dir):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.images_save_dir = save_dir+'/images/'
        self.models_save_dir = save_dir+'/model/'

        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment, log_dir=log_dir)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):
        if d_error.requires_grad:
            d_error = d_error.data.cpu().numpy()
        if g_error.requires_grad:
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar('{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar('{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        # input images are expected in format (NCHW)
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        # out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(self.images_save_dir)

        # save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        # out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(self.images_save_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(self.images_save_dir, comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        if d_error.requires_grad:
            d_error = d_error.data.cpu().numpy()
        if g_error.requires_grad:
            g_error = g_error.data.cpu().numpy()
        if d_pred_real.requires_grad:
            d_pred_real = d_pred_real.data
        if d_pred_fake.requires_grad:
            d_pred_fake = d_pred_fake.data
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, n_batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        # out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(self.models_save_dir)
        torch.save(generator.state_dict(), '{}/G_epoch_{}'.format(self.models_save_dir, epoch))
        torch.save(discriminator.state_dict(), '{}/D_epoch_{}'.format(self.models_save_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size):
    return torch.randn(size, 100)


def real_data_target(size):
    # Tensor containing ones, with shape = size
    return torch.ones(size, 1)


def fake_data_target(size):
    # Tensor containing zeros, with shape = size
    return torch.zeros(size, 1)
