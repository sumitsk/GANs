import torch
import torch.nn as nn

from utils import real_data_target, fake_data_target, Logger, noise


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        self.linear = torch.nn.Linear(100, 1024*4*4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4,stride=2, padding=1, bias=False)
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


class DCGAN(object):
    def __init__(self, args, data_loader):
        super(DCGAN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = GeneratorNet().to(self.device)
        self.disc = DiscriminatorNet().to(self.device)

        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        lr = args.lr
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lr)

        self.loss = torch.nn.BCELoss()

        self.num_test_samples = 16
        self.data_loader = data_loader
        self.logger = Logger(model_name='DCGAN', data_name='MNIST', save_dir=args.save_dir, log_dir=args.log_dir)

    def train_disc(self, real_data, fake_data):
        # Reset gradients
        self.disc_optim.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = self.disc(real_data)
        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, real_data_target(real_data.size(0)).to(self.device))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self.disc(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, fake_data_target(real_data.size(0)).to(self.device))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        self.disc_optim.step()
        
        # Return error
        return error_real + error_fake, prediction_real, prediction_fake
    
    def train_gen(self, fake_data):
        # Reset gradients
        self.gen_optim.zero_grad()
        # Sample noise and generate fake data
        prediction = self.disc(fake_data)
        # Calculate error and backpropagate
        error = self.loss(prediction, real_data_target(prediction.size(0)).to(self.device))
        error.backward()
        # Update weights with gradients
        self.gen_optim.step()
        # Return error
        return error

    def train(self, num_epochs):
        num_batches = len(self.data_loader)

        for epoch in range(num_epochs):
            for n_batch, (real_batch,_) in enumerate(self.data_loader):

                # 1. Train Discriminator
                real_data = real_batch.to(self.device)
                # Generate fake data
                with torch.no_grad():
                    fake_data = self.gen(noise(real_data.size(0)).to(self.device))
                
                # Train D
                d_error, d_pred_real, d_pred_fake = self.train_disc(real_data, fake_data)

                # 2. Train Generator
                # Generate fake data
                fake_data = self.gen(noise(real_batch.size(0)).to(self.device))
                # Train G
                g_error = self.train_gen(fake_data)
                # Log error
                self.logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Display Progress
                if n_batch % 100 == 0:
                    # display.clear_output(True)
                    # Display Images
                    test_images = self.test()
                    self.logger.log_images(test_images, self.num_test_samples, epoch, n_batch, num_batches)
                    # Display status Logs
                    self.logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)
                # Model Checkpoints
                self.logger.save_models(self.gen, self.disc, epoch)

    def test(self):
        test_noise = noise(self.num_test_samples)
        with torch.no_grad():
            test_images = self.gen(test_noise.to(self.device)).cpu()
        return test_images    
