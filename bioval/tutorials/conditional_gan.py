import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import bioval.utils.gpu_manager as gpu_manager
from bioval.metrics.conditional_evaluation import ConditionalEvaluation
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, img_size=32, num_classes=10):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim, 128 * (img_size // 4) ** 2)
        self.bn = nn.BatchNorm1d(128 * (img_size // 4) ** 2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, img_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        embeddings = self.label_emb(labels)
        x = torch.mul(z, embeddings)
        x = self.fc(x)
        x = self.bn(x)
        x = x.view(-1, 128, (self.img_size // 4), (self.img_size // 4))
        x = self.bn1(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=32, num_classes=10):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.conv1 = nn.Conv2d(img_channels + 1, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * (img_size // 4) ** 2, 1)

    def forward(self, x, labels):
        labels = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, labels], dim=1)
        x = nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = x.view(-1, 128 * (self.img_size // 4) ** 2)
        x = nn.functional.sigmoid(self.fc(x))
        return x

class GAN(object):
    def __init__(self, device):
        self.device = device
        self.latent_dim = 100
        self.img_size = 32
        self.img_channels = 3
        self.num_classes = 10
        self.generator = Generator(self.latent_dim, self.img_channels, self.img_size, self.num_classes).to(self.device)
        self.discriminator = Discriminator(self.img_channels, self.img_size, self.num_classes).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.d_losses = []
        self.g_losses = []

    def train(self, dataloader, epochs):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(dataloader):
                real_images = images.to(self.device)
                real_labels = labels.to(self.device)
                batch_size = real_images.size(0)
                # Train Discriminator
                self.optimizer_d.zero_grad()
                # Real images
                real_pred = self.discriminator(real_images, real_labels)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                d_loss_real = self.criterion(real_pred, real_labels)
                d_loss_real.backward()
                # Fake images
                z = torch.randn(self.num_classes*12, self.latent_dim).to(self.device)
                fake_labels = torch.repeat_interleave(torch.arange(self.num_classes), repeats=12).to(self.device)
                fake_images = self.generator(z, fake_labels)
                fake_pred = self.discriminator(fake_images, fake_labels)
                fake_labels = torch.zeros(self.num_classes*12, 1).to(self.device)
                d_loss_fake = self.criterion(fake_pred, fake_labels)
                d_loss_fake.backward()
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_d.step()
                # Train Generator
                self.optimizer_g.zero_grad()
                z = torch.randn(self.num_classes*12, self.latent_dim).to(self.device)
                gen_labels = torch.repeat_interleave(torch.arange(self.num_classes), repeats=12).to(self.device)
                gen_images = self.generator(z, gen_labels)
                gen_pred = self.discriminator(gen_images, gen_labels)
                gen_labels = torch.ones(self.num_classes*12, 1).to(self.device)
                g_loss = self.criterion(gen_pred, gen_labels)
                g_loss.backward()
                self.optimizer_g.step()
                
                # Add losses to list
                self.d_losses.append(d_loss.item())
                self.g_losses.append(g_loss.item())

                if i % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Batch {i}/{len(dataloader)}, d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")
                    self.save_images(epoch, i, real_images, real_labels)
        
        # Create plot of discriminator and generator losses
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.legend()
        plt.savefig('images/losses.png')
                    
    def save_images(self, epoch, batch, real_images, real_labels):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(self.num_classes*12, self.latent_dim).to(self.device)
            labels = torch.repeat_interleave(torch.arange(self.num_classes), repeats=12).to(self.device)
            gen_images = self.generator(z, labels).unsqueeze(1)
            gen_images = gen_images.view(self.num_classes, 12, self.img_channels, self.img_size, self.img_size)
            gen_images = gen_images.permute(0, 1, 3, 4, 2)#.reshape(self.num_classes*12, self.img_size, self.img_size, self.img_channels)
            real_images = real_images.view(self.num_classes, 12, self.img_channels, self.img_size, self.img_size)
            real_images = real_images.permute(0, 1, 3, 4, 2)#.reshape(self.num_classes*12, self.img_size, self.img_size, self.img_channels)
            topk = ConditionalEvaluation()
            # Compute top-k evaluation
            topk_eval = topk(real_images, gen_images, k_range=[1, 5])
            gen_images = gen_images.reshape(self.num_classes*12, self.img_size, self.img_size, self.img_channels)
            # reshape to format (batch, channels, height, width)
            gen_images = gen_images.permute(0, 3, 1, 2)
            save_image(gen_images, f"images/epoch_{epoch}_batch_{batch}.png", nrow=12, normalize=True)
        self.generator.train()
        print(f"Epoch [{epoch + 1}], Batch {batch}, Top-k Evaluation: {topk_eval}")


def create_dataloader(trainset, batch_size, num_batches):
    indices = []
    for label in range(10):
        class_indices = [i for i, (_, y) in enumerate(trainset) if y == label]
        indices.extend(class_indices[:12*num_batches])
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=2)

if __name__ == "__main__":
    best_gpu = gpu_manager.get_available_gpu()
    device = torch.device("cuda:"+str(best_gpu) if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 120
    num_batches = 10 # number of batches to load
    print("Loading data...")

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    print("Creating dataloader...")
    trainloader = create_dataloader(trainset, batch_size, num_batches)

    gan = GAN(device)
    print("Training...")
    gan.train(trainloader, epochs=100)

