import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image
import wandb

wandb.init(project="gan")

def load_and_preprocess_data():
    dog = load_dataset('amaye15/stanford-dogs')
    
    labels = dog['train'].features['label'].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    
    return dog, labels, label2id, id2label

def get_image_transforms():
    return tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class DOG100Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['pixel_values'].convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return {
            'pixel_values': image,
            'label': item['label']
        }

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate latent vector z with embedded labels
        gen_input = torch.cat((z, self.label_embedding(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten the image and concatenate with embedded labels
        img_flat = img.view(img.size(0), -1)
        d_input = torch.cat((img_flat, self.label_embedding(labels)), -1)
        validity = self.model(d_input)
        return validity

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_gan(generator, discriminator, dataloader, latent_dim, num_classes, n_epochs=229, lr=1e-5):
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    for epoch in range(n_epochs):
        if epoch % 5 == 0 and epoch > 0:
            discriminator.apply(weights_init)

        for i, imgs in enumerate(dataloader):
            batch_size = imgs['pixel_values'].size(0)
            real_labels = imgs['label'].to(device)

            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)

            real_imgs = imgs['pixel_values'].to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, real_labels), valid)
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            gen_imgs = generator(z, gen_labels)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            for _ in range(3):
                optimizer_G.zero_grad()

                z = torch.randn(batch_size, latent_dim).to(device)
                gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
                gen_imgs = generator(z, gen_labels)
                g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

                g_loss.backward()
                optimizer_G.step()

        print(f"Epoch [{epoch}/{n_epochs}] - D loss: {d_loss.item()} - G loss: {g_loss.item()}")

def main():
    dog, labels, label2id, id2label = load_and_preprocess_data()
    transform = get_image_transforms()
    train_dataset = DOG100Dataset(dog['train'], transform=transform)
    test_dataset = DOG100Dataset(dog['test'], transform=transform)

    # Define GAN parameters
    latent_dim = 100
    img_shape = (3, 224, 224)
    num_classes = len(labels)
    generator = Generator(latent_dim, num_classes, img_shape)
    discriminator = Discriminator(img_shape, num_classes)

    # Load data
    dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # Train GAN
    train_gan(generator, discriminator, dataloader, latent_dim, num_classes)

    # Save the trained generator model
    torch.save(generator.state_dict(), "generator.pth")

if __name__ == "__main__":
    main()
