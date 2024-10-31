import os
import torch
from gan import Generator  # Import your GAN generator class
import numpy as np

def load_generator(latent_dim, num_classes, img_shape, model_path="generator.pth"):
    generator = Generator(latent_dim, num_classes, img_shape)
    generator.load_state_dict(torch.load(model_path))
    generator.eval() 
    return generator

def generate_and_save_image(generator, latent_dim, label, img_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    z = torch.randn(1, latent_dim).to(device)
    label_tensor = torch.tensor([label]).to(device)
    
    with torch.no_grad():
        gen_img = generator(z, label_tensor).squeeze(0).cpu()
    
    np.save(img_path, gen_img)

def generate_images_for_all_classes(generator, latent_dim, num_classes, num_images_per_class=50, output_dir="./gan_images"):

    os.makedirs(output_dir, exist_ok=True)  

    for label in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)  

        for i in range(num_images_per_class):
            img_path = os.path.join(class_dir, f"{i}.npy")
            generate_and_save_image(generator, latent_dim, label, img_path)


if __name__ == '__main__':
    latent_dim = 100
    img_shape = (3, 224, 224)
    num_classes = 120

    generator = load_generator(latent_dim, num_classes, img_shape)
    generate_images_for_all_classes(generator, latent_dim, num_classes, num_images_per_class=50)
