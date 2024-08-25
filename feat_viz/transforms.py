import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def deprocess_image(tensor):
    tensor = tensor.squeeze().cpu().detach().numpy()
    tensor = (
        tensor * np.array([0.229, 0.224, 0.225])[:, None, None]
        + np.array([0.485, 0.456, 0.406])[:, None, None]
    )
    tensor = np.clip(tensor, 0, 1)
    return (tensor * 255).astype(np.uint8).transpose(1, 2, 0)


def rgb_to_decorrelated(image):
    matrix = np.array(
        [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
    ).astype("float32")
    return torch.matmul(
        image.permute(0, 2, 3, 1), torch.tensor(matrix).to(image.device)
    ).permute(0, 3, 1, 2)


def decorrelated_to_rgb(image):
    matrix = np.array(
        [[4.52, -1.07, -0.25], [-0.04, -1.22, 0.78], [-0.99, 3.39, 0.24]]
    ).astype("float32")
    return torch.matmul(
        image.permute(0, 2, 3, 1), torch.tensor(matrix).to(image.device)
    ).permute(0, 3, 1, 2)


def random_jitter(image, max_pixels=16):
    dx, dy = np.random.randint(-max_pixels, max_pixels + 1, 2)
    return F.affine(image, 0, (dx, dy), 1, 0)


def random_scale(image, min_scale=0.8, max_scale=1.2):
    scale = np.random.uniform(min_scale, max_scale)
    return F.affine(image, 0, (0, 0), scale, 0)


def random_rotate(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    return F.rotate(image, angle)


def apply_transformations(image):
    image = random_jitter(image)
    image = random_scale(image)
    image = random_rotate(image)
    return image
