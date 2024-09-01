from feat_viz.utils import create_simple_logger, T, M, A

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from typing import Literal, Tuple, List, Optional, Union

# Color correlation matrix and normalization
color_correlation_svd_sqrt = np.array(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def pixel_image(
    shape: Tuple[int, int, int, int],
    sd: Optional[float] = None,
    device: Optional[str] = None,
) -> T:
    """A naive, pixel-based image parameterization.
    Defaults to a random initialization, but can take a supplied init_val argument
    instead.

    Args:
      shape: shape of resulting image, [batch, width, height, channels].
      sd: standard deviation of param initialization noise.
      init_val: an initial value to use instead of a random initialization. Needs
        to have the same shape as the supplied shape argument.

    Returns:
      tensor with shape from first argument.
    """

    sd = sd or 0.5
    init_val = np.random.normal(size=shape, scale=sd).astype(np.float32)
    # convert to pytorch tensor
    image = torch.tensor(init_val, device=device)
    # convert to batch*channels*height*width
    image = image.permute(0, 3, 1, 2)
    return image


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    """An image parameterization using 2D Fourier coefficients."""
    sd = sd or 0.5
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape
    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    spectrum_real_imag_t = torch.tensor(init_val)
    spectrum_t = torch.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar learning rates to pixel-wise optimization.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    scale_t = torch.tensor(scale, dtype=torch.float32)
    scaled_spectrum_t = scale_t * spectrum_t

    # convert complex scaled spectrum to shape (batch, h, w, ch) image tensor
    image_t = torch.fft.irfft2(scaled_spectrum_t, s=(h, w))

    # in case of odd spatial input dimensions we need to crop
    image_t = image_t[:batch, :h, :w, :]
    image_t = image_t / 4.0
    return image_t


def linear_decorelate_color(t: T) -> T:
    """Multiply input by sqrt of empirical (ImageNet) color correlation matrix."""
    color_correlation_svd_sqrt = np.array(
        [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
    ).astype("float32")
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

    # check that inner dimension is 3?
    t_flat = torch.reshape(t, [-1, 3])
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    color_correlation_normalized = torch.tensor(color_correlation_normalized)
    t_flat = torch.matmul(t_flat, color_correlation_normalized.T)
    t = torch.reshape(t_flat, t.shape)
    return t


def convert_to_valid_rgb(
    t: T,
    decorrelate: bool = False,
    sigmoid: bool = True,
    scaling_method: Literal["min_max", "norm", "clamp"] = "min_max",
) -> T:
    """Transform inner dimension of t to valid rgb colors."""
    if decorrelate:
        t = linear_decorelate_color(t)
    if decorrelate and not sigmoid:
        color_mean = np.array([0.48, 0.46, 0.41]).reshape(1, 3, 1, 1)
        t += color_mean

    if sigmoid:
        return torch.sigmoid(t)

    if scaling_method == "min_max":
        # use min max normalization
        t = (t - t.min()) / (t.max() - t.min())
    elif scaling_method == "norm":
        t = t / torch.maximum(t.abs().max(), torch.tensor(1.0))
        t = (t + 1.0) / 2.0
    elif scaling_method == "clamp":
        t = torch.clamp(t, 0.0, 1.0)
    return t


def get_image(
    w: int,
    h: Optional[int] = None,
    batch: Optional[int] = 1,
    sd: Optional[float] = 0.5,
    decorrelate: bool = True,
    fft: bool = True,
    alpha=False,
    sigmoid: bool = True,
    scaling_method: Literal["min_max", "norm", "clamp"] = "min_max",
) -> T:
    h = h or w
    batch = batch or 1
    ch = 4 if alpha else 3
    shape = [batch, h, w, ch]
    param_f = fft_image if fft else pixel_image
    t = param_f(shape, sd=sd)
    output = convert_to_valid_rgb(
        t[:, :3, ...],
        decorrelate=decorrelate,
        sigmoid=sigmoid,
        scaling_method=scaling_method,
    )
    if alpha:
        a = torch.nn.functional.sigmoid(
            t[
                :,
                3:,
                ...,
            ]
        )
        output = torch.cat([output, a], -1)
    return output


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


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
