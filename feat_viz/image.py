from feat_viz.utils import create_simple_logger, T, M, A

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from typing import Literal, Tuple, List, Optional, Union, Callable

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_VERSION = torch.__version__
# Color correlation matrix and normalization
color_correlation_svd_sqrt = np.array(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def pixel_image(
    shape: Tuple[int, int, int, int],
    sd: Optional[float] = None,
    device: Optional[str] = DEFAULT_DEVICE,
) -> Tuple[T, Callable[[], T]]:
    """A naive, pixel-based image parameterization.
    Defaults to a random initialization, but can take a supplied init_val argument
    instead.

    Parameters
    ----------
    shape : Tuple[int, int, int, int]
        shape of resulting image, [batch, channels, height, width].
    sd : Optional[float], optional
        standard deviation of param initialization noise, by default None
    device : Optional[str], optional
        device to use for tensor, by default DEFAULT_DEVICE

    Returns
    -------
    Tuple[T, Callable[[], T]]
        Tuple containing the image tensor and a function to generate the image
    """

    sd = sd or 0.5
    init_val = np.random.normal(size=shape, scale=sd).astype(np.float32)
    # convert to pytorch tensor
    image = torch.tensor(init_val, device=device)
    image.requires_grad_(True)
    # convert to batch*channels*height*width
    # image = image.permute(0, 3, 1, 2)
    return [image], lambda: image


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


def fft_image(shape, sd=None, decay_power=1, device=DEFAULT_DEVICE):
    sd = sd or 0.05
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, channels) + freqs.shape + (2,)
    )  # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inverse():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft

            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        else:
            import torch

            image = torch.irfft(
                scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w)
            )
        image = image[:batch, :channels, :h, :w]
        magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image

    return [spectrum_real_imag_t], inverse


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
    sigmoid: bool = False,
    scaling_method: Literal["min_max", "norm", "clamp"] = "min_max",
    device: Optional[str] = DEFAULT_DEVICE,
) -> T:
    h = h or w
    batch = batch or 1
    c = 4 if alpha else 3
    shape = [batch, c, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)

    def image():
        t = image_f()
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
            output = torch.cat([output, a], 1)
        return output

    return params, image


def random_jitter(max_pixels=16):
    def inner(image):
        dx, dy = np.random.randint(-max_pixels, max_pixels + 1, 2)
        return F.affine(image, 0, (dx, dy), 1, 0)

    return inner


def random_scale(min_scale=0.8, max_scale=1.2):
    def inner(image):
        scale = np.random.uniform(min_scale, max_scale)
        return F.affine(image, 0, (0, 0), scale, 0)

    return inner


def random_rotate(max_angle=15):
    def inner(image):
        angle = np.random.uniform(-max_angle, max_angle)
        return F.rotate(image, angle)

    return inner


def pad_image(padding=16):
    def inner(image):
        return F.pad(image, padding)

    return inner


def crop_image(x, y):
    def inner(image):
        w = image.shape[-1] - x
        h = image.shape[-2] - y
        return F.crop(image, x, y, h, w)

    return inner


def normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def preprocess_inceptionv1():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


DEFAULT_TRANSFORMATIONS = [
    pad_image(16),
    random_jitter(16),
    random_scale(0.95, 1.05),
    random_rotate(5),
    random_jitter(16),
    crop_image(16, 16),
]


def apply_transformations(
    image: T, extra_transformations: Optional[List[A]] = None
) -> T:
    transformations = DEFAULT_TRANSFORMATIONS
    if extra_transformations:
        transformations = transformations + extra_transformations
    for transform in transformations:
        image = transform(image)
    return image
