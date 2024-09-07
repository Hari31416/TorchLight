import numpy as np
from feat_viz.image import (
    apply_transformations,
    get_image,
    normalize,
    preprocess_inceptionv1,
)
from feat_viz.io import show_image
from feat_viz.utils import (
    create_simple_logger,
    ImagePlotter,
    is_jupyter_notebook,
)
from feat_viz.objective import create_objective, Objective, T, M

import torch
import torch.optim as optim

from tqdm.auto import tqdm
from typing import Any, Iterable, Optional, List, Union, Tuple, Callable
import logging
from collections import OrderedDict

logger = create_simple_logger(__name__)


def remove_all_hooks(model: torch.nn.Module) -> None:
    """Remove all hooks from a neural network model."""
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                if child._forward_hooks != OrderedDict():
                    logger.debug(f"Removing forward hooks from {name}")
                    child._forward_hooks = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                if child._forward_pre_hooks != OrderedDict():
                    logger.debug(f"Removing forward pre hooks from {name}")
                    child._forward_pre_hooks = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                if child._backward_hooks != OrderedDict():
                    logger.debug(f"Removing backward hooks from {name}")
                    child._backward_hooks = OrderedDict()
            remove_all_hooks(child)


class FeatureViz:
    """Visualize features of a neural network using optimization.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to visualize.
    objective : Union[Objective, str]
        The objective function to optimize. If a string is passed, it will be used to create the objective function. The string must be in format "layer_name:channel_number". For example, "layer4:0" will optimize the first feature of the fourth layer. Have a look at the `feat_viz.objective.Hook` class for more information regarding `layer_name` and `channel_number`.
    log_level : str, optional
        The logging level, by default "warning".
    wandb_logger : Optional[Any], optional
        The Weights & Biases logger object, by default None. If None, no logging will be done.
    remove_existing_hooks : bool, optional
        Whether to remove existing hooks from the model, by default True. If True, all hooks will be removed before adding new ones.
    normalize_gradients : bool, optional
        Whether to normalize the gradients of the image, by default False. If True, the gradients will be normalized to have a norm of 1.
    clamp_param_value : Tuple[float, float], optional
        The range to clamp the image values, by default None. The image values will be clamped to this range after each optimization step. If None, no clamping will be done.
    """

    def __init__(
        self,
        model: M,
        objective: Union[Objective, str],
        log_level: str = "warning",
        wandb_logger: Optional[Any] = None,
        remove_existing_hooks: bool = True,
        normalize_gradients: bool = False,
        clamp_param_value: Tuple[float, float] = None,
    ) -> None:
        self.logger = create_simple_logger(self.__class__.__name__, log_level)

        self.model = model.eval()
        if remove_existing_hooks:
            remove_all_hooks(self.model)

        if isinstance(objective, str):
            objective = create_objective(objective)
        self.objective: Objective = objective

        # in case the objective was created using the Hook class,
        # we need to pass the model to the objective to register the hooks
        self.objective(self.model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"Using device: {self.device}")
        self.model.to(self.device)
        self.wandb_logger = wandb_logger
        self.normalize_gradients = normalize_gradients
        self.clamp_param_value = clamp_param_value

    def normalize_grad(self, tensor: List[T]) -> None:
        for t in tensor:
            t.grad.data.copy_(t.grad / torch.norm(t))

    def visualize(
        self,
        param_f: Optional[Callable[[], Tuple[List[T], Callable[[], T]]]] = None,
        thresholds: Optional[Iterable[int]] = (128,),
        lr: float = 0.05,
        freq: int = 20,
        use_decorrelated: bool = False,
        fft: bool = True,
        image_shape: Union[Tuple[int, int], Tuple[int, int, int], int] = (224, 224),
        plot_images: bool = False,
        show_last_image: bool = False,
        show_progress: bool = True,
        log_loss: bool = False,
        extra_transformations: Optional[List[Callable[[T], T]]] = None,
        preprocess: bool = True,
    ) -> List[T]:
        """Main method to visualize the features of the neural network.

        Parameters
        ----------
        param_f : Optional[Callable[[], Tuple[List[T], Callable[[], T]]]], optional
            The function to create the image. If None, a random image will be created, by default None. The function must return two values: the parameters that are to be optimzed and a function that creates the image.
        thresholds : Optional[Iterable[int]], optional
            The iterations on which to return the image, by default [128].
        lr : float, optional
            The learning rate of the optimizer, by default 0.1.
        freq : int, optional
            The frequency of logging the loss and saving the image, by default 20.
        use_decorrelated : bool, optional
            Whether to use the decorrelated color space, by default False.
        fft : bool, optional
            Whether to use the Fast Fourier Transform (FFT) to create the image, by default True.
        image_shape : Union[Tuple[int, int], Tuple[int, int, int], int], optional
            The shape of the image to optimize, by default (224, 224). If an integer is passed, the image will have the same width and height.
        plot_images : bool, optional
            Whether to plot the images while optimizing, by default False. If True, the images will be plotted at an interval of `freq`.
        show_last_image : bool, optional
            Whether to show the last image after optimization, by default False. If True, the last image will be displayed.
        show_progress : bool, optional
            Whether to show the progress bar, by default True.
        log_loss : bool, optional
            Whether to log the loss, by default False. If True, the loss will be logged at each iteration divisible by `freq`.
        extra_transformations : Optional[List[Callable[[T], T]], optional
            A list of extra transformations to apply to the image, by default None.
        preprocess : bool, optional
            Whether to apply the default preprocessing to the image, by default True. If True, the default preprocessing will be applied to the image before passing it to the neural network.

        Returns
        -------
        List[T]
            A list of images optimized at different steps.
        """
        if plot_images:
            image_plotter = ImagePlotter()
        images = []
        if param_f is None:
            if isinstance(image_shape, int):
                image_shape = (1, 3, image_shape, image_shape)
            elif len(image_shape) == 2:
                image_shape = (1, 3, *image_shape)
            elif len(image_shape) == 3:
                image_shape = (1, *image_shape)
            batch, channels, height, width = image_shape
            params, image_f = get_image(
                w=width,
                h=height,
                batch=batch,
                sd=0.5,
                decorrelate=use_decorrelated,
                fft=fft,
                alpha=channels == 4,
                sigmoid=True,
                scaling_method="min_max",
                device=self.device,
            )
        else:
            params, image_f = param_f()

        for p in params:
            p.requires_grad_(True)
            p.to(self.device)

        optimizer = optim.Adam(params, lr=lr)
        num_iterations = max(thresholds) + 1
        for i in tqdm(range(num_iterations), disable=not show_progress):

            if extra_transformations is None:
                extra_transformations = []
            if preprocess:
                if self.model._get_name() == "InceptionV1":
                    extra_transformations.append(preprocess_inceptionv1)
                else:
                    extra_transformations.append(normalize)

            transformed_image = apply_transformations(
                image_f(), extra_transformations=extra_transformations
            ).to(self.device)

            optimizer.zero_grad()
            try:
                self.model(transformed_image)
            except RuntimeError as ex:
                if i == 0:
                    # Only display the warning message on the first iteration, no need to do that every iteration
                    self.logger.warning(
                        f"Some layers could not be computed because the size of the image is not big enough. It is fine, as long as the non computed layers are not used in the objective function.\nException: {ex}"
                    )
            loss = self.objective(self.model)
            loss.backward()

            if self.normalize_gradients:
                self.normalize_grad(params)
            optimizer.step()

            if self.clamp_param_value is not None:
                with torch.no_grad():
                    for p in params:
                        p.data.clamp_(*self.clamp_param_value)

            if i % freq == 0:
                if log_loss:
                    self.logger.info(f"Loss: {loss.item()}")
                else:
                    self.logger.debug(f"Loss: {loss.item()}")

                image_to_log = (
                    image_f()[0].clone().detach().cpu().numpy()
                )  # show only the first image in the batch
                image_to_log = np.transpose(image_to_log, (1, 2, 0))
                if is_jupyter_notebook() and plot_images:
                    image_plotter.update_image(
                        image_to_log,
                        title=f"It: {i} | Loss: {loss.item():.2f}",
                    )
                if self.wandb_logger:
                    self.wandb_logger.log(
                        {
                            "loss": loss.item(),
                            "image": self.wandb_logger.Image(image_to_log),
                        }
                    )
            if i in thresholds:
                img = image_f().squeeze(0).detach().cpu().numpy()
                rank = len(img.shape)
                if rank == 3:
                    img = np.transpose(img, (1, 2, 0))
                else:
                    # multiple images
                    img = np.transpose(img, (0, 2, 3, 1))
                images.append(img)

        if show_last_image:
            if is_jupyter_notebook():
                show_image(images[-1])
            else:
                logger.warning(
                    "The last image can only be displayed in a Jupyter Notebook."
                )
        return images
