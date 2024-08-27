from feat_viz.image import (
    apply_transformations,
    get_image,
)
from feat_viz.utils import (
    create_simple_logger,
    ImagePlotter,
    is_jupyter_notebook,
)
from feat_viz.objective import create_objective, Objective, T, M

import torch
import torch.optim as optim

from tqdm.auto import tqdm
from typing import Optional, List, Union, Tuple
import logging

logger = create_simple_logger(__name__)


class FeatureViz:
    """Visualize features of a neural network using optimization.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to visualize.
    objective : Union[Objective, str]
        The objective function to optimize. If a string is passed, it will be used to create the objective function. The string must be in format "layer_name:channel_number". For example, "layer4:0" will optimize the first feature of the fourth layer. Have a look at the `feat_viz.objective.Hook` class for more information regarding `layer_name` and `channel_number`.
    logger : Optional[logging.Logger], optional
        The logger object to use, by default None. If None, a simple logger will be created.
    """

    def __init__(
        self,
        model: M,
        objective: Union[Objective, str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or create_simple_logger(__name__)

        self.model = model.eval()
        if isinstance(objective, str):
            objective = create_objective(objective)
        self.objective: Objective = objective

        # in case the objective was created using the Hook class,
        # we need to pass the model to the objective to register the hooks
        self.objective(self.model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

    def normalize_grad(self, tensor: T) -> None:
        tensor.grad.data.copy_(tensor.grad / torch.norm(tensor))

    def visualize(
        self,
        image: Optional[T] = None,
        num_iterations: int = 100,
        lr: float = 0.1,
        freq: int = 20,
        use_decorrelated: bool = False,
        image_shape: Union[Tuple[int, int], Tuple[int, int, int], int] = (224, 224),
        plot_images: bool = True,
    ) -> List[T]:
        """Main method to visualize the features of the neural network.

        Parameters
        ----------
        image : Optional[T], optional
            The initial image to optimize, by default None. If None, a random image will be created using `torch.randn` and the `image_shape` parameter.
        num_iterations : int, optional
            The number of optimization steps, by default 100.
        lr : float, optional
            The learning rate of the optimizer, by default 0.1.
        freq : int, optional
            The frequency of logging the loss and saving the image, by default 20.
        use_decorrelated : bool, optional
            Whether to use the decorrelated color space, by default False.
        image_shape : Union[Tuple[int, int], Tuple[int, int, int], int], optional
            The shape of the image to optimize, by default (224, 224). If an integer is passed, the image will have the same width and height.

        Returns
        -------
        List[T]
            A list of images optimized at different steps.
        """
        if plot_images:
            image_plotter = ImagePlotter()
        images = []
        if image is None:
            if isinstance(image_shape, int):
                image_shape = (1, 3, image_shape, image_shape)
            elif len(image_shape) == 2:
                image_shape = (1, 3, *image_shape)
            elif len(image_shape) == 3:
                image_shape = (1, *image_shape)
            batch, channels, height, width = image_shape
            image = get_image(
                w=width,
                h=height,
                batch=batch,
                sd=0.5,
                decorrelate=use_decorrelated,
                fft=True,
                alpha=channels == 4,
                sigmoid=True,
                scaling_method="min_max",
            )

        image = image.to(self.device)
        image.requires_grad_(True)

        optimizer = optim.Adam([image], lr=lr)

        for i in tqdm(range(num_iterations)):
            transformed_image = apply_transformations(image).to(self.device)
            optimizer.zero_grad()
            try:
                self.model(transformed_image)
            except RuntimeError as ex:
                if i == 0:
                    # Only display the warning message
                    # on the first iteration, no need to do that every iteration
                    self.logger.warning(
                        f"Some layers could not be computed because the size of the image is not big enough. It is fine, as long as the non computed layers are not used in the objective function.\nException: {ex}"
                    )
            loss = self.objective(self.model)
            loss.backward()
            self.normalize_grad(image)
            optimizer.step()

            # Clip values to keep them in a valid range
            with torch.no_grad():
                image.clamp_(-1, 1)

            if i % freq == 0:
                self.logger.info(f"Loss: {loss.item()}")
                image_to_log = image.clone().detach().cpu()
                images.append(image_to_log)

                if is_jupyter_notebook() and plot_images:
                    image_plotter.update_image(
                        image_to_log,
                        title=f"It: {i} | Loss: {loss.item():.2f}",
                    )
        image.detach_()
        return images
