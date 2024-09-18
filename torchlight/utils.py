import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import get_model, get_model_weights
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import List, Union, Any, Callable, Optional, Dict
import logging
from PIL import Image
from IPython.display import display
from collections import OrderedDict


M = nn.Module
T = torch.Tensor
A = np.ndarray
END = "\033[0m"
BOLD = "\033[1m"
BROWN = "\033[0;33m"
ITALIC = "\033[3m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_logger_level_to_all_local(level: int) -> None:
    """Sets the level of all local loggers to the given level.

    Parameters
    ----------
    level : int, optional
        The level to set the loggers to, by default logging.DEBUG.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]

    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if hasattr(logger, "local"):
                logger.setLevel(level)


def create_simple_logger(
    logger_name: str, level: str = "info", set_level_to_all_loggers: bool = False
) -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical".

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.local = True
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if set_level_to_all_loggers:
        set_logger_level_to_all_local(level)
    return logger


logger = create_simple_logger(__name__)


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


class ImagePlotter:
    """A class to display images. It can be used to display images in a loop."""

    def __init__(
        self,
        cmap: str = "viridis",
        **kwargs: dict[str, any],
    ):
        """Initializes the ImagePlotter object.

        Parameters
        ----------
        title : str, optional
            The title of the figure. Default is "".
        cmap : str, optional
            The colormap to be used. Default is "viridis".
        kwargs
            Additional keyword arguments to be passed to the `plt.subplots` method.
        """

        self.fig, self.ax = plt.subplots(**kwargs)
        self.im = None
        self.cmap = cmap

    def update_image(
        self,
        image: Union[np.ndarray, Image.Image, T],
        title: str = "",
        path_to_save: Union[str, None] = None,
    ) -> None:
        # convert pil image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        # convert tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.squeeze().detach().cpu().numpy()
            # permute the channels to the last dimension
            image = np.transpose(image, (1, 2, 0))

        channels = image.shape[-1]
        if channels == 1 and self.cmap not in ["gray", "Greys"]:
            cmap = "gray"
        else:
            cmap = self.cmap

        if self.im is None:
            self.im = self.ax.imshow(image, cmap=cmap)
        else:
            self.im.set_data(image)
        self.ax.set_title(title)
        self.ax.title.set_fontsize(15)
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.01)
        # display the figure if running in a Jupyter notebook
        if is_jupyter_notebook():
            display(self.fig, clear=True)
        if path_to_save is not None:
            self.fig.savefig(path_to_save)


def create_wandb_logger(
    name: Union[str, None] = None,
    project: Union[str, None] = None,
    config: Union[dict[str, any], None] = None,
    tags: Union[list[str], None] = None,
    notes: str = "",
    group: Union[str, None] = None,
    job_type: str = "",
    logger: Union[logging.Logger, None] = None,
) -> Any:
    """Creates a new run on Weights & Biases and returns the run object.

    Parameters
    ----------
    project : str | None, optional
        The name of the project. If None, it must be provided in the config. Default is None.
    name : str | None, optional
        The name of the run. If None, it must be provided in the config. Default is None.
    config : dict[str, any] | None, optional
        The configuration to be logged. Default is None. If `project` and `name` are not provided, they must be present in the config.
    tags : list[str] | None, optional
        The tags to be added to the run. Default is None.
    notes : str, optional
        The notes to be added to the run. Default is "".
    group : str | None, optional
        The name of the group to which the run belongs. Default is None.
    job_type : str, optional
        The type of job. Default is "train".
    logger : logging.Logger | None, optional
        The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.

    Returns
    -------
    wandb.Run
        The run object.
    """
    import wandb

    logger = logger or create_simple_logger("create_wandb_logger")
    if config is None:
        logger.debug("No config provided. Using an empty config.")
        config = {}

    if name is None and "name" not in config.keys():
        m = "Run name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    if project is None and "project" not in config.keys():
        m = "Project name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    # If the arguments are provided, they take precedence over the config
    name = name or config.get("name")
    project = project or config.get("project")
    notes = notes or config.get("notes")
    tags = tags or config.get("tags")
    group = group or config.get("group")
    job_type = job_type or config.get("job_type")

    logger.info(
        f"Initializing Weights & Biases for project {project} with run name {name}."
    )
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
    )
    return wandb


def show_tensor_image(tensor: torch.Tensor, ax: Optional[matplotlib.axes.Axes] = None):
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    if ax:
        ax.imshow(img)
    else:
        plt.imshow(img)


def visualize_grid_images(
    images: Union[List[np.ndarray], List[torch.Tensor]],
    function_to_apply: Optional[Callable] = None,
    title_prefix: str = "",
) -> plt.Figure:
    if isinstance(images[0], torch.Tensor):
        images = [img.squeeze().cpu().detach().numpy() for img in images]
        # permute the channels to the last dimension
        images = [np.transpose(img, (1, 2, 0)) for img in images]
    num_images = len(images)
    columns = 5
    rows = int(np.ceil(num_images / columns))
    fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    for i, ax in enumerate(axs.flatten()):
        if i >= num_images:
            continue

        image = images[i]
        if function_to_apply:
            image = function_to_apply(image)
        image = np.clip(image, 0, 1)
        # plot the image
        ax.imshow(image)
        if title_prefix:
            ax.set_title(f"{title_prefix} {i}")
        ax.axis("off")
    plt.tight_layout()
    fig.show()
    return fig


def create_grid_image_from_tensor_images(
    images: List[torch.Tensor], nrow: int = 5
) -> torch.Tensor:
    grid_image = torchvision.utils.make_grid(images, nrow=nrow)
    return grid_image


def get_model_tree(
    model: nn.Module,
) -> Dict[str, Union[nn.Module, Dict[str, nn.Module]]]:
    model_tree = OrderedDict()
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            model_tree[name] = get_model_tree(module)
        else:
            # if the module has no children, it is a leaf node, add it to the tree
            model_tree[name] = module

    return model_tree


def print_model_tree(
    model_tree: Union[Dict[str, Union[nn.Module, Dict[str, nn.Module]]], nn.Module],
    indent: int = 0,
    add_modules: bool = False,
):
    if isinstance(model_tree, nn.Module):
        model_tree = get_model_tree(model_tree)

    for name, module in model_tree.items():
        ended = False
        print(" " * indent + f"{BOLD}{name}{END}:", end="")

        if isinstance(module, dict):
            if not ended:
                print()
            print_model_tree(module, indent + 2, add_modules=add_modules)
        else:
            if add_modules:
                print(f"{' ' * (indent+2)}{ITALIC}{module}{END}", end="")
        if not ended:
            print()


def create_all_possible_submodule_keys(
    tree: Union[Dict[str, Union[nn.Module, Dict[str, nn.Module]]], nn.Module],
    prefix: str = "",
) -> List[str]:
    if isinstance(tree, nn.Module):
        tree = get_model_tree(tree)
    keys = []
    for name, module in tree.items():
        keys.append(f"{prefix}{name}")
        if isinstance(module, dict):
            keys.extend(
                create_all_possible_submodule_keys(module, prefix=f"{prefix}{name}.")
            )
    return keys


def load_model(model_name: str) -> nn.Module:
    model = get_model(model_name, weights=get_model_weights(model_name).IMAGENET1K_V1)
    return model
