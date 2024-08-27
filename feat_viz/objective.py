from feat_viz.utils import create_simple_logger, M, T

import torch
from typing import List, Tuple, Union, Optional, Callable

logger = create_simple_logger(__name__)


def mean_loss(x: T):
    return -x.mean()


def norm_loss(x: T):
    return x.norm()


def sum_loss(x: T):
    return -x.sum()


def get_nth_matrix_element(matrix: T, n: int):
    """Gets the nth element from an 2D and matrix.
    If 2D matrix, return the element at index n, determined by traversing the matrix row by row. For example, for a 3x3 matrix, the elements are:
    
    0 1 2\\
    3 4 5\\
    6 7 8
    
    The element at index 4 is 4.

    For a 3D matrix, a slice of the matrix is returned. The slice is determined by the last dimension of the matrix. For example, for a 3x3x3 matrix, the elements are:

    0 1 2\\
    3 4 5\\
    6 7 8
    
    9 10 11\\
    12 13 14\\
    15 16 17
    
    18 19 20\\
    21 22 23\\
    24 25 26

    The slice at index 1 is: 1 10 19
    """
    if len(matrix.shape) > 3:
        raise ValueError(
            f"Matrix with more than 3 dimensions not supported. Got {len(matrix.shape)} dimensions"
        )

    if n >= matrix.numel():
        msg = f"Index {n} out of bounds for matrix of size {matrix.shape}"
        raise ValueError(msg)

    row = n // matrix.shape[-1]
    col = n % matrix.shape[-1]
    if len(matrix.shape) == 2:
        return matrix[row, col]
    return matrix[:, row, col]


class Hook:
    """A object that adds a forward hook to a layer in a model. The hook extracts features from the layer output and calculates the loss using a loss function. The loss function can be a simple function or a custom function. The hook can be called on a model to calculate the loss.

    Parameters
    ----------
    layer_name : str
        The name of the layer in the model. Must be a valid layer name in the model.
    channel_number : Optional[Union[int, List[int], Tuple[int, int], slice]], optional
        The channel number to extract from the layer output. The possible types are:

        - int: Select a single channel from the output. The output shape will be 1 x height x width.
        - List[int]: Concatenate multiple channels from the output. The output shape will be n x height x width.
        - slice: Slice channels from the output. The output shape will be n x height x width where n is the number of channels in the slice.
        - Tuple[int, int]: Slice channels from the output. The output shape will be n x height x width where `n = silce(channel_number[0], channel_number[1])`.

    neuron_number : Optional[int], optional
        The neuron number to extract from the channel output. If provided, the output shape will be 1 x width. By default None. To better understand how the neuron number is extracted, see the `get_nth_matrix_element` function.
    loss_function : Callable[[T], T], optional
        The loss function to calculate the loss from the extracted features. The loss function should take a tensor as input and return a scalar value. The default is `mean_loss` which calculates the mean of the tensor.

    Methods
    -------
    __call__(model: M) -> T:
        Calls the hook on the model to calculate the loss. Returns the loss value.
    has_hook() -> bool:
        Returns True if the hook has been added to the model, False otherwise.
    """

    def __init__(
        self,
        layer_name: str,
        channel_number: Optional[Union[int, List[int], Tuple[int, int], slice]] = None,
        neuron_number: Optional[int] = None,
        loss_function: Callable[[T], T] = mean_loss,
    ):
        self.layer_name = layer_name
        self.channel_number = channel_number
        self.neuron_number = neuron_number
        self.loss_function = loss_function
        self.feature: T = None
        self.loss: T = 0
        self.__hook = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_name={self.layer_name}, channel_number={self.channel_number}, neuron_number={self.neuron_number})"

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}(layer_name={self.layer_name}"
        if self.channel_number is not None:
            string += f", channel_number={self.channel_number}"
        if self.neuron_number is not None:
            string += f", neuron_number={self.neuron_number}"
        string += ")"
        return string

    def __call__(self, model: M):
        """Call the objective function on the model to calculate the loss. Accepts a model and returns the loss value. If the hook has not been added to the model, it will be added first."""
        if not self.has_hook():
            self.add_hook(model)
        return self.loss

    def has_hook(self) -> bool:
        return self.__hook is not None

    def add_hook(
        self,
        model: M,
    ):
        module = model.get_submodule(self.layer_name)
        if module is None:
            raise ValueError(f"Layer {self.layer_name} not found in model")
        if self.channel_number is None:
            logger.info(f"Adding hook to {self.layer_name}")
        else:
            logger.info(
                f"Adding hook to {self.layer_name} using {self.channel_number} channel"
            )

        def hook_fn(module: M, input: T, output: T) -> None:
            # shape will be batch_size x channels x height x width
            # take the first element from batch_size
            output = output[0]  # shape will be channels x height x width

            # if no channel number specified, use the whole output
            if self.channel_number is None:
                if self.neuron_number is not None:
                    # shape will be channels x width
                    output = get_nth_matrix_element(self.feature, self.neuron_number)
                self.feature = output
                output_shape = self.feature.shape
                logger.debug(f"Output shape: {output_shape}")
                self.loss = mean_loss(output)
                return

            # for a list of channel numbers, concatenate the channels
            if isinstance(self.channel_number, list):
                logger.debug(f"Channel number is list, concatenating channels")
                # convert to channel x height x width
                output = torch.cat(
                    [output[c] for c in self.channel_number], dim=0
                ).reshape(-1, *output.shape[1:])
            # for a slice of channel numbers, slice the channels
            elif isinstance(self.channel_number, slice):
                logger.debug(f"Channel number is slice, slicing channels")
                # shape will be n x height x width where n is the number of channels in the slice
                output = output[self.channel_number]

            elif isinstance(self.channel_number, tuple):
                logger.debug(
                    f"Channel number is tuple, slicing channels from {self.channel_number[0]} to {self.channel_number[1]}"
                )
                # shape will be n x height x width where n = silce(channel_number[0], channel_number[1])
                output = output[self.channel_number[0] : self.channel_number[1]]

            # for a single channel number, select the channel
            else:
                logger.debug(f"Channel number is int, selecting channel")
                # select and convert to 1 x height x width for consistency
                output = output[self.channel_number].reshape(-1, *output.shape[1:])
            if self.neuron_number is not None:
                # the shape will be n x width where n is the number of channels in the slice, 1 for a single channel
                output = get_nth_matrix_element(output, self.neuron_number)
            self.feature = output
            output_shape = self.feature.shape
            logger.debug(f"Output shape: {output_shape}")
            self.loss = mean_loss(self.feature)

        self.__hook = module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.__hook is not None:
            self.__hook.remove()
            self.__hook = None


class Objective:
    """A class to represent an objective function for optimization. An objective function is a function that takes a model and returns a scalar value. This will be used to optimize the model using gradient descent. The objective function can be a simple function or a hook that extracts features from a model. The objective function can be combined using arithmetic operations.

    Parameters
    ----------
    objective_func : Union[Hook, Callable[[M], T]
        The objective function. If a Hook is provided, the __call__ method of the Hook will be used. If a Callable is provided, the Callable will be used as the objective function. Note that if using a Hook, you will need to call the hook first on the model before passing it to the Objective to register the forward hook. If not, the loss for the first time will be zero since at that time the hook has not been called yet. From the second time onwards, the loss will be calculated correctly.
    name : Optional[str], optional
        The name of the objective function, by default None

    """

    def __init__(
        self,
        objective_func: Union[Hook, Callable[[M], T]],
        name: Optional[str] = None,
    ):
        if isinstance(objective_func, Hook):
            # use the hook's __call__ method
            self.objective_func = objective_func.__call__
        else:
            self.objective_func = objective_func

        self.name = name if name is not None else ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __call__(self, model: M):
        """Call the objective function on the model to calculate the loss. Accepts a model and returns the loss value."""
        return self.objective_func(model)

    def __add__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) + other
            new_name = f"{self.name}+{other}"
            return Objective(new_hook, new_name)

        new_hook = lambda model: self.objective_func(model) + other.objective_func(
            model
        )
        new_name = f"{self.name}+{other.name}"
        return Objective(new_hook, new_name)

    def __neg__(self) -> "Objective":
        new_hook = lambda model: -self.objective_func(model)
        new_name = f"-{self.name}"
        return Objective(new_hook, new_name)

    def __mul__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) * other
            new_name = f"{self.name}*{other}"
            return Objective(new_hook, new_name)

        new_hook = lambda model: self.objective_func(model) * other.objective_func(
            model
        )
        new_name = f"{self.name}*{other.name}"
        return Objective(new_hook, new_name)

    def __truediv__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) / other
            new_name = f"{self.name}/{other}"
            return Objective(new_hook, new_name)

        new_hook = lambda model: self.objective_func(model) / other.objective_func(
            model
        )
        new_name = f"{self.name}/{other.name}"
        return Objective(new_hook, new_name)

    def __rmul__(self, other: Union["Objective", float]) -> "Objective":
        return self.__mul__(other)

    def __radd__(self, other: Union["Objective", float]) -> "Objective":
        return self.__add__(other)

    @staticmethod
    def sum(objectives: List["Objective"]) -> "Objective":
        new_hook = lambda model: sum([obj.objective_func(model) for obj in objectives])
        new_name = "+".join([obj.name for obj in objectives])
        return Objective(new_hook, new_name)


def _parse_layer_channel_neuron_string(
    layer_channel_string: str,
) -> Tuple[str, Optional[Union[int, List[int], Tuple[int, int]]]]:
    """Parses a string in the format `layer_name:channel_number:neuron_number` where `channel_number` can be a single channel number, a list of channel numbers, or a tuple of channel numbers. The `neuron_number` is optional and can be used to extract a single neuron from the channel output.

    This can be used to specify the layer and channel number to extract features from using the `Hook` class.
    """
    if ":" in layer_channel_string:
        splits = layer_channel_string.split(":")
        if len(splits) == 2:
            logger.info(
                f"Found layer name and channel number in {layer_channel_string}"
            )
            layer_name, channel_number = splits
            neuron_number = None

        elif len(splits) == 3:
            logger.info(
                f"Found layer name, channel number, and neuron number in {layer_channel_string}"
            )
            layer_name, channel_number, neuron_number = splits
            neuron_number = int(neuron_number)
        else:
            raise ValueError(
                f"Invalid format for layer_channel_string: {layer_channel_string}"
            )

        # List of channel numbers
        if "," in channel_number:
            channel_number = [int(c) for c in channel_number.split(",")]
        # range of channel numbers
        elif "-" in channel_number:
            channel_number = tuple([int(c) for c in channel_number.split("-")])
        # single channel number
        else:
            channel_number = int(channel_number)
    else:
        logger.info(f"Selecting all channels from {layer_channel_string}")
        layer_name = layer_channel_string
        channel_number = None
        neuron_number = None
    return layer_name, channel_number, neuron_number


def create_objective(
    layer_channel_string: str,
    loss_function: Callable[[T], T] = mean_loss,
    name: Optional[str] = None,
) -> Objective:
    """Creates an objective function using the layer name, channel number and neuron number string (last two are optional). The layer name and channel number string should be in the format `layer_name:channel_number:neuron_number` where `channel_number` can be a single channel number, a list of channel numbers, or a tuple of channel numbers. The `neuron_number` is optional and can be used to extract a single neuron from the channel output. To better understand how the neuron number is extracted, see the `get_nth_matrix_element` function. The loss function should be a function that takes a tensor as input and returns a scalar value. The objective function will be created using the `Hook` class.

    Parameters
    ----------
    layer_channel_string : str
        The layer name and channel number string in the format `layer_name:channel_number`.
    loss_function : Callable[[T], T]
        The loss function to calculate the loss from the extracted features. By default, it is `mean_loss` which calculates the mean of the tensor.
    name : Optional[str], optional
        The name of the objective function, by default None
    """
    layer_name, channel_number, neuron_number = _parse_layer_channel_neuron_string(
        layer_channel_string
    )
    hook = Hook(layer_name, channel_number, neuron_number, loss_function)
    return Objective(hook.__call__, name or layer_channel_string)
