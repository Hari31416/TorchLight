from torchlight.utils import create_simple_logger, M, T, A

import torch
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

logger = create_simple_logger(__name__)


def mean_loss(x: T):
    return -x.mean()


def norm_loss(x: T):
    return x.norm()


def sum_loss(x: T):
    return -x.sum()


def get_matrix_element_around_position(
    matrix: T,
    x: Optional[int] = None,
    y: Optional[int] = None,
    wx: int = 1,
    wy: int = 1,
) -> T:
    if len(matrix.shape) < 2:
        raise ValueError(
            f"Matrix with less than 2 dimensions not supported. Got {len(matrix.shape)} dimensions"
        )

    if x is None:
        x = matrix.shape[-2] // 2
    if y is None:
        y = matrix.shape[-1] // 2
    return matrix[..., x : x + wx, y : y + wy]  # handle 3D+ matrices


def get_matrix_element_at_position(
    matrix: T, x: Optional[int] = None, y: Optional[int] = None
) -> T:
    """Gets the element at position (x, y) from a 2D+ matrix. If x and y are not provided, the center element is returned. Output dimensionality will be the same as the input matrix."""
    return get_matrix_element_around_position(matrix, x, y, 1, 1)


def get_nth_matrix_element(matrix: T, n: int) -> T:
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

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to extract the element from.
    n : int
        The index of the element to extract. The index is determined by traversing the matrix row by row.

    Returns
    -------
    torch.Tensor
        The element at index n from the matrix. Note that the dimensionality of the output will remain the same as the input matrix.
    """
    if n >= matrix.numel():
        msg = f"Index {n} out of bounds for matrix of size {matrix.shape}"
        raise ValueError(msg)

    row = n // matrix.shape[-1]
    col = n % matrix.shape[-1]
    return get_matrix_element_at_position(matrix, row, col)


def diversity_loss(layer_output: T) -> T:
    # Source: https://github.com/greentfrapp/lucent/blob/dev/lucent/optvis/objectives.py#L318
    batch, channels, _, _ = layer_output.shape
    flattened = layer_output.view(batch, channels, -1)
    grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
    grams = torch.nn.functional.normalize(grams, p=2, dim=(1, 2))
    return (
        -sum(
            [
                sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                for i in range(batch)
            ]
        )
        / batch
    )


def alignment_loss(layer_output: T, decay_ratio: int = 2) -> T:
    # Source: https://github.com/greentfrapp/lucent/blob/dev/lucent/optvis/objectives.py#L304
    batch_n = list(layer_output.shape)[0]
    accum = 0
    for d in [1, 2, 3, 4]:
        for i in range(batch_n - d):
            a, b = i, i + d
            arr_a, arr_b = layer_output[a], layer_output[b]
            accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
    return accum


def neuron_direction_in_layer(layer_output: T, direction: Union[A, T]) -> T:
    if isinstance(direction, A):
        direction = torch.tensor(
            direction, dtype=layer_output.dtype, device=layer_output.device
        )
    else:
        direction = direction.to(layer_output.device)

    layer_mean = layer_output.mean(
        dim=(-1, -2)
    )  # last two dimensions are height and width
    # reshape to make b*c*1*1
    layer_mean = layer_mean.view(layer_mean.shape[0], layer_mean.shape[1], 1, 1)
    direction = direction.view(1, -1, 1, 1)
    if layer_mean.shape[1] != direction.shape[1]:
        msg = f"Direction shape {direction.shape} does not match layer output shape {layer_mean.shape}"
        logger.error(msg)
        raise ValueError(msg)

    out = torch.nn.functional.cosine_similarity(layer_mean, direction, dim=1)
    logger.debug(
        f"Layer mean shape: {layer_mean.shape}, Direction shape: {direction.shape}, Cosine similarity shape: {out.shape}"
    )
    return -torch.mean(out)


def neuron_direction_in_layer_at_pos(
    layer_output: T, x: int, y: int, direction: Union[A, T]
) -> T:
    if isinstance(direction, A):
        direction = torch.tensor(
            direction, dtype=layer_output.dtype, device=layer_output.device
        )
    else:
        direction = direction.to(layer_output.device)

    neuron_output = get_matrix_element_at_position(layer_output, x, y)
    neuron_output = neuron_output.view(
        neuron_output.shape[0], neuron_output.shape[1], 1, 1
    )
    direction = direction.view(1, -1, 1, 1)
    if neuron_output.shape[1] != direction.shape[1]:
        msg = f"Direction shape {direction.shape} does not match layer output shape {neuron_output.shape}"
        logger.error(msg)
        raise ValueError(msg)

    out = torch.nn.functional.cosine_similarity(neuron_output, direction, dim=1)
    logger.debug(
        f"Neuron output shape: {neuron_output.shape}, Direction shape: {direction.shape}, Cosine similarity shape: {out.shape}"
    )
    return -torch.mean(out)


def neuron_direction_in_channel(
    layer_output: T, channel: int, direction: Union[A, T]
) -> T:
    if isinstance(direction, A):
        direction = torch.tensor(
            direction, dtype=layer_output.dtype, device=layer_output.device
        )
    else:
        direction = direction.to(layer_output.device)

    channel_output = layer_output[:, channel, :, :]
    # flatten the last two dimensions
    channel_output = channel_output.view(channel_output.shape[0], 1, -1)

    direction = direction.view(1, 1, -1)

    if channel_output.shape[-1] != direction.shape[-1]:
        msg = f"Direction shape {direction.shape} does not match layer output shape {channel_output.shape}"
        logger.error(msg)
        raise ValueError(msg)

    out = torch.nn.functional.cosine_similarity(channel_output, direction, dim=1)
    logger.debug(
        f"Channel output shape: {channel_output.shape}, Direction shape: {direction.shape}, Cosine similarity shape: {out.shape}"
    )
    return -torch.mean(out)


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

    neuron_number : Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]
        The neuron number to extract from the channel output. The possible types are:

        - int: Select a single neuron from the channel output or activations along channel with position determined by `neuron_number` by traversing the matrix row by row (See `get_nth_matrix_element` method for details). The output shape will be `width x 1 x 1` where width is the number of channels in the slice, 1 for a single channel.
        - Tuple[int, int]: Selects a single neuron from the channel output or activations along channel with position determined by the tuple (see `get_matrix_element_at_position` method for details). The output shape will be `width x 1 x 1` where width is the number of channels in the slice, 1 for a single channel.
        - Tuple[Tuple[int, int], Tuple[int, int]]: Selects a slice of neurons from the channel output or activations along channel where position is determined by the first two elements of the tuple and width and height is determined by the last twp elements. (see `get_matrix_element_around_position` method for details). The output shape will be `width x m x n` where width is determined by the number of channels in the slice, m and n are determined by the last two elements of the tuple.

    loss_function : Callable[[T], T], optional
        The loss function to calculate the loss from the extracted features. The loss function should take a tensor as input and return a scalar value. The default is `mean_loss` which calculates the mean of the tensor.
    extractor_function : Optional[Callable[[T], T]], optional
        A custom function to extract features from the layer output. The function should take a tensor as input and return a tensor. By default None. The input tensor will be the output of the layer with shape batch_size x channels x height x width. The function must return a scalar tensor that is used as the loss.
    extractor_function_kwargs : Optional[dict], optional
        Keyword arguments to pass to the `extractor_function`, by default None.
    batch : Optional[int], optional
        The batch number to extract from the output. By default None which means the first batch is used. In case of None and an `extractor_function` is used, the function must handle the batch dimension itself, the input to the function will be the output of the layer with shape `batch x channels x height x width`. Even if the batch is provided, the output will still be `1 x channels x height x width` and not `channels x height x width`.

    Methods
    -------
    __call__(model: M) -> T:
        Calls the hook on the model to calculate the loss. Returns the loss value.
    has_hook() -> bool:
        Returns True if the hook has been added to the model, False otherwise.

    Notes
    -----
    Some possible ways to use the Hook class:

    1. Extract a single channel from the output of a layer using the channel number:
        ```
        hook = Hook("layer4", channel_number=0)
        ```
    2. Extract a list of channels from the output of a layer using a list of channel numbers:
        ```
        hook = Hook("layer4", channel_number=[0, 1, 2])
        ```
    3. Extract a slice of activations along the channel dimension:
        ```
        hook = Hook("layer4", channel_number=None, neuron_number=5)
        ```
    4. Extract a slice of neurons from multiple channel:
        ```
        hook = Hook("layer4", channel_number=[0, 1, 2], neuron_number=(0, 0, 3, 3))
        ```
    """

    def __init__(
        self,
        layer_name: str,
        channel_number: Optional[Union[int, List[int], Tuple[int, int], slice]] = None,
        neuron_number: Optional[
            Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]
        ] = None,
        loss_function: Callable[[T], T] = mean_loss,
        extractor_function: Optional[Callable[[T], T]] = None,
        extractor_function_kwargs: Optional[dict] = {},
        batch: Optional[int] = None,
    ):
        self.layer_name = layer_name
        self.channel_number = channel_number
        self.neuron_number = neuron_number
        self.loss_function = loss_function
        self.extractor_function = extractor_function
        self.extractor_function_kwargs = extractor_function_kwargs
        self.feature: T = None
        self.loss: T = 0
        self.__hook = None
        self.batch = batch

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

        if self.extractor_function is not None:
            logger.info(
                f"Adding hook to {self.layer_name} using custom extractor function"
            )

            def hook_fn(module: M, input: T, output: T) -> None:
                if self.batch is not None:
                    logger.info(f"Extracting batch {self.batch} from output")
                    output = output[
                        self.batch : self.batch + 1
                    ]  # take the batch from the output

                self.feature = self.extractor_function(
                    output, **self.extractor_function_kwargs
                )
                # in case a function is used, do not use the loss_function
                self.loss = self.feature

            self.__hook = module.register_forward_hook(hook_fn)
            return

        if self.neuron_number is not None:
            if isinstance(self.neuron_number, tuple):
                if isinstance(self.neuron_number[0], tuple):
                    logger.debug(
                        f"Adding hook to {self.layer_name} using slice of neurons"
                    )
                    function_for_slice = lambda x: get_matrix_element_around_position(
                        x,
                        *self.neuron_number[0],  # x and y
                        *self.neuron_number[1],  # width and height
                    )
                else:
                    if len(self.neuron_number) != 2:
                        msg = f"Invalid neuron number: {self.neuron_number}. Must be a tuple of length 2."
                        raise ValueError(msg)

                    logger.debug(
                        f"Adding hook to {self.layer_name} using a single neuron from position"
                    )
                    function_for_slice = lambda x: get_matrix_element_at_position(
                        x, *self.neuron_number
                    )
            else:
                logger.debug(
                    f"Adding hook to {self.layer_name} using neuron number {self.neuron_number}"
                )
                function_for_slice = lambda x: get_nth_matrix_element(
                    x, self.neuron_number
                )

        if self.channel_number is None:
            logger.info(f"Adding hook to {self.layer_name}")
        else:
            logger.info(
                f"Adding hook to {self.layer_name} using {self.channel_number} channel"
            )

        def hook_fn(module: M, input: T, output: T) -> None:
            # shape will be batch_size x channels x height x width
            # take the first element from batch_size
            if self.batch is not None:
                logger.info(f"Extracting batch {self.batch} from output")
                output = output[self.batch]  # shape will be channels x height x width
            else:
                output = output[0]  # shape will be channels x height x width

            # if no channel number specified, use the whole output
            if self.channel_number is None:
                if self.neuron_number is not None:
                    # shape will be channels x width
                    output = function_for_slice(output)
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
                # the shape will be width x m x n where width is the number of channels in the slice, m and n
                # possibly 1, are the width of slices for neurons
                output = function_for_slice(output)
            self.feature = output
            output_shape = self.feature.shape
            logger.debug(f"Output shape: {output_shape}")
            self.loss = mean_loss(self.feature)

        self.__hook = module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.__hook is not None:
            self.__hook.remove()
            self.__hook = None


class ModuleHook:
    def __init__(self, module: M, batch: Optional[int] = None) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module: M = None
        self.features: T = None
        self.batch = batch

    def hook_fn(self, module, input, output):
        self.module: M = module
        self.features: T = output
        if self.batch is not None:
            self.features = self.features[self.batch : self.batch + 1]


class MultiHook:
    """A class that adds multiple hooks to a model. The hooks extract features from multiple layers in the model and calculate the loss using a loss function. This class can be used for the cases when features from multiple layers are required to calculate the loss. If features from a only a single layer is required, use the `Hook` class.

    Parameters
    ----------
    layer_names : List[str]
        The list of layer names in the model. Must be valid layer names in the model.
    extractor_function : Callable[[T], T]
        A custom function to extract features from the layer output. The function should take a list of tensors as input and return a tensor. The input will be a list of tensors where each tensor is the output of the corresponding layer in the `layer_names` list. The function must return a scalar tensor that is used as the loss.
    extractor_function_kwargs : Optional[dict], optional
        Keyword arguments to pass to the `extractor_function`, by default None.
    batch : Optional[int], optional
        The batch number to extract from the output. By default None which means that the whole output is passed to the extractor function (with shape batch*channels*height*width). If not None, the batch number is extracted from the output and passed to the extractor function. The resulting shape will be 1*channels*height*width.
    """

    def __init__(
        self,
        layer_names: List[str],
        extractor_function: Callable[[T], List[T]],
        extractor_function_kwargs: Optional[dict] = {},
        batch: Optional[int] = None,
    ) -> None:
        self.layer_names = layer_names
        self.extractor_function = extractor_function
        self.extractor_function_kwargs = extractor_function_kwargs
        self.batch = batch
        self.features: OrderedDictType[str, ModuleHook] = OrderedDict()
        self.loss = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_names={self.layer_names})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(layer_names={self.layer_names})"

    def has_hook(self) -> bool:
        return len(self.features) > 0

    def __call__(self, model: M) -> T:
        if not self.has_hook():
            logger.info("Adding hooks to model")
            self.add_hook_multiple(model)

        features = [hook.features for hook in self.features.values()]
        if None in features:
            msg = "No features found in any of the hooks. Make a forward pass on the model first."
            logger.info(msg)
            return 0

        self.loss = self.extractor_function(features, **self.extractor_function_kwargs)
        return self.loss

    def add_hook_multiple(
        self,
        model: M,
    ):
        def hook_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():

                    # hook the layer if it is in the layer_names list
                    if layer is None or name not in self.layer_names:
                        continue

                    self.features[".".join(prefix + [name])] = ModuleHook(
                        layer, self.batch
                    )
                    hook_layers(layer, prefix=prefix + [name])

        hook_layers(model)

        # check if all layers have been hooked and warn if not
        layers_with_hook = list(self.features.keys())
        if len(layers_with_hook) == 0:
            msg = f"Could not find any layers in model with names: {self.layer_names}"
            logger.error(msg)
            raise ValueError(msg)

        layers_without_hook = [
            layer for layer in self.layer_names if layer not in layers_with_hook
        ]

        if len(layers_without_hook) > 0:
            logger.warning(f"Could not find layers: {layers_without_hook}")


class Objective:
    """A class to represent an objective function for optimization. An objective function is a function that takes a model and returns a scalar value. This will be used to optimize the model using gradient descent. The objective function can be a simple function or a hook that extracts features from a model. The objective function can be combined using arithmetic operations.

    Parameters
    ----------
    objective_func : Union[Hook, Callable[[M], T]
        The objective function. If a Hook is provided, the __call__ method of the Hook will be used. If a Callable is provided, the Callable will be used as the objective function. Note that if using a Hook, you will need to call the hook first on the model before passing it to the Objective to register the forward hook. If not, the loss for the first time will be zero since at that time the hook has not been called yet. From the second time onwards, the loss will be calculated correctly.
    name : Optional[str], optional
        The name of the objective function, by default None
    hook_objects : Optional[Union[Hook, List[Hook]], optional
        The hook objects to add to the objective function. By default None. If two objective functions are combined via arithmetic operations, the hook objects will be combined as well.

    """

    def __init__(
        self,
        objective_func: Union[Hook, Callable[[M], T]],
        name: Optional[str] = None,
        hook_objects: Optional[Union[Hook, List[Hook]]] = None,
    ):
        if isinstance(objective_func, Hook):
            # use the hook's __call__ method
            self.objective_func = objective_func.__call__
        else:
            self.objective_func = objective_func
        if hook_objects is None:
            self.hook_objects = []
        elif isinstance(hook_objects, Hook):
            self.hook_objects = [hook_objects]
        else:
            self.hook_objects = hook_objects

        self.name = name if name is not None else ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __call__(self, model: M):
        """Call the objective function on the model to calculate the loss. Accepts a model and returns the loss value."""
        # model is only required if hook is to be added
        # if hook is already added, model is not required and will not be used
        return self.objective_func(model)

    def __add__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) + other
            new_name = f"{self.name}+{other}"
            return Objective(new_hook, new_name, self.hook_objects)

        new_hook = lambda model: self.objective_func(model) + other.objective_func(
            model
        )
        new_name = f"{self.name}+{other.name}"
        hook_objects = self.hook_objects + other.hook_objects
        return Objective(new_hook, new_name, hook_objects)

    def __neg__(self) -> "Objective":
        new_hook = lambda model: -self.objective_func(model)
        new_name = f"-{self.name}"
        return Objective(new_hook, new_name, self.hook_objects)

    def __sub__(self, other: Union["Objective", float]) -> "Objective":
        return self + (-1 * other)

    def __mul__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) * other
            new_name = f"{self.name}*{other}"
            return Objective(new_hook, new_name, self.hook_objects)

        new_hook = lambda model: self.objective_func(model) * other.objective_func(
            model
        )
        new_name = f"{self.name}*{other.name}"
        return Objective(new_hook, new_name, self.hook_objects + other.hook_objects)

    def __truediv__(self, other: Union["Objective", float]) -> "Objective":
        if isinstance(other, (int, float)):
            new_hook = lambda model: self.objective_func(model) / other
            new_name = f"{self.name}/{other}"
            return Objective(new_hook, new_name, self.hook_objects)

        new_hook = lambda model: self.objective_func(model) / other.objective_func(
            model
        )
        new_name = f"{self.name}/{other.name}"
        return Objective(new_hook, new_name, self.hook_objects + other.hook_objects)

    def __rmul__(self, other: Union["Objective", float]) -> "Objective":
        return self.__mul__(other)

    def __radd__(self, other: Union["Objective", float]) -> "Objective":
        return self.__add__(other)

    @staticmethod
    def sum(objectives: List["Objective"]) -> "Objective":
        new_hook = lambda model: sum([obj.objective_func(model) for obj in objectives])
        new_name = "+".join([obj.name for obj in objectives])
        hook_objects = [hook for obj in objectives for hook in obj.hook_objects]
        return Objective(new_hook, new_name, hook_objects)


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
    batch: Optional[int] = None,
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
    batch : Optional[int], optional
        The batch number to extract from the output. By default None which means the first batch is used. See the `Hook` class for more details.
    """
    layer_name, channel_number, neuron_number = _parse_layer_channel_neuron_string(
        layer_channel_string
    )
    hook = Hook(layer_name, channel_number, neuron_number, loss_function, batch=batch)
    return Objective(hook.__call__, name or layer_channel_string, hook_objects=hook)


def create_objective_from_function(
    extractor_function: Callable[[T], T],
    layer_name: str,
    batch: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> Objective:
    """Creates an objective function using a custom function. The custom function should take a tensor as input and return a tensor. The objective function will be created using the `Hook` class. The `extractor_function` must take a tensor as input and return a scalar tensor as output to be used as loss. The `layer_name` is the name of the layer in the model. The `kwargs` are keyword arguments to pass to the `extractor_function` via `extractor_function_kwargs` parameter. For more details, see the `Hook` class. The `batch` parameter is the batch number to extract from the output. If None, the first batch is used. Look at the `Hook` class for more details."""
    hook = Hook(
        layer_name,
        extractor_function=extractor_function,
        extractor_function_kwargs=kwargs,
        batch=batch,
    )
    o = Objective(hook, name=extractor_function.__name__, hook_objects=hook)
    return o
