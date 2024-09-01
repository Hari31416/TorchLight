# Sources
# https://github.com/greentfrapp/lucent/blob/dev/lucent/misc/io/serialize_array.py
# https://github.com/greentfrapp/lucent/blob/dev/lucent/misc/io/showing.py
from .utils import create_simple_logger, T, A

import base64
import numpy as np
import PIL.Image
from io import BytesIO
import IPython.display
from string import Template
from typing import List, Optional, Tuple, Union

logger = create_simple_logger(__name__)


def _normalize_array(array: A, domain: Optional[Tuple[float, float]] = None) -> A:
    """Given an arbitrary rank-3 NumPy array, produce one representing an image.

    This ensures the resulting array has a dtype of uint8 and a domain of 0-255.

    Parameters
    ----------
    array : np.ndarray
        The array to normalize.
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.

    Returns
    -------
    np.ndarray
        The normalized array.
    """
    # first copy the input so we're never mutating the user's data
    array = np.array(array)
    # squeeze helps both with batch=1 and B/W and PIL's mode inference
    array = np.squeeze(array)
    assert len(array.shape) <= 3
    assert np.issubdtype(array.dtype, np.number)
    assert not np.isnan(array).any()

    low, high = np.min(array), np.max(array)
    if domain is None:
        message = f"No domain specified, normalizing from measured range (~{low:.2f}, ~{high:.2f})."
        logger.debug(message)
        domain = (low, high)

    # clip values if domain was specified and array contains values outside of it
    if low < domain[0] or high > domain[1]:
        message = f"Clipping domain from ({low:.2f}, {high:.2f}) to ({domain[0]}, {domain[1]})."
        logger.debug(message)
        array = array.clip(*domain)

    min_value, max_value = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max  # 0, 255
    # convert signed to unsigned if needed
    if np.issubdtype(array.dtype, np.inexact):
        offset = domain[0]
        if offset != 0:
            array -= offset
        if domain[0] != domain[1]:
            scalar = max_value / (domain[1] - domain[0])
            if scalar != 1:
                array *= scalar

    return array.clip(min_value, max_value).astype(np.uint8)


def _serialize_normalized_array(
    array: A, fmt: str = "png", quality: int = 70
) -> BytesIO:
    """Given a normalized array, returns byte representation of image encoding.

    Parameters
    ----------
    array : np.ndarray
        The normalized array of dtype uint8 and range 0 to 255
    fmt : str, optional
        The desired file format, by default 'png'
    quality : int, optional
        The compression quality from 0 to 100 for lossy formats, by default 70

    Returns
    -------
    BytesIO
        The image data as a BytesIO buffer.
    """
    dtype = array.dtype
    assert np.issubdtype(dtype, np.unsignedinteger)
    assert np.max(array) <= np.iinfo(dtype).max
    assert array.shape[-1] > 1  # array dims must have been squeezed

    image = PIL.Image.fromarray(array)
    image_bytes = BytesIO()
    image.save(image_bytes, fmt, quality=quality)
    image_data = image_bytes.getvalue()
    return image_data


def serialize_array(
    array: A,
    domain: Optional[Tuple[float, float]] = None,
    fmt: str = "png",
    quality: int = 70,
) -> BytesIO:
    """Given an arbitrary rank-3 NumPy array, returns the byte representation of the encoded image.

    Parameters
    ----------
    array : np.ndarray
        The normalized array of dtype uint8 and range 0 to 255
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.
    fmt : str, optional
        The desired file format, by default 'png'
    quality : int, optional
        The compression quality from 0 to 100 for lossy formats, by default 70
    """
    normalized = _normalize_array(array, domain=domain)
    return _serialize_normalized_array(normalized, fmt=fmt, quality=quality)


JS_ARRAY_TYPES = {
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
}


def array_to_jsbuffer(array: A) -> str:
    """Serialize 1d NumPy array to JS TypedArray.

    Data is serialized to base64-encoded string, which is much faster
    and memory-efficient than json list serialization.

    Parameters
    ----------
    array : np.ndarray
        The array to serialize.

    Returns
    -------
    str
        The serialized array as a JS TypedArray.
    """

    if array.ndim != 1:
        raise TypeError("Only 1d arrays can be converted JS TypedArray.")
    if array.dtype.name not in JS_ARRAY_TYPES:
        raise TypeError("Array dtype not supported by JS TypedArray.")
    js_type_name = array.dtype.name.capitalize() + "Array"
    data_base64 = base64.b64encode(array.tobytes()).decode("ascii")
    code = """
        (function() {
            const data = atob("%s");
            const buf = new Uint8Array(data.length);
            for (var i=0; i<data.length; ++i) {
                buf[i] = data.charCodeAt(i);
            }
            var array_type = %s;
            if (array_type == Uint8Array) {
                return buf;
            }
            return new array_type(buf.buffer);
        })()
    """ % (
        data_base64,
        js_type_name,
    )
    return code


def _display_html(html_str):
    IPython.display.display(IPython.display.HTML(html_str))


def _image_url(
    array: A,
    fmt: str = "png",
    mode: str = "data",
    quality: int = 90,
    domain: Optional[Tuple[float, float]] = None,
) -> str:
    """Create a data URL representing an image from a PIL.Image.

    Parameters
    ----------
    array : np.ndarray
        The array to convert to an image.
    fmt : str, optional
        The image format, by default 'png'
    mode : str, optional
        The mode of the image, by default 'data'
    quality : int, optional
        The compression quality from 0 to 100 for lossy formats, by default 90
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.

    Returns
    -------
    str
        The URL of the image.
    """
    supported_modes = "data"
    if mode not in supported_modes:
        message = "Unsupported mode '%s', should be one of '%s'."
        raise ValueError(message, mode, supported_modes)

    image_data = serialize_array(array, fmt=fmt, quality=quality, domain=domain)
    base64_byte_string = base64.b64encode(image_data).decode("ascii")
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def _image_html(
    array: A,
    width: Optional[int] = None,
    domain: Optional[Tuple[float, float]] = None,
    fmt: str = "png",
) -> str:
    url = _image_url(array, domain=domain, fmt=fmt)
    style = "image-rendering: pixelated;"
    if width is not None:
        style += "width: {width}px;".format(width=width)
    return """<img src="{url}" style="{style}">""".format(url=url, style=style)


def show_image(
    array: A,
    domain: Optional[Tuple[float, float]] = None,
    width: Optional[int] = None,
    fmt: str = "png",
) -> None:
    """Display an image.

    Parameters
    ----------
    array : np.ndarray
        The array to display.
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.
    width : Optional[int], optional
        The width of the output image, by default None. If None, the size will be unchanged.
    fmt : str, optional
        The image format, by default 'png'
    """
    _display_html(_image_html(array, width=width, domain=domain, fmt=fmt))


def _create_image_table(
    images: List[A],
    labels: Optional[List[str]] = None,
    domain: Optional[Tuple[float, float]] = None,
    width: Optional[int] = None,
    fmt: str = "png",
    n_cols: Optional[int] = None,
):
    """Create an HTML table of images.

     Parameters
    ----------
    images : List[np.ndarray]
        A list of NumPy images representing images.
    labels : Optional[List[str]], optional
        A list of strings to label each image, by default None. If None, the index will be shown.
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.
    width : Optional[int], optional
        The width of the output image, by default None. If None, the size will be unchanged.
    n_cols : Optional[int], optional
        The number of columns in the output table, by default None. If None, the number of columns will be the square root of the number of images.

    Returns
    -------
    str
        The HTML table of images.
    """
    n_cols = n_cols or np.ceil(np.sqrt(len(images))).astype(int)
    n_rows = len(images) // n_cols
    if n_rows * n_cols < len(images):
        n_rows += 1
    images_html = "<table>"
    for i in range(n_rows):
        images_html += "<tr>"
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(images):
                # add label
                images_html += f"<td style='text-align: center;'>{labels[idx]} <br>{_image_html(images[idx], domain=domain, width=width, fmt=fmt)}</td>"
        images_html += "</tr>"
    images_html += "</table>"
    return images_html


def show_images(
    images: List[A],
    labels: Optional[List[str]] = None,
    domain: Optional[Tuple[float, float]] = None,
    width: Optional[int] = None,
    fmt: str = "png",
    n_cols: Optional[int] = None,
) -> None:
    """Display a list of images with optional labels.

    Parameters
    ----------
    images : List[np.ndarray]
        A list of NumPy images representing images.
    labels : Optional[List[str]], optional
        A list of strings to label each image, by default None. If None, the index will be shown.
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.
    width : Optional[int], optional
        The width of the output image, by default None. If None, the size will be unchanged.
    n_cols : Optional[int], optional
        The number of columns in the output table, by default None. If None, the number of columns will be the square root of the number of images.
    """
    string = '<div style="display: flex; flex-direction: row;">'
    labels = labels or list(range(1, len(images) + 1))
    images_html = _create_image_table(
        images=images, labels=labels, domain=domain, width=width, fmt=fmt, n_cols=n_cols
    )
    string += f"""<div style="margin-right:10px; margin-top: 4px;">
                        {images_html}
                    </div>"""
    string += "</div>"
    _display_html(string)


def animate_sequence(
    sequence: Union[A, List[A]],
    domain: Optional[Tuple[float, float]] = None,
    fmt: str = "png",
) -> None:
    """Animate a sequence of images.

    Parameters
    ----------
    sequence : np.ndarray
        The sequence of images to animate.
    domain : Tuple[float, float], optional
        The domain of the input array, by default None. If None, the domain will be inferred from the array.
    fmt : str, optional
        The image format, by default 'png'
    """
    if isinstance(sequence, list):
        sequence = np.array(sequence)

    steps, height, width, _ = sequence.shape
    sequence = np.concatenate(sequence, 1)
    code = Template(
        """
    <style> 
        #animation {
            width: ${width}px;
            height: ${height}px;
            background: url('$image_url') left center;
            animation: play 1s steps($steps) infinite alternate;
        }
        @keyframes play {
            100% { background-position: -${sequence_width}px; }
        }
    </style><div id='animation'></div>
    """
    ).substitute(
        image_url=_image_url(sequence, domain=domain, fmt=fmt),
        sequence_width=width * steps,
        width=width,
        height=height,
        steps=steps,
    )
    _display_html(code)
