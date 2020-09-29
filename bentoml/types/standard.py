from typing import Dict, List, Mapping, Sequence, Union

from bentoml.utils.lazy_loader import LazyLoader

# https://tools.ietf.org/html/rfc7159#section-3
JSON = JsonSerializable = Union[
    int, float, bool, str, None, Sequence["JSON"], Mapping[str, "JSON"],
]

# https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
AwsLambdaEvent = Union[Dict, List, str, int, float, None]

numpy = LazyLoader('numpy', globals(), 'numpy')
Ndarray = Union["numpy.ndarray"]

pandas = LazyLoader('pandas', globals(), 'pandas')
DataFrame = Union["pandas.DataFrame"]

tensorflow = LazyLoader('tensorflow', globals(), 'tensorflow')
TfTensor = Union["tensorflow.Tensor"]

imageio = LazyLoader('imageio', globals(), 'imageio')
ImageIOArray = Union['imageio.core.Array']
