# type: ignore[reportMissingTypeStubs]
import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

import attr
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.frameworks import ModelRunner

from ..types import PathType
from ..models import Model
from ..models import SAVE_NAMESPACE
from ..runner.utils import Params
from ..configuration.containers import BentoMLContainer
from ..configuration.containers import BentoRunnerContainer

SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
ONNX_EXT: str = ".onnx"

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    raise MissingDependencyException(
        """\
`onnx` is required in order to use the module `bentoml.onnx`, do `pip install onnx`.
For more information, refers to https://onnx.ai/get-started.html
`onnxruntime` is also required by `bentoml.onnx`. Refer to https://onnxruntime.ai/ for
more information.
        """
    )

if TYPE_CHECKING:
    import numpy as np
    import torch
    from pandas.core.frame import DataFrame
    from tensorflow.python.framework.ops import Tensor as TFTensor

    from ..models import ModelStore

    _ProviderType = t.List[t.Union[str, t.Tuple[str, t.Dict[str, t.Any]]]]
    _GPUProviderType = t.List[
        t.Tuple[
            t.Literal["CUDAExecutionProvider"],
            t.Union[t.Dict[str, t.Union[int, str, bool]], str],
        ]
    ]


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

_PACKAGE = ["onnxruntime", "onnxruntime-gpu"]
for p in _PACKAGE:
    try:
        _onnxruntime_version = importlib_metadata.version(p)
        break
    except importlib_metadata.PackageNotFoundError:
        pass
_onnx_version = importlib_metadata.version("onnx")


logger = logging.getLogger(__name__)


# helper methods
def _yield_providers(
    iterable: t.Sequence[t.Any],
) -> t.Generator[str, None, None]:  # pragma: no cover
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst: t.List[t.Any]) -> t.List[str]:  # pragma: no cover
    return [k for i in lst for k in _yield_providers(i)]


def _get_model_info(
    tag: t.Union[str, Tag],
    model_store: "ModelStore",
) -> t.Tuple["Model", str]:
    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}")
    return model, model_file


@inject
def load(
    tag: t.Union[str, Tag],
    backend: t.Optional[str] = "onnxruntime",
    providers: t.Optional[t.Union["_ProviderType", "_GPUProviderType"]] = None,
    session_options: t.Optional["ort.SessionOptions"] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ort.InferenceSession":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        backend (`str`, `optional`, default to `onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted `onnxruntime`
             and `onnxruntime-gpu`.
        providers (`List[Union[str, t.Tuple[str, Dict[str, Any]]`, `optional`, default to `None`):
            Different providers provided by users. By default BentoML will use `onnxruntime.get_available_providers()`
             when loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to `None`):
            SessionOptions per usecase. If not specified, then default to None.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of Onnx model from BentoML modelstore.

    Examples::
    """  # noqa
    _, model_file = _get_model_info(tag, model_store)

    if backend not in SUPPORTED_ONNX_BACKEND:
        raise BentoMLException(
            f"'{backend}' runtime is currently not supported for ONNXModel"
        )
    if providers:
        if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
            raise BentoMLException(f"'{providers}' cannot be parsed by `onnxruntime`")
    else:
        providers = ort.get_available_providers()

    return ort.InferenceSession(
        model_file, sess_options=session_options, providers=providers  # type: ignore[reportGeneralTypeIssues] # noqa: LN001
    )


@inject
def save(
    name: str,
    model: t.Union[onnx.ModelProto, PathType],
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model:
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "onnx",
        "pip_dependencies": [
            f"onnx=={_onnx_version}",
            f"onnxruntime=={_onnxruntime_version}",
        ],
    }

    _model = Model.create(
        name,
        module=__name__,
        metadata=metadata,
        context=context,
    )

    if isinstance(model, onnx.ModelProto):
        onnx.save_model(model, _model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}"))
    else:
        shutil.copyfile(model, _model.path_of(f"{SAVE_NAMESPACE}{ONNX_EXT}"))

    _model.save(model_store)

    return _model.tag


@attr.define(kw_only=True)
class ONNXRunner(ModelRunner):
    backend: str
    disable_copy_in_default_stream: bool
    providers: t.Optional["_ProviderType"]

    def _check_params(self):
        if self.backend not in SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f"'{self.backend}' runtime is currently not supported for ONNXModel"
            )
        if self.providers is not None:
            if not all(
                i in ort.get_all_providers() for i in flatten_list(self.providers)
            ):
                raise BentoMLException(
                    f"'{self.providers}' cannot be parsed by `onnxruntime`"
                )

    @staticmethod
    def _get_default_providers(
        gpu_device_id: int, disable_copy_in_default_stream: bool
    ) -> "_ProviderType":
        from ..utils.resource import get_gpu_memory

        if gpu_device_id != -1:
            _, free = get_gpu_memory(gpu_device_id)
            gpu_ = {
                "device_id": gpu_device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": free,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            if disable_copy_in_default_stream:
                logger.warning(
                    "`disable_copy_in_default_stream=True` will set `do_copy_in_default_stream=False`."
                    " There are race conditions and possibly better performance."
                )
                gpu_["do_copy_in_default_stream"] = False
            providers = [
                ("CUDAExecutionProvider", gpu_),
                "CPUExecutionProvider",
            ]
        else:
            providers = ort.get_available_providers()  # type: ignore[]
        return providers  # type: ignore[return-value]

    @property
    def num_concurrency_per_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    @property
    def _gpu_device(self) -> int:
        replica_id = BentoRunnerContainer.replica_id.get()
        if replica_id is not None:
            return int(self.resource_quota.gpus[replica_id - 1])
        return int(self.resource_quota.gpus[0])

    def _setup(self) -> None:
        if self.providers is not None:
            if not all(
                i in ort.get_all_providers() for i in flatten_list(self.providers)
            ):
                raise BentoMLException(
                    f"'{self.providers}' cannot be parsed by `onnxruntime`"
                )
            providers = self.providers
        else:
            providers = self._get_default_providers(
                self._gpu_device, self.disable_copy_in_default_stream
            )
        session_options = ort.SessionOptions()  # type:ignore
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.intra_op_num_threads = self.num_concurrency_per_replica
        session_options.inter_op_num_threads = self.num_concurrency_per_replica
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self._model = load(
            self.tag,
            backend=self.backend,
            providers=providers,
            session_options=session_options,  # type: ignore
            model_store=self.model_store,
        )
        self._infer_func = getattr(self._model, "run")

    @staticmethod
    def _to_ndarray(
        item: t.Union[
            "torch.Tensor",
            "TFTensor",
            "np.ndarray",
            "DataFrame",
        ]
    ) -> t.Any:
        # TODO: check if imported before actual eval
        import numpy as np
        import torch
        import pandas as pd
        import tensorflow as tf

        if isinstance(item, np.ndarray):
            item = item.astype(np.float32)
        elif isinstance(item, pd.DataFrame):
            item = item.to_numpy()
        elif isinstance(item, (tf.Tensor, torch.Tensor)):
            item = item.numpy()
        else:
            raise TypeError(
                f"`_run_batch` of {ONNXRunner.__name__} only takes "
                "`numpy.ndarray` or `pd.DataFrame` as input parameters"
            )
        return item

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        *args: t.Union[
            "np.ndarray",
            "DataFrame",
            "torch.Tensor",
            "TFTensor",
        ],
    ) -> t.Any:
        params = Params(*args)
        params = params.map(self._to_ndarray)
        input_names = {
            i.name: val for i, val in zip(self._model.get_inputs(), params.args)
        }
        output_names = [i.name for i in self._model.get_outputs()]
        return self._infer_func(output_names, input_names)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    name: t.Optional[str] = None,
    backend: str = "onnxruntime",
    disable_copy_in_default_stream: bool = False,
    providers: t.Optional["_ProviderType"] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ONNXRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.onnx.load_runner` implements a Runner class that
    wrap around an ONNX model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        disable_copy_in_default_stream (`bool`, `optional`, default to `False`):
            Whether to do copies in the default stream or use separate streams. Refers to
             https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#do_copy_in_default_stream
        backend (`str`, `optional`, default to `onnxruntime`):
            Different backend runtime supported by ONNX. Currently only accepted `onnxruntime`
             and `onnxruntime-gpu`.
        providers (`List[Union[str, t.Tuple[str, Dict[str, Any]]`, `optional`, default to `None`):
            Different providers provided by users. By default BentoML will use `CPUExecutionProvider` when
             loading a model
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.onnx` model

    Examples::
    """
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return ONNXRunner(  # pylint: disable=unexpected-keyword-arg
        tag=tag,
        backend=backend,
        disable_copy_in_default_stream=disable_copy_in_default_stream,
        providers=providers,
        model_store=model_store,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
    )
