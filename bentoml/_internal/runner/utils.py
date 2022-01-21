import re
import typing as t
import logging
import itertools
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Some constants taken from cuda.h


if TYPE_CHECKING:
    from ctypes import c_int
    from ctypes import c_char_p
    from ctypes import c_size_t
    from ctypes import c_void_p

    CDataType = t.Union[c_int, c_void_p, c_size_t, c_char_p]

    from aiohttp import MultipartWriter
    from starlette.requests import Request

    from ..runner.container import Payload

    class PlcType(t.TypedDict):
        err: c_char_p
        device: c_int
        num_gpus: c_int
        context: c_void_p
        free_mem: c_size_t
        total_mem: c_size_t


T = t.TypeVar("T")
To = t.TypeVar("To")


CUDA_SUCCESS = 0


class Params(t.Generic[T]):
    def __init__(self, *args: T, **kwargs: T):
        self.args: t.Tuple[T, ...] = args
        self.kwargs: t.Dict[str, T] = kwargs

    def map(self, function: t.Callable[[T], To]) -> "Params[To]":
        args = tuple(function(a) for a in self.args)
        kwargs = {k: function(v) for k, v in self.kwargs.items()}
        return Params[To](*args, **kwargs)

    def imap(
        self, function: t.Callable[[T], t.Iterable[To]]
    ) -> "t.Iterator[Params[To]]":
        args_iter = tuple(iter(function(a)) for a in self.args)
        kwargs_iter = {k: iter(function(v)) for k, v in self.kwargs.items()}

        try:
            while True:
                args = tuple(next(a) for a in args_iter)
                kwargs = {k: next(v) for k, v in kwargs_iter.items()}
                yield Params[To](*args, **kwargs)
        except StopIteration:
            pass

    def items(self) -> t.Iterator[t.Tuple[t.Union[int, str], T]]:
        return itertools.chain(enumerate(self.args), self.kwargs.items())

    @classmethod
    def agg(
        cls,
        params_list: t.Sequence["Params[T]"],
        agg_func: t.Callable[[t.Sequence[T]], To] = lambda i: i,
    ) -> "Params[To]":
        if not params_list:
            return t.cast(Params[To], [])

        args: t.List[To] = []
        kwargs: t.Dict[str, To] = {}

        for j, _ in enumerate(params_list[0].args):
            arg: t.List[T] = []
            for params in params_list:
                arg.append(params.args[j])
            args.append(agg_func(arg))
        for k in params_list[0].kwargs:
            kwarg: t.List[T] = []
            for params in params_list:
                kwarg.append(params.kwargs[k])
            kwargs[k] = agg_func(kwarg)
        return Params(*tuple(args), **kwargs)

    @property
    def sample(self) -> T:
        if self.args:
            return self.args[0]
        return next(iter(self.kwargs.values()))


PAYLOAD_META_HEADER = "Bento-Payload-Meta"


def payload_params_to_multipart(params: Params["Payload"]) -> "MultipartWriter":
    import json

    from multidict import CIMultiDict
    from aiohttp.multipart import MultipartWriter

    multipart = MultipartWriter(subtype="form-data")
    for key, payload in params.items():
        multipart.append(
            payload.data,
            headers=CIMultiDict(
                (
                    (PAYLOAD_META_HEADER, json.dumps(payload.meta)),
                    ("Content-Disposition", f'form-data; name="{key}"'),
                )
            ),
        )
    return multipart


async def multipart_to_payload_params(request: "Request") -> Params["Payload"]:
    import json

    from bentoml._internal.runner.container import Payload
    from bentoml._internal.utils.formparser import populate_multipart_requests

    parts = await populate_multipart_requests(request)
    max_arg_index = -1
    kwargs: t.Dict[str, Payload] = {}
    args_map: t.Dict[int, Payload] = {}
    for field_name, req in parts.items():
        payload = Payload(
            data=await req.body(),
            meta=json.loads(req.headers[PAYLOAD_META_HEADER]),
        )
        if field_name.isdigit():
            arg_index = int(field_name)
            args_map[arg_index] = payload
            max_arg_index = max(max_arg_index, arg_index)
        else:
            kwargs[field_name] = payload
    args = tuple(args_map[i] for i in range(max_arg_index + 1))
    return Params(*args, **kwargs)


class TypeRef:
    def __init__(self, module: str, qualname: str):
        self.module = module
        self.qualname = qualname

    @classmethod
    def from_instance(cls, instance: object) -> "TypeRef":
        klass = type(instance)
        return cls.from_type(klass)

    @classmethod
    def from_type(cls, klass: t.Union["TypeRef", type]) -> "TypeRef":
        if isinstance(klass, type):
            return cls(klass.__module__, klass.__qualname__)
        return klass

    def evaluate(self) -> type:
        import importlib

        m = importlib.import_module(self.module)
        ref = t.ForwardRef(f"m.{self.qualname}")
        localns = {"m": m}

        if hasattr(t, "_eval_type"):  # python3.7, 3.8 & 3.9
            _eval_type = getattr(t, "_eval_type")
            return t.cast(type, _eval_type(ref, globals(), localns))

        raise SystemError("unsupported Python version")

    def __eq__(self, o: object) -> bool:
        """
        TypeRef("numpy", "ndarray") == np.ndarray
        TypeRef("numpy", "ndarray") == TypeRef.from_instance(np.random.randint([2, 3]))
        """
        if isinstance(o, type):
            o = self.from_type(o)

        if isinstance(o, TypeRef):
            return self.module == o.module and self.qualname == o.qualname

        return False

    def __hash__(self) -> int:
        return hash(f"{self.module}.{self.qualname}")

    def __repr__(self) -> str:
        return f'TypeRef("{self.module}", "{self.qualname}")'


def cpu_converter(cpu: t.Union[int, float, str]) -> float:
    if isinstance(cpu, (int, float)):
        return float(cpu)

    if isinstance(cpu, str):
        milli_match = re.match("([0-9]+)m", cpu)
        if milli_match:
            return int(milli_match[1]) / 1000.0

    raise ValueError(f"Invalid CPU resource limit '{cpu}'")


def mem_converter(mem: t.Union[int, str]) -> int:
    if isinstance(mem, int):
        return mem

    if isinstance(mem, str):
        unit_match = re.match("([0-9]+)([A-Za-z]{1,2})", mem)
        mem_multipliers = {
            "k": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
            "T": 1000 ** 4,
            "P": 1000 ** 5,
            "E": 1000 ** 6,
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
        }
        if unit_match:
            base = int(unit_match[1])
            unit = unit_match[2]
            if unit in mem_multipliers:
                return base * mem_multipliers[unit]

    raise ValueError(f"Invalid MEM resource limit '{mem}'")


def _gpu_formater(i: t.Union[int, str]) -> str:
    if isinstance(i, int):
        return str(i)
    return i


def gpu_converter(
    gpus: t.Optional[t.Union[int, str, t.Sequence[t.Union[int, str]]]]
) -> t.List[str]:
    if gpus is None:
        return []
    if isinstance(gpus, (int, str)):
        return [_gpu_formater(gpus)]
    if isinstance(gpus, (list, tuple)):
        return [_gpu_formater(g) for g in gpus]
    raise ValueError(f"Invalid GPUS resource '{gpus}'")
