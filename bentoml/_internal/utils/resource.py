import os
import math
import ctypes
import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di.providers import SingletonFactory

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

# Some constants taken from cuda.h
import psutil

if TYPE_CHECKING:
    from ctypes import c_int
    from ctypes import c_char_p
    from ctypes import c_size_t
    from ctypes import c_void_p

    CDataType = t.Union[c_int, c_void_p, c_size_t, c_char_p]

    class VMem(t.Protocol):
        total: int
        free: int
        available: int
        used: int
        percent: int

    class PlcType(t.TypedDict):
        err: c_char_p
        device: c_int
        num_gpus: c_int
        context: c_void_p
        free_mem: c_size_t
        total_mem: c_size_t


CUDA_SUCCESS = 0


@SingletonFactory
def _cgroup_cpu_count() -> float:
    # Query active cpu processor count using cgroup v1 API, based on OpenJDK
    # implementation for `active_processor_count` using cgroup v1:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupSubsystem_linux.cpp
    # For cgroup v2, see:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupV2Subsystem_linux.cpp
    def _read_integer_file(filename: str) -> int:
        with open(filename, "r", encoding="utf-8") as f:
            return int(f.read().rstrip())

    cgroup_root = "/sys/fs/cgroup/"
    cfs_quota_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us")
    cfs_period_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us")
    shares_file = os.path.join(cgroup_root, "cpu", "cpu.shares")

    quota = shares = period = -1
    if os.path.isfile(cfs_quota_us_file):
        quota = _read_integer_file(cfs_quota_us_file)

    if os.path.isfile(shares_file):
        shares = _read_integer_file(shares_file)
        if shares == 1024:
            shares = -1

    if os.path.isfile(cfs_period_us_file):
        period = _read_integer_file(cfs_period_us_file)

    os_cpu_count = float(os.cpu_count() or 1)

    limit_count = math.inf
    quota_count = 0.0
    share_count = 0.0

    if quota > -1 and period > 0:
        quota_count = float(quota) / float(period)
    if shares > -1:
        share_count = float(shares) / float(1024)

    if quota_count != 0 and share_count != 0:
        limit_count = min(quota_count, share_count)
    if quota_count != 0:
        limit_count = quota_count
    if share_count != 0:
        limit_count = share_count

    return float(min(limit_count, os_cpu_count))


@SingletonFactory
def _cuda_lib() -> "ctypes.CDLL":
    libs = ("libcuda.so", "cuda.dll")
    for lib in libs:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    raise OSError(f"could not load any of: {' '.join(libs)}")


@SingletonFactory
def _init_var() -> t.Tuple["ctypes.CDLL", "PlcType"]:
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
    # TODO: add threads_per_core, cores, Compute Capability
    err = ctypes.c_char_p()
    plc: PlcType = {
        "err": err,
        "device": ctypes.c_int(),
        "num_gpus": ctypes.c_int(),
        "context": ctypes.c_void_p(),
        "free_mem": ctypes.c_size_t(),
        "total_mem": ctypes.c_size_t(),
    }

    try:
        _drv = _cuda_lib.get()
        res = _drv.cuInit(0)
        if res != CUDA_SUCCESS:
            _drv.cuGetErrorString(res, ctypes.byref(err))
            logger.error(f"cuInit failed with error code {res}: {str(err.value)}")
        return _drv, plc
    except OSError as e:
        raise BentoMLException(
            f"{e}\nMake sure to have CUDA "
            f"installed you are intending "
            f"to use GPUs with BentoML."
        )


def get_system_cpu() -> float:
    # Default to the total CPU count available in current node or cgroup
    if psutil.POSIX:
        return _cgroup_cpu_count.get()
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return float(cpu_count)
        raise ValueError("CPU count is NoneType")


def get_system_mem() -> int:
    # Default to the total memory available
    from psutil import virtual_memory

    mem: VMem = virtual_memory()
    return mem.total


def get_system_gpus() -> t.List[str]:
    drv, plc = _init_var.get()

    res = drv.cuDeviceGetCount(ctypes.byref(plc["num_gpus"]))
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            f"cuDeviceGetCount failed with error code {res}: {str(plc['err'].value)}"
        )

    return [str(dev) for dev in range(plc["num_gpus"].value)]


def get_gpu_memory(dev: int) -> t.Tuple[int, int]:
    """Return Total Memory and Free Memory in given GPU device. in MiB"""
    drv, plc = _init_var.get()

    res = drv.cuDeviceGet(ctypes.byref(plc["device"]), dev)
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            "cuDeviceGet failed " f"with error code {res}: {str(plc['err'].value)}"
        )
    try:
        res = drv.cuCtxCreate_v2(ctypes.byref(plc["context"]), 0, plc["device"])
    except AttributeError:
        res = drv.cuCtxCreate(ctypes.byref(plc["context"]), 0, plc["device"])
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            f"cuCtxCreate failed with error code {res}: {str(plc['err'].value)}"
        )

    try:
        res = drv.cuMemGetInfo_v2(
            ctypes.byref(plc["free_mem"]), ctypes.byref(plc["total_mem"])
        )
    except AttributeError:
        res = drv.cuMemGetInfo(
            ctypes.byref(plc["free_mem"]), ctypes.byref(plc["total_mem"])
        )
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            f"cuMemGetInfo failed with error code {res}: " f"{str(plc['err'].value)}"
        )
    _total_mem = plc["total_mem"].value
    _free_mem = plc["free_mem"].value
    logger.debug(f"Total Memory: {_total_mem} MiB\nFree Memory: {_free_mem} MiB")
    drv.cuCtxDetach(plc["context"])
    return _total_mem, _free_mem
