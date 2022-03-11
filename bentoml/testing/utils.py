import io
import os
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING
from pathlib import Path

from plumbum import local
from plumbum.cmd import grep
from plumbum.commands.processes import ProcessExecutionError

from bentoml.exceptions import BentoMLException

logger = logging.getLogger("bentoml.tests")


if TYPE_CHECKING:
    from aiohttp.typedefs import LooseHeaders
    from plumbum.machines import LocalCommand
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData


def get_bin(bin: str) -> "LocalCommand":
    if "docker" in bin:
        if Path("/usr/local/bin/docker").exists():
            bin_name = local["/usr/local/bin/docker"]
        elif Path("/usr/bin/docker").exists():
            bin_name = local["/usr/bin/docker"]
        else:
            raise BentoMLException("Can't find docker executable!")
    else:
        bin_name = local[bin]
    return bin_name


def run(bin: str, *args: str, shell: bool = False) -> t.Tuple[str, str, str]:
    stdout, stderr = "", ""
    bin_name = get_bin(bin)
    try:
        p = bin_name.popen(
            args=args, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for _ in io.TextIOWrapper(p.stdout, encoding="utf-8"):
            logger.info(_)
            stdout += _
        for _ in io.TextIOWrapper(p.stderr, encoding="utf-8"):
            logger.error(_)
            stderr += _
    except Exception as err:  # pylint: disable=broad-except
        logger.error(err)
        raise err

    p.communicate()
    return p.returncode, stdout, stderr


async def parse_multipart_form(headers: "Headers", body: bytes) -> "FormData":
    """
    parse starlette forms from headers and body
    """

    from starlette.formparsers import MultiPartParser

    async def async_bytesio(bytes_: bytes) -> t.AsyncGenerator[bytes, None]:
        yield bytes_
        yield b""
        return

    parser = MultiPartParser(headers=headers, stream=async_bytesio(body))
    return await parser.parse()


async def async_request(
    method: str,
    url: str,
    headers: t.Union[None, t.Tuple[t.Tuple[str, str], ...], "LooseHeaders"] = None,
    data: t.Any = None,
    timeout: t.Optional[int] = None,
    assert_status: t.Union[int, t.Callable[[int], bool], None] = None,
    assert_data: t.Union[bytes, t.Callable[[bytes], bool], None] = None,
    assert_headers: t.Optional[t.Callable[[t.Any], bool]] = None,
) -> t.Tuple[int, "Headers", bytes]:
    """
    raw async request client
    """
    import aiohttp
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            method, url, data=data, headers=headers, timeout=timeout
        ) as r:
            r_body = await r.read()

    if assert_status is not None:
        if callable(assert_status):
            assert assert_status(r.status), f"{r.status} {repr(r_body)}"
        else:
            assert r.status == assert_status, f"{r.status} {repr(r_body)}"

    if assert_data is not None:
        if callable(assert_data):
            assert assert_data(r_body), r_body
        else:
            assert r_body == assert_data, r_body

    if assert_headers is not None:
        assert assert_headers(r.headers), repr(r.headers)

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def build_e2e_docker_image(
    python_version: str,
    test_environment: t.Literal["local", "ci"] = "local",
    verbose: bool = False,
) -> None:
    docker = get_bin("docker")
    e2e_tag = "bentoml/bento-server:e2e"
    check_e2e_pipe: t.Callable[[], str] = (
        docker["images"] | grep["-E", "^bentoml/bento-server(\\s)+e2e*"]
    )
    try:
        _: str = check_e2e_pipe()
        logger.info(
            "e2e image already built, skipping.\n"
            "Make sure to use [bold yellow]bentoml/bento-server:e2e[/] under "
            "docker fields for bentofile.yaml",
            extra={"markup": True},
        )
    except ProcessExecutionError:
        logger.error("Couldn't find e2e images. Running `build_e2e_docker_image`...")
        testing_utils = Path(os.path.dirname(__file__))
        git_repo = testing_utils.parent.parent.parent
        docker_build_cmd = [
            "build",
            "--build-arg",
            f"PYTHON_VERSION={python_version}",
            "--build-arg",
            f"TEST_ENVIRONMENT={test_environment}",
            "--progress",
            "plain",
            "-t",
            e2e_tag,
            "-f",
            testing_utils.joinpath("e2e.Dockerfile").as_posix(),
            git_repo.as_posix(),
        ]
        rc, stdout, stderr = run("docker", *docker_build_cmd)
        if verbose:
            logger.info(f"returncode: {rc}\nLogs:\n\t{stdout}")
            if stderr:
                logger.error(stderr)
