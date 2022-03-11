# pylint: disable=redefined-outer-name # pragma: no cover
import os
import sys
import time
import typing as t
import urllib
import logging
import contextlib
import subprocess
import urllib.request
from sys import version_info as pyver
from typing import TYPE_CHECKING
from contextlib import contextmanager

from .utils import build_e2e_docker_image
from .._internal.utils import reserve_free_port
from .._internal.utils import cached_contextmanager

logger = logging.getLogger("bentoml.tests")


if TYPE_CHECKING:
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}"

clean_context = contextlib.ExitStack()


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
    headers: t.Optional["LooseHeaders"] = None,
    data: t.Any = None,
    timeout: t.Optional[int] = None,
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

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def _wait_until_api_server_ready(
    host_url: str,
    timeout: float,
    check_interval: float = 1,
    popen: t.Optional["subprocess.Popen[bytes]"] = None,
) -> bool:
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    ex = None
    while time.time() - start_time < timeout:
        try:
            if popen:
                popen.poll()
                if popen.returncode is not None:
                    return False
            elif opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
                return True
            else:
                logger.info("Waiting for host %s to be ready..", host_url)
                time.sleep(check_interval)
        except ConnectionError as e:  # pylint:disable=broad-except
            logger.info(f"[{e}]retrying to connect to the host {host_url}...")
            ex = e
            time.sleep(check_interval)
    else:
        logger.info(
            f"Timed out waiting {timeout} seconds for Server {host_url} to be ready, "
            f"exception: {ex}"
        )
        return False


@contextmanager
def export_service_bundle(_):
    """
    Export a bentoml service to a temporary directory, yield the path.
    Delete the temporary directory on close.
    """
    yield


@cached_contextmanager("{project_path}, {image_tag}")
def build_api_server_docker_image(
    project_path: str,
    image_tag: t.Optional[str] = None,
    *,
    python_version: str,
    test_environment: t.Literal["local", "ci"] = "local",
) -> t.Generator[str, None, None]:
    """
    Build the docker image for a saved bentoml bundle, yield the docker image object.
    """
    import bentoml

    build_e2e_docker_image(
        python_version=python_version, test_environment=test_environment
    )

    bento = bentoml.bentos.build_bentofile(build_ctx=project_path)
    tag = bento.info.tag
    if image_tag is None:
        image_tag = tag.name

    logger.info(f"Building API server docker image from build context: {project_path}")
    subprocess.check_call(["bentoml", "containerize", str(tag), "-t", image_tag])

    yield image_tag
    subprocess.check_call(["docker", "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}")
def run_api_server_in_docker(
    image_tag: str,
    config_file: t.Optional[str] = None,
    timeout: float = 40,
):
    """
    Launch a bentoml service container from a docker image, yields the host URL.
    """
    container_name = f"bentoml-test-{image_tag}"
    with reserve_free_port() as port:
        pass

    cmd = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--publish",
        f"{port}:3000",
        "--env",
        "BENTOML_LOG_STDOUT=true",
        "--env",
        "BENTOML_LOG_STDERR=true",
    ]

    if config_file is not None:
        cmd.extend(["--env", "BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/bentoml/bentoml_config.yml"]
        )
    cmd.append(image_tag)

    logger.info(f"Running API server docker image: {cmd}")
    proc = subprocess.Popen(cmd)  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        host_url = f"127.0.0.1:{port}"
        if _wait_until_api_server_ready(host_url, timeout, popen=proc):
            yield host_url
        else:
            raise RuntimeError(
                f"API server {host_url} failed to start within {timeout} seconds"
            )
    finally:
        proc.kill()
        proc.wait()
        logger.info(
            f"API server container {container_name} exited with code {proc.returncode}"
        )
        # err = proc.communicate()[1].decode("utf-8")
        # logger.info(f"API server stdout: {err}")
        time.sleep(0.5)  # make sure container stopped & deleted
        subprocess.call(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.5)  # make sure container stopped & deleted


@contextmanager
def run_api_server(
    bento: str,
    workdir: t.Optional[str] = None,
    config_file: t.Optional[str] = None,
    dev_server: bool = False,
    timeout: float = 90,
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    workdir = workdir if workdir is not None else "./"

    serve_cmd = "serve"

    my_env = os.environ.copy()

    with reserve_free_port() as port:
        cmd = [sys.executable, "-m", "bentoml", serve_cmd]

        if not dev_server:
            cmd += ["--production"]

        if port:
            cmd += ["--port", f"{port}"]
        cmd += [bento]
        cmd += ["--working-dir", workdir]

    print(cmd)

    # def print_log(p: "subprocess.Popen"):
    # try:
    # for line in p.stdout:
    # print(line.decode(), end="")
    # except ValueError:
    # pass

    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)

    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=my_env,
    )
    try:
        # threading.Thread(target=print_log, args=(p,), daemon=True).start()
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout=timeout)
        yield host_url
    finally:
        # TODO: can not terminate the subprocess on Windows
        p.terminate()
        p.wait()


@cached_contextmanager("{bento}, {config_file}, {workdir}, {docker}, {dev_server}")
def host_bento(
    bento: str,
    config_file: str = "bentoml_config.yml",
    workdir: str = "./",
    docker: bool = False,
    dev_server: bool = False,
) -> t.Generator[str, None, None]:
    if not os.path.exists(config_file):
        raise Exception(f"config file not found: {config_file}")
    if os.environ.get("GITHUB_REF", None) is not None:
        test_environment = "ci"
    else:
        test_environment = "local"

    if docker:
        image = clean_context.enter_context(
            build_api_server_docker_image(
                bento,
                "example_service",
                python_version=PYTHON_VERSION,
                test_environment=test_environment,
            )  # TODO
        )
        host = clean_context.enter_context(
            run_api_server_in_docker(
                image,
                config_file=config_file,
            )
        )
        yield host
    elif dev_server:
        host = clean_context.enter_context(
            run_api_server(
                bento,
                config_file=config_file,
                workdir=workdir,
                dev_server=True,
            )
        )
        yield host
    else:
        host = clean_context.enter_context(
            run_api_server(
                bento,
                config_file=config_file,
                workdir=workdir,
            )
        )
        yield host
