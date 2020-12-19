# pylint: disable=redefined-outer-name
import json

import pytest


@pytest.mark.asyncio
async def test_api_server_meta(host):
    await pytest.assert_request("GET", f"http://{host}/")
    await pytest.assert_request("GET", f"http://{host}/healthz")
    await pytest.assert_request("GET", f"http://{host}/docs.json")


@pytest.mark.asyncio
async def test_customized_route(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/$~!@%^&*()_-+=[]\\|;:,./predict_json",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps("hello"),
        assert_data=bytes('"hello"', 'ascii'),
    )


@pytest.mark.asyncio
async def test_customized_request_schema(host):
    await pytest.assert_request(
        "GET",
        f"http://{host}/docs.json",
        headers=(("Content-Type", "application/json"),),
        assert_data="",
    )
