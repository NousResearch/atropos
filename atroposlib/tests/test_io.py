"""Tests for parse_http_response error handling."""

import logging

import pytest

from atroposlib.utils.io import parse_http_response


class _FakeHTTPError(Exception):
    """Stand-in for aiohttp.ClientResponseError (has .status/.message)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class _FakeResponse:
    def __init__(self, status, body=None, text_exc=None):
        self._status = status
        self._body = body
        self._text_exc = text_exc

    def raise_for_status(self):
        if self._status >= 400:
            raise _FakeHTTPError(self._status, "Server Error")

    async def json(self):
        return {"ok": True}

    async def text(self):
        if self._text_exc is not None:
            raise self._text_exc
        return self._body


async def test_parse_http_response_success():
    resp = _FakeResponse(200)
    assert await parse_http_response(resp) == {"ok": True}


async def test_parse_http_response_logs_body_and_reraises_original(caplog):
    resp = _FakeResponse(500, body="boom detail")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_FakeHTTPError) as exc_info:
            await parse_http_response(resp)
    assert exc_info.value.status == 500
    assert "boom detail" in caplog.text


async def test_parse_http_response_reraises_original_when_body_read_fails(caplog):
    """If reading the body fails (e.g. connection released by
    raise_for_status), the original HTTP error must still propagate and the
    error must still be logged.
    """
    resp = _FakeResponse(503, text_exc=RuntimeError("Connection closed"))
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_FakeHTTPError) as exc_info:
            await parse_http_response(resp)
    assert exc_info.value.status == 503
    assert "Status: 503" in caplog.text
    assert "<response body unavailable>" in caplog.text
