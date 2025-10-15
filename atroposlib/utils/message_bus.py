import asyncio
from typing import Any, Dict

import zmq
import zmq.asyncio


class MessageBusClient:
    """ZMQ client used by environments to publish payloads."""

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        linger: int = 0,
        snd_hwm: int = 0,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._context = zmq.asyncio.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, linger)
        if snd_hwm >= 0:
            self._socket.setsockopt(zmq.SNDHWM, snd_hwm)
        self._socket.connect(endpoint)
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def token(self) -> str:
        return self._token

    async def send_json(self, payload: Dict[str, Any]) -> None:
        """Send a JSON serialisable payload over the message bus."""
        if self._closed:
            return
        message = dict(payload)
        message.setdefault("token", self._token)
        async with self._lock:
            await self._socket.send_json(message)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)

    def __del__(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
