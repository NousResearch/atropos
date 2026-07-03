"""Regression tests for BaseEnv.handle_env when an item uuid is missing.

handle_env coroutines are scheduled with asyncio.create_task before they run,
and self.running_items can be reset or popped by several concurrent code paths
(the STOP_TRAIN eval branch, the off-policy overflow branch, the worker-reduction
and worker-timeout paths). A uuid can therefore be evicted before its scheduled
handle_env first looks it up, so the lookup must tolerate a missing key.
"""

import asyncio

from atroposlib.envs.base import BaseEnv


class _MinimalEnv(BaseEnv):
    """Concrete BaseEnv with the abstract methods stubbed out.

    Instances are created without running BaseEnv.__init__ so that handle_env can
    be exercised in isolation without the full config/tokenizer/server setup.
    """

    async def get_next_item(self):
        raise NotImplementedError

    async def evaluate(self, *args, **kwargs):
        raise NotImplementedError


def _make_env():
    return _MinimalEnv.__new__(_MinimalEnv)


def test_handle_env_missing_item_returns_none():
    env = _make_env()
    env.running_items = {}

    result = asyncio.run(env.handle_env("does-not-exist"))

    assert result is None


def test_handle_env_present_item_is_processed():
    env = _make_env()
    sentinel = object()
    env.running_items = {"uuid-1": {"item": sentinel}}
    env.backlog = []
    env.task_duration = []
    env.task_successful = []
    env.succeeded_task_duration = []
    env.failed_task_duration = []

    seen = {}

    async def fake_collect_trajectories(item):
        seen["item"] = item
        return ["traj"], []

    async def fake_postprocess_histories(trajectories):
        return trajectories

    async def fake_handle_send_to_api(to_postprocess, item):
        seen["sent"] = to_postprocess

    async def fake_cleanup():
        return None

    env.collect_trajectories = fake_collect_trajectories
    env.postprocess_histories = fake_postprocess_histories
    env.handle_send_to_api = fake_handle_send_to_api
    env.cleanup = fake_cleanup

    result = asyncio.run(env.handle_env("uuid-1"))

    assert seen["item"] is sentinel
    assert result == ["traj"]
    assert "uuid-1" not in env.running_items
