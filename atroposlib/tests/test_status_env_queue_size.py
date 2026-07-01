"""Tests /status-env queue accounting when the max group size is 1."""

from atroposlib.api import server as api_server
from atroposlib.api.server import EnvIdentifier, get_status_env


def _seq():
    return [1, 2, 3]  # a single sequence (list of token ids)


def _env(registered_id):
    return {
        "max_context_len": 100,
        "weight": 1.0,
        "connected": True,
        "group_size": 1,
        "registered_id": registered_id,
        "min_batch_allocation": None,
    }


async def test_status_env_counts_other_envs_when_group_size_one():
    state = api_server.app.state
    state.envs = [_env(0), _env(1)]
    # 1 sequence for the requesting env (id 0) and 5 for the other (id 1);
    # every queued group has size 1 so max_group_size stays 1.
    state.queue = [{"env_id": 0, "tokens": [_seq()]}] + [
        {"env_id": 1, "tokens": [_seq()]} for _ in range(5)
    ]
    state.batchsize = 8
    state.status_dict = {"step": 0}

    result = await get_status_env(EnvIdentifier(env_id=0))

    assert result["max_group_size"] == 1
    # 1 (own) + 5 (other env), group_size 1 -> 6, not just the 1 own sequence.
    assert result["queue_size"] == 6
