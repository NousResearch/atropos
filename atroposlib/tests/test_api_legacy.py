import pytest
import requests

from atroposlib.tests.api_test_utils import launch_api_for_testing


def register_data(
    base_url: str, group: str = "test", proj: str = "test", batch_size: int = 32
) -> requests.Response:
    x = requests.post(
        f"{base_url}/register",
        json={
            "wandb_group": group,
            "wandb_project": proj,
            "batch_size": batch_size,
            "max_token_len": 512,
            "checkpoint_dir": "/tmp/test",
            "save_checkpoint_interval": 100,
            "starting_step": 0,
            "num_steps": 1000,
        },
    )
    return x


def post_scored_data(
    base_url: str,
    tokens=((0,),),
    masks=((0,),),
    scores=(0,),
    ref_logprobs=((0,),),
) -> requests.Response:
    data = {
        "tokens": tokens,
        "masks": masks,
        "scores": scores,
    }
    if ref_logprobs is not None:
        data["ref_logprobs"] = ref_logprobs
    x = requests.post(f"{base_url}/scored_data", json=data)
    return x


def reset(base_url: str) -> requests.Response:
    x = requests.get(f"{base_url}/reset_data")
    return x


@pytest.fixture(scope="session")
def api():
    proc, base_url = launch_api_for_testing()
    yield base_url
    proc.terminate()
    proc.wait()


def test_register(api):
    x = register_data(api)
    assert x.status_code == 200, x.text
    data = x.json()
    assert "uuid" in data


def test_reset(api):
    x = register_data(api)
    assert x.status_code == 200, x.text
    data = x.json()
    assert "uuid" in data
    x = post_scored_data(api)
    assert x.status_code == 200, x.text
    x = reset(api)
    print("0-0-0-0-0-0-0-0", flush=True)
    print(x.text, flush=True)
    print("0-0-0-0-0-0-0-0", flush=True)
    assert x.status_code == 200, x.text
    x = requests.get(f"{api}/info")
    assert x.status_code == 200
    assert x.json()["batch_size"] == -1
    x = requests.get(f"{api}/status")
    assert x.status_code == 200, x.text
    data = x.json()
    assert data["current_step"] == 0
    assert data["queue_size"] == 0
    x = requests.get(f"{api}/wandb_info")
    assert x.status_code == 200, x.text
    data = x.json()
    assert data["group"] is None
    assert data["project"] is None


def test_batch_size(api):
    x = register_data(api)
    assert x.status_code == 200, x.text
    # get the batch size
    x = requests.get(f"{api}/info")
