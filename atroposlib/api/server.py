import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator

from atroposlib.api.utils import (
    find_groups_summing_to_target,
    grab_batch_with_minimum_allocations,
    grab_exact_from_heterogeneous_queue,
)

# Message import removed - using Dict[str, Any] for more flexible validation

app = FastAPI(title="AtroposLib API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AtroposLib API"}


class Registration(BaseModel):
    wandb_group: str
    wandb_project: str
    batch_size: int
    max_token_len: int
    checkpoint_dir: str
    save_checkpoint_interval: int
    starting_step: int
    num_steps: int


class RegisterEnv(BaseModel):
    max_token_length: int
    desired_name: str
    weight: float
    group_size: int
    min_batch_allocation: Optional[float] = (
        None  # Minimum proportion of a batch this env should be allocated (0.0-1.0)
    )


class EnvIdentifier(BaseModel):
    env_id: int


class ScoredData(BaseModel):
    tokens: List[List[int]]
    masks: List[List[int]]
    scores: List[float]
    advantages: Optional[List[List[float]]] = None
    ref_logprobs: Optional[List[List[float]]] = None
    messages: Optional[List[List[Dict[str, Any]]]] = (
        None  # Changed from Message TypedDict to Dict
    )
    overrides: Optional[List[dict]] = None
    group_overrides: Optional[dict] = None
    images: Optional[Any] = None
    env_id: Optional[int] = None  # ID of the environment that generated this data

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, v):
        """Validate messages field to ensure required fields are present.

        This validator only checks that messages have 'role' and 'content' fields.
        The 'reward' field is completely optional.
        """
        if v is None:
            return None

        for message_list in v:
            for msg in message_list:
                # Ensure the message has the required fields
                if "role" not in msg or "content" not in msg:
                    raise ValueError("Message must have 'role' and 'content' fields")

        return v


class Status(BaseModel):
    """
    basemodel for status information of the current server
    """

    current_step: int
    queue_size: int


class Info(BaseModel):
    """
    basemodel for useful information
    """

    batch_size: int = -1


@app.post("/register")
async def register(registration: Registration):
    try:
        isinstance(app.state.queue, list)
    except AttributeError:
        app.state.queue = []
        app.state.group = registration.wandb_group
        app.state.project = registration.wandb_project
        app.state.batchsize = int(registration.batch_size)
        app.state.max_token_len = int(registration.max_token_len)
        app.state.status_dict = {"step": registration.starting_step}
        app.state.checkpoint_dir = registration.checkpoint_dir
        app.state.save_checkpoint_interval = registration.save_checkpoint_interval
        app.state.num_steps = registration.num_steps
        app.state.curr_batch = []
        app.state.started = False
        app.state.envs = []
        app.state.buffer = {}  # Buffer for mixed-size groups per environment
    try:
        app.state.requesters.append(uuid.uuid4().int)
    except AttributeError:
        # If requesters doesn't exist, create it
        app.state.requesters = [uuid.uuid4().int]
    return {"uuid": app.state.requesters[-1]}


@app.post("/register-env")
async def register_env_url(register_env: RegisterEnv):
    try:
        if not app.state.started:
            return {
                "status": "wait for trainer to start",
            }
    except AttributeError:
        return {
            "status": "wait for trainer to start",
        }
    try:
        isinstance(app.state.envs, list)
    except AttributeError:
        app.state.envs = []
    checkpoint_dir = ""
    try:
        checkpoint_dir = app.state.checkpoint_dir
    except AttributeError:
        pass
    real_name = (
        f"{register_env.desired_name}_"
        f"{len([x for x in app.state.envs if x['desired_name'] == register_env.desired_name])}"
    )
    registered_id = len(app.state.envs)
    app.state.envs.append(
        {
            "max_context_len": register_env.max_token_length,
            "weight": register_env.weight if register_env.weight is not None else 1.0,
            "desired_name": register_env.desired_name,
            "real_name": real_name,
            "registered_id": registered_id,
            "last_update": time.time(),
            "connected": True,
            "min_batch_allocation": register_env.min_batch_allocation,
            "group_size": register_env.group_size,
        }
    )
    return {
        "status": "success",
        "env_id": registered_id,
        "wandb_name": real_name,
        "checkpoint_dir": checkpoint_dir,
        "starting_step": app.state.status_dict["step"],
        "checkpoint_interval": app.state.save_checkpoint_interval,
        "num_steps": app.state.num_steps,
    }


@app.post("/disconnect-env")
async def disconnect_env(disconnect_env: EnvIdentifier):
    try:
        app.state.envs[disconnect_env.env_id]["connected"] = False
        return {"status": "success"}
    except (AttributeError, IndexError) as e:
        return {"status": "failure", "error": str(e)}


@app.get("/wandb_info")
async def wandb_info():
    try:
        return {"group": app.state.group, "project": app.state.project}
    except AttributeError:
        return {"group": None, "project": None}


@app.get("/info")
async def info():
    try:
        return {
            "batch_size": app.state.batchsize,
            "max_token_len": app.state.max_token_len,
        }
    except AttributeError:
        return {"batch_size": -1, "max_token_len": -1}


@app.get("/batch")
async def get_batch():
    if not app.state.started:
        app.state.started = True

    if len(app.state.curr_batch) > 0:
        return {"batch": app.state.curr_batch.pop()}
    else:
        new_batches = []
        # Check if any envs have minimum allocations
        has_min_allocations = any(
            env.get("min_batch_allocation") is not None
            for env in getattr(app.state, "envs", [])
        )

        if has_min_allocations:
            batch, app.state.queue = grab_batch_with_minimum_allocations(
                app.state.queue, app.state.batchsize, app.state.envs
            )
        else:
            batch, app.state.queue = grab_exact_from_heterogeneous_queue(
                app.state.queue, app.state.batchsize
            )

        while batch is not None:
            new_batches.append(batch)
            if has_min_allocations:
                batch, app.state.queue = grab_batch_with_minimum_allocations(
                    app.state.queue, app.state.batchsize, app.state.envs
                )
            else:
                batch, app.state.queue = grab_exact_from_heterogeneous_queue(
                    app.state.queue, app.state.batchsize
                )
        steps_to_take = len(new_batches)
        if steps_to_take == 0:
            return {"batch": None}
        app.state.status_dict["step"] += steps_to_take
        # chunk it
        for batch in new_batches:
            app.state.curr_batch.append(batch)
        curr_batch = app.state.curr_batch.pop()
        # check length before sending
        print(f"Sending batch of {sum(len(x['tokens']) for x in curr_batch)} sequences")
        return {"batch": curr_batch}


@app.get("/latest_example")
async def get_latest_example():
    try:
        return app.state.latest
    except AttributeError:
        return {
            "tokens": [],
            "masks": [],
            "scores": [],
            "advantages": [],
            "ref_logprobs": [],
            "messages": [],
            "images": [],
        }


@app.post("/scored_data")
async def scored_data(scored_data: ScoredData):
    data_dict = {
        "tokens": scored_data.tokens,
        "masks": scored_data.masks,
        "scores": scored_data.scores,
        "advantages": scored_data.advantages,
        "ref_logprobs": scored_data.ref_logprobs,
        "messages": scored_data.messages,
        "overrides": scored_data.overrides,
        "group_overrides": scored_data.group_overrides,
        "images": scored_data.images,
        "env_id": scored_data.env_id,
    }

    # Check if this is a mixed-size group
    env_id = scored_data.env_id
    if env_id is not None and env_id < len(app.state.envs):
        expected_group_size = app.state.envs[env_id].get("group_size", 1)
        actual_group_size = len(scored_data.tokens)

        if actual_group_size != expected_group_size:
            # Mixed size group - add to buffer
            if env_id not in app.state.buffer:
                app.state.buffer[env_id] = []

            app.state.buffer[env_id].append(data_dict)

            # Try to find groups that sum to expected_group_size
            indices = find_groups_summing_to_target(
                app.state.buffer[env_id], expected_group_size
            )

            if indices:
                # Add these groups to queue in order
                groups_to_add = []
                for idx in sorted(indices, reverse=True):
                    groups_to_add.append(app.state.buffer[env_id].pop(idx))

                # Add in FIFO order
                for group in reversed(groups_to_add):
                    app.state.queue.append(group)
                    app.state.latest = group

            return {
                "status": "buffered",
                "buffer_size": sum(
                    len(g["tokens"]) for g in app.state.buffer.get(env_id, [])
                ),
            }

    # Normal path - correct size or no env info
    app.state.queue.append(data_dict)
    app.state.latest = data_dict
    return {"status": "received"}


@app.post("/scored_data_list")
async def scored_data_list(scored_data_list: List[ScoredData]):
    """Handle a list of ScoredData objects for step-based learning"""

    # Process each scored data item
    for scored_data in scored_data_list:
        data_dict = {
            "tokens": scored_data.tokens,
            "masks": scored_data.masks,
            "scores": scored_data.scores,
            "advantages": scored_data.advantages,
            "ref_logprobs": scored_data.ref_logprobs,
            "images": scored_data.images,
            "messages": scored_data.messages,
            "overrides": scored_data.overrides,
            "group_overrides": scored_data.group_overrides,
            "env_id": scored_data.env_id,
        }

        # Check if this is a mixed-size group
        env_id = scored_data.env_id
        if env_id is not None and env_id < len(app.state.envs):
            expected_group_size = app.state.envs[env_id].get("group_size", 1)
            actual_group_size = len(scored_data.tokens)

            if actual_group_size != expected_group_size:
                # Mixed size group - add to buffer
                if env_id not in app.state.buffer:
                    app.state.buffer[env_id] = []

                app.state.buffer[env_id].append(data_dict)

                # Try to find groups that sum to expected_group_size
                indices = find_groups_summing_to_target(
                    app.state.buffer[env_id], expected_group_size
                )

                if indices:
                    # Add these groups to queue in order
                    groups_to_add = []
                    for idx in sorted(indices, reverse=True):
                        groups_to_add.append(app.state.buffer[env_id].pop(idx))

                    # Add in FIFO order
                    for group in reversed(groups_to_add):
                        app.state.queue.append(group)
                        app.state.latest = group
            else:
                # Normal size - add directly to queue
                app.state.queue.append(data_dict)
                app.state.latest = data_dict
        else:
            # No env info or normal path - add directly to queue
            app.state.queue.append(data_dict)
            app.state.latest = data_dict

    return {"status": "received", "groups_processed": len(scored_data_list)}


@app.get("/status")
async def get_status():
    try:
        return {
            "current_step": app.state.status_dict["step"],
            "queue_size": len(app.state.queue),
        }
    except AttributeError:
        return {"current_step": 0, "queue_size": 0}


@app.get("/status-env")
async def get_status_env(env: EnvIdentifier):
    total = sum(
        [
            x["max_context_len"] * max(0.0, x["weight"])
            for x in app.state.envs
            if x["connected"]
        ]
    )
    env_group_size = app.state.envs[env.env_id]["group_size"]
    env_weight = (
        app.state.envs[env.env_id]["max_context_len"]
        * app.state.envs[env.env_id]["weight"]
        / total
    )
    env_weight = max(
        0.01, env_weight
    )  # Minimum weight of 0.01 :) TODO: try to figure out a better way to do this

    # Calculate total minimum allocations
    total_min_allocation = 0.0
    for env_config in app.state.envs:
        if (
            env_config.get("connected", False)
            and env_config.get("min_batch_allocation") is not None
        ):
            total_min_allocation += env_config["min_batch_allocation"]

    # Calculate unallocated fraction
    unallocated_fraction = 1.0 - min(total_min_allocation, 1.0)

    # Find the maximum group size across all items in queue
    queue = getattr(app.state, "queue", [])
    max_group_size = 1
    num_self_sequences_in_queue = 0
    for item in queue:
        group_size = len(item.get("tokens", []))
        if group_size > max_group_size:
            max_group_size = group_size
        if item.get("env_id") == env.env_id:
            # update the group size for the requesting env, handle cases where the group size may be dynamic with max
            env_group_size = max(env_group_size, group_size)
            num_self_sequences_in_queue += group_size

    # update the group size for the requesting env
    app.state.envs[env.env_id]["group_size"] = env_group_size

    # Calculate minimum sequences allocated to each environment
    batch_size = getattr(app.state, "batchsize", 0)
    min_sequences_by_env = {}
    for env_config in app.state.envs:
        if (
            env_config.get("connected", False)
            and env_config.get("min_batch_allocation") is not None
        ):
            env_id = env_config["registered_id"]
            min_sequences = int(batch_size * env_config["min_batch_allocation"])
            min_sequences_by_env[env_id] = min_sequences

    # Count sequences and calculate packed groups for each environment
    import math

    sequences_by_env = {}
    packed_groups_by_env = {}
    curr_env_total_sequences = 0

    for item in queue:
        env_id = item.get("env_id")
        seq_count = len(item.get("tokens", []))

        # Special handling for the requesting environment
        if env_id == env.env_id:
            curr_env_total_sequences += seq_count
        else:
            if env_id not in sequences_by_env:
                sequences_by_env[env_id] = 0
            sequences_by_env[env_id] += seq_count

    # Calculate packed groups for each environment (excluding the requesting env)
    if max_group_size > 1:
        for env_id, seq_count in sequences_by_env.items():
            packed_groups_by_env[env_id] = math.ceil(seq_count / max_group_size)

    # Calculate adjusted queue size
    # (curr_env_total_sequences + sum of available sequences from other envs after their minimums)
    available_from_others = 0
    for env_id in packed_groups_by_env:
        packed_sequences = packed_groups_by_env[env_id] * max_group_size
        min_sequences = min_sequences_by_env.get(env_id, 0)
        available_from_others += max(0, packed_sequences - min_sequences)

    env_queue_size = curr_env_total_sequences + available_from_others

    try:
        ret_dict = {
            "current_step": app.state.status_dict["step"],
            "queue_size": env_queue_size // env_group_size,
            "unallocated_fraction": unallocated_fraction,
            "self_queue_size": num_self_sequences_in_queue // env_group_size,
            "max_group_size": max_group_size,
        }
    except AttributeError:
        ret_dict = {
            "current_step": 0,
            "queue_size": 0,
            "unallocated_fraction": 1.0,
            "num_self_sequences_in_queue": 0,
        }
    ret_dict["env_weight"] = env_weight
    return ret_dict


@app.get("/reset_data")
async def reset_data():
    try:
        del app.state.queue
        app.state.group = None
        app.state.project = None
        app.state.batchsize = -1
        app.state.num_steps = -1
        app.state.status_dict = {"step": 0}
        app.state.curr_batch = []
        app.state.started = False
        app.state.requesters = []
        app.state.envs = []
        app.state.buffer = {}
    except KeyError:
        pass
    return PlainTextResponse("Reset successful", status_code=status.HTTP_200_OK)
