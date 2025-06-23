print("--- PING: app/main.py top level ---")
import os

# Path handling for importing from spatial_rl_mvp
import sys
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Assuming /padres_app is the WORKDIR and spatial_rl_mvp is at /padres_app/spatial_rl_mvp
# and this file is /padres_app/app/main.py
# Adding /padres_app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_rl_mvp.spatial_env import ObjectState, SpatialEnvironmentMVP, SpatialTask

app = FastAPI(title="Padres Simulation Service API (Live V2)")

spatial_env_instance: Optional[SpatialEnvironmentMVP] = None

# --- Helper: Neutralize websocket calls for now ---
# This is a bit of a hack. A cleaner way would be to pass a flag to SpatialEnvironmentMVP
# or modify its methods to optionally skip notifications.
original_notify_visualization_clients = None


async def dummy_notify_visualization_clients_func(scene_state: Any):
    # print("DEBUG: notify_visualization_clients (dummy) called, doing nothing.")
    pass


# We'll apply monkeypatch after ensuring module is loaded, e.g. in startup.
# This is because the module might not be loaded at the exact moment of this script parsing.


@app.on_event("startup")
async def startup_event():
    global spatial_env_instance
    global original_notify_visualization_clients

    # Attempt to monkeypatch after imports are surely done and module loaded by SpatialEnvironmentMVP
    if "spatial_rl_mvp.spatial_env" in sys.modules:
        if hasattr(
            sys.modules["spatial_rl_mvp.spatial_env"], "notify_visualization_clients"
        ):
            original_notify_visualization_clients = sys.modules[
                "spatial_rl_mvp.spatial_env"
            ].notify_visualization_clients
            sys.modules["spatial_rl_mvp.spatial_env"].notify_visualization_clients = (
                dummy_notify_visualization_clients_func
            )
            print(
                "INFO: spatial_rl_mvp.spatial_env.notify_visualization_clients has been temporarily neutralized during startup."
            )
        else:
            print(
                "WARN (startup): spatial_rl_mvp.spatial_env.notify_visualization_clients not found for neutralization."
            )
    else:
        print(
            "WARN (startup): spatial_rl_mvp.spatial_env module not found in sys.modules for websocket neutralization."
        )

    try:
        spatial_env_instance = SpatialEnvironmentMVP()
        print("INFO: SpatialEnvironmentMVP instance created on FastAPI startup.")
    except Exception as e:
        print(f"FATAL: Failed to create SpatialEnvironmentMVP instance on startup: {e}")
        import traceback

        traceback.print_exc()
        spatial_env_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    if (
        spatial_env_instance
        and hasattr(spatial_env_instance, "simulator")
        and spatial_env_instance.simulator
    ):
        spatial_env_instance.simulator.cleanup()
        print("INFO: PyBullet simulation cleaned up on FastAPI shutdown.")

    # Restore original if it was patched
    if (
        original_notify_visualization_clients
        and "spatial_rl_mvp.spatial_env" in sys.modules
    ):
        if hasattr(
            sys.modules["spatial_rl_mvp.spatial_env"], "notify_visualization_clients"
        ):
            sys.modules["spatial_rl_mvp.spatial_env"].notify_visualization_clients = (
                original_notify_visualization_clients
            )
            print("INFO: Restored original notify_visualization_clients on shutdown.")


@app.get("/")
async def root():
    return {"message": "Padres API is alive!"}


@app.get("/status")
async def get_status_endpoint():  # Renamed from get_status to avoid conflict if imported elsewhere
    global spatial_env_instance
    env_status = "Not initialized or instance creation failed"
    task_id = "N/A"
    pybullet_client_id = "N/A"

    if spatial_env_instance:
        env_status = "Instance created"
        if (
            hasattr(spatial_env_instance, "simulator")
            and spatial_env_instance.simulator
        ):
            pybullet_client_id = str(spatial_env_instance.simulator.client_id)
            if spatial_env_instance.simulator.client_id != -1:
                env_status += ", PyBullet Client Connected"
            else:
                env_status += ", PyBullet Client NOT Connected"
        else:
            env_status += ", Simulator object not found"

        if (
            hasattr(spatial_env_instance, "current_task")
            and spatial_env_instance.current_task
        ):
            task_id = spatial_env_instance.current_task.task_id
            env_status += ", Task "{task_id}' loaded"
        else:
            env_status += ", No task loaded"

    return {
        "api_status": "OPERATIONAL",
        "simulation_status_summary": env_status,
        "pybullet_direct_client_id": pybullet_client_id,
        "current_task_id": task_id,
        "notes": "This is the LIVE API wrapping spatial_rl_mvp (V2 structure).",
    }


@app.post("/setup_environment", status_code=201)
async def setup_environment_endpoint():
    global spatial_env_instance
    if not spatial_env_instance:
        print(
            "ERROR: /setup_environment called but spatial_env_instance is None. Startup likely failed."
        )
        raise HTTPException(
            status_code=500,
            detail="Spatial environment instance not available. Check server logs for startup errors.",
        )

    # Define a hardcoded task for Day 1
    objects = [
        ObjectState(
            id="red_cube",
            type="cube",
            position=[0.5, 0.0, 0.2],
            scale=[0.2, 0.2, 0.2],
            color_rgba=[1, 0, 0, 1],
        ),
        ObjectState(
            id="blue_sphere",
            type="sphere",
            position=[-0.5, 0.0, 0.2],
            scale=[0.2, 0.2, 0.2],
            color_rgba=[0, 0, 1, 1],
        ),
    ]
    task_id = f"hardcoded_task_day1_{uuid.uuid4().hex[:4]}"
    current_default_task = SpatialTask(
        task_id=task_id,
        description="Move red cube near blue sphere.",
        initial_objects=objects,
        goal_description="Red cube within 0.3 units of blue sphere.",
        target_object_id="red_cube",
        reference_object_id="blue_sphere",
        target_distance=0.3,
    )

    try:
        print(f"API CALL: /setup_environment - Initializing task: {task_id}")
        await spatial_env_instance.initialize_task(current_default_task)
        print("API CALL: /setup_environment - Task "{task_id}' initialized.")
        return {
            "message": "Environment initialized with hardcoded task.",
            "task_id": task_id,
            "status": "SUCCESS",
        }
    except Exception as e:
        print(f"ERROR in /setup_environment: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize environment: {str(e)}"
        )


@app.post("/execute_action")
async def execute_action_endpoint():
    global spatial_env_instance
    if not spatial_env_instance:
        print("ERROR: /execute_action called but spatial_env_instance is None.")
        raise HTTPException(
            status_code=500, detail="Spatial environment instance not available."
        )
    if (
        not hasattr(spatial_env_instance, "current_task")
        or not spatial_env_instance.current_task
    ):
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized with a task. Call /setup_environment first.",
        )

    target_obj_id = spatial_env_instance.current_task.target_object_id
    ref_obj_initial_pos = [0.0, 0.0, 0.0]  # Default if not found
    # Find the reference object's initial position to make the action relative
    for obj_state in spatial_env_instance.current_task.initial_objects:
        if obj_state.id == spatial_env_instance.current_task.reference_object_id:
            ref_obj_initial_pos = obj_state.position
            break

    # Hardcoded action: move target_obj_id slightly towards where reference_object_id was initially
    action_to_apply = {
        "action_type": "move_object",
        "object_id": target_obj_id,
        "target_position": [
            (
                ref_obj_initial_pos[0] + 0.1
                if target_obj_id == "red_cube"
                else ref_obj_initial_pos[0] - 0.1
            ),  # Simplistic move towards center
            ref_obj_initial_pos[1],
            ref_obj_initial_pos[2],
        ],
    }

    try:
        task_id = spatial_env_instance.current_task.task_id
        print(
            "API CALL: /execute_action for task "{task_id}' - Applying action: {action_to_apply}"
        )
        outcome = await spatial_env_instance.apply_action_and_get_outcome(
            action_to_apply
        )
        obs_msg = outcome.get("observation", "No observation string found in outcome.")
        print(
            "API CALL: /execute_action for task "{task_id}' - Observation: {obs_msg}"
        )

        return {
            "message": "Action executed.",
            "task_id": task_id,
            "action_applied": action_to_apply,
            "observation": obs_msg,
            "reward": outcome.get("reward"),
            "done": outcome.get("done"),
            "full_outcome_debug": outcome,  # Useful for debugging
        }
    except Exception as e:
        print(f"ERROR in /execute_action: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to execute action: {str(e)}"
        )


# Old mock definitions and Pydantic models from previous version are removed.
# The uvicorn.run call is handled by Dockerfile CMD.
