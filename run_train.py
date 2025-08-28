import sky
import textwrap
from dotenv import dotenv_values, load_dotenv
from sky import ClusterStatus
import os

load_dotenv()

def launch_model():
    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env
        """
    )

    # Remove --no-managed-python and revert to python 3.12 once https://github.com/astral-sh/python-build-standalone/pull/667#issuecomment-3059073433 is addressed.
    # uv pip install "git+https://github.com/JonesAndrew/ART.git@12e1dfe#egg=openpipe-art[backend,langgraph]"
    run_script = textwrap.dedent(f"""
        uv remove openpipe-art
        uv add 'openpipe-art[backend,langgraph]'
        uv remove wandb
        uv add wandb==0.21.0

        uv run python train.py
    """)

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"deep-re-sft",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators="H200-SXM:1"))

    # Generate cluster name
    cluster_name = f"deep-re-sft"
    # Add cluster prefix if defined in environment
    cluster_prefix = os.environ.get("CLUSTER_PREFIX")
    if cluster_prefix:
        cluster_name = f"{cluster_prefix}-{cluster_name}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task; stream_and_get blocks until the task starts running, but
    # running this in its own thread means all models run in parallel.
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
        )
    )

    print(f"Job submitted(ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} finished with exit code {exit_code}.")


launch_model()
