import os
import re
import docker
import argparse
import textwrap

def _calc_memswap(mem_limit: str | None) -> str | None:
    """
    Docker best-practice: give container +2 GB swap if a hard memory
    limit is set and the unit is GiB.
    """
    if not mem_limit:
        return None
    m = re.match(r"^(\d+)([gGmM])$", mem_limit.strip())
    if not m:
        return None          # unknown format → let Docker decide
    value, unit = int(m.group(1)), m.group(2).lower()
    if unit == "g":
        return f"{value + 2}g"
    return None              # skip for “m” because +2 GB > mem_limit

def create_container(
    container_name: str = "attackbox",
    mem_limit: str | None = None,
    host_directory: str = "./attackbox_data",
    image: str = "python:3.12-slim",
):
    client = docker.from_env()
    host_directory = os.path.abspath(host_directory)
    os.makedirs(host_directory, exist_ok=True)

    # Re-use container if it already exists
    existing = client.containers.list(filters={"name": container_name}, all=True)
    if existing:
        container = existing[0]
        if container.status != "running":
            container.start()
        print(f"[ATTACKBOX] Existing container '{container_name}' is ready.")
        return container

    print(f"[ATTACKBOX] Pulling {image} …")
    client.images.pull(image)

    container = client.containers.run(
        image,
        detach=True,
        tty=True,
        name=container_name,
        mem_limit=mem_limit,
        memswap_limit=_calc_memswap(mem_limit),
        volumes={host_directory: {"bind": "/data", "mode": "rw"}},
        environment={"DEBIAN_FRONTEND": "noninteractive"},
        stdin_open=True,
    )

    setup_commands = '''
apt update && \
apt -y install cargo git python3 python3-venv build-essential && \
pip install uv --break-system-packages && \
if [ ! -d "/opt/mcp-python" ]; then
  git clone https://github.com/MuzsaiLajos/mcp-python-sandboxed /opt/mcp-python && \
  cd /opt/mcp-python && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install uv && \
  uv pip install -r pyproject.toml && \
  pip install .
fi && \
mkdir -p /tmp/mcp_code \
'''

    print("[ATTACKBOX] Installing MCP server inside the container …")
    result = container.exec_run(f'/bin/bash -c "{setup_commands}"',
                                stdout=True, stderr=True)
    print(result.output.decode())

    print(f"[ATTACKBOX] Container '{container_name}' is up and running.")
    return container


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or start an attackbox Docker container."
    )
    parser.add_argument("--name", default="attackbox",
                        help="Name of the Docker container")
    parser.add_argument("--mem", default=None,
                        help="Memory limit (e.g. 4g, 2048m)")
    parser.add_argument("--data", default="./attackbox_data",
                        help="Host directory to mount at /data")
    parser.add_argument("--image", default="python:3.12-slim",
                        help="Docker image to use (default python:3.12-slim)")
    args = parser.parse_args()

    create_container(
        container_name=args.name,
        mem_limit=args.mem,
        host_directory=args.data,
        image=args.image,
    )
