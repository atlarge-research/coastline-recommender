"""Container wiring: the compose services read the mounted config files and can save tuned models.

Static checks of docker-compose.yml and the Dockerfile. Each config path a service is given must
resolve, through that service's own volume mounts, to a file in this repository, and the directory
`coastline utils tune` saves into must be writable: a read-write mount under compose, and a
directory the image's user owns in the plain image.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SERVICES = yaml.safe_load((_REPO / "docker-compose.yml").read_text())["services"]


def _environment(service: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in _SERVICES[service].get("environment", []))


def _mounts(service: str) -> list[tuple[str, str, str]]:
    """(host, container, mode) for each volume of ``service``."""
    mounts = []
    for volume in _SERVICES[service].get("volumes", []):
        host, container, *mode = volume.split(":")
        mounts.append((host, container, mode[0] if mode else "rw"))
    return mounts


def _host_path(service: str, container_path: str) -> Path:
    """The repo file a container path maps to through the service's mounts."""
    for host, container, _ in _mounts(service):
        if container_path == container or container_path.startswith(container.rstrip("/") + "/"):
            return (_REPO / host / container_path[len(container) :].lstrip("/")).resolve()
    raise AssertionError(f"{service}: {container_path} is not inside any mounted volume")


@pytest.mark.parametrize("service", ["recommender", "api"])
@pytest.mark.parametrize("variable", ["INFRASTRUCTURE_CONFIG", "EXPERIMENT_CONFIG"])
def test_each_service_reads_the_mounted_config(service, variable) -> None:
    environment = _environment(service)

    assert variable in environment, f"{service} does not set {variable}, so the mounted file is never read"
    assert _host_path(service, environment[variable]).is_file()


def test_the_recommender_can_save_tuned_models() -> None:
    portfolio = _environment("recommender")["PORTFOLIO_DIR"]
    modes = [mode for _, container, mode in _mounts("recommender") if container == portfolio]

    assert modes == ["rw"], "utils tune saves into PORTFOLIO_DIR/custom, so the mount must be writable"


def test_the_plain_image_points_tuned_models_at_its_user_home() -> None:
    dockerfile = (_REPO / "Dockerfile").read_text()
    user = re.search(r"^USER (\S+)", dockerfile, re.MULTILINE)
    portfolio = re.search(r"^ENV PORTFOLIO_DIR=(\S+)", dockerfile, re.MULTILINE)

    assert user and portfolio
    assert portfolio.group(1).startswith(f"/home/{user.group(1)}/")
    assert portfolio.start() > user.start()  # in the runtime stage, where the user is created
