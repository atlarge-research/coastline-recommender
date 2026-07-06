"""Java home detection utilities."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_java_home() -> str:
    """Detect and return JAVA_HOME; raises RuntimeError if Java cannot be found."""
    if "JAVA_HOME" in os.environ:
        java_home = os.environ["JAVA_HOME"]
        java_bin = Path(java_home) / "bin" / "java"
        if java_bin.is_file() and os.access(java_bin, os.X_OK):
            return java_home

    # macOS
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home"],
            capture_output=True,
            text=True,
            check=True,
        )
        java_home = result.stdout.strip()
        if java_home and Path(java_home).exists():
            return java_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Linux readlink
    try:
        result = subprocess.run(
            ["readlink", "-f", "/usr/bin/java"],
            capture_output=True,
            text=True,
            check=True,
        )
        java_binary = result.stdout.strip()
        java_home = str(Path(java_binary).parent.parent)
        if java_home and Path(java_home).exists():
            return java_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    common_paths = [
        "/usr/lib/jvm/java-21-openjdk-arm64",
        "/usr/lib/jvm/java-21-openjdk-amd64",
        "/usr/lib/jvm/default-java",
        "/usr/lib/jvm/java-21",
    ]
    for path in common_paths:
        if Path(path).exists():
            return path

    raise RuntimeError("Could not detect JAVA_HOME. Please set the JAVA_HOME environment variable.")
