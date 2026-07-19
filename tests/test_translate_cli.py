import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", (["-m", "src.translate"], ["run.py"]))
def test_translation_cli_exposes_only_the_local_backend(command: list[str]) -> None:
    # Given: Either supported translation command.

    # When: The user requests its command-line interface.
    result = subprocess.run(  # noqa: S603
        [sys.executable, *command, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    # Then: Model and request sizing remain fixed by the managed backend.
    assert "--provider" not in result.stdout
    assert "--batch-size" not in result.stdout
