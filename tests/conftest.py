import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def tiny_connectome(tmp_path_factory):
    """A 200-neuron random network in the reference file format."""
    return synthetic.write(tmp_path_factory.mktemp("conn"), n_neurons=200, n_connections=4000, seed=1)
