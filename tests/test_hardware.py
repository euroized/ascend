import json

from ascend import hardware


def test_info_is_complete_and_serializable():
    info = hardware.info()
    for k in ("os", "cpu", "cpu_logical_cores", "ram_gb", "gpus", "python", "packages", "repo"):
        assert k in info
    assert "numpy" in info["packages"]
    json.dumps(info)
