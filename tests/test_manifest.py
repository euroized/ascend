import re

from ascend import data, stimuli


def test_manifest_entries_are_pinned():
    for group in data.load_manifest()["groups"].values():
        for e in group:
            assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"])
            assert re.search(r"/[0-9a-f]{40}/", e["url"]), "URLs must pin a commit"
            assert e["bytes"] > 0


def test_connectome_counts_match_brief():
    c = data.load_manifest()["connectome"]
    assert (c["neurons"], c["connections"], c["synapses"]) == (138639, 15091983, 54492922)


def test_stimuli_783_mirror_630():
    old, new = stimuli.table(630), stimuli.table(783)
    assert old["stimuli"].keys() == new["stimuli"].keys()
    for k in old["stimuli"]:
        assert len(new["stimuli"][k]) == len(old["stimuli"][k])
        assert len(set(new["stimuli"][k])) == len(new["stimuli"][k])
    assert old["readouts"].keys() == new["readouts"].keys()
    assert all(m["accepted"] for m in new["remapped_v630_ids"].values())


def test_composite_and_empty_stimuli():
    assert len(stimuli.stimulus("jon_all")) == 70 + 60 + 16
    assert stimuli.stimulus("none") == []
