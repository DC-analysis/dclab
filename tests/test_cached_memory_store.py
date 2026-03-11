import time

from dclab import cached


def test_memory_store():
    ms = cached.MemoryStore()
    ms["peter"] = "hans"
    assert "peter" in ms
    assert ms["peter"] == "hans"
    assert ms.data["peter"][0] == "hans"
    time.sleep(.1)
    assert ms.data["peter"][1] < time.monotonic()
    assert len(ms) == 1
    ms.clear()
    assert len(ms) == 0
    ms["a"] = 1
    time.sleep(.1)
    ms["b"] = 2
    time.sleep(.1)
    assert len(ms) == 2
    ms.pop("a")
    assert len(ms) == 1
    assert "a" not in ms
    assert "b" in ms
    assert ms.items() == [(key, value)
                          for (key, (value, _)) in ms.data.items()]
    ms["c"] = 3
    time.sleep(.1)
    ms["d"] = 4
    time.sleep(.1)
    ms["e"] = 5
    time.sleep(.1)
    assert len(ms) == 4
    ms.remove_least_used_keys(2)
    assert "b" not in ms
    assert "c" not in ms
    assert "d" in ms
    assert "e" in ms
    ms["f"] = 6
    time.sleep(.1)
    ms["g"] = 7
    time.sleep(.1)
    # access d
    assert ms["d"] == 4
    ms.remove_least_used_keys(2)
    assert "d" in ms
    assert "g" in ms
    assert "f" not in ms
    assert "e" not in ms
    assert ms.pop("g") == 7
