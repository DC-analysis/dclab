from dclab import cached


def test_store_keeper_clear(tmp_path):
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_disk_store_path(tmp_path)
    assert sk.disk_store
    sk.memory_store["hans/peter"] = "franz"
    sk.disk_store["mans/reeder"] = "proin"
    sk.perform_tasks()
    sk.clear()

    assert "hans/peter" not in sk.memory_store
    assert "hans/peter" not in sk.disk_store
    assert "mans/reeder" not in sk.memory_store
    assert "mans/reeder" not in sk.disk_store


def test_store_keeper_mem_to_disk(tmp_path):
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_disk_store_path(tmp_path)
    assert sk.disk_store
    sk.memory_store["hans/peter"] = "franz"
    sk.perform_tasks()
    assert sk.disk_store["hans/peter"] == "franz"


def test_store_keeper_remove_memory_store_items():
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_memory_store_size(10)

    for ii in range(20):
        sk.memory_store[f"hans/peter{ii}"] = f"franz{ii}"

    assert len(sk.memory_store) == 20
    sk.perform_tasks()
    assert len(sk.memory_store) == 10


def test_store_keeper_start_join(tmp_path):
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_disk_store_path(tmp_path)
    sk.start()
    assert sk.disk_store.path == tmp_path
    sk.close()


def test_store_keeper_set_disk_store_size_bytes(tmp_path):
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_disk_store_path(tmp_path)
    sk.set_disk_store_size_bytes(1234)
    assert sk.disk_store_size_bytes == 1234


def test_store_keeper_set_interval():
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_interval(11)
    assert sk.interval == 11
