import sys
import threading
import time

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


def test_store_keeper_concurrent_memory_store_access(tmp_path):
    """Writing to the memory store must not kill the StoreKeeper thread"""
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_disk_store_path(tmp_path)
    sk.set_interval(0.0005)
    sk.set_memory_store_size(2000)

    errors = []
    perform_tasks = sk.perform_tasks

    def perform_tasks_record_errors():
        try:
            perform_tasks()
        except BaseException as e:
            errors.append(e)
            raise

    sk.perform_tasks = perform_tasks_record_errors

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        sk.start()
        stop = time.monotonic() + 1

        def writer(tid):
            ii = 0
            while time.monotonic() < stop and not errors:
                sk.memory_store[f"writer/{tid}/{ii}"] = ii
                ii += 1

        threads = [threading.Thread(target=writer, args=(tid,))
                   for tid in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        sys.setswitchinterval(switch_interval)
        sk.close()

    assert not errors


def test_store_keeper_exception_does_not_stop_thread(tmp_path):
    """An exception in `perform_tasks` must not stop the thread"""
    # create a disconnected store keeper instance
    sk = cached.StoreKeeper()
    sk.set_interval(0.001)
    calls = []

    def perform_tasks_failing():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Test error in perform_tasks")

    sk.perform_tasks = perform_tasks_failing
    sk.start()
    try:
        for _ in range(1000):
            if len(calls) >= 3:
                break
            time.sleep(0.01)
        assert sk.is_alive()
        assert len(calls) >= 3
    finally:
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
