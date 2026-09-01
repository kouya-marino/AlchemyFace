"""The background detection worker, tested with a fake detector.

The worker knows nothing about Tk and nothing about OpenCV: it is handed a
callable that turns a path into faces. So the concurrency — priorities, stale
work, error handling, shutdown — is testable without a display or a model.

Threading tests are written to be deterministic: nothing sleeps hoping for a
result, everything waits on an event or polls with a deadline.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from alchemyface.gui.detect_worker import DetectionResult, DetectionWorker
from alchemyface.types import Face

TIMEOUT = 10.0
"""Generous: a slow CI box must not turn a pass into a flake."""


def a_face(x: int = 0) -> Face:
    return Face(
        bbox=(x, 0, 10, 10),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        confidence=0.9,
    )


def collect(worker: DetectionWorker, count: int, timeout: float = TIMEOUT) -> list[DetectionResult]:
    """Drain until `count` results have arrived, or the deadline passes."""
    got: list[DetectionResult] = []
    deadline = time.monotonic() + timeout
    while len(got) < count and time.monotonic() < deadline:
        got.extend(worker.drain())
        if len(got) < count:
            time.sleep(0.005)
    return got


@pytest.fixture()
def worker_factory():  # type: ignore[no-untyped-def]
    """Build workers and guarantee they are shut down after the test."""
    created: list[DetectionWorker] = []

    def build(detect, **kw):  # type: ignore[no-untyped-def]
        worker = DetectionWorker(detect=detect, **kw)
        created.append(worker)
        worker.start()
        return worker

    yield build
    for worker in created:
        worker.shutdown(wait=True)


# --------------------------------------------------------------- the basics


def test_a_submitted_job_comes_back(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [a_face()])
    worker.submit(0, Path("a.jpg"))
    (result,) = collect(worker, 1)
    assert result.index == 0
    assert result.error is None
    assert result.faces is not None and len(result.faces) == 1


def test_no_faces_is_an_empty_list_not_an_error(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [])
    worker.submit(0, Path("a.jpg"))
    (result,) = collect(worker, 1)
    assert result.faces == []
    assert result.error is None


def test_drain_is_non_blocking_when_nothing_is_ready(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [a_face()])
    assert worker.drain() == []


def test_many_jobs_all_come_back(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [a_face()])
    for i in range(25):
        worker.submit(i, Path(f"{i}.jpg"))
    results = collect(worker, 25)
    assert sorted(r.index for r in results) == list(range(25))


def test_drain_respects_its_limit(worker_factory) -> None:  # type: ignore[no-untyped-def]
    # The Tk thread must not be monopolised when a big burst lands.
    worker = worker_factory(lambda _p: [a_face()])
    for i in range(30):
        worker.submit(i, Path(f"{i}.jpg"))
    collect(worker, 30)
    worker._results.queue.clear()  # noqa: SLF001
    for i in range(30):
        worker._results.put(DetectionResult(i, worker.generation, [], None))  # noqa: SLF001
    assert len(worker.drain(limit=8)) == 8


# ------------------------------------------------------------- priorities


def test_a_foreground_job_jumps_the_queue(worker_factory) -> None:  # type: ignore[no-untyped-def]
    """The image the user is looking at must not wait behind a folder prefetch."""
    release = threading.Event()
    started = threading.Event()
    order: list[int] = []

    def detect(path: Path) -> list[Face]:
        index = int(path.stem)
        if index == 0:  # the first job holds the worker
            started.set()
            release.wait(TIMEOUT)
        order.append(index)
        return []

    worker = worker_factory(detect)
    worker.submit(0, Path("0.jpg"))
    assert started.wait(TIMEOUT), "worker never picked up the first job"

    # Queue background work, then a foreground job behind it.
    for i in range(1, 6):
        worker.submit(i, Path(f"{i}.jpg"), foreground=False)
    worker.submit(9, Path("9.jpg"), foreground=True)

    release.set()
    collect(worker, 7)
    # 0 ran first because it was already in flight; 9 must precede the rest.
    assert order[0] == 0
    assert order[1] == 9, order


# ----------------------------------------------------------- generations


def test_bumping_the_generation_discards_queued_work(worker_factory) -> None:  # type: ignore[no-untyped-def]
    release = threading.Event()
    started = threading.Event()
    seen: list[int] = []

    def detect(path: Path) -> list[Face]:
        if path.stem == "0":
            started.set()
            release.wait(TIMEOUT)
        seen.append(int(path.stem))
        return []

    worker = worker_factory(detect)
    worker.submit(0, Path("0.jpg"))
    assert started.wait(TIMEOUT)
    for i in range(1, 10):
        worker.submit(i, Path(f"{i}.jpg"), foreground=False)

    worker.new_generation()  # user opened a different folder
    release.set()
    collect(worker, 1)
    time.sleep(0.2)  # give any survivor a chance to run
    assert seen == [0], f"stale jobs ran: {seen}"


def test_results_from_a_previous_generation_are_dropped(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [a_face()])
    stale = worker.generation
    worker.submit(0, Path("a.jpg"))
    collect(worker, 1)
    worker.new_generation()
    # A result minted under the old generation must not survive a drain.
    worker._results.put(DetectionResult(0, stale, [], None))  # noqa: SLF001
    assert worker.drain() == []


def test_new_generation_returns_an_increasing_number(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [])
    first = worker.generation
    assert worker.new_generation() > first
    assert worker.generation > first


# ----------------------------------------------------------------- errors


def test_a_failing_detector_is_reported_not_raised(worker_factory) -> None:  # type: ignore[no-untyped-def]
    def detect(_p: Path) -> list[Face]:
        raise RuntimeError("model exploded")

    worker = worker_factory(detect)
    worker.submit(0, Path("a.jpg"))
    (result,) = collect(worker, 1)
    assert result.faces is None
    assert result.error is not None and "model exploded" in result.error


def test_one_failure_does_not_kill_the_worker(worker_factory) -> None:  # type: ignore[no-untyped-def]
    def detect(path: Path) -> list[Face]:
        if path.stem == "1":
            raise RuntimeError("boom")
        return [a_face()]

    worker = worker_factory(detect)
    for i in range(4):
        worker.submit(i, Path(f"{i}.jpg"))
    results = collect(worker, 4)
    assert sorted(r.index for r in results) == [0, 1, 2, 3]
    failed = [r for r in results if r.error]
    assert len(failed) == 1 and failed[0].index == 1


def test_a_detector_returning_none_is_treated_as_no_faces(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: None)
    worker.submit(0, Path("a.jpg"))
    (result,) = collect(worker, 1)
    assert result.faces == []
    assert result.error is None


# --------------------------------------------------------------- lifecycle


def test_duplicate_submits_are_deduped(worker_factory) -> None:  # type: ignore[no-untyped-def]
    release = threading.Event()
    started = threading.Event()
    calls: list[int] = []

    def detect(path: Path) -> list[Face]:
        if path.stem == "0":
            started.set()
            release.wait(TIMEOUT)
        calls.append(int(path.stem))
        return []

    worker = worker_factory(detect)
    worker.submit(0, Path("0.jpg"))
    assert started.wait(TIMEOUT)
    for _ in range(5):
        worker.submit(1, Path("1.jpg"), foreground=False)
    release.set()
    collect(worker, 2)
    time.sleep(0.15)
    assert calls.count(1) == 1, calls


def test_shutdown_stops_the_thread(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [])
    worker.shutdown(wait=True)
    assert not worker.is_running


def test_shutdown_is_idempotent(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [])
    worker.shutdown(wait=True)
    worker.shutdown(wait=True)
    worker.shutdown()


def test_submitting_after_shutdown_is_ignored(worker_factory) -> None:  # type: ignore[no-untyped-def]
    worker = worker_factory(lambda _p: [a_face()])
    worker.shutdown(wait=True)
    worker.submit(0, Path("a.jpg"))
    assert worker.drain() == []


def test_the_worker_thread_is_a_daemon(worker_factory) -> None:  # type: ignore[no-untyped-def]
    # A non-daemon worker would hang interpreter exit if shutdown were missed.
    worker = worker_factory(lambda _p: [])
    assert worker.is_running
    assert worker._thread is not None and worker._thread.daemon  # noqa: SLF001
