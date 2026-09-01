"""The store is pure numpy, so it is tested with hand-made unit vectors and
no models at all."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alchemyface.store import InMemoryStore


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def basis(index: int, dim: int = 4) -> np.ndarray:
    """A one-hot unit vector — orthogonal to every other basis vector, so
    cosine similarity between two different ones is exactly 0."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_new_store_is_empty() -> None:
    assert len(InMemoryStore(dim=4)) == 0


def test_add_returns_a_unique_id() -> None:
    store = InMemoryStore(dim=4)
    first = store.add("ada", basis(0))
    second = store.add("grace", basis(1))
    assert first != second
    assert len(store) == 2


def test_search_on_an_empty_store_returns_nothing() -> None:
    assert InMemoryStore(dim=4).search(basis(0)) == []


def test_search_finds_the_exact_vector_with_score_one() -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0), {"team": "analytical"})
    (match,) = store.search(basis(0))
    assert match.label == "ada"
    assert match.entry_id == entry_id
    assert match.metadata == {"team": "analytical"}
    assert match.score == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0))
    (match,) = store.search(basis(1))
    assert match.score == pytest.approx(0.0)


def test_search_ranks_by_descending_score() -> None:
    store = InMemoryStore(dim=4)
    store.add("far", basis(1))
    store.add("near", unit(1.0, 0.1, 0.0, 0.0))
    store.add("middle", unit(1.0, 1.0, 0.0, 0.0))
    labels = [m.label for m in store.search(basis(0), k=3)]
    assert labels == ["near", "middle", "far"]


def test_k_larger_than_the_gallery_is_not_an_error() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0))
    assert len(store.search(basis(0), k=50)) == 1


def test_unnormalised_input_is_normalised_on_the_way_in() -> None:
    store = InMemoryStore(dim=4)
    store.add("ada", np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32))
    (match,) = store.search(np.array([9.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert match.score == pytest.approx(1.0)


def test_wrong_dimension_is_rejected() -> None:
    store = InMemoryStore(dim=4)
    with pytest.raises(ValueError, match="dimension"):
        store.add("ada", np.zeros(8, dtype=np.float32))


def test_zero_vector_is_rejected() -> None:
    # A zero vector has no direction, so cosine similarity is undefined.
    store = InMemoryStore(dim=4)
    with pytest.raises(ValueError, match="zero"):
        store.add("ada", np.zeros(4, dtype=np.float32))


def test_a_2d_row_vector_is_accepted() -> None:
    # cv2's feature() hands back (1, 128); callers should not have to ravel it.
    store = InMemoryStore(dim=4)
    store.add("ada", basis(0).reshape(1, 4))
    assert store.search(basis(0).reshape(1, 4))[0].label == "ada"


def test_remove_drops_the_entry_and_its_vector() -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0))
    store.add("grace", basis(1))
    store.remove(entry_id)
    assert len(store) == 1
    assert [m.label for m in store.search(basis(1), k=5)] == ["grace"]


def test_remove_of_an_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        InMemoryStore(dim=4).remove("nope")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = InMemoryStore(dim=4)
    entry_id = store.add("ada", basis(0), {"team": "analytical", "n": 1})
    store.add("grace", basis(1))
    path = tmp_path / "gallery.npz"
    store.save(path)

    restored = InMemoryStore(dim=4)
    restored.load(path)
    assert len(restored) == 2
    (match,) = restored.search(basis(0), k=1)
    assert match.label == "ada"
    assert match.entry_id == entry_id
    assert match.metadata == {"team": "analytical", "n": 1}


def test_load_replaces_rather_than_appends(tmp_path: Path) -> None:
    source = InMemoryStore(dim=4)
    source.add("ada", basis(0))
    path = tmp_path / "gallery.npz"
    source.save(path)

    target = InMemoryStore(dim=4)
    target.add("grace", basis(1))
    target.load(path)
    assert len(target) == 1
    assert target.search(basis(0))[0].label == "ada"


def test_load_rejects_a_dimension_mismatch(tmp_path: Path) -> None:
    source = InMemoryStore(dim=8)
    source.add("ada", basis(0, dim=8))
    path = tmp_path / "gallery.npz"
    source.save(path)
    with pytest.raises(ValueError, match="dimension"):
        InMemoryStore(dim=4).load(path)


def test_save_of_an_empty_store_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    InMemoryStore(dim=4).save(path)
    restored = InMemoryStore(dim=4)
    restored.load(path)
    assert len(restored) == 0


def test_satisfies_the_facestore_protocol() -> None:
    from alchemyface.store.base import FaceStore

    assert isinstance(InMemoryStore(dim=4), FaceStore)
