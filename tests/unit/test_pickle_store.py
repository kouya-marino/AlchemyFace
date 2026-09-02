"""PickleStore: the robot's list[(id, name, group, vector)] pickle as a FaceStore.

Two things separate it from InMemoryStore and both are deliberate:

- It stores vectors **exactly as given**, unnormalised, because the robot's
  databases hold raw SFace output and a round trip must not alter them.
- It is forgiving on read. The three production databases disagree with their
  own documented schema: `id` is int in one and str in two, and the vector is
  (1, 128) in one and (128,) in two. A stricter reader would refuse a database
  the robot loads today.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from alchemyface.errors import AlchemyFaceError, PickleSchemaError
from alchemyface.store import PickleStore


def raw(*values: float, scale: float = 10.0) -> np.ndarray:
    """A vector with an SFace-like magnitude, not unit length."""
    v = np.array(values, dtype=np.float32)
    return (v / np.linalg.norm(v) * scale).astype(np.float32)


def basis(i: int, dim: int = 4, scale: float = 10.0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = scale
    return v


# --------------------------------------------------------------- basics


def test_new_store_is_empty() -> None:
    assert len(PickleStore(dim=4)) == 0


def test_add_preserves_the_vector_exactly() -> None:
    # The whole reason this store exists: no normalisation on the way in.
    store = PickleStore(dim=4)
    v = basis(0, scale=13.5)
    store.add("ada", v, {"group": "staff"})
    (stored,) = store.vectors()
    np.testing.assert_array_equal(stored, v)
    assert float(np.linalg.norm(stored)) == pytest.approx(13.5)


def test_add_returns_unique_ids() -> None:
    store = PickleStore(dim=4)
    assert store.add("ada", basis(0)) != store.add("grace", basis(1))
    assert len(store) == 2


def test_group_travels_in_metadata() -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0), {"group": "ceo"})
    (m,) = store.search(basis(0))
    assert m.label == "ada"
    assert m.metadata["group"] == "ceo"


def test_missing_group_becomes_empty_string() -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0))
    (m,) = store.search(basis(0))
    assert m.metadata["group"] == ""


# --------------------------------------------------------------- search


def test_search_on_empty_store_returns_nothing() -> None:
    assert PickleStore(dim=4).search(basis(0)) == []


def test_search_ranks_correctly_despite_raw_vectors() -> None:
    # Stored vectors have differing magnitudes; cosine must ignore that.
    store = PickleStore(dim=4)
    store.add("far", basis(1, scale=50.0))
    store.add("near", raw(1.0, 0.1, 0.0, 0.0, scale=0.5))
    store.add("middle", raw(1.0, 1.0, 0.0, 0.0, scale=99.0))
    assert [m.label for m in store.search(basis(0), k=3)] == ["near", "middle", "far"]


def test_exact_match_scores_one_regardless_of_scale() -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0, scale=13.0))
    (m,) = store.search(basis(0, scale=2.0))
    assert m.score == pytest.approx(1.0)


def test_orthogonal_scores_zero() -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0))
    (m,) = store.search(basis(1))
    assert m.score == pytest.approx(0.0)


def test_remove_drops_the_entry() -> None:
    store = PickleStore(dim=4)
    eid = store.add("ada", basis(0))
    store.add("grace", basis(1))
    store.remove(eid)
    assert len(store) == 1
    assert [m.label for m in store.search(basis(1), k=5)] == ["grace"]


def test_remove_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        PickleStore(dim=4).remove("nope")


# --------------------------------------------------------------- round trip


def test_save_load_round_trip_is_lossless(tmp_path: Path) -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0, scale=11.0), {"group": "ceo"})
    store.add("grace", basis(1, scale=12.0), {"group": "staff"})
    p = tmp_path / "db.pkl"
    store.save(p)

    back = PickleStore(dim=4)
    back.load(p)
    assert len(back) == 2
    np.testing.assert_array_equal(back.vectors()[0], basis(0, scale=11.0))
    labels = [m.label for m in back.search(basis(0), k=2)]
    assert labels[0] == "ada"


def test_save_writes_the_robot_schema(tmp_path: Path) -> None:
    store = PickleStore(dim=4)
    store.add("ada", basis(0), {"group": "staff"})
    p = tmp_path / "db.pkl"
    store.save(p)

    with open(p, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list)
    (entry,) = data
    assert isinstance(entry, tuple) and len(entry) == 4
    eid, name, group, vec = entry
    assert isinstance(eid, str)
    assert (name, group) == ("ada", "staff")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1


def test_save_renumbers_ids_from_zero(tmp_path: Path) -> None:
    store = PickleStore(dim=4)
    for i in range(3):
        store.add(f"p{i}", basis(i % 4))
    p = tmp_path / "db.pkl"
    store.save(p)
    with open(p, "rb") as f:
        data = pickle.load(f)
    assert [e[0] for e in data] == ["0", "1", "2"]


def test_save_of_empty_store_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "empty.pkl"
    PickleStore(dim=4).save(p)
    back = PickleStore(dim=4)
    back.load(p)
    assert len(back) == 0


def test_load_replaces_rather_than_appends(tmp_path: Path) -> None:
    src = PickleStore(dim=4)
    src.add("ada", basis(0))
    p = tmp_path / "db.pkl"
    src.save(p)

    dst = PickleStore(dim=4)
    dst.add("grace", basis(1))
    dst.load(p)
    assert len(dst) == 1
    assert dst.search(basis(0))[0].label == "ada"


# --------------------------------------- real-world schema variations


def write_raw(path: Path, data: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_load_accepts_int_ids(tmp_path: Path) -> None:
    # One production database stores int ids.
    p = tmp_path / "int_ids.pkl"
    write_raw(p, [(0, "ada", "ceo", basis(0)), (1, "grace", "staff", basis(1))])
    store = PickleStore(dim=4)
    store.load(p)
    assert len(store) == 2
    assert store.search(basis(0))[0].label == "ada"


def test_load_accepts_nested_1xN_vectors(tmp_path: Path) -> None:
    # One production database stores (1, 128), not (128,).
    p = tmp_path / "nested.pkl"
    write_raw(p, [("0", "ada", "ceo", basis(0).reshape(1, 4))])
    store = PickleStore(dim=4)
    store.load(p)
    assert store.vectors()[0].shape == (4,)
    assert store.search(basis(0))[0].label == "ada"


def test_load_accepts_the_dict_form(tmp_path: Path) -> None:
    p = tmp_path / "dict.pkl"
    write_raw(p, {"ada": basis(0), "grace": basis(1)})
    store = PickleStore(dim=4)
    store.load(p)
    assert len(store) == 2
    (m,) = store.search(basis(0))
    assert m.label == "ada"
    assert m.metadata["group"] == ""


def test_load_infers_dimension_from_the_file(tmp_path: Path) -> None:
    p = tmp_path / "eight.pkl"
    write_raw(p, [("0", "ada", "", np.arange(8, dtype=np.float32))])
    store = PickleStore(dim=4)  # constructed for 4, file holds 8
    store.load(p)
    assert store.dim == 8


# --------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "payload",
    [
        42,
        "not a db",
        [("0", "ada", "staff")],  # tuple too short
        [("0", "ada", "staff", basis(0), "extra")],  # tuple too long
        [["0", "ada", "staff", basis(0)]],  # list, not tuple
        [("0", "ada", "staff", "not a vector")],  # unconvertible vector
    ],
)
def test_load_rejects_malformed_payloads(tmp_path: Path, payload: object) -> None:
    p = tmp_path / "bad.pkl"
    write_raw(p, payload)
    with pytest.raises(PickleSchemaError):
        PickleStore(dim=4).load(p)


def test_load_rejects_mixed_dimensions(tmp_path: Path) -> None:
    p = tmp_path / "mixed.pkl"
    # Non-zero, or the zero-vector check fires first and masks this one.
    write_raw(
        p,
        [
            ("0", "ada", "", basis(0, dim=4)),
            ("1", "grace", "", basis(0, dim=8)),
        ],
    )
    with pytest.raises(PickleSchemaError, match="dimension"):
        PickleStore(dim=4).load(p)


def test_load_rejects_a_zero_vector(tmp_path: Path) -> None:
    p = tmp_path / "zero.pkl"
    write_raw(p, [("0", "ada", "", np.zeros(4, dtype=np.float32))])
    with pytest.raises(PickleSchemaError, match="zero"):
        PickleStore(dim=4).load(p)


def test_load_of_a_non_pickle_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "notapickle.pkl"
    p.write_bytes(b"this is not a pickle")
    with pytest.raises(PickleSchemaError):
        PickleStore(dim=4).load(p)


def test_schema_error_is_an_alchemyface_error() -> None:
    assert issubclass(PickleSchemaError, AlchemyFaceError)


# ------------------------------------------- the real production files


def test_real_databases_load(pkl_dir: Path) -> None:
    """The three production databases must load, with the counts we measured."""
    # Fixture names are deliberately neutral: the originals embedded a client
    # name, which reached a public repository through this file.
    expected = {"db_mixed_ids.pkl": 53, "db_thirty.pkl": 30, "db_small.pkl": 7}
    for name, count in expected.items():
        path = pkl_dir / name
        if not path.is_file():
            pytest.skip(f"fixture missing: {name}")
        store = PickleStore()
        store.load(path)
        assert len(store) == count, name
        assert store.dim == 128, name
        # Every one stores raw vectors — that is why normalize=False exists.
        norms = [float(np.linalg.norm(v)) for v in store.vectors()]
        assert min(norms) > 2.0, f"{name}: vectors look normalised"


# ------------------------------------------------- protocol + invariants


def test_satisfies_the_facestore_protocol() -> None:
    from alchemyface.store.base import FaceStore

    # The class claims to implement it; nothing asserted so before now.
    assert isinstance(PickleStore(dim=4), FaceStore)


def test_vectors_returns_copies_not_references() -> None:
    # A caller mutating what it was handed must not corrupt the gallery. If it
    # could, a zero written that way would make search() return NaN.
    store = PickleStore(dim=4)
    store.add("ada", basis(0, scale=10.0))
    handed_out = store.vectors()[0]
    handed_out[0] = 999.0
    assert store.vectors()[0][0] == pytest.approx(10.0)


def test_add_refuses_a_zero_vector() -> None:
    # The other half of the boundary; load() is covered separately.
    store = PickleStore(dim=4)
    with pytest.raises(AlchemyFaceError, match="zero"):
        store.add("ada", np.zeros(4, dtype=np.float32))


def test_add_refuses_a_wrong_dimension() -> None:
    store = PickleStore(dim=4)
    with pytest.raises(AlchemyFaceError, match="dimension"):
        store.add("ada", basis(0, dim=8))


def test_search_never_returns_nan_for_valid_input() -> None:
    store = PickleStore(dim=4)
    for i in range(4):
        store.add(f"p{i}", basis(i, scale=float(i + 1) * 3.0))
    scores = [m.score for m in store.search(basis(0), k=4)]
    assert all(np.isfinite(s) for s in scores)


def test_load_wraps_an_uninstalled_module_reference(tmp_path: Path) -> None:
    """A pickle whose GLOBAL opcode names an absent module used to escape.

    load() enumerated the exceptions it expected, so ModuleNotFoundError went
    straight past the wrapper and out of the library as itself.
    """
    p = tmp_path / "ghost.pkl"
    p.write_bytes(b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x11definitely_absent\x94\x8c\x01X\x94\x93\x94.")
    with pytest.raises(PickleSchemaError):
        PickleStore(dim=4).load(p)


def test_load_of_a_missing_file_is_still_a_schema_error(tmp_path: Path) -> None:
    """OSError keeps its own clearer message rather than the generic one."""
    with pytest.raises(PickleSchemaError, match="cannot be read"):
        PickleStore(dim=4).load(tmp_path / "absent.pkl")


def test_load_of_a_directory_is_a_schema_error(tmp_path: Path) -> None:
    with pytest.raises(PickleSchemaError):
        PickleStore(dim=4).load(tmp_path)


# ------------------------------------------------------- the lenient read path
#
# A database is opened in the inspector because something is suspected wrong
# with it, and in the editor to delete the offending rows. Refusing to load one
# with a bad vector withheld the diagnosis and made repair impossible.


def _broken_database(path: Path) -> Path:
    good = np.ones(128, dtype=np.float32)
    nan = np.full(128, np.nan, dtype=np.float32)
    short = np.ones(64, dtype=np.float32)
    with open(path, "wb") as handle:
        pickle.dump(
            [("1", "ada", "ceo", good), ("2", "grace", "staff", nan), ("3", "linus", "staff", short)],
            handle,
        )
    return path


def test_a_strict_load_still_refuses_a_bad_vector(tmp_path: Path) -> None:
    store = PickleStore()
    with pytest.raises(PickleSchemaError, match="NaN or infinity"):
        store.load(_broken_database(tmp_path / "broken.pkl"))


def test_a_lenient_load_shows_every_entry_and_names_the_problems(tmp_path: Path) -> None:
    store = PickleStore()
    problems = store.load_leniently(_broken_database(tmp_path / "broken.pkl"))
    assert len(store) == 3
    assert [entry.label for entry in store.entries()] == ["ada", "grace", "linus"]
    assert any("NaN" in problem for problem in problems)
    assert any("dimension 64" in problem for problem in problems)


def test_a_lenient_load_of_a_clean_database_complains_about_nothing(tmp_path: Path) -> None:
    store = PickleStore()
    store.add("ada", np.ones(128, dtype=np.float32), {"group": "ceo"})
    store.save(tmp_path / "clean.pkl")
    reader = PickleStore()
    assert reader.load_leniently(tmp_path / "clean.pkl") == []
    assert len(reader) == 1


def test_a_lenient_load_still_refuses_something_that_is_not_a_database(tmp_path: Path) -> None:
    """Tolerance is for bad vectors, not for a file that is not a database."""
    path = tmp_path / "nonsense.pkl"
    with open(path, "wb") as handle:
        pickle.dump("just a string", handle)
    with pytest.raises(PickleSchemaError, match="expected a list"):
        PickleStore().load_leniently(path)


def test_the_dict_form_numbers_entries_by_position(tmp_path: Path) -> None:
    """A random uuid differed on every load, so the ID column could not be used
    to refer to a row or to compare two views of the same file."""
    path = tmp_path / "dict.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"ada": np.ones(128, dtype=np.float32), "grace": np.full(128, 2.0, dtype=np.float32)}, handle)

    first = PickleStore()
    first.load(path)
    ids = [entry.entry_id for entry in first.entries()]
    assert ids == ["0", "1"]

    second = PickleStore()
    second.load(path)
    assert [entry.entry_id for entry in second.entries()] == ids
