# Reception greeter (reference, not migrated)

The reception demo that AlchemyFace was extracted from: a camera loop
that recognises a face, logs the sighting to Postgres and plays an mp3 greeting
by name.

**These files are the original prototype, copied verbatim and not yet
refactored.** They do not import `alchemyface` and will not run as-is — they
expect a Postgres instance with `pgvector`, a working directory containing
`sounds/` and `onnx/`, and a `recorded_faces/` directory that the original code
never creates. They are kept as the behavioural reference for the eventual
rewrite on top of the library.

Nothing here is packaged: `examples/` is excluded from the wheel and sdist.

| File | What it does | Known problems |
|---|---|---|
| `face_recognizer.py` | The whole pipeline in one `main()` — capture, detect, embed, match, draw, log, greet | Hardcoded DSN, thresholds and paths; writes to `./recorded_faces/` without creating it; `if True:` at line 170 disables the unknown-face branch; YOLO block entirely commented out |
| `data_capsule.py` | SQLAlchemy models `FaceData` (`Vector(128)`) and `EventData` | Creates its schema under `__main__` against a hardcoded DSN |
| `entry_db.py` | Bulk-enrols embeddings from `.npy` + csv | `id=43` is hardcoded inside the loop, so every row collides on the primary key; reads `id2map2.csv`, which is absent |
| `get_eventlog.py` | Dumps the event table as JSON | Hardcoded DSN |

## Running it

```bash
make install_greeter          # adds sqlalchemy, pgvector, psycopg2, pygame, pandas
export DATABASE_URL=postgresql://user:password@localhost:5432/faces
```

The connection string comes from `DATABASE_URL`, falling back to
`postgresql://localhost:5432/faces`. The prototype had credentials hardcoded in
four places; they are gone.

Assets live in the git-ignored `_local/` directory (`sounds/`, `onnx/`,
`face_data/`, `data/`). They contain personal data — see the note in the root
README.
