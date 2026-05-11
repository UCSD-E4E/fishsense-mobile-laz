# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`fishsense-laz` — a single-binary Rust CLI that converts
[fishsense-mobile](https://github.com/UCSD-E4E/fishsense-mobile) SQLite captures
(RGB JPEG + ARKit LiDAR depth map) into LAZ point clouds for the
[e4e-point-cloud-viewer](https://github.com/UCSD-E4E/e4e-point-cloud-viewer).
One photo row → one `.laz` file. See [README.md](README.md) for user-facing docs.

## Commands

```sh
cargo build              # build
cargo build --release    # release binary at target/release/fishsense-laz
cargo test               # all tests (unit + integration)
cargo clippy             # lints — workspace denies unsafe, warns on pedantic
cargo fmt
```

There is no CI config yet. The `las` crate is pinned to `0.9` deliberately —
that's the version the viewer reads with, so the write path must stay
compatible.

## Architecture

```
SQLite photos row
  → db.rs        schema-aware read (PRAGMA table_info; migrations are additive)
  → decode.rs    depth (f32 LE m), confidence (u8 ARKit levels), intrinsics (9×f64 K)
  → (image crate) load the RGB JPEG referenced by rgb_path
  → upsample.rs  (optional, --upsample) joint bilateral upsample of depth, RGB-guided
  → unproject.rs scale K to depth resolution, back-project, color, filter
  → write.rs     LAS point format 2, LAZ-compressed via `las`
lib.rs orchestrates per-photo (convert_one); main.rs is the CLI.
```

| File | Purpose |
|---|---|
| `crates/fishsense-laz/src/lib.rs` | `convert_one`, `ConvertOptions`, `resolve_intrinsics` |
| `crates/fishsense-laz/src/db.rs` | `rusqlite` reads; `Schema::probe`; `PhotoRow` / `PhotoListEntry` |
| `crates/fishsense-laz/src/decode.rs` | BLOB → typed (`Intrinsics`, `Confidence`, depth `Vec<f32>`) |
| `crates/fishsense-laz/src/upsample.rs` | `joint_bilateral_upsample()` — RGB-guided depth densification (rayon) |
| `crates/fishsense-laz/src/unproject.rs` | `unproject()` → `Vec<ColoredPoint>` |
| `crates/fishsense-laz/src/write.rs` | `write_laz()` |
| `crates/fishsense-laz/src/main.rs` | clap CLI: `list` and `convert` subcommands |
| `crates/fishsense-laz/tests/end_to_end.rs` | full pipeline against a synthetic DB + JPEG |

## fishsense-mobile data model (what this tool reads)

The `photos` table. Schema migrations are **strictly additive** — each version
just `ALTER TABLE ADD COLUMN`s — so older databases lack newer columns. Always
probe with `PRAGMA table_info(photos)` rather than trusting `user_version`
(the original Swift app never bumped it). Columns relevant here:

- `rgb_path` — usually a bare JPEG filename, relative to the DB's directory.
- `depth_bytes` (BLOB) — row-major **f32 little-endian**, meters, dims
  `depth_width × depth_height` (ARKit `sceneDepth` is a fixed 256×192).
  NaN/inf = invalid.
- `confidence_bytes` (BLOB) — row-major **u8**, ARKit `ARConfidenceLevel`
  0/1/2 = low/med/high. Same dims as depth (when present).
- `intrinsics_bytes` (BLOB, schema v7+) — 9 × **f64 LE**, row-major K
  (`fx,0,cx, 0,fy,cy, 0,0,1`), at the captured **RGB resolution** (≈1920×1440),
  *not* the depth resolution. Older DBs don't have this column; the caller
  supplies a fallback (`--intrinsics` or `--hfov-degrees`).

Known iPhone/iPad calibrations (from the oceans-2025 processing notebook) live
in `main.rs`'s `--intrinsics` help text — keep them there if more devices show up.

## Unprojection convention

Per valid depth pixel `(u,v)` with depth `z`, using K scaled to depth
resolution:

```
x = (u + 0.5 - cx) * z / fx
y = (v + 0.5 - cy) * z / fy
z = z
```

ARKit sensor-native camera frame (+X right, +Y down, +Z forward). No axis flip
is applied — the viewer recenters/reorients on import, so getting the absolute
frame "right" here isn't worth the complexity.

## Conventions / gotchas

- Per-row failures in `convert --all` are logged to stderr and the batch
  continues; exit non-zero only if *every* requested row failed.
- On-disk RGB checks (in `list`) do **one** `read_dir` of the DB's directory
  and answer per-row via a `HashSet<OsString>` — never `Path::exists` per row.
  These databases are often on a slow/networked mount (a FUSE-over-HTTP NAS),
  where N stat() calls means minutes of wall time.
- `*.laz` is gitignored — don't commit converted outputs.
- Comments: explain *why*, not *what*. The existing files err toward terse;
  match that.
- Don't widen scope speculatively (e.g. the `mask_bytes` fish segmentation is
  intentionally unused for now — add a `--fish-only` flag only if asked).
- `--upsample` produces *interpolated* depth, not measured — keep that framing
  in any docs/messages. Default is 1 (off); the honest sparse cloud stays the
  default. The viewer turns the denser cloud into smaller splats on its own.
