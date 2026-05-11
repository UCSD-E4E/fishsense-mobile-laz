# fishsense-laz

Convert [fishsense-mobile](https://github.com/UCSD-E4E/fishsense-mobile) SQLite
captures — an RGB JPEG plus an ARKit LiDAR depth map — into LAZ point clouds for
the [e4e-point-cloud-viewer](https://github.com/UCSD-E4E/e4e-point-cloud-viewer).

Each photo row becomes one `.laz` file. The depth map is back-projected through
the camera intrinsics into a colored 3D point cloud (one point per valid depth
pixel, colored by the corresponding RGB pixel). The output is LAS point format 2
(XYZ + RGB), LAZ-compressed via the same `las` crate the viewer reads on import,
so the viewer's normalize/recenter step handles units and orientation.

## Build

```sh
cargo build --release
# binary at target/release/fishsense-laz
```

## Usage

### List photos

```sh
fishsense-laz list --db /path/to/database.sqlite
```

Prints one row per photo: id, UTC capture time, fish length (if recorded),
whether depth + intrinsics are present, and whether the RGB JPEG is on disk.
The JPEGs are expected in the same directory as the database.

Rows whose JPEG is missing are hidden by default (they can't be converted).
Pass `--show-missing-rgb` to include them — useful when diagnosing an
incomplete copy off the device.

### Convert

```sh
# specific photos
fishsense-laz convert --db /path/to/database.sqlite --ids 18,19,20 --out ./clouds

# everything with a usable depth + intrinsics blob
fishsense-laz convert --db /path/to/database.sqlite --all --out ./clouds
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--ids 1,2,3` | — | Comma-separated photo ids. Mutually exclusive with `--all`. |
| `--all` | — | Convert every photo with a usable depth + intrinsics blob. |
| `--out DIR` | `.` | Output directory (created if missing). |
| `--rgb-root DIR` | DB's directory | Where the JPEGs referenced by `rgb_path` live. |
| `--min-confidence low\|medium\|high` | `medium` | Drop depth pixels below this ARKit confidence level. |
| `--intrinsics fx,fy,cx,cy` | — | Fallback K matrix at RGB resolution (see below). |
| `--hfov-degrees N` | — | Fallback: synthesize a K from horizontal field of view. |
| `--upsample N` | `1` | Densify the depth map by an integer factor before unprojecting (see below). |

Output files are named `photo-{id:06}-{utc_unix_timestamp}.laz`. Per-row
failures (missing JPEG, zero valid points, …) are logged and the batch
continues; the process exits non-zero only if every requested row failed.

## Camera intrinsics

The conversion needs the camera's 3×3 intrinsic matrix `K`. fishsense-mobile
schema v7+ stores it per row in `intrinsics_bytes` and it's used automatically.
**Older databases don't have that column** — for those you must supply a
fallback, in priority order:

1. `--intrinsics fx,fy,cx,cy` — an exact calibration at the captured RGB
   resolution. Known values from the
   [oceans-2025 processing pipeline](https://github.com/UCSD-E4E/fishsense-mobile-oceans-2025/blob/main/scripts/01_process.ipynb):

   | Device | `fx,fy,cx,cy` |
   |---|---|
   | iPhone Pro | `1375.0719,1375.0719,968.6433,723.04926` |
   | iPad Pro | `1604.2147,1604.2147,956.5816,717.7617` |

2. `--hfov-degrees N` — synthesize a square-pixel K with the principal point
   at the image center. iPhone 12–15 Pro wide camera (which ARKit's `sceneDepth`
   uses) is ≈ 73°. Less accurate than option 1 — edges of the cloud will be
   distorted if the FOV guess is off.

Example for an old iPhone-captured database:

```sh
fishsense-laz convert --db database.sqlite --all --out ./clouds \
  --intrinsics 1375.0719,1375.0719,968.6433,723.04926
```

## Denser clouds: `--upsample`

ARKit `sceneDepth` is a fixed 256×192 grid — ~49k points at most, fewer after
the confidence filter. `--upsample N` runs **joint bilateral upsampling** of
that depth map using the high-res RGB image as an edge guide before
unprojecting: each output pixel's depth is a weighted blend of nearby measured
samples, where the weight drops off both with distance *and* with RGB-color
difference, so depth edges snap to color edges instead of smearing across them.

```sh
fishsense-laz convert --db database.sqlite --all --out ./clouds \
  --intrinsics 1375.0719,1375.0719,968.6433,723.04926 \
  --upsample 8
```

Factor 8 puts you near the RGB resolution (~3M points). The viewer sizes each
splat from local point density (kNN), so a denser cloud automatically renders
with smaller splats — no viewer changes needed.

Caveats: the extra depth is **interpolated, not measured**. Flat regions
upsample cleanly; depth discontinuities can still bleed where the color guide
doesn't separate them; and output pixels with no measured depth nearby are
dropped rather than invented. It's also slower (parallelized across cores, but
still seconds per photo at factor 8). Default is off (`--upsample 1`), which
gives the honest sparse cloud. Allowed range: 1–16.

## How it works

For each valid depth pixel `(u, v)` with depth `z` (meters):

```
K is scaled from RGB resolution down to the depth-map resolution.
x = (u + 0.5 - cx) * z / fx
y = (v + 0.5 - cy) * z / fy
z = z
color = RGB image sampled at the corresponding pixel (nearest neighbor)
```

Pixels with non-finite or non-positive depth, or confidence below
`--min-confidence`, are skipped. Coordinates are in ARKit's sensor-native
camera frame; the viewer recenters and reorients on import.

## Notes

- ARKit `sceneDepth` is a fixed 256×192 grid (~49k points max per capture,
  fewer after the confidence filter) — see `--upsample` above for denser output.
- `*.laz` is gitignored, so converted outputs won't be accidentally committed.
- The fish segmentation mask (`mask_bytes`, when present) is not used yet.

## Tests

```sh
cargo test
```

Unit tests cover blob decoding, intrinsic scaling, unprojection, and LAZ
round-tripping; an integration test runs the full pipeline against a synthetic
SQLite database + JPEG.
