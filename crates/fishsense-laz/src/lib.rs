//! Convert fishsense-mobile SQLite captures into LAZ point clouds for
//! the e4e-point-cloud-viewer.
//!
//! Pipeline: SQLite row → decode BLOBs → load JPEG → unproject depth
//! through camera intrinsics → write LAZ.

pub mod db;
pub mod decode;
pub mod unproject;
pub mod upsample;
pub mod write;

use anyhow::{Context, Result};
use image::ImageReader;
use rusqlite::Connection;
use std::path::{Path, PathBuf};

pub use decode::{Confidence, Intrinsics};

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub min_confidence: Confidence,
    pub rgb_root: PathBuf,
    /// Fallback K matrix at RGB resolution — used when the DB lacks
    /// `intrinsics_bytes` (pre-schema-v7) and an exact calibration is
    /// available. Takes precedence over `fallback_hfov_degrees`.
    pub fallback_intrinsics: Option<Intrinsics>,
    /// Fallback horizontal field-of-view (degrees) used when the DB lacks
    /// `intrinsics_bytes` and no `fallback_intrinsics` is given.
    /// Synthesizes a square-pixel K centered in the captured RGB frame.
    pub fallback_hfov_degrees: Option<f64>,
    /// Upsample the depth map by this integer factor (guided by the RGB
    /// image) before unprojecting. 1 = no upsampling. The extra depth is
    /// interpolated, not measured — see [`upsample`].
    pub upsample_factor: u32,
}

fn synthesize_intrinsics(rgb_w: u32, rgb_h: u32, hfov_degrees: f64) -> Intrinsics {
    let hfov_rad = hfov_degrees.to_radians();
    let fx = f64::from(rgb_w) / (2.0 * (hfov_rad / 2.0).tan());
    Intrinsics {
        fx,
        fy: fx, // square pixels; iPhone wide camera is essentially isotropic.
        cx: f64::from(rgb_w) / 2.0,
        cy: f64::from(rgb_h) / 2.0,
    }
}

/// Resolve the camera intrinsic matrix for a photo: prefer the per-row
/// `intrinsics_bytes` (schema v7+), else the caller's explicit override,
/// else a horizontal-FOV-derived K; error if none is available.
fn resolve_intrinsics(
    row: &db::PhotoRow,
    rgb_w: u32,
    rgb_h: u32,
    opts: &ConvertOptions,
) -> Result<Intrinsics> {
    match row.intrinsics_bytes.as_deref() {
        Some(bytes) if !bytes.is_empty() => decode::decode_intrinsics(bytes)
            .with_context(|| format!("decode intrinsics for photo {}", row.id)),
        _ => opts
            .fallback_intrinsics
            .or_else(|| {
                opts.fallback_hfov_degrees
                    .map(|h| synthesize_intrinsics(rgb_w, rgb_h, h))
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "photo {} has no intrinsics_bytes (pre-schema-v7 capture); \
                     pass --intrinsics fx,fy,cx,cy or --hfov-degrees",
                    row.id
                )
            }),
    }
}

/// Filename for the output LAZ — stable across runs so re-running over the
/// same DB overwrites without piling up duplicates.
#[must_use]
pub fn default_output_name(id: i64, utc_unix_timestamp: i64) -> String {
    format!("photo-{id:06}-{utc_unix_timestamp}.laz")
}

/// Convert one photo row to a LAZ file at `out_path`. Returns the number
/// of points written.
pub fn convert_one(conn: &Connection, id: i64, out_path: &Path, opts: &ConvertOptions) -> Result<usize> {
    let row = db::fetch_photo(conn, id).with_context(|| format!("fetch photo id={id}"))?;

    if row.depth_bytes.is_empty() {
        anyhow::bail!("photo {id} has no depth_bytes (pre-LiDAR or migration-stale row)");
    }

    let depth = decode::decode_depth(&row.depth_bytes, row.depth_width, row.depth_height)
        .with_context(|| format!("decode depth for photo {id}"))?;

    let confidence = match (row.confidence_bytes.as_deref(), row.confidence_width, row.confidence_height) {
        (Some(bytes), w, h) if w == row.depth_width && h == row.depth_height => Some(
            decode::decode_confidence(bytes, w, h)
                .with_context(|| format!("decode confidence for photo {id}"))?,
        ),
        // No confidence stored, or dimensions mismatch the depth grid — skip filtering.
        _ => None,
    };

    let rgb_path = db::resolve_rgb_path(&row.rgb_path, &opts.rgb_root);
    let rgb_img = ImageReader::open(&rgb_path)
        .with_context(|| format!("open rgb image at {}", rgb_path.display()))?
        .with_guessed_format()
        .with_context(|| format!("sniff format of {}", rgb_path.display()))?
        .decode()
        .with_context(|| format!("decode rgb image at {}", rgb_path.display()))?
        .to_rgb8();
    let (rgb_w, rgb_h) = rgb_img.dimensions();

    let intrinsics = resolve_intrinsics(&row, rgb_w, rgb_h, opts)?;

    // Optionally upsample the (confidence-filtered) depth, guided by the
    // RGB image, before unprojecting. The upsampled grid carries no
    // per-pixel confidence — filtering happens up front by NaN-ing the
    // anchors — so the unprojection step gets `None` for confidence.
    let (work_depth, work_w, work_h, work_conf): (
        std::borrow::Cow<'_, [f32]>,
        u32,
        u32,
        Option<&[decode::Confidence]>,
    ) = if opts.upsample_factor > 1 {
        let mut anchors = depth.clone();
        for (i, d) in anchors.iter_mut().enumerate() {
            let keep = d.is_finite()
                && *d > 0.0
                && confidence
                    .as_ref()
                    .is_none_or(|c| c[i].at_least(opts.min_confidence));
            if !keep {
                *d = f32::NAN;
            }
        }
        let up = upsample::joint_bilateral_upsample(
            &anchors,
            row.depth_width,
            row.depth_height,
            rgb_img.as_raw(),
            rgb_w,
            rgb_h,
            opts.upsample_factor,
        );
        (std::borrow::Cow::Owned(up.depth), up.width, up.height, None)
    } else {
        (
            std::borrow::Cow::Borrowed(depth.as_slice()),
            row.depth_width,
            row.depth_height,
            confidence.as_deref(),
        )
    };

    let params = unproject::UnprojectParams {
        rgb_width: rgb_w,
        rgb_height: rgb_h,
        depth_width: work_w,
        depth_height: work_h,
        min_confidence: opts.min_confidence,
    };

    let pts = unproject::unproject(
        rgb_img.as_raw(),
        &work_depth,
        work_conf,
        &intrinsics,
        &params,
    );

    let count = pts.len();
    if count == 0 {
        anyhow::bail!(
            "photo {id} produced 0 points after filtering (confidence threshold={:?})",
            opts.min_confidence
        );
    }

    write::write_laz(out_path, &pts).with_context(|| format!("write {}", out_path.display()))?;
    Ok(count)
}
