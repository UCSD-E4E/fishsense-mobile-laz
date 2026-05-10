//! Convert fishsense-mobile SQLite captures into LAZ point clouds for
//! the e4e-point-cloud-viewer.
//!
//! Pipeline: SQLite row → decode BLOBs → load JPEG → unproject depth
//! through camera intrinsics → write LAZ.

pub mod db;
pub mod decode;
pub mod unproject;
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
}

fn synthesize_intrinsics(rgb_w: u32, rgb_h: u32, hfov_degrees: f64) -> decode::Intrinsics {
    let hfov_rad = hfov_degrees.to_radians();
    let fx = f64::from(rgb_w) / (2.0 * (hfov_rad / 2.0).tan());
    decode::Intrinsics {
        fx,
        fy: fx, // square pixels; iPhone wide camera is essentially isotropic.
        cx: f64::from(rgb_w) / 2.0,
        cy: f64::from(rgb_h) / 2.0,
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

    let intrinsics = match row.intrinsics_bytes.as_deref() {
        Some(bytes) if !bytes.is_empty() => decode::decode_intrinsics(bytes)
            .with_context(|| format!("decode intrinsics for photo {id}"))?,
        _ => {
            if let Some(intr) = opts.fallback_intrinsics {
                intr
            } else if let Some(hfov) = opts.fallback_hfov_degrees {
                synthesize_intrinsics(rgb_w, rgb_h, hfov)
            } else {
                anyhow::bail!(
                    "photo {id} has no intrinsics_bytes (pre-schema-v7 capture); \
                     pass --intrinsics fx,fy,cx,cy or --hfov-degrees"
                );
            }
        }
    };

    let params = unproject::UnprojectParams {
        rgb_width: rgb_w,
        rgb_height: rgb_h,
        depth_width: row.depth_width,
        depth_height: row.depth_height,
        min_confidence: opts.min_confidence,
    };

    let pts = unproject::unproject(
        rgb_img.as_raw(),
        &depth,
        confidence.as_deref(),
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
