//! Read photo rows from a fishsense-mobile SQLite database.
//!
//! Schema v7 fields used:
//! - `id`, `utc_unix_timestamp`
//! - `rgb_path` — relative path to the JPEG on iOS Documents.
//! - `depth_bytes` + `depth_width` + `depth_height` — float32 depth grid in meters.
//! - `confidence_bytes` + `confidence_width` + `confidence_height` — u8 ARKit confidence.
//! - `intrinsics_bytes` — 9 × f64 LE row-major K at RGB resolution.
//! - `place_name`, `fish_length` — listing metadata only.

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("photo id {0} not found")]
    NotFound(i64),
}

/// Columns present in this DB's `photos` table. fishsense-mobile schema
/// migrations are strictly additive (each version `ALTER TABLE ADD COLUMN`),
/// so older databases simply lack the newer columns. We probe instead of
/// trusting `PRAGMA user_version`, since the original Swift app didn't bump
/// it.
#[derive(Debug, Clone)]
pub struct Schema {
    columns: HashSet<String>,
}

impl Schema {
    pub fn probe(conn: &Connection) -> Result<Self, DbError> {
        let mut stmt = conn.prepare("PRAGMA table_info(photos)")?;
        let cols = stmt
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<Result<HashSet<_>, _>>()?;
        Ok(Self { columns: cols })
    }

    #[must_use]
    pub fn has(&self, col: &str) -> bool {
        self.columns.contains(col)
    }
}

/// Subset of the `photos` row needed to render a listing line. No BLOBs.
#[derive(Debug, Clone)]
pub struct PhotoListEntry {
    pub id: i64,
    pub utc_unix_timestamp: i64,
    pub rgb_path: String,
    pub place_name: Option<String>,
    pub fish_length_m: Option<f64>,
    pub has_depth: bool,
    pub has_intrinsics: bool,
}

/// Full row needed to convert one photo to a point cloud.
#[derive(Debug, Clone)]
pub struct PhotoRow {
    pub id: i64,
    pub utc_unix_timestamp: i64,
    pub rgb_path: String,
    pub depth_bytes: Vec<u8>,
    pub depth_width: u32,
    pub depth_height: u32,
    pub confidence_bytes: Option<Vec<u8>>,
    pub confidence_width: u32,
    pub confidence_height: u32,
    /// `None` on pre-v7 databases (column does not exist) or when the row
    /// itself has a NULL — the caller must supply a fallback K.
    pub intrinsics_bytes: Option<Vec<u8>>,
}

pub fn open(path: &Path) -> Result<Connection, DbError> {
    Ok(Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?)
}

pub fn list_photos(conn: &Connection) -> Result<Vec<PhotoListEntry>, DbError> {
    let schema = Schema::probe(conn)?;
    // Build the SELECT against whatever columns the DB actually has —
    // pre-v6 has no `place_name`, pre-v3 has no `fish_length`, pre-v7
    // has no `intrinsics_bytes`. Use a literal NULL placeholder for any
    // missing column so the query result indexing stays stable.
    let place_expr = if schema.has("place_name") {
        "place_name"
    } else {
        "NULL"
    };
    let fish_expr = if schema.has("fish_length") {
        "fish_length"
    } else {
        "NULL"
    };
    let intr_expr = if schema.has("intrinsics_bytes") {
        "intrinsics_bytes IS NOT NULL"
    } else {
        "0"
    };
    let sql = format!(
        "SELECT id, utc_unix_timestamp, rgb_path, {place_expr}, {fish_expr}, \
                depth_bytes IS NOT NULL, {intr_expr} \
         FROM photos ORDER BY id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map([], |r| {
            Ok(PhotoListEntry {
                id: r.get(0)?,
                utc_unix_timestamp: r.get(1)?,
                rgb_path: r.get(2)?,
                place_name: r.get(3)?,
                fish_length_m: r.get(4)?,
                has_depth: r.get(5)?,
                has_intrinsics: r.get(6)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn fetch_photo(conn: &Connection, id: i64) -> Result<PhotoRow, DbError> {
    let schema = Schema::probe(conn)?;
    let intr_expr = if schema.has("intrinsics_bytes") {
        "intrinsics_bytes"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT id, utc_unix_timestamp, rgb_path, \
                depth_bytes, depth_width, depth_height, \
                confidence_bytes, confidence_width, confidence_height, \
                {intr_expr} \
         FROM photos WHERE id = ?1"
    );
    let row = conn
        .query_row(&sql, [id], |r| {
            Ok(PhotoRow {
                id: r.get(0)?,
                utc_unix_timestamp: r.get(1)?,
                rgb_path: r.get(2)?,
                depth_bytes: r.get::<_, Option<Vec<u8>>>(3)?.unwrap_or_default(),
                depth_width: r.get::<_, i64>(4)? as u32,
                depth_height: r.get::<_, i64>(5)? as u32,
                confidence_bytes: r.get::<_, Option<Vec<u8>>>(6)?,
                confidence_width: r.get::<_, i64>(7)? as u32,
                confidence_height: r.get::<_, i64>(8)? as u32,
                intrinsics_bytes: r.get::<_, Option<Vec<u8>>>(9)?,
            })
        })
        .optional()?;
    row.ok_or(DbError::NotFound(id))
}

/// Resolve `rgb_path` against `--rgb-root` (defaulting to the DB's parent
/// directory). Absolute paths pass through unchanged. fishsense-mobile
/// stores paths as bare filenames relative to iOS Documents — when the DB
/// is copied off-device the user usually copies the JPEGs alongside it.
#[must_use]
pub fn resolve_rgb_path(rgb_path: &str, root: &Path) -> PathBuf {
    let p = Path::new(rgb_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_absolute_passes_through() {
        let resolved = resolve_rgb_path("/tmp/foo.jpg", Path::new("/var/lib"));
        assert_eq!(resolved, PathBuf::from("/tmp/foo.jpg"));
    }

    #[test]
    fn resolve_relative_joins_root() {
        let resolved = resolve_rgb_path("foo.jpg", Path::new("/var/lib"));
        assert_eq!(resolved, PathBuf::from("/var/lib/foo.jpg"));
    }
}
