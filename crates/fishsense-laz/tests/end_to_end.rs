//! End-to-end pipeline test: build a synthetic SQLite DB matching the
//! fishsense-mobile schema (subset), point the convert pipeline at it,
//! and verify the output `.laz` round-trips through `las::Reader`.

use fishsense_laz::{ConvertOptions, convert_one, db, decode::Confidence};
use image::{ImageBuffer, Rgb};
use rusqlite::Connection;
use std::path::Path;
use tempfile::tempdir;

fn write_synthetic_db(db_path: &Path, rgb_filename: &str, depth_w: u32, depth_h: u32) {
    let conn = Connection::open(db_path).unwrap();
    // Mirror the columns convert_one reads. Other v7 columns are present
    // so a `SELECT *`-style schema mismatch wouldn't slip past us.
    conn.execute_batch(
        "CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            utc_unix_timestamp INTEGER NOT NULL,
            rgb_path TEXT NOT NULL,
            depth_bytes BLOB,
            depth_width INTEGER NOT NULL,
            depth_height INTEGER NOT NULL,
            confidence_bytes BLOB,
            confidence_width INTEGER NOT NULL,
            confidence_height INTEGER NOT NULL,
            intrinsics_bytes BLOB
        );",
    )
    .unwrap();

    // Constant-depth plane at z=2.0 m, 4x4 grid.
    let depth_pixels = vec![2.0_f32; (depth_w * depth_h) as usize];
    let depth_bytes: Vec<u8> = depth_pixels.iter().flat_map(|f| f.to_le_bytes()).collect();
    let confidence_bytes = vec![2u8; (depth_w * depth_h) as usize];

    // K at "RGB" resolution = same as depth here so scaling is a no-op.
    let k: [f64; 9] = [
        depth_w as f64,
        0.0,
        depth_w as f64 / 2.0,
        0.0,
        depth_h as f64,
        depth_h as f64 / 2.0,
        0.0,
        0.0,
        1.0,
    ];
    let intrinsics_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();

    conn.execute(
        "INSERT INTO photos (
            utc_unix_timestamp, rgb_path,
            depth_bytes, depth_width, depth_height,
            confidence_bytes, confidence_width, confidence_height,
            intrinsics_bytes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            1_700_000_000_i64,
            rgb_filename,
            depth_bytes,
            depth_w as i64,
            depth_h as i64,
            confidence_bytes,
            depth_w as i64,
            depth_h as i64,
            intrinsics_bytes,
        ],
    )
    .unwrap();
}

fn write_solid_jpeg(path: &Path, w: u32, h: u32, color: [u8; 3]) {
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w, h);
    for px in img.pixels_mut() {
        *px = Rgb(color);
    }
    img.save(path).unwrap();
}

#[test]
fn fixture_db_converts_to_laz() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("fixture.sqlite");
    let rgb_filename = "frame.jpg";
    let depth_w = 4;
    let depth_h = 4;

    write_synthetic_db(&db_path, rgb_filename, depth_w, depth_h);
    write_solid_jpeg(
        &dir.path().join(rgb_filename),
        depth_w,
        depth_h,
        [200, 100, 50],
    );

    let conn = db::open(&db_path).unwrap();
    let out = dir.path().join("out.laz");
    let opts = ConvertOptions {
        min_confidence: Confidence::Medium,
        rgb_root: dir.path().to_path_buf(),
        fallback_intrinsics: None,
        fallback_hfov_degrees: None,
    };
    let n = convert_one(&conn, 1, &out, &opts).unwrap();
    assert_eq!(n, (depth_w * depth_h) as usize);

    let mut reader = las::Reader::from_path(&out).unwrap();
    let pts: Vec<las::Point> = reader.points().collect::<Result<_, _>>().unwrap();
    assert_eq!(pts.len(), 16);

    // Every point sits at the same depth (z = 2.0). The viewer doesn't
    // care about the camera frame's sign convention, but we can at
    // least pin z down to the value we put in.
    for p in &pts {
        assert!((p.z - 2.0).abs() < 1e-3);
        let c = p.color.expect("color");
        // JPEG is lossy — solid color survives well, but allow a few
        // codes of slop. The encoder also uses 4:2:0 chroma subsampling
        // so chroma can drift on tiny images.
        assert!((i32::from(c.red >> 8) - 200).abs() <= 5);
        assert!((i32::from(c.green >> 8) - 100).abs() <= 5);
        assert!((i32::from(c.blue >> 8) - 50).abs() <= 5);
    }

    // Bounding box matches the corner unprojection. With our K
    // (fx=4, cx=2, fy=4, cy=2, all in 4x4-pixel units) and z=2:
    // corner pixel (u=0,v=0): center (0.5,0.5) → x=y=(0.5-2)*2/4=-0.75
    // corner pixel (u=3,v=3): center (3.5,3.5) → x=y=0.75
    let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p.y).collect();
    let xmin = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((xmin - -0.75).abs() < 1e-3);
    assert!((xmax - 0.75).abs() < 1e-3);
    assert!((ymin - -0.75).abs() < 1e-3);
    assert!((ymax - 0.75).abs() < 1e-3);
}
