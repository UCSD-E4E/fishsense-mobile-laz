//! Write a colored point cloud to a LAZ file using LAS Point Format 2
//! (XYZ + intensity + RGB). The e4e-point-cloud-viewer reads this format
//! via the same `las` crate, so what we write here is exactly what it
//! ingests on import.

use crate::unproject::ColoredPoint;
use anyhow::Result;
use las::{Builder, Color, Point, Transform, Vector, Version, Writer, point::Format};
use std::path::Path;

const SCALE_M: f64 = 0.001; // 1 mm quantization — well below ARKit depth noise.

pub fn write_laz(path: &Path, points: &[ColoredPoint]) -> Result<()> {
    if points.is_empty() {
        anyhow::bail!("no points to write — every depth pixel was skipped");
    }

    let (mut min_x, mut min_y, mut min_z) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y, mut max_z) =
        (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        min_z = min_z.min(p.z);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
        max_z = max_z.max(p.z);
    }

    // Center the offset so the quantized i32 (`(value - offset) / scale`)
    // stays well inside i32 range for any plausible single-capture cloud.
    let offset = Vector {
        x: f64::midpoint(min_x, max_x),
        y: f64::midpoint(min_y, max_y),
        z: f64::midpoint(min_z, max_z),
    };

    let mut builder = Builder::from(Version::new(1, 2));
    builder.point_format = Format::new(2)?;
    builder.transforms = Vector {
        x: Transform {
            scale: SCALE_M,
            offset: offset.x,
        },
        y: Transform {
            scale: SCALE_M,
            offset: offset.y,
        },
        z: Transform {
            scale: SCALE_M,
            offset: offset.z,
        },
    };

    let header = builder.into_header()?;
    let mut writer = Writer::from_path(path, header)?;

    for p in points {
        let point = Point {
            x: p.x,
            y: p.y,
            z: p.z,
            // LAS color channels are u16 per spec. 8-bit input gets stretched
            // to the full 16-bit range via `* 0x101` so 255 maps to 65535.
            color: Some(Color {
                red: u16::from(p.r) * 0x0101,
                green: u16::from(p.g) * 0x0101,
                blue: u16::from(p.b) * 0x0101,
            }),
            ..Default::default()
        };
        writer.write_point(point)?;
    }
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_xyz_and_color() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.laz");
        let points = vec![
            ColoredPoint {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                r: 10,
                g: 128,
                b: 255,
            },
            ColoredPoint {
                x: -0.5,
                y: 0.25,
                z: 0.0,
                r: 0,
                g: 255,
                b: 0,
            },
            ColoredPoint {
                x: 4.123,
                y: -2.456,
                z: 7.89,
                r: 50,
                g: 60,
                b: 70,
            },
        ];
        write_laz(&path, &points).unwrap();

        let mut reader = las::Reader::from_path(&path).unwrap();
        let read_back: Vec<Point> = reader.points().collect::<Result<_, _>>().unwrap();
        assert_eq!(read_back.len(), points.len());

        for (orig, got) in points.iter().zip(read_back.iter()) {
            assert!((orig.x - got.x).abs() < SCALE_M);
            assert!((orig.y - got.y).abs() < SCALE_M);
            assert!((orig.z - got.z).abs() < SCALE_M);
            let c = got.color.expect("color");
            assert_eq!(c.red, u16::from(orig.r) * 0x0101);
            assert_eq!(c.green, u16::from(orig.g) * 0x0101);
            assert_eq!(c.blue, u16::from(orig.b) * 0x0101);
        }
    }

    #[test]
    fn empty_cloud_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.laz");
        let err = write_laz(&path, &[]).unwrap_err();
        assert!(err.to_string().contains("no points"));
    }
}
