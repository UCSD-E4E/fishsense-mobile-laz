//! Decode the binary BLOBs stored by fishsense-mobile.
//!
//! Three blob shapes the SQLite schema (v7) emits:
//! - `depth_bytes`: row-major little-endian f32, units = meters.
//! - `confidence_bytes`: row-major u8, ARKit `ARConfidenceLevel` (0=low, 1=med, 2=high).
//! - `intrinsics_bytes`: 9 little-endian f64, row-major K (`fx,0,cx, 0,fy,cy, 0,0,1`).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("depth blob length {len} does not match {width}x{height}*4")]
    DepthSizeMismatch {
        len: usize,
        width: usize,
        height: usize,
    },
    #[error("confidence blob length {len} does not match {width}x{height}")]
    ConfidenceSizeMismatch {
        len: usize,
        width: usize,
        height: usize,
    },
    #[error("intrinsics blob length {0} != 72 (9 × f64)")]
    IntrinsicsSize(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl Confidence {
    fn from_byte(b: u8) -> Self {
        match b {
            2 => Self::High,
            1 => Self::Medium,
            _ => Self::Low,
        }
    }

    #[must_use]
    pub fn at_least(self, threshold: Self) -> bool {
        (self as u8) >= (threshold as u8)
    }
}

/// 3x3 row-major intrinsic matrix at the **RGB image** resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Intrinsics {
    /// Scale to a different image resolution (e.g. depth resolution).
    #[must_use]
    pub fn scaled(&self, from_w: u32, from_h: u32, to_w: u32, to_h: u32) -> Self {
        let sx = f64::from(to_w) / f64::from(from_w);
        let sy = f64::from(to_h) / f64::from(from_h);
        Self {
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
        }
    }
}

pub fn decode_depth(bytes: &[u8], width: u32, height: u32) -> Result<Vec<f32>, DecodeError> {
    let expected = (width as usize) * (height as usize) * 4;
    if bytes.len() != expected {
        return Err(DecodeError::DepthSizeMismatch {
            len: bytes.len(),
            width: width as usize,
            height: height as usize,
        });
    }
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn decode_confidence(
    bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<Confidence>, DecodeError> {
    let expected = (width as usize) * (height as usize);
    if bytes.len() != expected {
        return Err(DecodeError::ConfidenceSizeMismatch {
            len: bytes.len(),
            width: width as usize,
            height: height as usize,
        });
    }
    Ok(bytes.iter().map(|&b| Confidence::from_byte(b)).collect())
}

pub fn decode_intrinsics(bytes: &[u8]) -> Result<Intrinsics, DecodeError> {
    if bytes.len() != 72 {
        return Err(DecodeError::IntrinsicsSize(bytes.len()));
    }
    let mut k = [0.0f64; 9];
    for (i, slot) in k.iter_mut().enumerate() {
        let s = i * 8;
        *slot = f64::from_le_bytes(bytes[s..s + 8].try_into().unwrap());
    }
    Ok(Intrinsics {
        fx: k[0],
        cx: k[2],
        fy: k[4],
        cy: k[5],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_threshold() {
        assert!(Confidence::High.at_least(Confidence::Medium));
        assert!(Confidence::Medium.at_least(Confidence::Medium));
        assert!(!Confidence::Low.at_least(Confidence::Medium));
    }

    #[test]
    fn depth_roundtrip() {
        let pixels = [0.5_f32, 1.5, f32::NAN, 3.25];
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        let decoded = decode_depth(&bytes, 2, 2).unwrap();
        assert_eq!(decoded[0], 0.5);
        assert_eq!(decoded[1], 1.5);
        assert!(decoded[2].is_nan());
        assert_eq!(decoded[3], 3.25);
    }

    #[test]
    fn depth_size_mismatch() {
        let bytes = vec![0u8; 12];
        assert!(matches!(
            decode_depth(&bytes, 2, 2),
            Err(DecodeError::DepthSizeMismatch { .. })
        ));
    }

    #[test]
    fn confidence_decodes_levels() {
        let bytes = [0u8, 1, 2, 7];
        let conf = decode_confidence(&bytes, 4, 1).unwrap();
        assert_eq!(conf[0], Confidence::Low);
        assert_eq!(conf[1], Confidence::Medium);
        assert_eq!(conf[2], Confidence::High);
        // Unknown values clamp to Low — keeps a malformed byte from being treated as High.
        assert_eq!(conf[3], Confidence::Low);
    }

    #[test]
    fn intrinsics_roundtrip() {
        let k: [f64; 9] = [1500.0, 0.0, 960.0, 0.0, 1500.0, 720.0, 0.0, 0.0, 1.0];
        let bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
        let intr = decode_intrinsics(&bytes).unwrap();
        assert_eq!(intr.fx, 1500.0);
        assert_eq!(intr.fy, 1500.0);
        assert_eq!(intr.cx, 960.0);
        assert_eq!(intr.cy, 720.0);
    }

    #[test]
    fn intrinsics_scale_to_depth_resolution() {
        let intr = Intrinsics {
            fx: 1500.0,
            fy: 1500.0,
            cx: 960.0,
            cy: 720.0,
        };
        let scaled = intr.scaled(1920, 1440, 256, 192);
        assert!((scaled.fx - 200.0).abs() < 1e-9);
        assert!((scaled.fy - 200.0).abs() < 1e-9);
        assert!((scaled.cx - 128.0).abs() < 1e-9);
        assert!((scaled.cy - 96.0).abs() < 1e-9);
    }
}
