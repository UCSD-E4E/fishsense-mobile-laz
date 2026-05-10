//! Back-project a depth + RGB pair through a 3x3 intrinsic matrix into
//! a colored point cloud in the camera's reference frame.
//!
//! Sign convention follows ARKit (sensor-native landscape):
//!   +X right, +Y down, +Z forward (into scene). The viewer auto-normalizes
//!   on import, so the absolute frame doesn't have to match anything else.

use crate::decode::{Confidence, Intrinsics};

#[derive(Debug, Clone, Copy)]
pub struct ColoredPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct UnprojectParams {
    /// Captured RGB image dimensions (intrinsics are at this resolution).
    pub rgb_width: u32,
    pub rgb_height: u32,
    /// Depth-map dimensions.
    pub depth_width: u32,
    pub depth_height: u32,
    /// Drop depth pixels whose ARKit confidence is below this threshold.
    pub min_confidence: Confidence,
}

/// `rgb_pixels` is row-major RGB8 of length `rgb_width * rgb_height * 3`.
/// `depth` is row-major f32 in meters of length `depth_width * depth_height`.
/// `confidence`, when supplied, is row-major u8 of the same length as `depth`;
/// when absent every depth sample is treated as "high".
#[must_use]
pub fn unproject(
    rgb_pixels: &[u8],
    depth: &[f32],
    confidence: Option<&[Confidence]>,
    intrinsics_at_rgb: &Intrinsics,
    params: &UnprojectParams,
) -> Vec<ColoredPoint> {
    let dw = params.depth_width as usize;
    let dh = params.depth_height as usize;
    let rw = params.rgb_width as usize;
    let rh = params.rgb_height as usize;

    debug_assert_eq!(depth.len(), dw * dh);
    debug_assert_eq!(rgb_pixels.len(), rw * rh * 3);
    if let Some(c) = confidence {
        debug_assert_eq!(c.len(), dw * dh);
    }

    let intr_d = intrinsics_at_rgb.scaled(
        params.rgb_width,
        params.rgb_height,
        params.depth_width,
        params.depth_height,
    );

    // RGB sampling step: nearest-neighbor lookup. Depth resolution is the
    // limiting factor (256x192 vs 1920x1440), so bilinear sampling adds
    // cost without adding fidelity.
    let sx = rw as f64 / dw as f64;
    let sy = rh as f64 / dh as f64;

    let mut out = Vec::with_capacity(dw * dh);
    for v in 0..dh {
        for u in 0..dw {
            let idx = v * dw + u;
            let z = depth[idx];
            if !z.is_finite() || z <= 0.0 {
                continue;
            }
            if let Some(conf) = confidence {
                if !conf[idx].at_least(params.min_confidence) {
                    continue;
                }
            }

            // Pixel-center convention: u + 0.5, v + 0.5.
            let uc = u as f64 + 0.5;
            let vc = v as f64 + 0.5;
            let zd = z as f64;
            let x = (uc - intr_d.cx) * zd / intr_d.fx;
            let y = (vc - intr_d.cy) * zd / intr_d.fy;

            // RGB sample location for this depth pixel.
            let ru = (uc * sx) as usize;
            let rv = (vc * sy) as usize;
            let ru = ru.min(rw - 1);
            let rv = rv.min(rh - 1);
            let pi = (rv * rw + ru) * 3;

            out.push(ColoredPoint {
                x,
                y,
                z: zd,
                r: rgb_pixels[pi],
                g: rgb_pixels[pi + 1],
                b: rgb_pixels[pi + 2],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn flat_plane_at_constant_depth_unprojects_to_a_grid() {
        // 4x4 depth, all z = 2.0. Use intrinsics at the same resolution
        // (rgb = depth) so scaling is a no-op and the math is easy to check.
        let dw = 4u32;
        let dh = 4u32;
        let intr = Intrinsics {
            fx: 4.0,
            fy: 4.0,
            cx: 2.0,
            cy: 2.0,
        };
        let depth = vec![2.0_f32; (dw * dh) as usize];
        let rgb = solid_rgb(dw, dh, [128, 64, 32]);
        let params = UnprojectParams {
            rgb_width: dw,
            rgb_height: dh,
            depth_width: dw,
            depth_height: dh,
            min_confidence: Confidence::High,
        };
        let pts = unproject(&rgb, &depth, None, &intr, &params);
        assert_eq!(pts.len(), 16);

        // Pixel (u=0, v=0): center (0.5, 0.5).
        // x = (0.5 - 2) * 2 / 4 = -0.75; y = -0.75; z = 2.0
        let p0 = pts[0];
        assert!((p0.x - -0.75).abs() < 1e-9);
        assert!((p0.y - -0.75).abs() < 1e-9);
        assert!((p0.z - 2.0).abs() < 1e-9);
        assert_eq!((p0.r, p0.g, p0.b), (128, 64, 32));

        // Pixel (u=3, v=3): center (3.5, 3.5).
        // x = (3.5 - 2) * 2 / 4 = 0.75; y = 0.75
        let p_last = pts.last().unwrap();
        assert!((p_last.x - 0.75).abs() < 1e-9);
        assert!((p_last.y - 0.75).abs() < 1e-9);
    }

    #[test]
    fn nan_and_nonpositive_depth_are_skipped() {
        let dw = 2u32;
        let dh = 2u32;
        let depth = vec![1.0_f32, f32::NAN, -1.0, 0.0];
        let rgb = solid_rgb(dw, dh, [255, 255, 255]);
        let intr = Intrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 1.0,
            cy: 1.0,
        };
        let params = UnprojectParams {
            rgb_width: dw,
            rgb_height: dh,
            depth_width: dw,
            depth_height: dh,
            min_confidence: Confidence::High,
        };
        let pts = unproject(&rgb, &depth, None, &intr, &params);
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn confidence_threshold_filters() {
        let dw = 2u32;
        let dh = 2u32;
        let depth = vec![1.0_f32; 4];
        let rgb = solid_rgb(dw, dh, [10, 20, 30]);
        let conf = vec![
            Confidence::Low,
            Confidence::Medium,
            Confidence::High,
            Confidence::Medium,
        ];
        let intr = Intrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 1.0,
            cy: 1.0,
        };
        let params = UnprojectParams {
            rgb_width: dw,
            rgb_height: dh,
            depth_width: dw,
            depth_height: dh,
            min_confidence: Confidence::Medium,
        };
        let pts = unproject(&rgb, &depth, Some(&conf), &intr, &params);
        // Three of the four samples are >= Medium.
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn rgb_at_higher_resolution_samples_correctly() {
        // Depth 2x2, RGB 4x4 with a checkerboard. The unprojection should
        // pick up four different colors — one per depth pixel — by mapping
        // each depth pixel to the corresponding 2x2 RGB block.
        let dw = 2u32;
        let dh = 2u32;
        let rw = 4u32;
        let rh = 4u32;
        // RGB rows: TL red, TR green, BL blue, BR white in 2x2 blocks.
        let mut rgb = vec![0u8; (rw * rh * 3) as usize];
        for ry in 0..rh {
            for rx in 0..rw {
                let block_x = rx / 2;
                let block_y = ry / 2;
                let color = match (block_x, block_y) {
                    (0, 0) => [255, 0, 0],
                    (1, 0) => [0, 255, 0],
                    (0, 1) => [0, 0, 255],
                    _ => [255, 255, 255],
                };
                let i = ((ry * rw + rx) * 3) as usize;
                rgb[i..i + 3].copy_from_slice(&color);
            }
        }
        let depth = vec![1.0_f32; 4];
        let intr = Intrinsics {
            fx: 4.0,
            fy: 4.0,
            cx: 2.0,
            cy: 2.0,
        };
        let params = UnprojectParams {
            rgb_width: rw,
            rgb_height: rh,
            depth_width: dw,
            depth_height: dh,
            min_confidence: Confidence::High,
        };
        let pts = unproject(&rgb, &depth, None, &intr, &params);
        assert_eq!(pts.len(), 4);
        assert_eq!((pts[0].r, pts[0].g, pts[0].b), (255, 0, 0));
        assert_eq!((pts[1].r, pts[1].g, pts[1].b), (0, 255, 0));
        assert_eq!((pts[2].r, pts[2].g, pts[2].b), (0, 0, 255));
        assert_eq!((pts[3].r, pts[3].g, pts[3].b), (255, 255, 255));
    }
}
