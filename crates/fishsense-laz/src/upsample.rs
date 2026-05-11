//! Joint bilateral upsampling of the sparse ARKit depth map, guided by
//! the high-resolution RGB image (Kopf et al., "Joint Bilateral
//! Upsampling", SIGGRAPH 2007).
//!
//! ARKit `sceneDepth` is a fixed 256x192 grid — ~49k points at most.
//! Upsampling toward the RGB resolution yields a much denser cloud, and
//! the viewer sizes each splat from local point density (kNN), so a
//! denser cloud automatically renders with smaller splats.
//!
//! This produces *interpolated* depth, not measured depth: flat regions
//! upsample cleanly; depth discontinuities can still bleed where the
//! colour guide fails to separate them; and output pixels with no valid
//! depth anchor in their search window are left NaN (dropped downstream)
//! rather than invented.

use rayon::prelude::*;

/// Search window radius, in low-resolution pixels, around each output
/// pixel. 3 → a 7x7 neighbourhood (up to 49 candidate anchors).
const WINDOW_RADIUS: i64 = 3;

/// Spatial Gaussian sigma, in low-resolution pixels.
const SIGMA_SPATIAL: f64 = 1.5;

/// Colour Gaussian sigma, on the Euclidean distance between two RGB
/// triples (each channel 0..=255). Smaller → sharper edge preservation.
const SIGMA_COLOR: f64 = 30.0;

pub struct UpsampledDepth {
    /// Row-major depth in metres; NaN where no anchor was found.
    pub depth: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Upsample `low_depth` (row-major, length `low_w * low_h`, in metres;
/// NaN entries are treated as "not an anchor" — the caller pre-filters
/// by confidence and finiteness) to a `low_w*factor × low_h*factor`
/// grid, using `guide_rgb` (row-major RGB8, length `guide_w*guide_h*3`)
/// as an edge guide. `factor` must be >= 1.
#[must_use]
pub fn joint_bilateral_upsample(
    low_depth: &[f32],
    low_w: u32,
    low_h: u32,
    guide_rgb: &[u8],
    guide_w: u32,
    guide_h: u32,
    factor: u32,
) -> UpsampledDepth {
    assert!(factor >= 1, "upsample factor must be >= 1");
    debug_assert_eq!(low_depth.len(), (low_w as usize) * (low_h as usize));
    debug_assert_eq!(
        guide_rgb.len(),
        (guide_w as usize) * (guide_h as usize) * 3
    );

    let lw = low_w as i64;
    let lh = low_h as i64;
    let gw = guide_w as i64;
    let gh = guide_h as i64;
    let f = f64::from(factor);
    let out_w = low_w * factor;
    let out_h = low_h * factor;

    // Map a low-res pixel centre to the nearest high-res guide pixel.
    let guide_px = |x: i64, x_n: i64, gx_n: i64| -> i64 {
        let g = ((x as f64 + 0.5) * gx_n as f64 / x_n as f64 - 0.5).round();
        g.clamp(0.0, (gx_n - 1) as f64) as i64
    };

    // Precompute the guide colour at each low-res anchor location once.
    let mut anchor_color = vec![[0.0_f64; 3]; (low_w as usize) * (low_h as usize)];
    for qy in 0..lh {
        let gy = guide_px(qy, lh, gh);
        for qx in 0..lw {
            let gx = guide_px(qx, lw, gw);
            let i = ((gy * gw + gx) * 3) as usize;
            anchor_color[(qy * lw + qx) as usize] = [
                f64::from(guide_rgb[i]),
                f64::from(guide_rgb[i + 1]),
                f64::from(guide_rgb[i + 2]),
            ];
        }
    }

    let two_ss2 = 2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL;
    let two_sc2 = 2.0 * SIGMA_COLOR * SIGMA_COLOR;

    let mut out = vec![f32::NAN; (out_w as usize) * (out_h as usize)];
    out.par_chunks_mut(out_w as usize)
        .enumerate()
        .for_each(|(oy_usize, row)| {
            let oy = oy_usize as i64;
            let ly = (oy as f64 + 0.5) / f - 0.5;
            let gy_t = guide_px(oy, out_h as i64, gh);
            for (ox_usize, slot) in row.iter_mut().enumerate() {
                let ox = ox_usize as i64;
                let lx = (ox as f64 + 0.5) / f - 0.5;
                let gx_t = guide_px(ox, out_w as i64, gw);
                let ti = ((gy_t * gw + gx_t) * 3) as usize;
                let ct = [
                    f64::from(guide_rgb[ti]),
                    f64::from(guide_rgb[ti + 1]),
                    f64::from(guide_rgb[ti + 2]),
                ];

                let cx = lx.round() as i64;
                let cy = ly.round() as i64;
                let mut wsum = 0.0_f64;
                let mut dsum = 0.0_f64;
                for qy in (cy - WINDOW_RADIUS)..=(cy + WINDOW_RADIUS) {
                    if qy < 0 || qy >= lh {
                        continue;
                    }
                    for qx in (cx - WINDOW_RADIUS)..=(cx + WINDOW_RADIUS) {
                        if qx < 0 || qx >= lw {
                            continue;
                        }
                        let s = low_depth[(qy * lw + qx) as usize];
                        if !s.is_finite() {
                            continue;
                        }
                        let dx = qx as f64 - lx;
                        let dy = qy as f64 - ly;
                        let ws = (-(dx * dx + dy * dy) / two_ss2).exp();
                        let cq = anchor_color[(qy * lw + qx) as usize];
                        let cd = (ct[0] - cq[0]).powi(2)
                            + (ct[1] - cq[1]).powi(2)
                            + (ct[2] - cq[2]).powi(2);
                        let wc = (-cd / two_sc2).exp();
                        let w = ws * wc;
                        wsum += w;
                        dsum += w * f64::from(s);
                    }
                }
                if wsum > 0.0 {
                    *slot = (dsum / wsum) as f32;
                }
            }
        });

    UpsampledDepth {
        depth: out,
        width: out_w,
        height: out_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_guide(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn flat_depth_upsamples_to_constant() {
        // 4x4 depth, all 2.0; uniform guide. Interpolating a constant
        // should give the constant everywhere.
        let low = vec![2.0_f32; 16];
        let guide = solid_guide(8, 8, [128, 128, 128]);
        let up = joint_bilateral_upsample(&low, 4, 4, &guide, 8, 8, 2);
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 8);
        for d in &up.depth {
            assert!((d - 2.0).abs() < 1e-4, "got {d}");
        }
    }

    #[test]
    fn colour_edge_blocks_cross_edge_bleeding() {
        // Two low-res depth samples: near (1.0) on the left, far (5.0)
        // on the right. Build two 8-px guides: one with a black|white
        // edge aligned to the depth boundary, one uniform grey.
        let low = vec![1.0_f32, 5.0_f32]; // 2x1
        let mut edge_guide = Vec::new();
        for x in 0..8u32 {
            let c = if x < 4 { [0, 0, 0] } else { [255, 255, 255] };
            edge_guide.extend_from_slice(&c);
        }
        let grey_guide = solid_guide(8, 1, [128, 128, 128]);

        let with_edge = joint_bilateral_upsample(&low, 2, 1, &edge_guide, 8, 1, 4);
        let without_edge = joint_bilateral_upsample(&low, 2, 1, &grey_guide, 8, 1, 4);

        // With the colour edge: a sharp step — left half ~1.0, right ~5.0.
        assert!(with_edge.depth[0] < 1.2, "left edge: {}", with_edge.depth[0]);
        assert!(with_edge.depth[3] < 1.5, "near boundary: {}", with_edge.depth[3]);
        assert!(with_edge.depth[4] > 4.5, "past boundary: {}", with_edge.depth[4]);
        assert!(with_edge.depth[7] > 4.8, "right edge: {}", with_edge.depth[7]);

        // Without it: a smooth gradient — the boundary pixel is a blend,
        // well away from either source value.
        assert!(
            without_edge.depth[3] > 2.0 && without_edge.depth[3] < 4.0,
            "uniform-guide boundary should blend: {}",
            without_edge.depth[3]
        );
    }

    #[test]
    fn no_anchors_yields_all_nan() {
        let low = vec![f32::NAN; 9];
        let guide = solid_guide(6, 6, [10, 20, 30]);
        let up = joint_bilateral_upsample(&low, 3, 3, &guide, 6, 6, 2);
        assert_eq!(up.depth.len(), 36);
        assert!(up.depth.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn factor_one_is_a_passthrough_grid() {
        let low = vec![1.0_f32, 2.0, 3.0, 4.0];
        let guide = solid_guide(2, 2, [50, 50, 50]);
        let up = joint_bilateral_upsample(&low, 2, 2, &guide, 2, 2, 1);
        assert_eq!((up.width, up.height), (2, 2));
        // With a uniform guide and factor 1 each output pixel still
        // averages its 7x7 (clamped) neighbourhood spatially; values
        // stay within the input range.
        for d in &up.depth {
            assert!(*d >= 1.0 && *d <= 4.0, "got {d}");
        }
    }
}
