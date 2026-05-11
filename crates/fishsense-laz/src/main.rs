use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use fishsense_laz::{
    ConvertOptions, Intrinsics, convert_one, db, default_output_name, decode::Confidence,
};
use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "fishsense-laz",
    about = "Convert fishsense-mobile SQLite captures (RGB + LiDAR depth) into LAZ point clouds.",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print one row per photo with id, timestamp, place, fish length,
    /// and whether depth + intrinsics are present (only those rows are
    /// convertible).
    List(ListArgs),
    /// Convert one or more photos to LAZ. Use `--ids 1,2,3` for a subset
    /// or `--all` for every convertible row.
    Convert(ConvertArgs),
}

#[derive(Debug, clap::Args)]
struct ListArgs {
    /// Path to the fishsense-mobile SQLite database. RGB JPEGs are
    /// expected in the same directory.
    #[arg(long)]
    db: PathBuf,

    /// Include rows whose RGB JPEG is not on disk. By default they are
    /// hidden, since they aren't convertible — useful when diagnosing
    /// a missing-file situation.
    #[arg(long)]
    show_missing_rgb: bool,
}

#[derive(Debug, clap::Args)]
struct ConvertArgs {
    /// Path to the fishsense-mobile SQLite database.
    #[arg(long)]
    db: PathBuf,

    /// Comma-separated photo ids to convert.
    #[arg(long, value_delimiter = ',', conflicts_with = "all")]
    ids: Vec<i64>,

    /// Convert every photo with a valid depth + intrinsics blob.
    #[arg(long, conflicts_with = "ids")]
    all: bool,

    /// Directory to write `.laz` files into. Created if missing.
    #[arg(long, default_value = ".")]
    out: PathBuf,

    /// Directory containing the JPEG files referenced by `rgb_path`.
    /// Defaults to the directory containing the SQLite database.
    #[arg(long)]
    rgb_root: Option<PathBuf>,

    /// Drop depth pixels whose ARKit confidence is below this threshold.
    #[arg(long, value_enum, default_value_t = ConfidenceArg::Medium)]
    min_confidence: ConfidenceArg,

    /// Fallback horizontal field-of-view in degrees, used only when the
    /// DB has no `intrinsics_bytes` column (pre-schema-v7 captures).
    /// iPhone 12-15 Pro wide camera ≈ 73°.
    #[arg(long, conflicts_with = "intrinsics")]
    hfov_degrees: Option<f64>,

    /// Fallback K matrix at RGB resolution, as `fx,fy,cx,cy`. Used only
    /// when the DB has no `intrinsics_bytes`. Known calibrations from
    /// the fishsense-mobile-oceans-2025 pipeline:
    ///   iPhone Pro: 1375.0719,1375.0719,968.6433,723.04926
    ///   iPad Pro:   1604.2147,1604.2147,956.5816,717.7617
    #[arg(long, value_delimiter = ',', conflicts_with = "hfov_degrees")]
    intrinsics: Option<Vec<f64>>,

    /// Upsample the depth map by this integer factor before
    /// unprojecting, using the RGB image as an edge guide (joint
    /// bilateral upsampling), for a much denser cloud. ARKit depth is
    /// 256x192; factor 8 gets you near RGB resolution (~3M points).
    /// The added depth is interpolated, not measured. 1 = off (default).
    #[arg(long, default_value_t = 1)]
    upsample: u32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConfidenceArg {
    Low,
    Medium,
    High,
}

impl From<ConfidenceArg> for Confidence {
    fn from(c: ConfidenceArg) -> Self {
        match c {
            ConfidenceArg::Low => Self::Low,
            ConfidenceArg::Medium => Self::Medium,
            ConfidenceArg::High => Self::High,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::List(args) => run_list(&args),
        Cmd::Convert(args) => run_convert(&args),
    }
}

fn run_list(args: &ListArgs) -> Result<()> {
    let conn = db::open(&args.db).with_context(|| format!("open db {}", args.db.display()))?;
    let entries = db::list_photos(&conn).context("list photos")?;
    if entries.is_empty() {
        println!("(no photos)");
        return Ok(());
    }

    let rgb_root = args
        .db
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    // Build a set of filenames in rgb_root once and check membership
    // against it — `Path::exists` per row is N round-trips on a network
    // mount, which can take many minutes for a few hundred captures.
    // fishsense-mobile stores rgb_path as a bare filename, so a flat
    // readdir is sufficient; absolute/nested paths fall back to stat.
    let rgb_dir_listing: HashSet<OsString> = std::fs::read_dir(&rgb_root)
        .with_context(|| format!("read rgb root {}", rgb_root.display()))?
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .collect();

    let rgb_present = |rgb_path: &str| -> bool {
        let p = Path::new(rgb_path);
        match p.file_name() {
            Some(name) if p.parent().map_or(true, |par| par.as_os_str().is_empty()) => {
                rgb_dir_listing.contains(name)
            }
            _ => db::resolve_rgb_path(rgb_path, &rgb_root).exists(),
        }
    };

    let (present, missing): (Vec<_>, Vec<_>) = entries
        .into_iter()
        .partition(|e| rgb_present(&e.rgb_path));

    let to_show: Vec<&db::PhotoListEntry> = if args.show_missing_rgb {
        present.iter().chain(missing.iter()).collect()
    } else {
        present.iter().collect()
    };

    println!(
        "{:>6}  {:<20}  {:>10}  {:>9}  {:>5}  {:<}",
        "id", "captured (utc)", "fish (cm)", "depth+K?", "rgb?", "place"
    );
    for e in to_show {
        let ts = format_iso8601(e.utc_unix_timestamp);
        let len = e
            .fish_length_m
            .map(|m| format!("{:>10.1}", m * 100.0))
            .unwrap_or_else(|| "         -".to_string());
        let convertible = if e.has_depth && e.has_intrinsics {
            "    yes"
        } else if e.has_depth {
            "  no(K)"
        } else {
            " no(d) "
        };
        let rgb_marker = if rgb_present(&e.rgb_path) {
            "  yes"
        } else {
            "   no"
        };
        let place = e.place_name.as_deref().unwrap_or("");
        println!(
            "{:>6}  {:<20}  {}  {:>9}  {}  {}",
            e.id, ts, len, convertible, rgb_marker, place
        );
    }

    if !args.show_missing_rgb && !missing.is_empty() {
        eprintln!(
            "({} hidden — RGB file missing under {}; pass --show-missing-rgb to include)",
            missing.len(),
            rgb_root.display()
        );
    }
    Ok(())
}

fn run_convert(args: &ConvertArgs) -> Result<()> {
    let conn = db::open(&args.db).with_context(|| format!("open db {}", args.db.display()))?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create out dir {}", args.out.display()))?;

    let rgb_root = args
        .rgb_root
        .clone()
        .or_else(|| args.db.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    let fallback_intrinsics = match args.intrinsics.as_deref() {
        Some([fx, fy, cx, cy]) => Some(Intrinsics {
            fx: *fx,
            fy: *fy,
            cx: *cx,
            cy: *cy,
        }),
        Some(_) => bail!("--intrinsics expects exactly 4 values: fx,fy,cx,cy"),
        None => None,
    };

    if !(1..=16).contains(&args.upsample) {
        bail!("--upsample must be between 1 and 16 (got {})", args.upsample);
    }

    let opts = ConvertOptions {
        min_confidence: args.min_confidence.into(),
        rgb_root,
        fallback_intrinsics,
        fallback_hfov_degrees: args.hfov_degrees,
        upsample_factor: args.upsample,
    };

    let ids = if args.all {
        db::list_photos(&conn)
            .context("list photos")?
            .into_iter()
            .filter(|e| e.has_depth && e.has_intrinsics)
            .map(|e| e.id)
            .collect()
    } else if args.ids.is_empty() {
        bail!("must specify --ids 1,2,3 or --all");
    } else {
        args.ids.clone()
    };

    if ids.is_empty() {
        eprintln!("nothing to convert");
        return Ok(());
    }

    let mut ok = 0usize;
    let mut failed = 0usize;
    for id in ids {
        // Look up the timestamp for the filename. fetch_photo loads the
        // BLOBs too, but it's the cleanest place to get the timestamp;
        // the row is small enough that the redundant read in convert_one
        // doesn't matter at human-driven batch sizes.
        let timestamp = match db::fetch_photo(&conn, id) {
            Ok(row) => row.utc_unix_timestamp,
            Err(e) => {
                eprintln!("photo {id}: {e}");
                failed += 1;
                continue;
            }
        };
        let out_path = args.out.join(default_output_name(id, timestamp));
        match convert_one(&conn, id, &out_path, &opts) {
            Ok(n) => {
                println!("photo {id}: wrote {n} points to {}", out_path.display());
                ok += 1;
            }
            Err(e) => {
                eprintln!("photo {id}: {e:#}");
                failed += 1;
            }
        }
    }

    println!("done — {ok} ok, {failed} failed");
    if ok == 0 && failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn format_iso8601(unix_seconds: i64) -> String {
    // The Dart `utc_unix_timestamp` is `millisecondsSinceEpoch` in some
    // places and seconds in others; the column name says "unix" so we
    // treat values that look millisecond-scale (> year 3000 in seconds)
    // as ms and divide.
    let secs = if unix_seconds > 32_503_680_000 {
        unix_seconds / 1000
    } else {
        unix_seconds
    };
    // Render UTC manually (no chrono dep): days since epoch + HMS.
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, m, s) = (tod / 3600, (tod / 60) % 60, tod % 60);
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}")
}

/// Converts days since 1970-01-01 to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Howard Hinnant's date algorithm: shift epoch from 1970-01-01 to
    // 0000-03-01 so leap-day handling falls out of integer division.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
