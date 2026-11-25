//! `cargo xtask perf` and `cargo xtask profile-diff`: what a run's log and
//! its `VICTORY_PROFILE` file say, and how two runs differ.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

/// The fields of the frame-rate line the host logs every five seconds
/// (`victory-gpu` host.rs, `Meter::frame`): name, and whether more is better
/// (`None`: neither, as more draws may be another scene).
const FIELDS: [(&str, Option<bool>); 16] = [
    ("fps", Some(true)),
    ("ms/frame", Some(false)),
    ("preparing ms", Some(false)),
    ("copies ms", Some(false)),
    ("copies MB", Some(false)),
    ("textures ms", Some(false)),
    ("textures MB", Some(false)),
    ("bindless plans ms", Some(false)),
    ("GPU ms", Some(false)),
    ("copy-back ms", Some(false)),
    ("copy-back MB", Some(false)),
    ("bindless records ms", Some(false)),
    ("draws/frame", None),
    ("dropped/frame", Some(false)),
    ("being made", Some(false)),
    ("indirect dropped", Some(false)),
];

fn field(name: &str) -> usize {
    FIELDS.iter().position(|f| f.0 == name).expect("a field of FIELDS")
}

/// Changes smaller than this (percent) are not marked better or worse.
const NOISE: f64 = 2.0;

/// One frame-rate line: its numbers in [`FIELDS`] order, and the
/// diagnostics lines of the same tick. The host logs the GPU times just
/// before the frame-rate line and the drop reasons just after it.
#[derive(Default)]
struct Sample {
    values: [f64; FIELDS.len()],
    /// Draws dropped without a host pipeline per frame, by reason (only
    /// reasons that dropped any are logged).
    drops: Vec<(String, f64)>,
    /// GPU ms per frame, by kind of command (with `VICTORY_GPU_TIMING`).
    gpu: Vec<(String, f64)>,
}

/// What one log holds.
#[derive(Default)]
struct Run {
    samples: Vec<Sample>,
    /// Frame-rate lines read, before a window cut them.
    read: usize,
    /// Lines that look like frame-rate lines but did not read: a format
    /// this tool does not know.
    unread: usize,
    /// Lines with warnings and errors worth counting, by kind.
    events: HashMap<&'static str, u32>,
    /// Slow list commands, by command (first word).
    slow_ops: HashMap<String, u32>,
}

/// Lines counted as events: what to look for and what to call it.
const EVENTS: [(&str, &str); 12] = [
    ("slow host submission", "slow host submissions"),
    ("slow list command", "slow list commands"),
    ("host pipeline refused", "host pipelines refused"),
    ("host mesh pipeline refused", "host mesh pipelines refused"),
    ("compiler crashed", "driver compiler crashes"),
    ("does not end a pipeline", "driver compiles stopped"),
    ("device removed", "device removed"),
    ("DXGI_ERROR", "DXGI errors"),
    ("no free dynamic descriptor slots", "dynamic descriptor slots out"),
    ("does not translate", "shaders not translated"),
    ("tiles given back", "tile sweeps"),
    ("unhandled exception", "unhandled exceptions"),
];

/// Which frame-rate lines of a log to keep: the first `skip` dropped (menus,
/// loading), then the `last` ones if given.
#[derive(Clone, Copy, Default)]
pub struct Window {
    pub skip: usize,
    pub last: Option<usize>,
}

pub fn run(logs: &[std::path::PathBuf], window: Window) -> Result<()> {
    let runs = logs
        .iter()
        .map(|p| read(p).map(|t| cut(parse_log(&t), window)))
        .collect::<Result<Vec<_>>>()?;
    let names: Vec<String> = logs.iter().map(|p| short(p)).collect();
    print!("{}", report(&names, &runs));
    Ok(())
}

pub fn profile_diff(a: &Path, b: &Path) -> Result<()> {
    let (pa, pb) = (parse_profile(&read(a)?), parse_profile(&read(b)?));
    print!("{}", diff_profiles(&pa, &pb, &short(a), &short(b)));
    Ok(())
}

fn read(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(text(&bytes))
}

/// A file's text: UTF-16 with its byte-order mark (Windows PowerShell 5.1's
/// `2>` writes that), else UTF-8.
fn text(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        let units: Vec<u16> = rest.as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(bytes).into_owned()
}

fn short(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// The line without terminal colour codes (a log captured from a terminal
/// keeps them).
fn plain(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The number written right before `pat`.
fn before(s: &str, pat: &str) -> Option<f64> {
    let head = &s[..s.find(pat)?];
    let word = head.rsplit(|c: char| c.is_whitespace() || c == '(' || c == '[').next()?;
    word.trim_end_matches([',', ';']).parse().ok()
}

/// The number written right after `pat`.
fn after(s: &str, pat: &str) -> Option<f64> {
    let tail = s[s.find(pat)? + pat.len()..].trim_start();
    let word = tail.split(|c: char| c.is_whitespace() || c == '/').next()?;
    word.trim_end_matches([',', ';', ')']).parse().ok()
}

/// Whether `line` is a frame-rate line, read or not.
fn is_frame_line(line: &str) -> bool {
    line.contains(" ms/frame (") && line.contains(" preparing")
}

fn parse_values(line: &str) -> Option<[f64; FIELDS.len()]> {
    let s = &line[line.find("] ").map_or(0, |i| i + 2)..];
    Some([
        before(s, " fps,")?,
        before(s, " ms/frame")?,
        before(s, " preparing")?,
        after(s, "copies ")?,
        before(s, " MB, textures")?,
        after(s, "textures ")?,
        before(s, " MB, bindless plans")?,
        after(s, "bindless plans ")?,
        before(s, " on the GPU")?,
        before(s, " copying back")?,
        after(s, "copying back ")?,
        after(s, "bindless records ")?,
        before(s, " draws/frame")?,
        after(s, "dropped ")?,
        before(s, " being made")?,
        before(s, " indirect")?,
    ])
}

fn parse_log(text: &str) -> Run {
    let mut run = Run::default();
    // GPU times logged before the next frame-rate line, and whether drop
    // reasons after it go with the sample read last (not after a line this
    // tool could not read).
    let mut gpu = Vec::new();
    let mut attach = false;
    for line in text.lines() {
        let line = plain(line);
        if is_frame_line(&line) {
            attach = match parse_values(&line) {
                Some(values) => {
                    run.samples.push(Sample {
                        values,
                        drops: Vec::new(),
                        gpu: std::mem::take(&mut gpu),
                    });
                    true
                }
                None => {
                    run.unread += 1;
                    gpu.clear();
                    false
                }
            };
            continue;
        }
        if let Some((head, why)) = line.split_once("/frame without a host pipeline: ")
            && let Some(at) = head.rfind("dropped ")
            && let Some(rate) = after(&head[at..], "dropped ")
        {
            if attach && let Some(s) = run.samples.last_mut() {
                s.drops.push((why.trim().to_owned(), rate));
            }
            continue;
        }
        if let Some(at) = line.find("GPU time ")
            && let Some((head, key)) = line[at..].split_once(" x: ")
            && let Some(ms) = after(head, "GPU time ")
        {
            gpu.push((key.trim().to_owned(), ms));
            continue;
        }
        for (pat, name) in EVENTS {
            if line.contains(pat) {
                *run.events.entry(name).or_default() += 1;
            }
        }
        if let Some((_, op)) = line.split_once("slow list command, ")
            && let Some((_, op)) = op.split_once(": ")
        {
            let op = op.split_whitespace().next().unwrap_or("").to_owned();
            *run.slow_ops.entry(op).or_default() += 1;
        }
    }
    run
}

fn cut(mut run: Run, w: Window) -> Run {
    run.read = run.samples.len();
    run.samples.drain(..w.skip.min(run.samples.len()));
    if let Some(last) = w.last {
        let n = run.samples.len().saturating_sub(last);
        run.samples.drain(..n);
    }
    run
}

/// The `q` quantile of `v` (sorted in place), by nearest rank.
fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Samples once no pipeline is being made (the first launch makes them for
/// minutes, at a fraction of the rate), else all; and whether they are those.
fn steady(run: &Run) -> (Vec<&Sample>, bool) {
    let made = field("being made");
    let s: Vec<&Sample> = run.samples.iter().filter(|s| s.values[made] == 0.0).collect();
    if s.is_empty() { (run.samples.iter().collect(), false) } else { (s, true) }
}

fn column(samples: &[&Sample], i: usize) -> Vec<f64> {
    samples.iter().map(|s| s.values[i]).collect()
}

/// Per-frame averages over `samples` of keyed rates: a key missing from a
/// sample counts 0 there.
fn averages<'a>(samples: &[&'a Sample], of: impl Fn(&'a Sample) -> &'a [(String, f64)]) -> Vec<(f64, &'a str)> {
    let mut sum: HashMap<&str, f64> = HashMap::new();
    for s in samples {
        for (k, v) in of(s) {
            *sum.entry(k).or_default() += v;
        }
    }
    let mut out: Vec<(f64, &str)> = sum.into_iter().map(|(k, v)| (v / samples.len() as f64, k)).collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    out
}

fn report(names: &[String], runs: &[Run]) -> String {
    let mut out = String::new();
    for (name, run) in names.iter().zip(runs) {
        let (s, is_steady) = steady(run);
        let _ = writeln!(
            out,
            "== {name}: {} frame-rate lines kept, {} of them {}",
            run.samples.len(),
            s.len(),
            if is_steady { "with no pipeline being made (steady)" } else { "used (no steady part)" }
        );
        if run.unread > 0 {
            let _ = writeln!(
                out,
                "   {} frame-rate lines in a format this tool does not read: update xtask/src/perf.rs",
                run.unread
            );
        }
        if s.is_empty() {
            let why = if run.read > 0 {
                "the window (--skip, --last) leaves none of its lines"
            } else {
                "no frame-rate lines: was the log captured (2> file)?"
            };
            let _ = writeln!(out, "   {why}");
            continue;
        }
        // Each line averages about the second before it, one second in five.
        let _ = writeln!(out, "   {:<22}{:>9}{:>9}{:>9}{:>10}", "", "median", "p10", "p90", "worst 1s");
        for (i, (f, more_better)) in FIELDS.iter().enumerate() {
            let mut v = column(&s, i);
            let (p10, med, p90) = (quantile(&mut v, 0.1), quantile(&mut v, 0.5), quantile(&mut v, 0.9));
            let worst = match more_better {
                Some(true) => format!("{:.1}", v[0]),
                Some(false) => format!("{:.1}", v[v.len() - 1]),
                None => String::new(),
            };
            let _ = writeln!(out, "   {f:<22}{med:>9.1}{p10:>9.1}{p90:>9.1}{worst:>10}");
        }
        // Each line's parts as shares of its own frame, then their medians;
        // records run while draws are recorded, on the title's threads.
        let share = |part: &str| {
            let (p, ms) = (field(part), field("ms/frame"));
            let mut v: Vec<f64> =
                s.iter().filter(|x| x.values[ms] > 0.0).map(|x| 100.0 * x.values[p] / x.values[ms]).collect();
            quantile(&mut v, 0.5)
        };
        let _ = writeln!(
            out,
            "   median share of a frame: preparing {:.0}%, GPU {:.0}%, copy-back {:.0}%, bindless records {:.0}%",
            share("preparing ms"),
            share("GPU ms"),
            share("copy-back ms"),
            share("bindless records ms"),
        );
        let drops = averages(&s, |x| &x.drops);
        if !drops.is_empty() {
            let _ = writeln!(out, "   draws dropped without a host pipeline, per frame, by reason:");
            for (rate, why) in drops.into_iter().take(12) {
                let _ = writeln!(out, "   {rate:>9.1}  {why}");
            }
        }
        let timed: Vec<&Sample> = s.iter().copied().filter(|x| !x.gpu.is_empty()).collect();
        let gpu = averages(&timed, |x| &x.gpu);
        if !gpu.is_empty() {
            let _ = writeln!(out, "   GPU ms per frame by command ({} timed lines):", timed.len());
            for (ms, key) in gpu.into_iter().take(15) {
                let _ = writeln!(out, "   {ms:>9.2}  {key}");
            }
        }
        if !run.events.is_empty() {
            let mut e: Vec<_> = run.events.iter().collect();
            e.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let text: Vec<String> = e.iter().map(|(k, n)| format!("{k} {n}")).collect();
            let _ = writeln!(out, "   lines of the whole log: {}", text.join(", "));
        }
        if !run.slow_ops.is_empty() {
            let mut o: Vec<_> = run.slow_ops.iter().collect();
            o.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let _ = writeln!(out, "   slow list commands, by kind:");
            for (op, n) in o.into_iter().take(8) {
                let _ = writeln!(out, "   {n:>9}  {op}");
            }
        }
    }
    if let [a, b] = runs {
        let ((sa, ta), (sb, tb)) = (steady(a), steady(b));
        if sa.is_empty() || sb.is_empty() {
        } else if !(ta && tb) {
            let which = match (ta, tb) {
                (true, _) => names[1].clone(),
                (_, true) => names[0].clone(),
                _ => format!("{} and {}", names[0], names[1]),
            };
            let verb = if ta || tb { "has" } else { "have" };
            let _ = writeln!(
                out,
                "== no comparison: {which} {verb} no steady part (pipelines were still being made throughout)"
            );
        } else {
            let _ = writeln!(out, "== {} -> {} (steady medians)", names[0], names[1]);
            let _ = writeln!(out, "   {:<22}{:>9}{:>9}{:>9}", "", "before", "after", "change");
            for (i, (f, more_better)) in FIELDS.iter().enumerate() {
                let (x, y) = (quantile(&mut column(&sa, i), 0.5), quantile(&mut column(&sb, i), 0.5));
                let (pct, mark) = if x == 0.0 {
                    (if y == 0.0 { "0%".to_owned() } else { "new".to_owned() }, "")
                } else {
                    let pct = 100.0 * (y - x) / x;
                    let mark = match more_better {
                        _ if pct.abs() < NOISE => "",
                        Some(m) if (y > x) == *m => "better",
                        Some(_) => "worse",
                        None => "",
                    };
                    (format!("{pct:.0}%"), mark)
                };
                let _ = writeln!(out, "   {f:<22}{x:>9.1}{y:>9.1}{pct:>9} {mark}");
            }
        }
    }
    out
}

/// A `VICTORY_PROFILE` file: samples, and each section's shares by function.
#[derive(Default)]
struct Profile {
    samples: u64,
    sections: Vec<(String, HashMap<String, f64>)>,
}

/// A function's name without the source line `profile.rs` appends
/// (`module!function:line`), which moves whenever the code above it does.
fn function(name: &str) -> &str {
    match name.rsplit_once(':') {
        Some((head, line)) if !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => name,
    }
}

fn parse_profile(text: &str) -> Profile {
    let mut p = Profile::default();
    for line in text.lines() {
        if let Some(n) = line.strip_suffix(" samples") {
            p.samples = n.trim().parse().unwrap_or(0);
        } else if let Some(title) = line.strip_prefix("-- ") {
            p.sections.push((title.trim().to_owned(), HashMap::new()));
        } else if let Some((share, name)) = line.trim_start().split_once("% ")
            && let Ok(share) = share.parse::<f64>()
            && let Some((_, s)) = p.sections.last_mut()
        {
            *s.entry(function(name.trim()).to_owned()).or_default() += share;
        }
    }
    p
}

fn diff_profiles(a: &Profile, b: &Profile, an: &str, bn: &str) -> String {
    let mut out = format!("== {an} ({} samples) -> {bn} ({} samples)\n", a.samples, b.samples);
    // Function shares only: whole stacks are too long to line up.
    for (title, sa) in a.sections.iter().filter(|s| s.0 == "in itself" || s.0 == "with what it calls") {
        let Some((_, sb)) = b.sections.iter().find(|s| &s.0 == title) else { continue };
        let mut names: Vec<&String> = sa.keys().chain(sb.keys()).collect();
        names.sort();
        names.dedup();
        let mut rows: Vec<(f64, f64, &String)> = names
            .into_iter()
            .map(|n| (sa.get(n).copied().unwrap_or(0.0), sb.get(n).copied().unwrap_or(0.0), n))
            .collect();
        rows.sort_by(|x, y| (y.1 - y.0).abs().total_cmp(&(x.1 - x.0).abs()).then(x.2.cmp(y.2)));
        let _ = writeln!(
            out,
            "-- {title}: largest changes in share (a function past a file's list counts 0)"
        );
        for (x, y, name) in rows.into_iter().take(25) {
            let _ = writeln!(out, "{x:6.1}% {y:6.1}% {:+6.1}  {name}", y - x);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = "2026-11-25T14:02:11.120Z  INFO victory_gpu::host: [0 0] 2.2 fps, 457.6 ms/frame \
        (253.2 preparing: copies 59.0 ms 206.6 MB, textures 11.7 ms 8.4 MB, bindless plans 82.1 ms; \
        121.0 on the GPU; 78.2 copying back 31.0 MB; bindless records 3.8 ms), 2729 draws/frame, \
        dropped 425 without a host pipeline (0 being made) and 1 indirect";

    #[test]
    fn the_frame_rate_line_is_read_whole() {
        let v = parse_values(LINE).unwrap();
        assert_eq!(
            v,
            [2.2, 457.6, 253.2, 59.0, 206.6, 11.7, 8.4, 82.1, 121.0, 78.2, 31.0, 3.8, 2729.0, 425.0, 0.0, 1.0]
        );
        let coloured = format!("\u{1b}[2m2026\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m {}", &LINE[25..]);
        assert_eq!(parse_values(&plain(&coloured)), Some(v));
        // A field renamed: not read, and counted as such.
        let run = parse_log(&LINE.replace(" on the GPU", " on the card"));
        assert_eq!((run.samples.len(), run.unread), (0, 1));
    }

    #[test]
    fn utf16_logs_read_as_text() {
        let mut bytes = vec![0xFF, 0xFE];
        bytes.extend(LINE.encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(parse_log(&text(&bytes)).samples.len(), 1);
    }

    #[test]
    fn diagnostics_lines_belong_to_their_tick() {
        let warm = LINE.replace("(0 being made)", "(889 being made)").replace("2.2 fps", "0.4 fps");
        let text = format!(
            "x INFO victory_gpu::host: GPU time    9.00 ms/frame,    12.0 x: draw indexed\n\
             {warm}\n\
             x INFO victory_gpu::host: dropped 50.0/frame without a host pipeline: being made\n\
             x INFO victory_gpu::host: GPU time    4.00 ms/frame,    12.0 x: draw indexed\n\
             {LINE}\n\
             x INFO victory_gpu::host: dropped 3.0/frame without a host pipeline: a table not made\n\
             {LINE}\n\
             x WARN victory_gpu::host: slow list command, 31.0 ms: dispatch 12 1 1\n\
             x WARN victory_gpu::host: slow list command, 31.0 ms: dispatch 12 2 1\n\
             x WARN victory_gpu::pipeline::host: host pipeline refused\n"
        );
        let run = parse_log(&text);
        assert_eq!(run.samples.len(), 3);
        let (s, is_steady) = steady(&run);
        assert!(is_steady);
        assert_eq!(s.len(), 2);
        // Over both steady lines, the second of which dropped nothing: 1.5;
        // the warm-up line's reason is not among them.
        assert_eq!(averages(&s, |x| &x.drops), [(1.5, "a table not made")]);
        // Only the steady line was timed, at 4 ms.
        let timed: Vec<&Sample> = s.iter().copied().filter(|x| !x.gpu.is_empty()).collect();
        assert_eq!(averages(&timed, |x| &x.gpu), [(4.0, "draw indexed")]);
        assert_eq!(run.slow_ops["dispatch"], 2);
        assert_eq!(run.events["host pipelines refused"], 1);
        let r = report(&["a".into(), "b".into()], &[parse_log(&warm), parse_log(&text)]);
        assert!(r.contains("no steady part"), "{r}");
        assert!(r.contains("== no comparison: a has no steady part"), "{r}");
        assert!(!r.contains("steady medians"), "{r}");
        assert!(r.contains("1.5  a table not made"), "{r}");
    }

    #[test]
    fn a_window_skips_the_menus_and_keeps_the_last_lines() {
        let text = (0..10).map(|i| LINE.replace("2729 draws", &format!("{i} draws"))).collect::<Vec<_>>().join("\n");
        let run = cut(parse_log(&text), Window { skip: 2, last: Some(3) });
        let draws: Vec<f64> = run.samples.iter().map(|s| s.values[field("draws/frame")]).collect();
        assert_eq!(draws, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn small_changes_are_not_marked_and_new_ones_say_so() {
        let a = parse_log(LINE);
        let b = parse_log(&LINE.replace("2.2 fps", "2.21 fps").replace("dropped 425", "dropped 0").replace("and 1 indirect", "and 0 indirect"));
        let c = parse_log(&LINE.replace("and 1 indirect", "and 0 indirect"));
        let r = report(&["a".into(), "b".into()], &[a, b]);
        let fps = r.lines().find(|l| l.trim_start().starts_with("fps") && l.contains('%')).unwrap();
        assert!(!fps.contains("better"), "{r}");
        assert!(r.lines().any(|l| l.contains("dropped/frame") && l.contains("better")), "{r}");
        let r = report(&["c".into(), "a".into()], &[c, parse_log(LINE)]);
        assert!(r.lines().any(|l| l.contains("indirect dropped") && l.contains("new")), "{r}");
    }

    #[test]
    fn profiles_are_compared_by_function_whatever_its_line() {
        let a = parse_profile(
            "1000 samples\n-- in itself\n 30.0% victory_gpu!copy:120\n 10.0% ntdll!wait\n\
             -- with what it calls\n 60.0% victory_gpu!submit:88\n-- stacks\n 5 a <- b\n",
        );
        let b = parse_profile(
            "900 samples\n-- in itself\n 12.0% victory_gpu!copy:131\n 11.0% ntdll!wait\n 4.0% victory_gpu!new:7\n",
        );
        assert_eq!(a.samples, 1000);
        assert_eq!(a.sections.len(), 3);
        assert_eq!(function("C:\\x\\a.rs!f:12"), "C:\\x\\a.rs!f");
        assert_eq!(function("ntdll!wait"), "ntdll!wait");
        let d = diff_profiles(&a, &b, "a", "b");
        let first = d.lines().nth(2).unwrap();
        assert!(first.ends_with("victory_gpu!copy") && first.contains("-18.0"), "{d}");
        assert!(d.contains("victory_gpu!new") && !d.contains(":131"), "{d}");
    }
}
