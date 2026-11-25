//! Title memory the title mapped anew while a submission was in flight. The
//! submission's copy-back must not put the GPU's bytes into the new mapping.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard};

/// Disjoint remapped `[start, end)` intervals, by start, with the submission
/// count when they were remapped (a later remap of the same bytes wins).
struct Remaps {
    intervals: BTreeMap<usize, (usize, u64)>,
}

impl Remaps {
    const fn new() -> Self {
        Self {
            intervals: BTreeMap::new(),
        }
    }

    /// Splits the interval holding `at` there.
    fn split(&mut self, at: usize) {
        if let Some((&from, &(to, stamp))) = self.intervals.range(..at).next_back()
            && at < to
        {
            self.intervals.insert(from, (at, stamp));
            self.intervals.insert(at, (to, stamp));
        }
    }

    /// Forgets what only submissions up to `finished` could have written.
    fn prune(&mut self, finished: u64) {
        self.intervals.retain(|_, &mut (_, stamp)| stamp > finished);
    }

    fn record(&mut self, start: usize, end: usize, started: u64, finished: u64) {
        self.prune(finished);
        if start >= end || started <= finished {
            return;
        }
        self.split(start);
        self.split(end);
        let over: Vec<usize> = self
            .intervals
            .range(start..end)
            .map(|(&at, _)| at)
            .collect();
        for at in over {
            self.intervals.remove(&at);
        }
        let (mut from, mut to) = (start, end);
        // A stream of single-page remaps stays one interval.
        if let Some((&a, &(b, stamp))) = self.intervals.range(..start).next_back()
            && b == start
            && stamp == started
        {
            self.intervals.remove(&a);
            from = a;
        }
        if let Some(&(b, stamp)) = self.intervals.get(&end)
            && stamp == started
        {
            self.intervals.remove(&end);
            to = b;
        }
        self.intervals.insert(from, (to, started));
    }

    /// Calls `f` with the parts of `[from, to)` remapped after `submission`
    /// started, in order.
    fn since(&self, submission: u64, from: usize, to: usize, mut f: impl FnMut(usize, usize)) {
        if self.intervals.is_empty() || from >= to {
            return;
        }
        let first = self
            .intervals
            .range(..=from)
            .next_back()
            .map_or(from, |(&at, _)| at);
        for (&a, &(b, stamp)) in self.intervals.range(first..to) {
            if b > from && stamp >= submission {
                f(a.max(from), b.min(to));
            }
        }
    }
}

static REMAPS: RwLock<Remaps> = RwLock::new(Remaps::new());
/// The last submission whose copy-back is done (they finish in order).
static FINISHED: AtomicU64 = AtomicU64::new(0);

/// The title mapped or unmapped `[start, end)`. `started` reads the submission
/// count, under the lock so stamps grow in the order they are recorded.
/// Never call with the watch guards' or views' lock held: writes under [`held`] take them.
pub(crate) fn record(start: usize, end: usize, started: impl FnOnce() -> u64) {
    let mut remaps = REMAPS.write().unwrap_or_else(|e| e.into_inner());
    let finished = FINISHED.load(Ordering::Acquire);
    remaps.record(start, end, started(), finished);
}

/// Submission `submission` has been copied back.
pub(crate) fn finished(submission: u64) {
    if FINISHED.fetch_max(submission, Ordering::AcqRel) < submission {
        let mut remaps = REMAPS.write().unwrap_or_else(|e| e.into_inner());
        if !remaps.intervals.is_empty() {
            remaps.prune(submission);
        }
    }
}

/// Holds remaps off: a remap is recorded before the unmap, so bytes checked
/// and written under this land before it.
pub(crate) struct Held(RwLockReadGuard<'static, Remaps>);

pub(crate) fn held() -> Held {
    Held(REMAPS.read().unwrap_or_else(|e| e.into_inner()))
}

impl Held {
    /// [`Remaps::since`]: the parts of `[from, to)` whose GPU bytes from
    /// `submission` must stay out of title memory.
    pub(crate) fn since(
        &self,
        submission: u64,
        from: usize,
        to: usize,
        f: impl FnMut(usize, usize),
    ) {
        self.0.since(submission, from, to, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(r: &Remaps, submission: u64, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        r.since(submission, from, to, |a, b| v.push((a, b)));
        v
    }

    #[test]
    fn only_remaps_after_a_submission_started_are_withheld() {
        let mut r = Remaps::new();
        r.record(100, 200, 5, 4);
        assert_eq!(parts(&r, 5, 0, 1000), [(100, 200)]);
        assert_eq!(parts(&r, 4, 150, 1000), [(150, 200)]);
        assert!(parts(&r, 6, 0, 1000).is_empty());
        assert!(parts(&r, 5, 200, 300).is_empty());
        assert!(parts(&r, 5, 0, 100).is_empty());
    }

    #[test]
    fn nothing_in_flight_records_nothing_and_finishing_forgets() {
        let mut r = Remaps::new();
        r.record(100, 200, 4, 4);
        assert!(r.intervals.is_empty());
        r.record(100, 200, 5, 4);
        r.record(300, 400, 6, 4);
        r.prune(5);
        assert_eq!(parts(&r, 0, 0, 1000), [(300, 400)]);
        r.record(0, 1, 7, 6);
        assert!(parts(&r, 0, 100, 500).is_empty());
    }

    #[test]
    fn a_later_remap_restamps_the_overlap_and_neighbours_coalesce() {
        let mut r = Remaps::new();
        r.record(100, 200, 5, 4);
        r.record(150, 250, 6, 4);
        assert_eq!(parts(&r, 5, 0, 1000), [(100, 150), (150, 250)]);
        assert_eq!(parts(&r, 6, 0, 1000), [(150, 250)]);
        r.record(120, 130, 6, 4);
        assert_eq!(parts(&r, 6, 0, 1000), [(120, 130), (150, 250)]);
        assert_eq!(
            parts(&r, 5, 0, 1000),
            [(100, 120), (120, 130), (130, 150), (150, 250)]
        );
        let mut pages = Remaps::new();
        for p in 0..64 {
            pages.record(p * 4096, (p + 1) * 4096, 9, 8);
        }
        assert_eq!(pages.intervals.len(), 1);
        assert_eq!(parts(&pages, 9, 4000, 8192), [(4000, 8192)]);
        pages.record(64 * 4096, 65 * 4096, 10, 8);
        assert_eq!(pages.intervals.len(), 2);
    }
}
