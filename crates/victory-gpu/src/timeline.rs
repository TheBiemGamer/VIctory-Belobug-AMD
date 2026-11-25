//! The title's queues: each runs its work in submission order, and holds it
//! back behind a wait (for a fence, or for a value a list waits for in
//! memory) until that is met.
//!
//! Work runs on the thread that submits it, as long as nothing holds the
//! queue back; else the submission returns, and the work runs on the thread
//! that meets the wait. One title thread submits to all queues, waits before
//! what they wait for, so a submission never blocks.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

thread_local! {
    /// Nesting depth of `Timeline::step` on the current thread. Work that
    /// submits or pumps re-entrantly must not wait on a queue this thread
    /// already holds `busy`.
    static STEPPING: Cell<u32> = const { Cell::new(0) };
}

use crate::fence::Held;

/// Lists of one submission still to run.
pub(crate) trait Lists: Send {
    /// How many of the first may run now: those that wait for nothing (what
    /// an earlier one writes counts once that ran).
    fn ready(&self) -> usize;
    fn len(&self) -> usize;
    /// Writes what the first, not ready, writes before its wait, if it does
    /// no work before that; whether it wrote anything. Two queues may meet
    /// this way halfway through a list each.
    fn early(&mut self) -> bool {
        false
    }
    /// Takes the first `n`, to be run.
    fn take(&mut self, n: usize) -> Box<dyn FnOnce()>;
    /// What the first waits for (for the log).
    fn waits(&self) -> String;
}

/// One step of a queue's work.
pub(crate) enum Op {
    /// Holds back what follows until the fence reaches the value.
    Wait(Held, u64),
    Signal(Held, u64),
    Lists(Box<dyn Lists>),
    Work(Box<dyn FnOnce() + Send>),
}

pub(crate) struct Timeline {
    state: Mutex<State>,
    /// Notified whenever `busy` clears so submitters waiting for another
    /// thread's in-progress step wake immediately.
    done: Condvar,
    late: Late,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            state: Mutex::default(),
            done: Condvar::new(),
            late: Late {
                signals: async_signal(),
                waits: gpu_waits(),
                owed: &HostWork,
            },
        }
    }
}

/// What a queue leaves to the finishing thread ([`Finisher`]) instead of
/// waiting for the host GPU on the thread that steps it.
#[derive(Clone, Copy)]
struct Late {
    /// `VICTORY_ASYNC_SIGNAL`: its signals.
    signals: bool,
    /// `VICTORY_GPU_WAITS`: finishing the host work a wait may be owed by.
    waits: bool,
    owed: &'static dyn Owed,
}

/// Whether a queue's signals are given by the finishing thread once the
/// host work before them is finished (`VICTORY_ASYNC_SIGNAL=1`), so the
/// thread that submits them does not wait for the host GPU.
fn async_signal() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_SIGNAL").is_some_and(|v| v == "1"))
}

/// Whether a wait that unwaited host work may still owe holds its queue
/// back while the finishing thread finishes that work
/// (`VICTORY_GPU_WAITS=1`), instead of the submitting thread doing it.
fn gpu_waits() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_WAITS").is_some_and(|v| v == "1"))
}

/// Host GPU work still to finish.
trait Owed: Sync {
    /// The host fence value of the newest submission still to finish; 0
    /// if none.
    fn owed(&self) -> u64;
    /// Waits until the host GPU reached `target`, then finishes what ran
    /// up to it (puts back what it wrote, then its bottom-of-pipe writes).
    fn finish(&self, target: u64);
}

struct HostWork;

impl Owed for HostWork {
    fn owed(&self) -> u64 {
        #[cfg(windows)]
        return crate::host::owed();
        #[cfg(not(windows))]
        0
    }
    fn finish(&self, target: u64) {
        #[cfg(windows)]
        crate::host::finish_to(target);
        #[cfg(not(windows))]
        let _ = target;
    }
}

/// One thread finishes host work for the queues, in the order asked: one
/// queue's signals are given in order, after what was submitted before.
struct Finisher {
    jobs: Mutex<Jobs>,
    more: Condvar,
}

struct Jobs {
    queue: VecDeque<Job>,
    /// The job being done: what it finishes, up to which fence value.
    running: Option<(&'static dyn Owed, u64)>,
}

struct Job {
    owed: &'static dyn Owed,
    target: u64,
    signal: Option<(Held, u64)>,
}

impl Jobs {
    fn idle(&self) -> bool {
        self.queue.is_empty() && self.running.is_none()
    }

    /// Whether host work up to `target` is already being finished.
    fn asked(&self, owed: &'static dyn Owed, target: u64) -> bool {
        self.queue
            .iter()
            .map(|j| (j.owed, j.target))
            .chain(self.running)
            .any(|(o, t)| std::ptr::addr_eq(o, owed) && t >= target)
    }
}

static FINISHER: Finisher = Finisher {
    jobs: Mutex::new(Jobs {
        queue: VecDeque::new(),
        running: None,
    }),
    more: Condvar::new(),
};

impl Finisher {
    fn jobs(&self) -> MutexGuard<'_, Jobs> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the finishing thread runs (started on first use).
    fn started(&'static self) -> bool {
        static STARTED: OnceLock<bool> = OnceLock::new();
        *STARTED.get_or_init(|| {
            let started = std::thread::Builder::new()
                .name("victory-finish".into())
                .spawn(move || self.run());
            if let Err(error) = &started {
                tracing::warn!(%error, "no thread to finish host work; queues wait for it");
            }
            started.is_ok()
        })
    }

    fn run(&self) {
        loop {
            let job = {
                let mut jobs = self.jobs();
                jobs.running = None;
                loop {
                    if let Some(job) = jobs.queue.pop_front() {
                        jobs.running = Some((job.owed, job.target));
                        break job;
                    }
                    jobs = self.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                }
            };
            job.owed.finish(job.target);
            if let Some((fence, value)) = job.signal {
                fence.signal(value);
            }
            progress();
            crate::fence::wake();
            kick();
        }
    }

    /// Hands the thread host work to finish up to `target`, then `signal`;
    /// the signal back if there is no thread.
    fn push(
        &'static self,
        jobs: &mut Jobs,
        owed: &'static dyn Owed,
        target: u64,
        signal: Option<(Held, u64)>,
    ) -> Result<(), Option<(Held, u64)>> {
        if !self.started() {
            return Err(signal);
        }
        jobs.queue.push_back(Job {
            owed,
            target,
            signal,
        });
        self.more.notify_one();
        Ok(())
    }
}

/// A queue's signal with `VICTORY_ASYNC_SIGNAL`: given at once if no host
/// work is owed and no earlier signal waits to be given, else by the
/// finishing thread once the owed work is finished.
fn signal_late(owed: &'static dyn Owed, fence: Held, value: u64) {
    let target = owed.owed();
    let mut jobs = FINISHER.jobs();
    // Given under the lock: a signal handed to the thread meanwhile must come after it.
    if target == 0 && jobs.idle() {
        fence.signal(value);
        return;
    }
    let signal = Some((fence, value));
    if let Err(Some((fence, value))) = FINISHER.push(&mut jobs, owed, target, signal) {
        drop(jobs);
        owed.finish(target);
        fence.signal(value);
    }
}

/// Whether the finishing thread has work left.
fn finishing_late() -> bool {
    !FINISHER.jobs().idle()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Idle,
    Moved,
    Busy,
}

struct BusyGuard<'a>(&'a Timeline);

impl<'a> BusyGuard<'a> {
    fn enter(timeline: &'a Timeline) -> Self {
        STEPPING.with(|n| n.set(n.get() + 1));
        Self(timeline)
    }
}

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        STEPPING.with(|n| n.set(n.get().saturating_sub(1)));
        let mut s = self.0.lock();
        s.busy = false;
        self.0.done.notify_all();
    }
}

#[derive(Default)]
struct State {
    ops: VecDeque<Op>,
    /// A thread runs the queue's work.
    busy: bool,
    /// Since when the first op is held back.
    held: Option<Instant>,
}

/// Every queue with a timeline.
static QUEUES: Mutex<Vec<Arc<Timeline>>> = Mutex::new(Vec::new());
/// How many queues are held back.
static HELD: AtomicU64 = AtomicU64::new(0);

/// How long queues may be held back while none of them moves. A wait whose
/// value only comes from later work of the same list, or from work that was
/// dropped, is never met: past this its list runs all the same.
const LIMIT: Duration = Duration::from_secs(2);

/// Whether an unwaited host submission is still to be finished.
fn finishing() -> bool {
    #[cfg(windows)]
    return crate::host::has_in_flight();
    #[cfg(not(windows))]
    false
}

/// Waits for the unwaited host submissions and finishes them.
fn finish() {
    #[cfg(windows)]
    crate::host::wait_in_flight_if_ready();
}

/// Whether any queue is held back.
pub(crate) fn held() -> bool {
    HELD.load(Ordering::Acquire) > 0
}

/// Whether any queue has work still to run.
pub(crate) fn active() -> bool {
    queues().iter().any(|q| !q.idle()) || finishing_late()
}

impl Timeline {
    pub(crate) fn new() -> Arc<Timeline> {
        let timeline = Arc::new(Timeline::default());
        queues().push(timeline.clone());
        timeline
    }

    #[cfg(test)]
    fn with(late: Late) -> Arc<Timeline> {
        let timeline = Arc::new(Timeline {
            late,
            ..Timeline::default()
        });
        queues().push(timeline.clone());
        timeline
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs what can run, on every queue.
    pub(crate) fn submit(&self, op: Op) {
        self.lock().ops.push_back(op);
        pump();
    }

    /// Whether work submitted now would run at once.
    pub(crate) fn idle(&self) -> bool {
        let s = self.lock();
        s.ops.is_empty() && !s.busy
    }

    /// Waits until no other thread is mid-step on this queue.
    fn wait_not_busy(&self) {
        let mut s = self.lock();
        while s.busy {
            s = self.done.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Runs what can run on this queue; whether anything did or another thread
    /// is currently running it. `force`: the first op runs whether or not its
    /// wait is met.
    fn step(&self, mut force: bool) -> Step {
        let mut moved = false;
        loop {
            let mut s = self.lock();
            if s.busy {
                return if moved { Step::Moved } else { Step::Busy };
            }
            let work: Box<dyn FnOnce()> = match s.ops.front_mut() {
                None => {
                    s.hold(false);
                    return if moved { Step::Moved } else { Step::Idle };
                }
                Some(Op::Wait(fence, value)) => {
                    if !fence.reached(*value) && !force {
                        // What it waits for may be what an unwaited
                        // submission still owes (its bottom-of-pipe writes):
                        // that one is finished first, then this is looked
                        // at again. Without this the queue stood until the
                        // wait was forced, two seconds a frame.
                        if self.finish_late() {
                            s.hold(true);
                            return if moved { Step::Moved } else { Step::Idle };
                        }
                        if !finishing() {
                            s.hold(true);
                            return if moved { Step::Moved } else { Step::Idle };
                        }
                        Box::new(finish)
                    } else {
                        let op = s.ops.pop_front();
                        Box::new(move || drop(op))
                    }
                }
                Some(Op::Lists(lists)) => {
                    let (ready, len) = (lists.ready(), lists.len());
                    let n = if force { ready.max(1) } else { ready }.min(len);
                    if n == 0 && len > 0 {
                        if lists.early() {
                            progress();
                            moved = true;
                            continue;
                        }
                        // As for a fence above.
                        if self.finish_late() {
                            s.hold(true);
                            return if moved { Step::Moved } else { Step::Idle };
                        }
                        if !finishing() {
                            s.hold(true);
                            return if moved { Step::Moved } else { Step::Idle };
                        }
                        Box::new(finish)
                    } else {
                        let work = lists.take(n);
                        if n == len {
                            s.ops.pop_front();
                        }
                        work
                    }
                }
                Some(Op::Signal(..) | Op::Work(_)) => match s.ops.pop_front() {
                    Some(Op::Signal(fence, value)) if self.late.signals => {
                        let owed = self.late.owed;
                        Box::new(move || signal_late(owed, fence, value))
                    }
                    Some(Op::Signal(fence, value)) => Box::new(move || {
                        #[cfg(windows)]
                        crate::host::wait_in_flight_if_ready();
                        fence.signal(value);
                    }),
                    Some(Op::Work(work)) => Box::new(move || {
                        #[cfg(windows)]
                        crate::host::wait_in_flight_if_ready();
                        work();
                    }),
                    _ => return if moved { Step::Moved } else { Step::Idle },
                },
            };
            s.hold(false);
            s.busy = true;
            let _busy = BusyGuard::enter(self);
            drop(s);
            work();
            drop(_busy);
            progress();
            (moved, force) = (true, false);
        }
    }

    /// With `VICTORY_GPU_WAITS`: hands the finishing thread the host work
    /// a wait not met may be owed by; whether it took it (or has it), and
    /// the queue is to be held back until the work is finished.
    fn finish_late(&self) -> bool {
        if !self.late.waits {
            return false;
        }
        let (owed, target) = (self.late.owed, self.late.owed.owed());
        if target == 0 {
            return false;
        }
        let mut jobs = FINISHER.jobs();
        jobs.asked(owed, target) || FINISHER.push(&mut jobs, owed, target, None).is_ok()
    }
}

impl State {
    fn hold(&mut self, hold: bool) {
        match (hold, self.held) {
            (true, None) => {
                self.held = Some(Instant::now());
                HELD.fetch_add(1, Ordering::AcqRel);
                watch();
            }
            (false, Some(since)) => {
                self.held = None;
                HELD.fetch_sub(1, Ordering::AcqRel);
                static COUNT: AtomicU64 = AtomicU64::new(0);
                let count = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                if count <= 8 || count.is_power_of_two() {
                    let ms = since.elapsed().as_millis();
                    tracing::info!(count, ms, "a queue waited for another's work");
                }
            }
            _ => {}
        }
    }
}

fn queues() -> MutexGuard<'static, Vec<Arc<Timeline>>> {
    QUEUES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs what can run on every queue, until none moves and (when called outside
/// `Timeline::step`) no other thread is mid-step on any queue: what one ran
/// may be what another waits for, and a submission does not return while its
/// ready work is still running elsewhere.
pub(crate) fn pump() {
    pump_inner(true);
}

fn pump_inner(wait: bool) {
    let wait = wait && STEPPING.with(|n| n.get() == 0);
    loop {
        // Queues the title released go once they have run everything.
        let all = {
            let mut queues = queues();
            queues.retain(|q| Arc::strong_count(q) > 1 || !q.idle());
            queues.clone()
        };
        let mut moved = false;
        let mut busy: Option<Arc<Timeline>> = None;
        for queue in &all {
            match queue.step(false) {
                Step::Moved => moved = true,
                Step::Busy if busy.is_none() => busy = Some(queue.clone()),
                Step::Busy | Step::Idle => {}
            }
        }
        if moved {
            crate::fence::wake();
            continue;
        }
        if wait && let Some(queue) = busy {
            queue.wait_not_busy();
            continue;
        }
        return;
    }
}

/// When a queue last ran something.
static MOVED: Mutex<Option<Instant>> = Mutex::new(None);

fn progress() {
    *MOVED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
}

/// Starts the thread that looks at what holds queues back: values the CPU
/// writes to memory wake nothing.
fn watch() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let started = std::thread::Builder::new()
        .name("victory-queues".into())
        .spawn(|| {
            loop {
                if !held() {
                    nap(Duration::from_millis(2));
                    continue;
                }
                nap(Duration::from_micros(500));
                pump_inner(false);
                force();
            }
        });
    if let Err(error) = started {
        tracing::warn!(%error, "no thread for held back queues");
    }
}

/// Set when the finishing thread finished work a held back queue may wait for.
static KICK: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Wakes the thread that looks at held back queues.
fn kick() {
    *KICK.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
    KICK.1.notify_one();
}

/// Sleeps for `time`, or until kicked.
fn nap(time: Duration) {
    let mut kicked = KICK.0.lock().unwrap_or_else(|e| e.into_inner());
    if !*kicked {
        kicked = KICK.1.wait_timeout(kicked, time).unwrap_or_else(|e| e.into_inner()).0;
    }
    *kicked = false;
}

/// Runs the op held back longest, if nothing moved for `LIMIT`.
fn force() {
    let quiet = MOVED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_none_or(|at| at.elapsed() > LIMIT);
    let all = queues().clone();
    // Work the finishing thread still does is not a wait never met.
    if !quiet || all.iter().any(|q| q.lock().busy) || finishing_late() {
        return;
    }
    let oldest = all
        .iter()
        .filter_map(|q| Some((q.lock().held?, q)))
        .min_by_key(|(since, _)| *since);
    let Some((since, queue)) = oldest.filter(|(since, _)| since.elapsed() > LIMIT) else {
        return;
    };
    static FORCED: AtomicU64 = AtomicU64::new(0);
    let forced = FORCED.fetch_add(1, Ordering::Relaxed) + 1;
    if forced <= 32 || forced.is_power_of_two() {
        let waits = match queue.lock().ops.front() {
            Some(Op::Lists(lists)) => lists.waits(),
            Some(Op::Wait(_, value)) => format!("a fence to reach {value}"),
            _ => String::new(),
        };
        for (i, q) in all.iter().enumerate() {
            let s = q.lock();
            for (k, op) in s.ops.iter().enumerate().take(12) {
                let text = match op {
                    Op::Lists(lists) => lists.waits(),
                    Op::Wait(_, v) => format!("fence wait {v}"),
                    Op::Signal(_, v) => format!("signal {v}"),
                    Op::Work(_) => "work".into(),
                };
                tracing::info!("queue {i} op {k}: {text}");
            }
        }
        tracing::warn!(
            forced,
            ms = since.elapsed().as_millis(),
            waits,
            "a wait was never met: its work runs all the same"
        );
    }
    queue.step(true);
    crate::fence::wake();
}

#[cfg(test)]
mod tests {
    use super::*;

    type Log = Arc<Mutex<Vec<(u64, u64)>>>;

    /// Lists as (value to wait for at the address, value to write there).
    struct Fake {
        at: Arc<AtomicU64>,
        lists: VecDeque<(u64, u64)>,
        log: Log,
    }

    impl Lists for Fake {
        fn ready(&self) -> usize {
            let value = self.at.load(Ordering::Acquire);
            self.lists.iter().take_while(|l| value >= l.0).count()
        }
        fn len(&self) -> usize {
            self.lists.len()
        }
        fn take(&mut self, n: usize) -> Box<dyn FnOnce()> {
            let run: Vec<_> = self.lists.drain(..n).collect();
            let (at, log) = (self.at.clone(), self.log.clone());
            Box::new(move || {
                for l in run {
                    log.lock().unwrap().push(l);
                    at.fetch_max(l.1, Ordering::AcqRel);
                }
            })
        }
        fn waits(&self) -> String {
            String::new()
        }
    }

    fn op(at: &Arc<AtomicU64>, log: &Log, lists: &[(u64, u64)]) -> Op {
        Op::Lists(Box::new(Fake {
            at: at.clone(),
            lists: lists.iter().copied().collect(),
            log: log.clone(),
        }))
    }

    fn until(what: impl Fn() -> bool) {
        let started = Instant::now();
        while !what() {
            assert!(started.elapsed() < Duration::from_secs(20), "still held back");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Compute submitted before the graphics it waits for, in two parts
    /// that wait for one another.
    #[test]
    fn work_runs_once_its_wait_is_met() {
        let (compute, graphics) = (Timeline::new(), Timeline::new());
        let at = Arc::new(AtomicU64::new(0));
        let log = Log::default();
        compute.submit(op(&at, &log, &[(1, 2), (3, 4)]));
        let done = Arc::new(AtomicBool::new(false));
        let mark = done.clone();
        compute.submit(Op::Work(Box::new(move || mark.store(true, Ordering::Release))));
        assert!(log.lock().unwrap().is_empty() && !compute.idle());
        graphics.submit(op(&at, &log, &[(0, 1), (2, 3)]));
        // The watching thread may run some of it.
        until(|| compute.idle() && graphics.idle());
        let order = log.lock().unwrap().clone();
        assert_eq!(order, [(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert!(done.load(Ordering::Acquire));
    }

    /// A value the CPU writes wakes the queue, and a wait never met does not
    /// hold it back for good.
    #[test]
    fn queues_move_without_another_submission() {
        let queue = Timeline::new();
        let at = Arc::new(AtomicU64::new(0));
        let log = Log::default();
        queue.submit(op(&at, &log, &[(5, 0)]));
        assert!(log.lock().unwrap().is_empty());
        at.store(5, Ordering::Release);
        until(|| log.lock().unwrap().len() == 1);
        queue.submit(op(&at, &log, &[(6, 0)]));
        until(|| log.lock().unwrap().len() == 2);
        assert_eq!(at.load(Ordering::Acquire), 5);
    }

    /// Submitting to a queue that another thread is currently stepping does not
    /// return before both the in-progress op and the newly submitted op finish.
    #[test]
    fn submit_waits_for_busy_queue_on_another_thread() {
        let queue = Timeline::new();
        let started = Arc::new(AtomicBool::new(false));
        let first_done = Arc::new(AtomicBool::new(false));
        let second_done = Arc::new(AtomicBool::new(false));

        let q = queue.clone();
        let s = started.clone();
        let f = first_done.clone();
        let worker = std::thread::spawn(move || {
            q.submit(Op::Work(Box::new(move || {
                s.store(true, Ordering::Release);
                std::thread::sleep(Duration::from_millis(30));
                f.store(true, Ordering::Release);
            })));
        });
        until(|| started.load(Ordering::Acquire));

        let d = second_done.clone();
        queue.submit(Op::Work(Box::new(move || {
            d.store(true, Ordering::Release);
        })));
        assert!(first_done.load(Ordering::Acquire));
        assert!(second_done.load(Ordering::Acquire));
        worker.join().unwrap();
    }

    /// Submitting re-entrantly from inside a queue's work does not deadlock
    /// waiting for the outer queue's `busy` flag.
    #[test]
    fn reentrant_submit_inside_work_does_not_deadlock() {
        let (first, second) = (Timeline::new(), Timeline::new());
        let nested = Arc::new(AtomicBool::new(false));
        let followup = Arc::new(AtomicBool::new(false));

        let f = first.clone();
        let s = second.clone();
        let n = nested.clone();
        let u = followup.clone();
        first.submit(Op::Work(Box::new(move || {
            s.submit(Op::Work(Box::new(move || n.store(true, Ordering::Release))));
            f.submit(Op::Work(Box::new(move || u.store(true, Ordering::Release))));
        })));
        assert!(nested.load(Ordering::Acquire));
        assert!(followup.load(Ordering::Acquire));
    }

    /// Host work the test runs by hand: `submit` owes a fence value,
    /// `run` lets the GPU reach one.
    #[derive(Default)]
    struct FakeHost {
        submitted: AtomicU64,
        ran: Mutex<u64>,
        ran_now: Condvar,
        finished: AtomicU64,
        /// A bottom-of-pipe write of the owed submission.
        writes: Mutex<Option<(Held, u64)>>,
    }

    impl Owed for FakeHost {
        fn owed(&self) -> u64 {
            let submitted = self.submitted.load(Ordering::Acquire);
            if submitted > self.finished.load(Ordering::Acquire) {
                submitted
            } else {
                0
            }
        }
        fn finish(&self, target: u64) {
            let mut ran = self.ran.lock().unwrap();
            while *ran < target {
                ran = self.ran_now.wait(ran).unwrap();
            }
            drop(ran);
            if self.finished.fetch_max(target, Ordering::AcqRel) < target
                && let Some((fence, value)) = self.writes.lock().unwrap().take()
            {
                fence.signal(value);
            }
        }
    }

    impl FakeHost {
        fn new() -> &'static FakeHost {
            Box::leak(Box::default())
        }
        fn run(&self, to: u64) {
            *self.ran.lock().unwrap() = to;
            self.ran_now.notify_all();
        }
        fn late(&'static self, signals: bool, waits: bool) -> Late {
            Late {
                signals,
                waits,
                owed: self,
            }
        }
    }

    use crate::object::{S_OK, This, slot};

    struct Fence(This);

    impl Fence {
        fn new() -> Fence {
            let mut f = std::ptr::null();
            let iid = &crate::fence::FENCE_IIDS[0];
            assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, iid, &mut f), S_OK);
            Fence(f)
        }
        fn held(&self) -> Held {
            Held::new(self.0)
        }
        fn value(&self) -> u64 {
            slot::<extern "system" fn(This) -> u64>(self.0, 0x40)(self.0)
        }
    }

    impl Drop for Fence {
        fn drop(&mut self) {
            slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
        }
    }

    /// `VICTORY_ASYNC_SIGNAL`: the signal returns at once, is given only
    /// once the work before it is finished, after an earlier one; a wait
    /// for it on another queue holds that queue back until then.
    #[test]
    fn late_signals_come_after_the_work_before_them_in_order() {
        let host = FakeHost::new();
        let (signals, waits) = (Timeline::with(host.late(true, false)), Timeline::new());
        let fence = Fence::new();
        host.submitted.store(1, Ordering::Release);
        signals.submit(Op::Signal(fence.held(), 1));
        // Owes nothing now, but must not overtake the signal before it.
        host.finished.store(1, Ordering::Release);
        host.submitted.store(1, Ordering::Release);
        signals.submit(Op::Signal(fence.held(), 2));
        assert_eq!(fence.value(), 0);
        let seen = Arc::new(AtomicU64::new(0));
        let see = seen.clone();
        waits.submit(Op::Wait(fence.held(), 2));
        waits.submit(Op::Work(Box::new(move || see.store(1, Ordering::Release))));
        assert_eq!(seen.load(Ordering::Acquire), 0);
        host.run(1);
        until(|| seen.load(Ordering::Acquire) == 1);
        assert_eq!(fence.value(), 2);
        until(|| FINISHER.jobs().idle());
    }

    /// Without owed work and earlier signals, a late signal is given at once.
    #[test]
    fn late_signal_without_owed_work_is_given_at_once() {
        let host = FakeHost::new();
        let queue = Timeline::with(host.late(true, false));
        let fence = Fence::new();
        until(|| FINISHER.jobs().idle());
        queue.submit(Op::Signal(fence.held(), 3));
        assert_eq!(fence.value(), 3);
    }

    /// `VICTORY_GPU_WAITS`: a wait owed by unfinished host work holds the
    /// queue back without the submission waiting for the host; what
    /// follows runs once that work is finished and its writes are back.
    #[test]
    fn owed_wait_does_not_block_the_submission() {
        let host = FakeHost::new();
        let queue = Timeline::with(host.late(false, true));
        let fence = Fence::new();
        host.submitted.store(4, Ordering::Release);
        *host.writes.lock().unwrap() = Some((fence.held(), 1));
        let ran = Arc::new(AtomicBool::new(false));
        let after_finish = Arc::new(AtomicBool::new(false));
        let (r, a) = (ran.clone(), after_finish.clone());
        let started = Instant::now();
        queue.submit(Op::Wait(fence.held(), 1));
        queue.submit(Op::Work(Box::new(move || {
            a.store(host.finished.load(Ordering::Acquire) >= 4, Ordering::Release);
            r.store(true, Ordering::Release);
        })));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!ran.load(Ordering::Acquire));
        host.run(4);
        until(|| ran.load(Ordering::Acquire));
        assert!(after_finish.load(Ordering::Acquire));
        until(|| FINISHER.jobs().idle());
    }
}
