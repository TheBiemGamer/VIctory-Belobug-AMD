//! Lazy copy-back of what shaders wrote into sparse copies
//! (`VICTORY_LAZY_BACK=1`). Without it each submission's GPU writes are
//! merged back into title memory before its fences are signalled. With it a
//! 64 KiB page the GPU changed is left without access instead, at every
//! address the title shows it at, and remembered as the bytes the GPU wrote
//! (one bit a byte) and where their values are (the host copy's tile). The
//! title's first touch of the page faults; the handler writes those bytes
//! into the page through a view of our own of its physical page, so no
//! other thread sees the page before it is whole, and gives the access
//! back. Pages nobody reads are never copied.
//!
//! Only physical-page views can be done so: private title memory has no
//! second view to write it through, so it is merged at once as before, as
//! are pages the host GPU uses in place, pages with more than
//! [`crate::watch::MANY_VIEWS`] views, and pages a kernel write is pinned
//! on.
//!
//! Our own reads and writes of title memory that do not fault into the
//! handler (the kernel's copies, the runtime's own views of physical pages,
//! kernel reads into title memory) put back the pages they touch first
//! ([`resolve`]). A submission that declares a page still owed for writing
//! takes no snapshot of it: the tile's bytes are kept ([`declare`]), which
//! serve as its snapshot and as the values the title still lacks, so a
//! fault while it runs never waits for the host GPU, and its own writes
//! join the page's when it finishes ([`finished`]).
//!
//! Locks: this module's, then the guards' (`watch`), then the views'. The
//! handler takes only those, and code holding them never touches title
//! memory.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::watch::PAGE;

/// One bit for each byte of a page.
pub(crate) type Mask = Box<[u64; PAGE / 64]>;

/// A mask with no byte set.
pub(crate) fn mask() -> Mask {
    Box::new([0; PAGE / 64])
}

/// Whether GPU writes into sparse copies go back lazily.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_BACK").is_some_and(|v| v == "1"))
}

/// Where a hidden page is: its section and offset there, and the addresses
/// it is shown at (all without access).
struct Hidden {
    section: usize,
    offset: u64,
    shown: Vec<usize>,
}

/// A page of title memory whose bytes the GPU wrote and the title lacks.
struct Record {
    /// The CPU's view of the host copy's tile showing it.
    tile: usize,
    /// The copy the tile is of ([`crate::shadow::Shadow::root_key`]).
    root: usize,
    /// The bytes owed and where the page is, while it is hidden.
    owed: Option<(Mask, Hidden)>,
    /// The tile as it was when a submission declared it for writing: the
    /// values of the bytes owed meanwhile, and that submission's snapshot.
    frozen: Option<Box<[u8]>>,
    /// That submission and what of the page it declared (offsets).
    next: Option<(u64, Vec<(usize, usize)>)>,
    /// What it wrote, found when it finished.
    fresh: Option<Mask>,
}

struct Lazy {
    /// By the page's address where its tile shows it.
    records: BTreeMap<usize, Record>,
    /// By each address hidden: the records hiding it.
    hiding: BTreeMap<usize, Vec<usize>>,
    /// By section: our view of all of it, and how many hidden pages use it.
    views: BTreeMap<usize, (usize, usize)>,
}

static LAZY: Mutex<Lazy> = Mutex::new(Lazy {
    records: BTreeMap::new(),
    hiding: BTreeMap::new(),
    views: BTreeMap::new(),
});

thread_local! {
    /// How many of our locks the fault handler also takes this thread holds.
    static INSIDE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Held while this thread holds a lock the fault handler takes: a fault
/// then is not ours to handle (it would wait for itself).
pub(crate) struct Inside;

impl Inside {
    pub(crate) fn enter() -> Inside {
        INSIDE.with(|c| c.set(c.get() + 1));
        Inside
    }
}

impl Drop for Inside {
    fn drop(&mut self) {
        INSIDE.with(|c| c.set(c.get() - 1));
    }
}

/// Our lock, marked held for the fault handler (fields drop in order).
struct Held {
    guard: std::sync::MutexGuard<'static, Lazy>,
    _inside: Inside,
}

impl std::ops::Deref for Held {
    type Target = Lazy;
    fn deref(&self) -> &Lazy {
        &self.guard
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Lazy {
        &mut self.guard
    }
}

fn lazy() -> Held {
    let inside = Inside::enter();
    Held {
        guard: LAZY.lock().unwrap_or_else(|e| e.into_inner()),
        _inside: inside,
    }
}

/// Pages hidden now.
static HIDDEN: AtomicUsize = AtomicUsize::new(0);
/// A page was ever hidden: from then on every access fault is looked at.
static EVER: AtomicBool = AtomicBool::new(false);

/// Counts since the last log line.
#[derive(Clone, Copy)]
enum S {
    /// Pages left owed after a submission.
    Owed,
    /// Pages the GPU changed that could not be (merged at once).
    Eager,
    /// Owed pages a later submission declared for writing (snapshot kept).
    Declared,
    /// Faults that put pages back.
    Faults,
    /// Pages put back on a fault.
    ByFault,
    /// Pages put back before our own access.
    ByUs,
    /// Bytes put back.
    Bytes,
}
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn add(s: S, n: u64) {
    STATS[s as usize].fetch_add(n, Ordering::Relaxed);
}

/// Pages the GPU wrote into a copy's tile, by copy and page: the asking
/// that noted them written for the other copies, after which the copy is
/// still current. Taken last: nothing else is locked while it is held.
static CURRENT: Mutex<BTreeMap<(usize, usize), u64>> = Mutex::new(BTreeMap::new());

/// The GPU wrote the page `at` of the copy `root`, shown at `shown`: every
/// other copy of it takes it again (reading it puts it back first), the
/// copy's own is current, unless another record owes the same page (the
/// GPU wrote it at another address too): then none is.
fn noted(l: &Lazy, root: usize, at: usize, shown: &[usize]) {
    let others: Vec<usize> = shown
        .iter()
        .filter_map(|a| l.hiding.get(a))
        .flatten()
        .copied()
        .filter(|&k| k != at)
        .collect();
    let now = crate::watch::noted_written(shown);
    let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    if current.len() >= 1 << 16 {
        current.clear();
    }
    if others.is_empty() {
        current.insert((root, at), now);
    } else {
        current.retain(|&(_, page), _| page != at && !others.contains(&page));
    }
}

/// Whether the copy `root` already holds the page at `page`, last noted
/// written at the asking `write`: it was put back from its tile then, and
/// not written since.
pub(crate) fn current(root: usize, page: usize, write: u64) -> bool {
    if !on() {
        return false;
    }
    let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    current.remove(&(root, page)) == Some(write)
}

/// Whether any page is owed now.
pub(crate) fn owing() -> bool {
    HIDDEN.load(Ordering::Acquire) != 0
}

/// The counts since the last call, and the pages owed now, for the frame
/// log.
pub(crate) fn text() -> String {
    let [owed, eager, declared, faults, by_fault, by_us, bytes] =
        std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed));
    format!(
        "{owed} pages left owed, {eager} merged at once (not lazily doable), {declared} declared again \
         while owed; {faults} faults put back {by_fault} pages, our own accesses {by_us}, {:.1} MB; \
         {} pages owed now",
        bytes as f64 / 1e6,
        HIDDEN.load(Ordering::Relaxed)
    )
}

/// Whether a page of the copy `root` is owed or declared (its tiles must
/// stay).
pub(crate) fn holds(root: usize) -> bool {
    lazy().records.values().any(|r| r.root == root)
}

/// The bits of `mask` set where `host` differs from `old` (both from `from`
/// on in the page, `n` bytes); whether any was.
pub(crate) fn differ(mask: &mut [u64; PAGE / 64], from: usize, host: &[u8], old: &[u8]) -> bool {
    let mut any = false;
    for (k, (h, o)) in host.chunks(8).zip(old.chunks(8)).enumerate() {
        if h == o {
            continue;
        }
        for (j, (a, b)) in h.iter().zip(o).enumerate() {
            if a != b {
                let at = from + k * 8 + j;
                mask[at / 64] |= 1 << (at % 64);
                any = true;
            }
        }
    }
    any
}

/// Writes the bytes `mask` names from `src` into `title` (a page each);
/// the bytes that changed.
///
/// # Safety
/// Both point at a page of readable memory, `title` writable.
unsafe fn apply(title: *mut u8, src: *const u8, mask: &[u64; PAGE / 64]) -> usize {
    let mut changed = 0;
    for (w, &bits) in mask.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let at = w * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            // SAFETY: inside both pages.
            unsafe {
                let v = src.add(at).read();
                if title.add(at).read() != v {
                    title.add(at).write(v);
                    changed += 1;
                }
            }
        }
    }
    changed
}

/// Where the page at `at` could be hidden: a view of a physical page with
/// few views, none used in place by the host GPU.
fn hideable(at: usize) -> Option<Hidden> {
    let (section, offset, shown) = crate::watch::physical(at)?;
    let page = PAGE as u64;
    if shown
        .iter()
        .any(|&a| crate::shadow::in_place(a as u64, a as u64 + page))
    {
        return None;
    }
    Some(Hidden {
        section,
        offset,
        shown,
    })
}

/// Our view of all of `section`, the page at `at` being its `offset`,
/// mapped on first use; one more user.
fn view(l: &mut Lazy, at: usize, section: usize, offset: u64) -> Option<usize> {
    if let Some((base, users)) = l.views.get_mut(&section) {
        *users += 1;
        return Some(*base);
    }
    let base = crate::watch::map_section(at, section, offset)?;
    l.views.insert(section, (base, 1));
    Some(base)
}

/// One user less of our view of `section`; unmapped after the last.
fn unview(l: &mut Lazy, section: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
    let Some((base, users)) = l.views.get_mut(&section) else {
        return;
    };
    *users -= 1;
    if *users == 0 {
        let base = *base;
        l.views.remove(&section);
        // SAFETY: our view, used by nothing now.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: base as *mut _,
            })
        };
    }
}

/// A page to leave owed: its address, its tile, its copy and the bytes the
/// GPU wrote.
pub(crate) struct Owe {
    pub(crate) at: usize,
    pub(crate) tile: usize,
    pub(crate) root: usize,
    pub(crate) mask: Mask,
}

/// Leaves the pages owed; for each, whether it was (if not, the caller
/// merges it now). A page owed already gets the new bytes added.
pub(crate) fn owe(pages: Vec<Owe>) -> Vec<bool> {
    let mut done = vec![false; pages.len()];
    let mut new = Vec::new();
    {
        let mut l = lazy();
        for (i, page) in pages.into_iter().enumerate() {
            let shown = match l.records.get_mut(&page.at).and_then(|r| r.owed.as_mut()) {
                Some((mask, hidden)) => {
                    for (a, b) in mask.iter_mut().zip(page.mask.iter()) {
                        *a |= b;
                    }
                    hidden.shown.clone()
                }
                None => {
                    new.push((i, page));
                    continue;
                }
            };
            noted(&l, page.root, page.at, &shown);
            done[i] = true;
        }
    }
    if new.is_empty() {
        return done;
    }
    // Asked outside our lock: the host GPU's windows are looked up under a
    // lock held while title memory is opened for them.
    let found: Vec<Option<Hidden>> = new.iter().map(|(_, p)| hideable(p.at)).collect();
    let mut l = lazy();
    let mut candidates = Vec::new();
    for ((i, page), hidden) in new.into_iter().zip(found) {
        let Some(hidden) = hidden else { continue };
        if view(&mut l, page.at, hidden.section, hidden.offset).is_none() {
            continue;
        }
        candidates.push((i, page, hidden));
    }
    // Counted and seen before any page is without access: the handler and
    // our own accesses look only once these say so.
    EVER.store(true, Ordering::Release);
    HIDDEN.fetch_add(candidates.len(), Ordering::AcqRel);
    let shown: Vec<Vec<usize>> = candidates.iter().map(|(_, _, h)| h.shown.clone()).collect();
    let hid = crate::watch::hide(&shown);
    let mut hidden_now = Vec::new();
    for ((i, page, hidden), hid) in candidates.into_iter().zip(hid) {
        if !hid {
            unview(&mut l, hidden.section);
            HIDDEN.fetch_sub(1, Ordering::AcqRel);
            continue;
        }
        for &at in &hidden.shown {
            l.hiding.entry(at).or_default().push(page.at);
        }
        noted(&l, page.root, page.at, &hidden.shown);
        hidden_now.push((page.at, hidden.section, hidden.offset, hidden.shown.clone()));
        let record = l.records.entry(page.at).or_insert(Record {
            tile: page.tile,
            root: page.root,
            owed: None,
            frozen: None,
            next: None,
            fresh: None,
        });
        record.owed = Some((page.mask, hidden));
        done[i] = true;
    }
    drop(l);
    // Looked at again now the pages are hidden: a window the host GPU uses
    // in place, or a view unmapped, that came meanwhile (a window cannot
    // take a hidden page, and a view going puts its pages back).
    for (at, section, offset, shown) in hidden_now {
        let same = hideable(at).is_some_and(|h| {
            h.section == section && h.offset == offset && h.shown.iter().all(|a| shown.contains(a))
        });
        if !same {
            resolve(at, PAGE);
        }
    }
    let owed = done.iter().filter(|d| **d).count() as u64;
    add(S::Owed, owed);
    add(S::Eager, done.len() as u64 - owed);
    done
}

/// Puts back the bytes the record at `key` owes, through our view of its
/// page, and gives its addresses their access back; whether a byte of title
/// memory changed.
fn settle(l: &mut Lazy, key: usize) -> bool {
    let Some(record) = l.records.get_mut(&key) else {
        return false;
    };
    let Some((mask, hidden)) = record.owed.take() else {
        return false;
    };
    let src = record
        .frozen
        .as_ref()
        .map_or(record.tile as *const u8, |f| f.as_ptr());
    let done = record.next.is_none() && record.fresh.is_none();
    let (tile, root) = (record.tile, record.root);
    let changed = match l.views.get(&hidden.section) {
        // SAFETY: our view of the page's section, and the tile's bytes (or
        // its frozen copy), a page each.
        Some(&(base, _)) => unsafe {
            apply((base + hidden.offset as usize) as *mut u8, src, &mask)
        },
        None => 0,
    };
    if done {
        l.records.remove(&key);
    }
    let stuck = crate::watch::show(&hidden.shown, false);
    for at in hidden.shown.iter().filter(|a| !stuck.contains(a)) {
        if let Some(keys) = l.hiding.get_mut(at) {
            keys.retain(|&k| k != key);
            if keys.is_empty() {
                l.hiding.remove(at);
            }
        }
    }
    if stuck.is_empty() {
        unview(l, hidden.section);
        HIDDEN.fetch_sub(1, Ordering::AcqRel);
    } else {
        // Still without access there: the next touch tries again, with
        // nothing more to write.
        tracing::warn!(
            at = format_args!("{key:#x}"),
            "a page lazy copy-back put back stayed without access"
        );
        let record = l.records.entry(key).or_insert(Record {
            tile,
            root,
            owed: None,
            frozen: None,
            next: None,
            fresh: None,
        });
        record.owed = Some((
            crate::lazy::mask(),
            Hidden {
                shown: stuck,
                ..hidden
            },
        ));
    }
    add(S::Bytes, changed as u64);
    changed > 0
}

/// Before our own access to title memory `[at, at + len)` that would not
/// fault into the handler: puts back the pages owed there (at any of their
/// addresses). Whether any was.
pub(crate) fn resolve(at: usize, len: usize) -> bool {
    if len == 0 || !owing() {
        return false;
    }
    let (first, last) = (at - at % PAGE, at.saturating_add(len));
    let mut l = lazy();
    let mut keys: Vec<usize> = l
        .hiding
        .range(first..last)
        .flat_map(|(_, k)| k.iter().copied())
        .collect();
    if keys.is_empty() {
        return false;
    }
    keys.sort_unstable();
    keys.dedup();
    for &key in &keys {
        settle(&mut l, key);
    }
    add(S::ByUs, keys.len() as u64);
    true
}

/// The access fault handler's part: an access at `at` (a write if `write`)
/// to a hidden page puts it back. Whether the access may be tried again.
pub(crate) fn fault(at: usize, write: bool) -> bool {
    // A fault while this thread holds our locks would wait for itself.
    if !EVER.load(Ordering::Acquire) || INSIDE.with(|c| c.get()) != 0 {
        return false;
    }
    {
        let mut l = lazy();
        let keys = l.hiding.get(&(at - at % PAGE)).cloned().unwrap_or_default();
        if !keys.is_empty() {
            for &key in &keys {
                settle(&mut l, key);
            }
            add(S::Faults, 1);
            add(S::ByFault, keys.len() as u64);
            return true;
        }
    }
    // Put back by another thread since this one faulted.
    allows(at, write)
}

/// Whether `at` is committed memory that allows the access now.
fn allows(at: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a query of our own address space.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
        || info.State != MEM_COMMIT
    {
        return false;
    }
    match info.Protect {
        PAGE_READWRITE | PAGE_EXECUTE_READWRITE => true,
        PAGE_READONLY | PAGE_EXECUTE_READ => !write,
        _ => false,
    }
}

/// At its preparation, submission `submission` declares parts of pages of
/// the copy for writing (`(title address, view, length)`). For each part,
/// whether its page has a record: then no snapshot is taken, the tile is
/// kept as it is now instead.
pub(crate) fn declare(submission: u64, parts: &[(usize, usize, usize)]) -> Vec<bool> {
    if !EVER.load(Ordering::Acquire) {
        return vec![false; parts.len()];
    }
    let mut l = lazy();
    if l.records.is_empty() {
        return vec![false; parts.len()];
    }
    parts
        .iter()
        .map(|&(at, _, n)| {
            let page = at - at % PAGE;
            let Some(record) = l.records.get_mut(&page) else {
                return false;
            };
            let range = (at % PAGE, at % PAGE + n);
            match &mut record.next {
                Some((s, ranges)) if *s == submission => ranges.push(range),
                _ => {
                    // SAFETY: the tile's mapping; no submission runs on it now.
                    let now = unsafe { std::slice::from_raw_parts(record.tile as *const u8, PAGE) };
                    record.frozen = Some(now.into());
                    record.next = Some((submission, vec![range]));
                    record.fresh = None;
                    add(S::Declared, 1);
                }
            }
            true
        })
        .collect()
}

/// After `submission` ran: for each part of a page (`(title address, view,
/// length)`), the tile as it kept it at [`declare`] (the snapshot to compare
/// with), if it did.
pub(crate) fn declared(submission: u64, parts: &[(usize, usize, usize)]) -> Vec<Option<usize>> {
    if !EVER.load(Ordering::Acquire) {
        return vec![None; parts.len()];
    }
    let l = lazy();
    parts
        .iter()
        .map(|&(at, _, _)| {
            let record = l.records.get(&(at - at % PAGE))?;
            match (&record.next, &record.frozen) {
                // The box stays until `finished`, on this thread.
                (Some((s, _)), Some(f)) if *s == submission => Some(f.as_ptr() as usize),
                _ => None,
            }
        })
        .collect()
}

/// What `submission` wrote of the declared page at `at`, inside what it
/// declared.
pub(crate) fn found(submission: u64, at: usize, mut wrote: Mask) {
    let mut l = lazy();
    let Some(record) = l.records.get_mut(&(at - at % PAGE)) else {
        return;
    };
    let Some((s, ranges)) = &record.next else {
        return;
    };
    if *s != submission {
        return;
    }
    // Only inside the declared parts: elsewhere the tile took the title's bytes.
    let mut inside = mask();
    for &(a, b) in ranges {
        for k in a..b {
            inside[k / 64] |= 1 << (k % 64);
        }
    }
    for (m, i) in wrote.iter_mut().zip(inside.iter()) {
        *m &= i;
    }
    match &mut record.fresh {
        Some(fresh) => {
            for (a, b) in fresh.iter_mut().zip(wrote.iter()) {
                *a |= b;
            }
        }
        None => record.fresh = Some(wrote),
    }
}

/// After `submission`'s writes were looked at: the pages it declared owe
/// what they owed and what it wrote, now from their tiles. Pages that can
/// no longer be hidden are merged now.
pub(crate) fn finished(submission: u64) {
    if !EVER.load(Ordering::Acquire) {
        return;
    }
    let mut again = Vec::new();
    {
        let mut l = lazy();
        let keys: Vec<usize> = l
            .records
            .iter()
            .filter(|(_, r)| r.next.as_ref().is_some_and(|(s, _)| *s == submission))
            .map(|(&k, _)| k)
            .collect();
        for key in keys {
            let Some(record) = l.records.get_mut(&key) else {
                continue;
            };
            record.next = None;
            record.frozen = None;
            let fresh = record.fresh.take();
            match (&mut record.owed, fresh) {
                (Some((mask, hidden)), Some(fresh)) => {
                    for (a, b) in mask.iter_mut().zip(fresh.iter()) {
                        *a |= b;
                    }
                    let (root, shown) = (record.root, hidden.shown.clone());
                    noted(&l, root, key, &shown);
                }
                (Some(_), None) => {}
                (None, fresh) => {
                    let (tile, root) = (record.tile, record.root);
                    l.records.remove(&key);
                    if let Some(mask) = fresh.filter(|m| m.iter().any(|&w| w != 0)) {
                        again.push(Owe {
                            at: key,
                            tile,
                            root,
                            mask,
                        });
                    }
                }
            }
        }
    }
    if again.is_empty() {
        return;
    }
    let left: Vec<(usize, usize, Mask)> = again
        .iter()
        .map(|p| (p.at, p.tile, p.mask.clone()))
        .collect();
    for ((at, tile, mask), owed) in left.into_iter().zip(owe(again)) {
        if !owed {
            write_now(at, tile, &mask);
        }
    }
}

/// Writes the bytes `mask` names from the tile into title memory now, as an
/// eager copy-back does.
fn write_now(at: usize, tile: usize, mask: &[u64; PAGE / 64]) {
    let set = |k: usize| mask[k / 64] >> (k % 64) & 1 != 0;
    let mut k = 0;
    while k < PAGE {
        if !set(k) {
            k += 1;
            continue;
        }
        let from = k;
        while k < PAGE && set(k) {
            k += 1;
        }
        crate::watch::writing(at + from, k - from);
        crate::host::write_mapped((at + from) as *mut u8, (tile + from) as *const u8, k - from);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        PAGE_READWRITE, UnmapViewOfFile,
    };

    /// Two views of one physical page, told as the runtime tells them.
    fn two_views() -> (usize, usize) {
        // SAFETY: a pagefile section of a page, mapped read-write twice.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE).Value as usize;
            assert!(one != 0 && two != 0);
            crate::watch::title_memory_view(one, PAGE, section as usize, 0);
            crate::watch::title_memory_view(two, PAGE, section as usize, 0);
            (one, two)
        }
    }

    fn drop_views(views: &[usize]) {
        for &view in views {
            crate::watch::title_memory_remapped(view, PAGE);
            crate::watch::title_memory_view_gone(view, PAGE);
            // SAFETY: our view.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
    }

    /// A tile whose bytes `bytes` the GPU set to `value`, and their mask.
    fn tile_with(bytes: &[usize], value: u8) -> (Vec<u8>, Mask) {
        let mut tile = vec![0u8; PAGE];
        let mut m = super::mask();
        for &k in bytes {
            tile[k] = value;
            m[k / 64] |= 1 << (k % 64);
        }
        (tile, m)
    }

    fn owe_one(at: usize, tile: &[u8], mask: Mask) -> bool {
        owe(vec![Owe {
            at,
            tile: tile.as_ptr() as usize,
            root: 1,
            mask,
        }])[0]
    }

    fn read(at: usize) -> u8 {
        // SAFETY: a byte of a view of ours; the handler opens it if hidden.
        unsafe { (at as *const u8).read_volatile() }
    }

    fn write(at: usize, v: u8) {
        // SAFETY: as above.
        unsafe { (at as *mut u8).write_volatile(v) }
    }

    /// The title reads a page the GPU wrote through another view of it: the
    /// fault puts the GPU's bytes in, and leaves the title's own alone.
    #[test]
    fn a_read_through_any_view_faults_the_gpu_bytes_in() {
        let (one, two) = two_views();
        write(one + 7, 0x55); // the title's, not the GPU's
        let (tile, mask) = tile_with(&[3, 9000], 0xaa);
        assert!(owe_one(one, &tile, mask));
        assert!(!crate::watch::tests_writable(two), "every view is hidden");
        assert_eq!(read(two + 9000), 0xaa);
        assert_eq!(read(one + 3), 0xaa);
        assert_eq!(read(one + 7), 0x55);
        assert!(crate::watch::tests_writable(one) && crate::watch::tests_writable(two));
        assert!(!owing());
        drop_views(&[one, two]);
    }

    /// A title write to an owed page lands after the GPU's bytes, over them.
    #[test]
    fn a_title_write_lands_over_the_gpu_bytes() {
        let (one, two) = two_views();
        let (tile, mask) = tile_with(&[100, 101], 0xbb);
        assert!(owe_one(one, &tile, mask));
        write(two + 100, 0x11);
        assert_eq!(read(one + 100), 0x11);
        assert_eq!(read(one + 101), 0xbb);
        drop_views(&[one, two]);
    }

    /// Our reads through the kernel, a remap and the kernel's writes put
    /// pages back first.
    #[test]
    fn our_own_accesses_put_pages_back_first() {
        use std::io::{Read, Seek, Write};
        let (one, two) = two_views();
        let (tile, mask) = tile_with(&[5], 0xcc);
        assert!(owe_one(one, &tile, mask));
        let mut byte = [0u8];
        assert!(crate::host::read_into((two + 5) as *const u8, &mut byte));
        assert_eq!(byte[0], 0xcc);
        // A remap puts the bytes into the old page.
        let (tile, mask) = tile_with(&[8], 0xee);
        assert!(owe_one(one, &tile, mask));
        crate::watch::title_memory_remapped(two, PAGE);
        assert!(!owing());
        assert_eq!(read(one + 8), 0xee);
        // A kernel read into a hidden page.
        let (tile, mask) = tile_with(&[6], 0xdd);
        assert!(owe_one(one, &tile, mask));
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"disk").unwrap();
        file.rewind().unwrap();
        crate::watch::writing(one + 200, 4);
        assert!(!owing());
        // SAFETY: 4 bytes of our view.
        let into = unsafe { std::slice::from_raw_parts_mut((one + 200) as *mut u8, 4) };
        assert!(file.read(into).is_ok_and(|n| n == 4));
        assert_eq!(read(two + 6), 0xdd);
        assert_eq!(read(two + 200), b'd');
        // Pinned for the kernel: not hidden meanwhile.
        let (tile, mask) = tile_with(&[9], 0xfe);
        assert!(!owe_one(two, &tile, mask));
        crate::watch::tests_unpin(one, PAGE);
        crate::watch::tests_unpin(two, PAGE);
        drop_views(&[one, two]);
    }

    /// A guard's check of a hidden page leaves it hidden.
    #[test]
    fn guards_do_not_open_a_hidden_page() {
        let (one, two) = two_views();
        let (tile, mask) = tile_with(&[1], 0x77);
        assert!(owe_one(one, &tile, mask));
        crate::watch::tests_guard(one, one + PAGE);
        crate::watch::tests_guard(two, two + PAGE);
        assert!(!crate::watch::tests_readable(one) && !crate::watch::tests_readable(two));
        assert_eq!(read(one + 1), 0x77);
        assert!(crate::watch::tests_readable(one) && crate::watch::tests_readable(two));
        drop_views(&[one, two]);
    }

    /// A submission declares an owed page: a fault while it runs puts back
    /// the bytes as they were, and what it writes is owed once it finished.
    #[test]
    fn a_declared_page_keeps_its_bytes_and_owes_the_next_ones() {
        let (one, two) = two_views();
        let (mut tile, mask) = tile_with(&[10], 0x10);
        assert!(owe_one(one, &tile, mask));
        let submission = 1 << 40;
        let parts = [(one, tile.as_ptr() as usize, PAGE)];
        assert_eq!(declare(submission, &parts), [true]);
        // It runs: the GPU writes the tile.
        tile[10] = 0x20;
        tile[20] = 0x21;
        assert_eq!(read(two + 10), 0x10, "the bytes owed before it ran");
        assert_eq!(read(one + 20), 0);
        let frozen = declared(submission, &parts)[0].expect("kept");
        let mut m = super::mask();
        // SAFETY: the kept page.
        let old = unsafe { std::slice::from_raw_parts(frozen as *const u8, PAGE) };
        assert!(differ(&mut m, 0, &tile, old));
        found(submission, one, m);
        finished(submission);
        assert!(owing());
        assert_eq!(read(one + 20), 0x21);
        assert_eq!(read(two + 10), 0x20);
        assert!(!owing());
        drop_views(&[one, two]);
    }
}
