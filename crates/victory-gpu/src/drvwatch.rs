//! The graphics driver's errors, seen as the system records them (after
//! fxceless a253706).
//!
//! On fxceless's NVIDIA runs every device loss was preceded by entries of
//! the card's driver in the system's event log, 2.7 to 55 seconds ahead,
//! and nothing in the layer saw them. Here the layer subscribes to that
//! log: each display-driver entry is logged when the system records it, so
//! it can be laid beside what the layer did then, and counted, so a device
//! removal says how many came before it. Whether AMD's driver announces a
//! loss the same way is not known yet.
//!
//! On by default; `VICTORY_DRIVER_WATCH=0` subscribes to nothing.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows_sys::Win32::System::EventLog::{
    EVT_HANDLE, EVT_SUBSCRIBE_NOTIFY_ACTION, EvtSubscribe, EvtSubscribeActionDeliver,
    EvtSubscribeToFutureEvents,
};

/// Display-driver entries seen since the start.
static ERRORS: AtomicU64 = AtomicU64::new(0);

/// Display-driver entries seen since the start.
pub(crate) fn errors() -> u64 {
    ERRORS.load(Relaxed)
}

fn seen() {
    let n = ERRORS.fetch_add(1, Relaxed) + 1;
    tracing::error!(n, "the graphics driver reported an error to the system's event log");
}

unsafe extern "system" fn on_event(
    action: EVT_SUBSCRIBE_NOTIFY_ACTION,
    _context: *const core::ffi::c_void,
    _event: EVT_HANDLE,
) -> u32 {
    if action == EvtSubscribeActionDeliver {
        seen();
    }
    0
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The kernel drivers of the three vendors, and the system's own "display
/// driver stopped responding" source.
const QUERY: &str = "*[System[Provider[@Name='amdkmdag' or @Name='amdwddmg' or @Name='nvlddmkm' \
                     or @Name='igfx' or @Name='igfxn' or @Name='Display']]]";

/// Subscribes to the display drivers' entries in the system's event log,
/// once; entries from before the call are not delivered.
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("VICTORY_DRIVER_WATCH").is_some_and(|v| v == "0") {
            return;
        }
        let (channel, query) = (wide("System"), wide(QUERY));
        // SAFETY: NUL-terminated strings that outlive the call; the callback
        // touches only statics; the subscription is kept for the process.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        if subscription == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "no subscription to the system's event log: graphics driver errors are not seen as they happen"
            );
        } else {
            tracing::info!(
                "graphics driver errors in the system's event log are watched (VICTORY_DRIVER_WATCH=0: not)"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The system accepts the channel and the query (one it cannot read is
    /// refused at once).
    #[test]
    fn the_system_takes_the_subscription() {
        let (channel, query) = (wide("System"), wide(QUERY));
        // SAFETY: as in `start`.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        assert_ne!(subscription, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: our own subscription.
        unsafe { windows_sys::Win32::System::EventLog::EvtClose(subscription) };
    }
}
