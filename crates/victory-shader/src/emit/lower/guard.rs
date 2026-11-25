//! Buffer accesses through descriptors taken from the resource heap itself.

use super::Lower;
use crate::emit::bitcode::{V, pred};
use crate::native::RootType;
use crate::resource::Source;

impl Lower<'_> {
    /// The heap index to read a buffer view at, for a descriptor `src` taken
    /// from the resource heap by an index from data: `index`, or the host's
    /// null raw buffer view ([`crate::emit::NULL_BUFFER_VIEW`], the UAV below
    /// it for unordered access) where the stride table holds 0 for the slot
    /// (no buffer view there). A texture descriptor read as a buffer one is a
    /// wild address and faults the device.
    pub(super) fn buffer_guard(&mut self, src: &Source, index: V, sampler: bool, props: (u32, u32)) -> V {
        // Typed, raw and structured buffers (DXIL resource kinds 10-12); off
        // where heap indices are not limited (test fixtures), as the clamp is.
        if sampler || !(10..=12).contains(&(props.0 & 0xFF)) || !crate::emit::limits_indices() {
            return index;
        }
        let heap = src.leaves().iter().any(|l| match l {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => self
                .scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|rp| rp.root == RootType::ResourceHeap),
            _ => false,
        });
        if !heap {
            return index;
        }
        let c = self.cur;
        let word = self.slot_dword(index, 0);
        let zero = self.s.m.i32(0);
        let none = self.s.m.cmp(c, pred::EQ, word, zero);
        let null = crate::emit::NULL_BUFFER_VIEW - u32::from(props.0 & 0x1000 != 0);
        let null = self.s.m.i32(null);
        self.s.m.select(c, none, null, index)
    }
}
