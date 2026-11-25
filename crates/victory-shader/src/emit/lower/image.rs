//! Image instructions: what the host's drivers need beyond `Lower::image`.

use super::Lower;
use crate::emit::bitcode::{Term, V, pred};

impl Lower<'_> {
    /// An MSAA load after a size query of an MSAA view in the same block
    /// starts a block of its own.
    pub(super) fn msaa_load_block(&mut self, name: &str, dim: u8) {
        // amdxc64.dll dereferences null on getDimensions before textureLoad of
        // one heap Texture2DMS(Array) in a block (xcl 425b869).
        if name == "image_load" && matches!(dim, 6 | 7) && self.msaa_sized == Some(self.cur) {
            let next = self.s.m.new_block();
            self.s.m.terminate(self.cur, Term::Br(next));
            self.cur = next;
        }
    }

    /// Whether `mip` is past the last level of the size query `d`: the
    /// console then reads 0 for every size and keeps the level count.
    pub(super) fn past_last_level(&mut self, c: usize, d: V, mip: V) -> V {
        // AMD's driver (RDNA4) answers with the sizes of a level it clamped
        // (xcl ea088a5).
        let levels = self.s.m.extract(c, d, 3);
        self.s.m.cmp(c, pred::IUGE, mip, levels)
    }
}
