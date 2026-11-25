//! Image instructions: the choices around `Lower::image_with`.

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Term, V, pred};
use crate::ir::Inst;
use crate::resource::Source;

impl<'a> Lower<'a> {
    /// An image instruction. Of two samplers given in the code that a branch
    /// picks one of, each is sampled with, the result taken by the edge the
    /// join was entered by (DexNom ddb7332).
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let sampler = self.descriptors.get(&((b, i), true)).copied();
        if let Some(Source::Phi { block, args }) = sampler
            && let [first @ Source::Constant(x), second @ Source::Constant(y)] = &args[..]
            && x != y
            && let [left, right] = self.x.p.blocks[*block].preds[..]
        {
            if self.block[left].is_none() {
                return self.image_with(inst, b, i, Some(second));
            }
            if self.block[right].is_none() {
                return self.image_with(inst, b, i, Some(first));
            }
            let edge = |l: &Self, q: usize| {
                let tail = l.tail[q]?;
                Some(l.edges.get(&(q, *block)).copied().unwrap_or(tail))
            };
            // A predecessor not lowered yet (a loop's back edge) keeps the
            // first sampler (`handle`).
            if let (Some(left), Some(right), Some(join)) =
                (edge(self, left), edge(self, right), self.block[*block])
                && left != right
            {
                let chosen = self.s.m.phi(join, self.s.t.i1);
                let (yes, no) = (self.s.m.bool(true), self.s.m.bool(false));
                self.s.m.set_phi(chosen, vec![(yes, left), (no, right)]);
                let written = |l: &Self| -> Vec<Option<V>> {
                    let get = |&(_, v)| match l.vals.get(&v) {
                        Some(&Val::I32(x)) => Some(x),
                        _ => None,
                    };
                    inst.writes.iter().map(get).collect()
                };
                self.image_with(inst, b, i, Some(first))?;
                let firsts = written(self);
                self.image_with(inst, b, i, Some(second))?;
                for ((&(_, v), a), b) in inst.writes.iter().zip(firsts).zip(written(self)) {
                    if let (Some(a), Some(b)) = (a, b) {
                        let x = self.s.m.select(self.cur, chosen, a, b);
                        self.vals.insert(v, Val::I32(x));
                    }
                }
                return Ok(());
            }
        }
        self.image_with(inst, b, i, sampler)
    }

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
