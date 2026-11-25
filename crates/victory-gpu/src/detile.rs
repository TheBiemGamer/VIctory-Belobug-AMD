//! Title textures detiled on the host GPU (`VICTORY_GPU_DETILE=1`): the
//! texture's tiled memory is copied as it is into an upload buffer, and a
//! compute shader writes its elements as linear rows into a buffer laid out
//! as the host texture's copyable footprints, which `CopyTextureRegion`
//! copies into the texture (block-compressed textures cannot be written as
//! unordered access).
//!
//! The shader reads each element's address from tables made once per
//! texture by [`victory_formats::texture::Layout::addresses`], the address
//! mapping the CPU detiling walks, so both give the same bytes.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{Host, transition};

/// Whether title textures are detiled on the GPU (`VICTORY_GPU_DETILE=1`).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_DETILE").is_some_and(|v| v == "1"))
}

/// One thread per dword of a linear row: (`SV_DispatchThreadID.x`, row,
/// depth slice) of one subresource.
const DETILE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=9, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Subresource : register(b0) {
    uint out_at;       // byte offset of its footprint in `rows`
    uint row_pitch;
    uint depth_pitch;
    uint dwords;       // dwords of one row's elements, rounded up
    uint width;        // elements
    uint height;
    uint shift;        // log2 of the element's bytes
    uint rows_at;      // index in `tables` of its first row's address
    uint columns_at;   // index in `tables` of its first column's address
};
ByteAddressBuffer tables : register(t0);
ByteAddressBuffer tiled : register(t1);
RWByteAddressBuffer rows : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= dwords) return;
    uint row = tables.Load((rows_at + id.z * height + id.y) * 4u);
    uint value = 0u;
    if (shift >= 2u) {
        uint byte = id.x * 4u;
        uint column = tables.Load((columns_at + (byte >> shift)) * 4u);
        value = tiled.Load(row + column + (byte & ((1u << shift) - 1u)));
    } else {
        uint per = 4u >> shift;
        uint bits = 8u << shift;
        uint mask = (1u << bits) - 1u;
        for (uint k = 0u; k < per; k++) {
            uint x = id.x * per + k;
            if (x < width) {
                uint at = row + tables.Load((columns_at + x) * 4u);
                uint word = tiled.Load(at & ~3u);
                value |= ((word >> ((at & 3u) * 8u)) & mask) << (k * bits);
            }
        }
    }
    rows.Store(out_at + id.z * depth_pitch + id.y * row_pitch + id.x * 4u, value);
}
"#;

/// The detiling pipeline, and the buffer it writes linear rows into: one
/// for all textures, each needing it only from its dispatch to its copies.
pub(crate) struct Detiler {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    out: Mutex<Option<(ID3D12Resource, u64)>>,
}

// SAFETY: D3D12 interfaces are free-threaded; `out` is behind its lock.
unsafe impl Send for Detiler {}
// SAFETY: as above.
unsafe impl Sync for Detiler {}

/// What one texture's dispatches need: the shader's constants and thread
/// groups per subresource, and where its tables and tiled bytes are.
pub(crate) struct Plan {
    subresources: Vec<([u32; 9], [u32; 3])>,
    /// Offsets in the upload buffer (its slot's added): the tables, then
    /// the tiled bytes.
    pub(crate) tables_at: u64,
    pub(crate) tiled_at: u64,
    /// Bytes of the upload buffer.
    pub(crate) upload: u64,
    /// Bytes of linear rows written.
    pub(crate) out: u64,
}

/// The tables and dispatches detiling `layout` (`tiled` bytes of its
/// memory read) into `footprints` (counted from 0), mip `skip` and smaller
/// of every slice; `None` for what the shader cannot address (offsets past
/// 32 bits, a subresource the CPU copies refuse), left to the CPU.
pub(crate) fn plan(
    layout: &victory_formats::texture::Layout,
    tiled: u64,
    mips: u32,
    skip: u32,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
) -> Option<(Vec<u32>, Plan)> {
    // Signed 32-bit room: `ByteAddressBuffer` offsets and dispatch sizes.
    const MOST: u64 = 1 << 31;
    let n = layout.desc.element_bytes;
    let mut tables = Vec::new();
    let mut subresources = Vec::with_capacity(footprints.len());
    let mut out = 0u64;
    for (i, (footprint, &count)) in footprints.iter().zip(rows).enumerate() {
        let (mip, slice) = (i as u32 % mips + skip, i as u32 / mips);
        let m = layout.mips.get(mip as usize)?;
        let f = footprint.Footprint;
        let bytes = u64::from(m.width) * u64::from(n);
        let depth_pitch = u64::from(f.RowPitch) * u64::from(count);
        if u64::from(m.height) > u64::from(count) || bytes > u64::from(f.RowPitch) {
            return None;
        }
        let (row_offsets, column_offsets) = layout.addresses(mip, slice).ok()?;
        let reach = row_offsets.iter().max()? + column_offsets.iter().max()? + u64::from(n);
        let end = footprint.Offset + depth_pitch * u64::from(m.depth);
        if reach > tiled || end > MOST || tables.len() as u64 > MOST {
            return None;
        }
        out = out.max(end);
        let rows_at = tables.len() as u32;
        tables.extend(row_offsets.iter().map(|&o| o as u32));
        let columns_at = tables.len() as u32;
        tables.extend(column_offsets.iter().map(|&o| o as u32));
        let dwords = bytes.div_ceil(4) as u32;
        subresources.push((
            [
                footprint.Offset as u32,
                f.RowPitch,
                depth_pitch as u32,
                dwords,
                m.width,
                m.height,
                n.trailing_zeros(),
                rows_at,
                columns_at,
            ],
            [dwords.div_ceil(64), m.height, m.depth],
        ));
    }
    let tables_bytes = (tables.len() as u64 * 4).next_multiple_of(256);
    let plan = Plan {
        subresources,
        tables_at: 0,
        tiled_at: tables_bytes,
        // Whole dwords: the shader reads the last element's.
        upload: tables_bytes + tiled.next_multiple_of(4),
        out: out.next_multiple_of(4),
    };
    Some((tables, plan))
}

impl Detiler {
    pub(crate) fn new(host: &Host) -> windows::core::Result<Self> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                DETILE_HLSL.as_ptr().cast(),
                DETILE_HLSL.len(),
                windows::core::s!("detile"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        let text = |b: &ID3DBlob| {
            // SAFETY: the blob's own bytes.
            unsafe {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            }
            .to_vec()
        };
        if let Err(e) = compiled {
            if let Some(errors) = errors {
                tracing::warn!(
                    "texture detiling: {}",
                    String::from_utf8_lossy(&text(&errors))
                );
            }
            return Err(e);
        }
        let bytecode = text(&code.ok_or_else(windows::core::Error::empty)?);
        let bytecode = bytecode.as_slice();
        // SAFETY: the bytecode embeds `[RootSignature(RS)]`.
        let root_sig: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytecode)? };
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.as_ptr().cast(),
                BytecodeLength: bytecode.len(),
            },
            ..Default::default()
        };
        // SAFETY: a whole compute pipeline description.
        let pso = unsafe { host.device.CreateComputePipelineState(&desc) };
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
        Ok(Self {
            root_sig,
            pso: pso?,
            out: Mutex::new(None),
        })
    }

    /// Makes the linear rows' buffer hold `bytes` at least.
    pub(crate) fn reserve(&self, host: &Host, bytes: u64) -> bool {
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        if out.as_ref().is_some_and(|(_, size)| *size >= bytes) {
            return true;
        }
        let size = bytes.next_power_of_two().max(4 << 20);
        let Some(buffer) = host.uav_buffer(size) else {
            return false;
        };
        crate::host::name(&buffer, || format!("detiled rows {size:#x}"));
        if let Some((old, _)) = out.replace((buffer, size)) {
            crate::host::retire(old);
        }
        true
    }

    /// Records the detiling of the tiled bytes in `upload` (where `plan`
    /// places them) and the copies of the rows into `texture` (in
    /// `COPY_DEST`).
    pub(crate) fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        upload: &ID3D12Resource,
        plan: &Plan,
        footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        texture: &ID3D12Resource,
    ) {
        let Some(out) = self
            .out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|o| o.0.clone())
        else {
            return;
        };
        // SAFETY: an open list; the buffers and texture are live, and the
        // rows' buffer is in the common state between uses.
        unsafe {
            let address = upload.GetGPUVirtualAddress();
            list.ResourceBarrier(&[transition(
                &out,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.SetComputeRootSignature(&self.root_sig);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootShaderResourceView(1, address + plan.tables_at);
            list.SetComputeRootShaderResourceView(2, address + plan.tiled_at);
            list.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
            for (constants, [x, y, z]) in &plan.subresources {
                list.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
                list.Dispatch(*x, *y, *z);
            }
            list.ResourceBarrier(&[transition(
                &out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            for (i, footprint) in footprints.iter().enumerate() {
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: a borrowed pointer the location never releases.
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: i as u32,
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: as above.
                    pResource: std::mem::transmute_copy(&out),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: *footprint,
                    },
                };
                list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
            list.ResourceBarrier(&[transition(
                &out,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }
}
