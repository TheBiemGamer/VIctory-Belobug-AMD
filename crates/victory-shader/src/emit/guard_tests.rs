//! Buffer accesses through descriptors the program takes from the resource
//! heap by index read the host's null buffer view where the slot holds no
//! buffer view (`lower::guard`).

use super::shader::OPS;
use super::{HwStage, NULL_BUFFER_VIEW, NativeProgram, lower};
use crate::native::{ComputeIo, ResourceType, RootParam, RootType, StageIo, UsedResource};

/// A compute program reading heap slot 1 (scalar and vector loads) and
/// writing heap slot 2 (a store and an atomic), slots picked by user data
/// s6 (the heap); s4 a table whose raw UAV takes what it read.
fn heap_program() -> NativeProgram {
    let words: &[u32] = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (the table's UAV)
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0403,
        0xFA00_0020, // s_load_dwordx4 s[16:19], s[6:7], 0x20 (heap slot 1)
        0xF408_0603,
        0xFA00_0040, // s_load_dwordx4 s[24:27], s[6:7], 0x40 (heap slot 2)
        0xF420_0508,
        0xFA00_0008, // s_buffer_load_dword s20, s[16:19], 0x8
        0x7E08_0280, // v_mov_b32 v4, 0
        0xE030_1000,
        0x8004_0504, // buffer_load_dword v5, v4, s[16:19], 0 offen
        0x7E00_0214, // v_mov_b32 v0, s20
        0x7E02_0305, // v_mov_b32 v1, v5
        0xE070_1000,
        0x8006_0004, // buffer_store_dword v0, v4, s[24:27], 0 offen
        0xE0C8_1000,
        0x8006_0004, // buffer_atomic_add v0, v4, s[24:27], 0 offen
        0xE074_1000,
        0x8002_0004, // buffer_store_dwordx2 v[0:1], v4, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 6,
        layout: vec![
            param(RootType::Constants, 4, 0, 0),
            param(RootType::Table, 1, 4, 1),
            param(RootType::ResourceHeap, 1, 6, 2),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 7 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` lowered, heap indices limited (`limits`) or not, and the DXIL
/// ops it called.
fn lowered(native: &NativeProgram, limits: bool) -> (lower::Lowered, Vec<String>) {
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
    let io = crate::io::raise(&p, native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    OPS.with(|o| o.borrow_mut().clear());
    super::LIMITS_IN_TESTS.with(|on| on.set(limits));
    let l = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        psv: None,
        console_cf: false,
        per_lane_only: false,
    });
    super::LIMITS_IN_TESTS.with(|on| on.set(false));
    (l.unwrap(), OPS.with(|o| o.take()))
}

/// Each handle of a heap-indexed buffer reads its slot's dword of the stride
/// table and picks the null SRV (reads) or UAV (writes, atomics) where it is
/// 0; table descriptors and unlimited fixtures are left as they were.
#[test]
fn heap_indexed_buffers_read_the_stride_table_and_pick_a_null_view() {
    let native = heap_program();
    let count = |ops: &[String], what: &str| ops.iter().filter(|o| o.as_str() == what).count();
    let (plain, plain_ops) = lowered(&native, false);
    assert!(!plain.stride_table, "no guard without limited indices");
    let (guarded, ops) = lowered(&native, true);
    assert!(guarded.stride_table, "the guard reads the stride table");
    // Slot 1's loads share a handle; slot 2's store and atomic take two
    // (an atomic needs an integer handle); the table's UAV takes none.
    let handles = count(&ops, "createHandleFromHeap");
    assert_eq!(handles, count(&plain_ops, "createHandleFromHeap"));
    let table_reads = count(&ops, "rawBufferLoad.f32") - count(&plain_ops, "rawBufferLoad.f32");
    assert_eq!(table_reads, handles - 1, "{ops:?}");

    let dxc = std::env::var_os("VICTORY_PC_DXC").map(std::path::PathBuf::from);
    #[cfg(windows)]
    let dxc = dxc.or_else(|| Some(super::validator::Validator::default_path()?.with_file_name("dxc.exe")));
    let Some(dxc) = dxc.filter(|p| p.exists()) else {
        return eprintln!("no dxc; DXIL check skipped");
    };
    let file = std::env::temp_dir().join(format!("victory-buffer-guard-{}.dxil", std::process::id()));
    std::fs::write(&file, &guarded.dxil).unwrap();
    let out = std::process::Command::new(&dxc).arg("-dumpbin").arg(&file).output().unwrap();
    let _ = std::fs::remove_file(&file);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let picks = |null: u32| {
        text.lines()
            .filter(|l| l.contains("= select i1 ") && l.contains(&format!(", i32 {null}, i32 ")))
            .count()
    };
    assert_eq!(picks(NULL_BUFFER_VIEW), 1, "the loads' null SRV: {text}");
    assert_eq!(picks(NULL_BUFFER_VIEW - 1), handles - 2, "the writes' null UAV: {text}");
}

/// On the GPU: a slot the stride table holds 0 for (a texture here) reads
/// zeros through the null view; a slot holding a buffer view reads it.
#[cfg(windows)]
#[test]
fn a_heap_buffer_read_where_no_buffer_view_is_reads_zeros() {
    use super::host::{Dispatch, Host, Root, Stride, View};
    let native = heap_program();
    let (l, _) = lowered(&native, true);
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone()).expect("DXIL validation failed");
    }
    let shader = super::HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        stride_table: l.stride_table,
    };
    let data: Vec<u32> = (0..16).map(|k| 0xC0DE_0000 + k).collect();
    let texel = [[1.0f32, 2.0, 3.0, 4.0]];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: {e}"));
        let run = |slot1: View, stride: u32| {
            let views = [View::RawUav(16), slot1, View::RawUav(4), View::NullBuffers];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        // User data, then the stride table: slots 0 and 2 hold
                        // buffer views.
                        roots: vec![
                            (0, Root::Constants(vec![0; 8])),
                            (1, Root::View { srv: true, data: vec![4, stride, 4] }),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            out[0][..2].to_vec()
        };
        let buffer = run(View::Buffer { data: &data, stride: Stride::Raw }, 4);
        assert_eq!(buffer, [data[2], data[0]], "{label}: a buffer view is read");
        let texture = run(View::Texture(1, 1, &texel), 0);
        assert_eq!(texture, [0, 0], "{label}: no buffer view, zeros");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
