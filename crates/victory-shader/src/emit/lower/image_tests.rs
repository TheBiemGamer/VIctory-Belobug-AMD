//! Image lowerings read back through `dxc -dumpbin` (`VICTORY_PC_DXC`, else
//! the Windows SDK's); skipped without one.

use super::tests::{lower_with, root};
use crate::native::{ResourceType, RootType, UsedResource};

const S_ENDPGM: u32 = 0xBF81_0000;

fn dxc() -> Option<std::path::PathBuf> {
    let dxc = std::env::var_os("VICTORY_PC_DXC").map(std::path::PathBuf::from);
    #[cfg(windows)]
    let dxc = dxc.or_else(|| {
        Some(super::super::validator::Validator::default_path()?.with_file_name("dxc.exe"))
    });
    dxc.filter(|p| p.exists())
}

fn dump(dxc: &std::path::Path, dxil: &[u8], tag: &str) -> String {
    let file =
        std::env::temp_dir().join(format!("victory-image-{}-{tag}.dxil", std::process::id()));
    std::fs::write(&file, dxil).unwrap();
    let out = std::process::Command::new(dxc)
        .arg("-dumpbin")
        .arg(&file)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&file);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A table of 2D textures at s0, its first descriptor loaded into s[12:19].
fn texture_table() -> (Vec<u32>, Vec<crate::native::RootParam>, Vec<UsedResource>) {
    let words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302, // s_load_dwordx8 s[12:19], s[4:5], 0
        0xFA00_0000,
    ];
    let used = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    };
    (words, vec![root(RootType::Table, 0, 0)], vec![used])
}

/// MIMG `op` of v[data:data+3] at addresses from v[addr] on, of s[12:19].
fn mimg(op: u32, dim: u32, data: u32, addr: u32) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | 1 << 12 | 0xF << 8 | dim << 3,
        addr | data << 8 | 3 << 16,
    ]
}

const IMAGE_LOAD: u32 = 0x00;
const IMAGE_GET_RESINFO: u32 = 0x0E;

/// `ds_write_b32 v0, v[data]`: keeps a result live.
fn keep(data: u32) -> [u32; 2] {
    [0xD834_0000, data << 8]
}

#[test]
fn msaa_loads_never_follow_a_size_query_in_one_block() {
    let Some(dxc) = dxc() else {
        return eprintln!("no dxc; DXIL check skipped");
    };
    for dim in [6, 7] {
        let (mut words, layout, used) = texture_table();
        for r in 1..=4 {
            words.push(0x7E00_0280 | r << 17); // v_mov_b32 v[r], 0
        }
        words.extend(mimg(IMAGE_GET_RESINFO, dim, 8, 1));
        words.extend(mimg(IMAGE_LOAD, dim, 12, 1));
        words.extend(mimg(IMAGE_GET_RESINFO, dim, 16, 1));
        words.extend(mimg(IMAGE_LOAD, dim, 20, 1));
        for r in [8, 12, 16, 20] {
            words.extend(keep(r));
        }
        words.push(S_ENDPGM);
        let dxil = lower_with(&words, layout, used).unwrap().dxil;
        let text = dump(&dxc, &dxil, &format!("msaa-{dim}"));
        let (mut sized, mut loads, mut queries) = (false, 0, 0);
        for line in text.lines().map(str::trim) {
            if line.starts_with("; <label>") || line.starts_with("define ") {
                sized = false;
            } else if line.contains("call %dx.types.Dimensions @dx.op.getDimensions(") {
                sized = true;
                queries += 1;
            } else if line.contains("call %dx.types.ResRet.f32 @dx.op.textureLoad.") {
                assert!(
                    !sized,
                    "a textureLoad after getDimensions in one block:\n{text}"
                );
                loads += 1;
            }
        }
        assert_eq!((queries, loads), (2, 2), "{text}");
    }
}

#[test]
fn a_level_past_the_last_has_zero_sizes() {
    let Some(dxc) = dxc() else {
        return eprintln!("no dxc; DXIL check skipped");
    };
    // 2D, 3D and cube: the level's sizes are selected against 0 by
    // `level >= levels`; the level count is the query's own.
    for dim in [1, 2, 3] {
        let (mut words, layout, used) = texture_table();
        words.extend([0x7E02_02FF, 5]); // v_mov_b32 v1, 5
        words.extend(mimg(IMAGE_GET_RESINFO, dim, 8, 1));
        for r in 8..12 {
            words.extend(keep(r));
        }
        words.push(S_ENDPGM);
        let dxil = lower_with(&words, layout, used).unwrap().dxil;
        let text = dump(&dxc, &dxil, &format!("levels-{dim}"));
        let call = text
            .lines()
            .find(|l| l.contains("call %dx.types.Dimensions @dx.op.getDimensions("))
            .unwrap_or_else(|| panic!("{text}"));
        let d = call.trim().split(' ').next().unwrap();
        let levels = format!("extractvalue %dx.types.Dimensions {d}, 3");
        let n = text
            .lines()
            .find(|l| l.contains(&levels))
            .unwrap_or_else(|| panic!("{text}"));
        let n = n.trim().split(' ').next().unwrap();
        let past = format!("icmp uge i32 5, {n}");
        let p = text
            .lines()
            .find(|l| l.contains(&past))
            .unwrap_or_else(|| panic!("{text}"));
        let p = p.trim().split(' ').next().unwrap();
        let zeroed = format!("select i1 {p}, i32 0, ");
        assert_eq!(text.matches(&zeroed).count(), 3, "{text}");
    }
    // An MSAA view's query names no level: nothing to compare.
    let (mut words, layout, used) = texture_table();
    words.extend(mimg(IMAGE_GET_RESINFO, 6, 8, 1));
    words.extend(keep(8));
    words.push(S_ENDPGM);
    let dxil = lower_with(&words, layout, used).unwrap().dxil;
    let text = dump(&dxc, &dxil, "levels-msaa");
    assert!(!text.contains("icmp uge"), "{text}");
}
