//! Repository maintenance. Run with `cargo xtask <command>`.

mod perf;
mod policy;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
    /// Summarise the frame-rate lines and warnings of run logs (stderr of
    /// `victory run`); with two logs, compare their steady frames.
    Perf {
        #[arg(required = true, num_args = 1..=2)]
        logs: Vec<PathBuf>,
        /// Leave out each log's first N frame-rate lines (menus, loading).
        #[arg(long, default_value_t = 0)]
        skip: usize,
        /// Keep only each log's last N frame-rate lines (after --skip).
        #[arg(long)]
        last: Option<usize>,
    },
    /// Compare two `VICTORY_PROFILE` files function by function.
    ProfileDiff { before: PathBuf, after: PathBuf },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::Perf { logs, skip, last } => perf::run(&logs, perf::Window { skip, last }),
        Cmd::ProfileDiff { before, after } => perf::profile_diff(&before, &after),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
