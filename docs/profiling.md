# Measuring a run

How to capture runs that can be compared, and the tools that read them.

## The build

```powershell
cargo build --profile diag -p victory --features diagnostics
```

The `diag` build logs the drop reasons and GPU times the tools below read.
Compare builds of the same profile only.

## A run worth comparing

1. Start from a warm cache: run the same save once first, until the log's
   frame-rate lines say `(0 being made)`. A first launch makes pipelines for
   minutes, at a fraction of the frame rate.
2. Same save, same spot, same switches, same window size. Stand still, or
   repeat one walk.
3. Keep the log (it goes to stderr). Capture it through `cmd`, which writes
   the lines as they are (Windows PowerShell 5.1's own `2>` can wrap and
   decorate them):

   ```powershell
   $env:VICTORY_PROFILE = 'profile.txt'   # optional: where the submitting thread's time goes
   cmd /c "target\diag\victory.exe run <title.exe> --execute --console series-x 2> run.log"
   ```

4. Stay in the world for two or three minutes after the last pipeline is
   made: the host logs a frame-rate line every five seconds. Each line
   averages about the second before it.

## Reading it

```powershell
cargo xtask perf run.log                 # one run
cargo xtask perf before.log after.log    # two runs: steady medians side by side
cargo xtask perf --skip 20 run.log      # past the menus and loading (20 lines: 100 s)
cargo xtask perf --last 30 run.log      # the last 150 s: close the window from the world
cargo xtask profile-diff before.txt after.txt
```

`perf` reads the frame-rate lines (`… fps, … ms/frame (… preparing: copies …`).
It keeps those logged once no pipeline is being made (steady). With a warm
cache that is every line, menus and loading included, so cut the log to the
stretch in the world with `--skip` and `--last` (counted in frame-rate
lines, five seconds each). It gives the median, 10th and 90th percentile,
and the worst one-second average, of each part:

- preparing: copies, textures and bindless plans;
- GPU;
- copy-back;
- bindless records;
- draws, and draws dropped.

Its other sections:

- each part's median share of its frame;
- in a diagnostics build, the drop reasons and the GPU time per command
  (with `VICTORY_GPU_TIMING=1`);
- a count of what went wrong: slow submissions and list commands, refused
  pipelines, driver compiler crashes, device removal.

Changes under 2% are not marked better or worse.

`profile-diff` lines up the function shares of two `VICTORY_PROFILE` files
("in itself" and "with what it calls"), largest change first. Source line
numbers are left out, so a function keeps its name across builds. The file
lists the 40 and 90 busiest lines of code, so a function's total there is
a lower bound.

## Other measures

- Shader translation, CPU only: `victory translate --list <game_dir>` lists
  each program and why it does not translate. The count of
  "mesh outputs beyond the host's 32 KiB" and of the other reasons shows what
  a translator change won.
- Per-command GPU time: `VICTORY_GPU_TIMING=1` runs and times each list on
  its own. It is slower, so it does not go with a frame-rate comparison.
- Faulting reads: in a diagnostics build, the line "reads of title memory
  since" counts the user-mode copies of title memory that faulted (each
  costs an exception) and the reads sent straight to the kernel
  because their 64 KiB block faulted in the last two seconds.
- Device loss: `VICTORY_D3D12_DRED=1` names the allocation hit.
  `VICTORY_KEEP_RETIRED=1` frees nothing retired. If DRED still says
  "inside freed" with it set, the access was a wild one, not a late use of a
  freed object.
- Driver errors: every build logs each display-driver entry the system
  writes to its event log as it is written ("the graphics driver reported
  an error"), and a device removal says how many came before it. On
  fxceless's NVIDIA runs they came 2.7 to 55 s before every loss.
  `VICTORY_DRIVER_WATCH=0` turns the watch off.
