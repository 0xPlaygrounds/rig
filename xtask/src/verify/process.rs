//! Serial subprocesses with visible output, logs, heartbeat, and cancellation.
use super::*;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    process::Stdio,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static PROBES: Mutex<BTreeMap<u32, (String, Instant)>> = Mutex::new(BTreeMap::new());
pub(super) fn install_interrupt_handler() -> Result<()> {
    ctrlc::set_handler(|| {
        INTERRUPTED.store(true, Ordering::SeqCst);
        // Version/metadata probes also own process groups. Cancel them even
        // while wait_with_output is draining pipes or waiting on Cargo locks.
        for pid in PROBES.lock().unwrap_or_else(|e| e.into_inner()).keys() {
            stop_group(*pid);
        }
    })
    .map_err(|e| invalid(format!("cannot install interruption handler: {e}")))?;
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(Duration::from_secs(15));
            for (label, started) in PROBES.lock().unwrap_or_else(|e| e.into_inner()).values() {
                if started.elapsed() >= Duration::from_secs(15) {
                    println!(
                        "BUSY {label}: {:.1}s elapsed; planning/preflight diagnostics on console",
                        started.elapsed().as_secs_f64()
                    );
                    let _ = std::io::stdout().flush();
                }
            }
        }
    });
    Ok(())
}

struct Probe(u32);
impl Drop for Probe {
    fn drop(&mut self) {
        PROBES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
    }
}
pub(super) fn capture(root: &Path, program: &str, args: &[&str]) -> Result<std::process::Output> {
    if interrupted() {
        return Err(invalid("verification interrupted"));
    }
    std::io::stdout().flush()?;
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd.spawn()?;
    let pid = child.id();
    let _probe = Probe(pid);
    PROBES.lock().unwrap_or_else(|e| e.into_inner()).insert(
        pid,
        (
            format!("{program} {}", args.first().unwrap_or(&"")),
            Instant::now(),
        ),
    );
    if interrupted() {
        stop_group(pid);
    }
    let result = child.wait_with_output();
    if result.is_err() {
        stop_group(pid);
    }
    if interrupted() {
        return Err(invalid("verification interrupted"));
    }
    Ok(result?)
}

pub(super) fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

struct Running(std::process::Child);
impl Drop for Running {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            stop(&mut self.0);
            let _ = self.0.wait();
        }
    }
}
fn stop_group(pid: u32) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }
}
fn stop(child: &mut std::process::Child) {
    stop_group(child.id());
    let _ = child.kill();
}

pub(super) fn run(root: &Path, step: &Step, log: &Path) -> Result<std::process::Output> {
    if interrupted() {
        return Err(invalid("verification interrupted"));
    }
    let mut cmd = execute::command(root, step);
    let stdout_path = PathBuf::from(format!("{}.stdout", log.display()));
    let stderr_path = PathBuf::from(format!("{}.stderr", log.display()));
    let stdout = File::create(&stdout_path)?;
    let stderr = File::create(&stderr_path)?;
    let mut out = File::open(&stdout_path)?;
    let mut err = File::open(&stderr_path)?;
    let mut combined = OpenOptions::new().create(true).append(true).open(log)?;
    writeln!(
        combined,
        "PHASE {} {:?} {:?}",
        step.program, step.args, step.env
    )?;
    println!(
        "PHASE {} {:?} {:?}; log: {}",
        step.program,
        step.args,
        step.env,
        log.display()
    );
    std::io::stdout().flush()?;
    cmd.stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = Running(cmd.spawn()?);
    let started = Instant::now();
    let mut heartbeat = Instant::now();
    let mut cancelled = false;
    let status = loop {
        if interrupted() && !cancelled {
            cancelled = true;
            // Each child owns its process group; never signal unrelated Cargo.
            stop(&mut child.0);
        }
        for stream in [&mut out, &mut err] {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            combined.write_all(&bytes)?;
            std::io::stdout().write_all(&bytes)?;
        }
        combined.flush()?;
        std::io::stdout().flush()?;
        if let Some(status) = child.0.try_wait()? {
            // Drain the final bytes after the child has closed its writers.
            for stream in [&mut out, &mut err] {
                let mut bytes = Vec::new();
                stream.read_to_end(&mut bytes)?;
                combined.write_all(&bytes)?;
                std::io::stdout().write_all(&bytes)?;
            }
            break status;
        }
        if heartbeat.elapsed() >= Duration::from_secs(15) {
            println!(
                "BUSY {} {:?}: {:.1}s elapsed; log: {}",
                step.program,
                step.args,
                started.elapsed().as_secs_f64(),
                log.display()
            );
            std::io::stdout().flush()?;
            heartbeat = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    combined.flush()?;
    std::io::stdout().flush()?;
    if cancelled || interrupted() {
        return Err(invalid(
            "verification interrupted; active subprocess group stopped",
        ));
    }
    Ok(std::process::Output {
        status,
        stdout: std::fs::read(stdout_path)?,
        stderr: std::fs::read(stderr_path)?,
    })
}
