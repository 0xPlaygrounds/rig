//! Serial subprocesses with visible output, logs, heartbeat, and cancellation.
use super::*;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
pub(super) fn install_interrupt_handler() -> Result<()> {
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .map_err(|e| invalid(format!("cannot install interruption handler: {e}")))
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
fn stop(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let group = format!("-{}", child.id());
        let _ = Command::new("kill").args(["-KILL", "--", &group]).status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .status();
    }
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
