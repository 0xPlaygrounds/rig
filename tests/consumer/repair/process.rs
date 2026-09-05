//! Execute named Rust validation steps with no inherited credentials or host writes.

use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use super::super::Error;

#[cfg(test)]
mod tests;

const OUTPUT_LIMIT: usize = 64 * 1024;

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Compile,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Output {
    pub code: Option<i32>,
    pub timed_out: bool,
    pub output_limit: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Compiler artifacts and runtime temporary files are separate: executed
/// model code cannot replace another test binary or poison a later Cargo run.
pub(super) struct Sandbox {
    pub toolchain: PathBuf,
    pub compile: assert_fs::TempDir,
    runtime: assert_fs::TempDir,
    deadline: Option<Instant>,
}

impl Sandbox {
    pub fn new() -> Result<Self, Error> {
        Self::new_until(None)
    }

    pub fn new_until(deadline: Option<Instant>) -> Result<Self, Error> {
        // Trusted toolchain discovery runs from the harness repository, before
        // model code exists, with bounded output/time and only toolchain env.
        let mut discovery = Command::new("rustc");
        discovery
            .args(["--print", "sysroot"])
            .current_dir(super::super::artifacts::root())
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for name in [
            "PATH",
            "HOME",
            "RUSTUP_HOME",
            "CARGO_HOME",
            "RUSTUP_TOOLCHAIN",
        ] {
            if let Some(value) = std::env::var_os(name) {
                discovery.env(name, value);
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            discovery.process_group(0);
        }
        let output = execute(discovery, remaining(Duration::from_secs(10), deadline))?;
        if output.code != Some(0) || output.timed_out || output.output_limit {
            return Err(Error::Invariant(
                "cannot locate the repository Rust toolchain".into(),
            ));
        }
        let toolchain = PathBuf::from(output.stdout.trim()).canonicalize()?;
        let temporary = || {
            assert_fs::TempDir::new()
                .map_err(|error| Error::Invariant(format!("create validation scratch: {error}")))
        };
        let sandbox = Self {
            toolchain,
            compile: temporary()?,
            runtime: temporary()?,
            deadline,
        };
        for directory in [&sandbox.compile, &sandbox.runtime] {
            for name in ["home", "cargo", "tmp", "target"] {
                std::fs::create_dir(directory.path().join(name))?;
            }
        }
        Ok(sandbox)
    }

    pub fn binary(&self, name: &str) -> PathBuf {
        self.toolchain.join("bin").join(name)
    }

    pub fn target(&self) -> PathBuf {
        self.compile.path().join("target")
    }

    pub fn runtime_root(&self) -> &Path {
        self.runtime.path()
    }

    pub fn run(
        &self,
        project: &Path,
        executable: &Path,
        arguments: &[String],
        mode: Mode,
        timeout: Duration,
    ) -> Result<Output, Error> {
        let project = project.canonicalize()?;
        let executable = executable.canonicalize()?;
        let compile = self.compile.path().canonicalize()?;
        let allowed = match mode {
            Mode::Compile => ["cargo", "rustc", "rustfmt"].iter().any(|name| {
                self.binary(name)
                    .canonicalize()
                    .is_ok_and(|path| path == executable)
            }),
            Mode::Test => executable.starts_with(compile.join("target")),
        };
        if !allowed {
            return Err(Error::Invariant(
                "validation executable is outside the toolchain/artifact allowlist".into(),
            ));
        }
        let writable = match mode {
            Mode::Compile => compile.clone(),
            Mode::Test => self.runtime.path().canonicalize()?,
        };
        let isolated = self.command(&project, &executable, &compile, &writable, mode)?;
        let mut command = Command::new("/usr/bin/python3");
        command
            .args([
                "-I",
                "-c",
                include_str!("limit_process.py"),
                match mode {
                    Mode::Compile => "compile",
                    Mode::Test => "test",
                },
            ])
            .arg(isolated.get_program())
            .args(isolated.get_args());
        command
            .args(arguments)
            .current_dir(&project)
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.toolchain.join("bin").display()),
            )
            .env("HOME", writable.join("home"))
            .env("CARGO_HOME", writable.join("cargo"))
            .env("CARGO_TARGET_DIR", compile.join("target"))
            .env("TMPDIR", writable.join("tmp"))
            .env("LC_ALL", "C")
            .env("RUST_BACKTRACE", "0")
            .env("CARGO_TERM_COLOR", "never")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        execute(command, remaining(timeout, self.deadline))
    }

    #[cfg(target_os = "macos")]
    fn command(
        &self,
        project: &Path,
        executable: &Path,
        compile: &Path,
        writable: &Path,
        mode: Mode,
    ) -> Result<Command, Error> {
        let quote =
            |path: &Path| serde_json::to_string(&path.to_string_lossy()).map_err(Error::from);
        let mut profile = String::from(
            "(version 1)\n(deny default)\n(allow file-read-metadata)\n(allow file-read* (literal \"/\") (literal \"/dev/null\") (literal \"/dev/random\") (literal \"/dev/urandom\") (literal \"/private/etc/ssl/openssl.cnf\"))\n(allow process-info* (target same-sandbox))\n(allow signal (target same-sandbox))\n(allow sysctl-read (sysctl-name-prefix \"hw.\") (sysctl-name-prefix \"kern.os\") (sysctl-name \"kern.usrstack64\") (sysctl-name \"kern.argmax\") (sysctl-name \"kern.maxfilesperproc\") (sysctl-name \"kern.hostname\") (sysctl-name \"kern.version\") (sysctl-name \"sysctl.proc_cputype\"))\n",
        );
        for path in [
            Path::new("/System/Library"),
            Path::new("/usr/bin"),
            Path::new("/usr/lib"),
            Path::new("/usr/share"),
            Path::new("/bin"),
            Path::new("/Library/Developer/CommandLineTools"),
            &self.toolchain,
            project,
            compile,
            writable,
        ] {
            profile.push_str(&format!("(allow file-read* (subpath {}))\n", quote(path)?));
        }
        profile.push_str(&format!(
            "(allow file-write* (subpath {}) (literal \"/dev/null\"))\n",
            quote(writable)?
        ));
        match mode {
            Mode::Compile => profile.push_str("(allow process-exec)\n(allow process-fork)\n"),
            Mode::Test => profile.push_str(&format!(
                "(allow process-exec (literal {}))\n",
                quote(executable)?
            )),
        }
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command.args(["-p", &profile]).arg(executable);
        Ok(command)
    }

    #[cfg(target_os = "linux")]
    fn command(
        &self,
        project: &Path,
        executable: &Path,
        compile: &Path,
        writable: &Path,
        _mode: Mode,
    ) -> Result<Command, Error> {
        let mut command = Command::new("/usr/bin/bwrap");
        command.args([
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]);
        for path in [
            Path::new("/usr"),
            Path::new("/lib"),
            Path::new("/lib64"),
            // Debian/Ubuntu's cc executable resolves through this symlink tree.
            Path::new("/etc/alternatives"),
            &self.toolchain,
            project,
            compile,
        ] {
            if path.exists() {
                command.arg("--ro-bind").arg(path).arg(path);
            }
        }
        command
            .arg("--bind")
            .arg(writable)
            .arg(writable)
            .arg("--chdir")
            .arg(project)
            .arg("--")
            .arg(executable);
        Ok(command)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn command(
        &self,
        _project: &Path,
        _executable: &Path,
        _compile: &Path,
        _writable: &Path,
        _mode: Mode,
    ) -> Result<Command, Error> {
        Err(Error::Invariant(
            "repository repair requires an implemented process isolation backend".into(),
        ))
    }
}

fn remaining(timeout: Duration, deadline: Option<Instant>) -> Duration {
    deadline.map_or(timeout, |deadline| {
        timeout.min(deadline.saturating_duration_since(Instant::now()))
    })
}

fn execute(mut command: Command, timeout: Duration) -> Result<Output, Error> {
    if timeout.is_zero() {
        return Ok(Output {
            code: None,
            timed_out: true,
            output_limit: false,
            stdout: String::new(),
            stderr: "validation execution deadline exhausted before launch".into(),
        });
    }
    let mut child = Running(Some(command.spawn()?));
    let exceeded = Arc::new(AtomicBool::new(false));
    let read = |pipe: Option<_>| -> Result<_, Error> {
        let pipe = pipe.ok_or_else(|| Error::Invariant("validation output pipe missing".into()))?;
        Ok(capture(pipe, exceeded.clone()))
    };
    let process = child
        .0
        .as_mut()
        .ok_or_else(|| Error::Invariant("validation child missing".into()))?;
    let stdout = read(process.stdout.take())?;
    let stderr = capture(
        process
            .stderr
            .take()
            .ok_or_else(|| Error::Invariant("validation stderr missing".into()))?,
        exceeded.clone(),
    );
    let deadline = Instant::now() + timeout;
    let (status, timed_out, output_limit) = loop {
        if let Some(status) = process.try_wait()? {
            break (status, false, exceeded.load(Ordering::SeqCst));
        }
        let timed_out = Instant::now() >= deadline;
        let output_limit = exceeded.load(Ordering::SeqCst);
        if timed_out || output_limit {
            terminate(process);
            break (process.wait()?, timed_out, output_limit);
        }
        thread::sleep(Duration::from_millis(5));
    };
    child.0 = None;
    let receive = |receiver: mpsc::Receiver<std::io::Result<Vec<u8>>>| -> Result<String, Error> {
        let bytes = receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| {
                Error::Invariant("validation output did not close after process exit".into())
            })??;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    };
    let stdout = receive(stdout)?;
    let stderr = receive(stderr)?;
    Ok(Output {
        code: status.code(),
        timed_out,
        output_limit: output_limit || exceeded.load(Ordering::SeqCst),
        stdout,
        stderr,
    })
}

fn capture(
    mut pipe: impl Read + Send + 'static,
    exceeded: Arc<AtomicBool>,
) -> mpsc::Receiver<std::io::Result<Vec<u8>>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let result = (|| {
            let mut output = Vec::new();
            let mut buffer = [0_u8; 8192];
            loop {
                let count = pipe.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let available = OUTPUT_LIMIT.saturating_sub(output.len());
                output.extend(buffer.iter().take(count.min(available)).copied());
                if count > available {
                    exceeded.store(true, Ordering::SeqCst);
                }
            }
            Ok(output)
        })();
        let _ = sender.send(result);
    });
    receiver
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        // The compiler's trusted subprocesses inherit this group. The PID
        // fallback also stops a test that moved itself to another group.
        let _ = Command::new("/bin/kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
}

struct Running(Option<Child>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            terminate(child);
            let _ = child.wait();
        }
    }
}
