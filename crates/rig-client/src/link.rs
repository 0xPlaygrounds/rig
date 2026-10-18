//! One wire to a server: the local socket (starting `rigd` when none
//! answers) or a remote host's transport running `rig connect --stdio`.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use rig_protocol::{Conn, Endpoint};

use crate::{Autostart, ClientError, Target};

/// How long a started server has to accept its first connection.
const START_FOR: Duration = Duration::from_secs(10);

/// The shell line a remote transport runs.
pub(crate) const REMOTE_COMMAND: &str = "rig connect --stdio";

pub(crate) struct Wire {
    pub reader: Box<dyn BufRead + Send>,
    pub writer: Box<dyn Write + Send>,
    pub closer: Closer,
}

/// Ends a wire from any thread, so its blocked reader returns.
pub(crate) enum Closer {
    Socket(Conn),
    Child(Arc<Mutex<Child>>),
}

impl Closer {
    pub fn close(&self) {
        match self {
            Closer::Socket(conn) => {
                let _ = conn.shutdown();
            }
            Closer::Child(child) => {
                if let Ok(mut child) = child.lock() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }
}

pub(crate) fn open(target: &Target, autostart: &Autostart) -> Result<Wire, ClientError> {
    match target {
        Target::Local { rig_home } => local(rig_home, autostart),
        Target::Remote { host, transport } => remote(host, transport),
    }
}

fn socket(conn: Conn) -> Result<Wire, ClientError> {
    let closer = Closer::Socket(conn.try_clone()?);
    let (reader, writer) = conn.split()?;
    Ok(Wire {
        reader: Box::new(reader),
        writer: Box::new(writer),
        closer,
    })
}

fn dead(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::NotFound | ErrorKind::ConnectionRefused
    )
}

fn local(rig_home: &Path, autostart: &Autostart) -> Result<Wire, ClientError> {
    let endpoint = Endpoint::for_home(rig_home);
    match Conn::connect(&endpoint) {
        Ok(conn) => return socket(conn),
        Err(error) if dead(&error) => {}
        Err(error) => {
            return Err(ClientError::Unavailable {
                next: format!(
                    "the rig server's socket {} refused us ({error}); check its owner and mode, or run `rig server status`",
                    endpoint.path.display()
                ),
            });
        }
    }
    let program = match autostart {
        Autostart::Off => {
            return Err(ClientError::Unavailable {
                next: "start the rig server with `rig serve`, or connect with autostart on".into(),
            });
        }
        Autostart::CurrentExe => (std::env::current_exe()?, Vec::new()),
        Autostart::Program { program, args } => (program.clone(), args.clone()),
    };
    let exited = start(rig_home, &program.0, &program.1)?;
    let deadline = Instant::now() + START_FOR;
    let mut pause = Duration::from_millis(5);
    loop {
        match Conn::connect(&endpoint) {
            Ok(conn) => return socket(conn),
            Err(error) if dead(&error) => {}
            Err(error) => return Err(error.into()),
        }
        let log = rig_home.join("run").join("server.log");
        if let Some(exited) = &exited
            && let Ok(status) = exited.try_recv()
        {
            // A second server that lost the lock race exits at once; the
            // winner may still be binding.
            if !status.success() {
                return Err(ClientError::Unavailable {
                    next: format!(
                        "the rig server exited at start ({status}); read {} and run `rig serve` to see why",
                        log.display()
                    ),
                });
            }
        }
        if Instant::now() >= deadline {
            return Err(ClientError::Unavailable {
                next: format!(
                    "the rig server did not start within {}s; read {} and run `rig serve` to see why",
                    START_FOR.as_secs(),
                    log.display()
                ),
            });
        }
        std::thread::sleep(pause);
        pause = (pause * 2).min(Duration::from_millis(100));
    }
}

/// Starts `rigd` unless one holds `run/server.lock`. Returns its exit
/// status channel when this call started it.
fn start(
    rig_home: &Path,
    program: &Path,
    args: &[OsString],
) -> Result<Option<mpsc::Receiver<std::process::ExitStatus>>, ClientError> {
    let rig_home = std::path::absolute(rig_home)?;
    let run = rig_home.join("run");
    let mut dir = std::fs::DirBuilder::new();
    dir.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut dir, 0o700);
    dir.create(&run)?;
    let lock = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(run.join("server.lock"))?;
    match lock.try_lock() {
        Ok(()) => drop(lock),
        // A server runs or is starting: wait for its socket.
        Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
        Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
    }
    let log = std::fs::File::options()
        .create(true)
        .append(true)
        .open(run.join("server.log"))?;
    let mut command = Command::new(program);
    command
        .args(args)
        .env("RIG_ROLE", "server")
        .env("RIG_HOME", &rig_home)
        .env_remove("RIG_SESSION")
        .env_remove("RIG_WORKER_TOKEN")
        .current_dir(&rig_home)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    let mut child = command.spawn().map_err(|error| ClientError::Unavailable {
        next: format!(
            "could not start the rig server {} ({error}); run `rig serve` to see why",
            program.display()
        ),
    })?;
    let (tx, rx) = mpsc::channel();
    // Reaps the server when it exits, so it never lingers as a zombie of a
    // long-lived client.
    std::thread::Builder::new()
        .name("rig-client-reaper".into())
        .spawn(move || {
            if let Ok(status) = child.wait() {
                let _ = tx.send(status);
            }
        })?;
    Ok(Some(rx))
}

fn remote(host: &str, transport: &[OsString]) -> Result<Wire, ClientError> {
    let Some((program, args)) = transport.split_first() else {
        return Err(ClientError::Unavailable {
            next: format!("give [remote.hosts] {host} a transport in settings.toml"),
        });
    };
    let mut child = Command::new(program)
        .args(args)
        .arg(REMOTE_COMMAND)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| ClientError::Unavailable {
            next: format!(
                "could not run the transport of {host} ({error}); check [remote.hosts] {host} in settings.toml"
            ),
        })?;
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = child.kill();
        return Err(ClientError::Io(std::io::Error::other(
            "the transport's pipes are missing",
        )));
    };
    Ok(Wire {
        reader: Box::new(BufReader::new(stdout)),
        writer: Box::new(stdin),
        closer: Closer::Child(Arc::new(Mutex::new(child))),
    })
}

/// `[remote.hosts].<host>` of `rig_home/settings.toml`: the transport argv.
pub(crate) fn transport(rig_home: &Path, host: &str) -> Result<Vec<OsString>, ClientError> {
    let settings: PathBuf = rig_home.join("settings.toml");
    let missing = || ClientError::Unavailable {
        next: format!(
            "add `[remote.hosts] {host} = [\"ssh\", \"{host}\"]` to {}",
            settings.display()
        ),
    };
    let text = match std::fs::read_to_string(&settings) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Err(missing()),
        Err(error) => return Err(error.into()),
    };
    let table: toml::Table = text.parse().map_err(|error| ClientError::Unavailable {
        next: format!("fix {}: {error}", settings.display()),
    })?;
    let argv = table
        .get("remote")
        .and_then(|remote| remote.get("hosts"))
        .and_then(|hosts| hosts.get(host))
        .and_then(toml::Value::as_array)
        .ok_or_else(missing)?;
    argv.iter()
        .map(|word| word.as_str().map(OsString::from).ok_or_else(missing))
        .collect()
}
