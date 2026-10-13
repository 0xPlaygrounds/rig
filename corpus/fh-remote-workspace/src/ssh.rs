//! [`Ssh`]: the [`Workspace`] of a directory on another machine. Every file
//! operation is a short `sh` script run there, and every process a local
//! `ssh` that runs it there, all over one multiplexed connection (OpenSSH's
//! `ControlMaster`). A dropped connection fails what was running with
//! "connection to host lost" (a file operation here, a command in Review,
//! from its exit status), is reconnected with backoff, and what comes next
//! waits for the reconnect.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::{self, ErrorKind, Write as _};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread;
use std::time::{Duration, Instant};

use futures::channel::mpsc::UnboundedSender;
use ignore::overrides::OverrideBuilder;
use rig_tools::process::{Launcher, Process, Spawn};
use rig_tools::workspace::{Contents, Stat, Workspace};

use crate::{RemoteSettings, Target, quote, quote_path};

/// The exit code with which `ssh` says the connection failed.
pub(crate) const SSH_FAILED: i32 = 255;
/// The first wait before a reconnect attempt; it doubles up to
/// [`MOST_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_millis(100);
/// The longest wait between reconnect attempts.
const MOST_BACKOFF: Duration = Duration::from_secs(5);

/// The exit codes the scripts below say what went wrong with.
const NOT_FOUND: i32 = 2;
const NOT_A_FILE: i32 = 3;
const READ_ONLY: i32 = 4;
const A_DIRECTORY: i32 = 5;
const ROOT_IS_A_FILE: i32 = 6;

/// A directory on another machine, reached over SSH.
#[derive(Clone)]
pub struct Ssh(Arc<Link>);

/// The connection an [`Ssh`] workspace and its clones share.
struct Link {
    target: Target,
    /// `ssh://user@host`, as [`Workspace::host`] says.
    url: String,
    settings: RemoteSettings,
    /// How the `ssh` of a file operation starts: the session's policy, in
    /// no directory of its own.
    launcher: Launcher,
    /// Whether calls go out now, and whether a reconnect runs.
    state: Mutex<State>,
    /// Signalled when the link is up again.
    up_again: Condvar,
    /// Where the link's news goes, for the app's notices.
    news: UnboundedSender<String>,
}

#[derive(Default)]
struct State {
    down: bool,
    reconnecting: bool,
}

/// What a remote script printed and how it ended.
struct Ran {
    code: Option<i32>,
    out: Vec<u8>,
    err: String,
}

impl fmt::Debug for Ssh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Ssh").field(&self.0.url).finish()
    }
}

impl Ssh {
    /// The workspace `target`, its `ssh` as `settings` say, started under
    /// `launcher`, telling its reconnects to `news`.
    pub fn new(
        target: Target,
        settings: RemoteSettings,
        mut launcher: Launcher,
        news: UnboundedSender<String>,
    ) -> Self {
        launcher.cwd = None;
        Self(Arc::new(Link {
            url: format!("ssh://{}", target.host),
            target,
            settings,
            launcher,
            state: Mutex::default(),
            up_again: Condvar::new(),
            news,
        }))
    }

    /// Runs `script` there, after a reconnect if one runs; an exit code
    /// the script names becomes its error, others an error with its stderr.
    fn run(&self, script: &str, stdin: Option<&[u8]>) -> io::Result<Vec<u8>> {
        let ran = self.0.run(script, stdin)?;
        match ran.code {
            Some(0) => Ok(ran.out),
            Some(NOT_FOUND) => Err(ErrorKind::NotFound.into()),
            Some(NOT_A_FILE) => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file",
            )),
            Some(READ_ONLY) => Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "the file is read-only",
            )),
            Some(A_DIRECTORY) => Err(ErrorKind::IsADirectory.into()),
            _ => Err(io::Error::other(ran.err.trim().to_owned())),
        }
    }

    /// Where it is.
    pub fn target(&self) -> &Target {
        &self.0.target
    }

    /// Marks the connection lost by a command it ran, starting a reconnect,
    /// and says what that command's call fails with.
    pub(crate) fn lost(&self) -> io::Error {
        self.0.went_down();
        self.0.lost()
    }

    /// A path quoted for the remote `sh`.
    fn quoted(path: &Path) -> String {
        quote_path(&path.to_string_lossy())
    }
}

impl Link {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// What a call that lost the connection fails with.
    fn lost(&self) -> io::Error {
        io::Error::new(
            ErrorKind::NotConnected,
            format!("connection to {} lost", self.target.host),
        )
    }

    /// Returns once the link is up, waiting for a reconnect at most as long
    /// as the settings say.
    fn ready(&self) -> io::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(self.settings.wait_secs);
        let mut state = self.state();
        while state.down {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(self.lost());
            }
            state = match self.up_again.wait_timeout(state, left) {
                Ok((state, _)) => state,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        Ok(())
    }

    /// `ssh` and its options, multiplexed over one connection per host,
    /// with a terminal for the remote command when `tty`, so it is hung up
    /// when the local end goes.
    fn argv(&self, tty: bool) -> Vec<String> {
        let mut argv = self.settings.ssh.clone();
        for option in [
            "ControlMaster=auto",
            "ControlPath=~/.ssh/rig-%C",
            "ControlPersist=600",
            "ServerAliveInterval=10",
            "ServerAliveCountMax=3",
            "BatchMode=yes",
        ] {
            argv.extend(["-o".to_owned(), option.to_owned()]);
        }
        if tty {
            argv.push("-tt".to_owned());
        }
        argv.push(self.target.host.clone());
        argv
    }

    /// Runs `script` with the remote `sh`, once the link is up. A dropped
    /// connection fails it and starts a reconnect.
    fn run(self: &Arc<Self>, script: &str, stdin: Option<&[u8]>) -> io::Result<Ran> {
        self.ready()?;
        let ran = self.once(script, stdin)?;
        if ran.code == Some(SSH_FAILED) {
            self.went_down();
            return Err(self.lost());
        }
        Ok(ran)
    }

    /// Runs `script` there once, as it goes.
    fn once(&self, script: &str, stdin: Option<&[u8]>) -> io::Result<Ran> {
        let argv = self.argv(false);
        let Some((program, args)) = argv.split_first() else {
            return Err(io::Error::other("the `ssh` setting is empty"));
        };
        let mut command = self.launcher.command(program);
        command
            .args(args)
            .arg(script)
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn()?;
        if let (Some(mut pipe), Some(bytes)) = (child.stdin.take(), stdin) {
            pipe.write_all(bytes).ok();
        }
        let output = child.wait_with_output()?;
        Ok(Ran {
            code: output.status.code(),
            out: output.stdout,
            err: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Marks the link down and reconnects it in the background, unless
    /// that already runs.
    fn went_down(self: &Arc<Self>) {
        {
            let mut state = self.state();
            if state.reconnecting {
                return;
            }
            state.down = true;
            state.reconnecting = true;
        }
        self.tell(format!(
            "Connection to {} lost; reconnecting.",
            self.target.host
        ));
        let link = Arc::downgrade(self);
        let spawned = thread::Builder::new()
            .name("rig-remote-reconnect".to_owned())
            .spawn(move || reconnect(&link));
        if spawned.is_err() {
            *self.state() = State::default();
            self.up_again.notify_all();
        }
    }

    fn tell(&self, news: String) {
        self.news.unbounded_send(news).ok();
    }
}

/// Tries `ssh host true` with a growing wait between tries, until it works
/// or no one uses the link any more.
fn reconnect(link: &Weak<Link>) {
    let mut backoff = FIRST_BACKOFF;
    loop {
        thread::sleep(backoff);
        let Some(link) = link.upgrade() else {
            return;
        };
        if link.once("true", None).is_ok_and(|ran| ran.code == Some(0)) {
            *link.state() = State::default();
            link.up_again.notify_all();
            link.tell(format!("Reconnected to {}.", link.target.host));
            return;
        }
        backoff = (backoff * 2).min(MOST_BACKOFF);
    }
}

impl Workspace for Ssh {
    fn host(&self) -> Option<&str> {
        Some(&self.0.url)
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        let script = format!(
            "p={}; if [ -f \"$p\" ]; then wc -c < \"$p\" || exit 1; \
             if [ -w \"$p\" ]; then echo w; else echo r; fi; \
             elif [ -d \"$p\" ]; then echo d; elif [ -e \"$p\" ]; then echo o; else echo m; fi",
            Self::quoted(path)
        );
        let out = String::from_utf8_lossy(&self.run(&script, None)?).into_owned();
        let mut words = out.split_whitespace();
        Ok(match (words.next(), words.next()) {
            (Some("d"), _) => Stat::Dir,
            (Some("o"), _) => Stat::Other,
            (Some("m"), _) => Stat::Missing,
            (Some(size), Some(access)) => Stat::File {
                size: size.parse().map_err(io::Error::other)?,
                read_only: access == "r",
                modified: None,
            },
            _ => return Err(io::Error::other(format!("stat said `{}`", out.trim()))),
        })
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        let script = format!(
            "p={}; [ -f \"$p\" ] || {{ [ -e \"$p\" ] && exit {NOT_A_FILE}; exit {NOT_FOUND}; }}; \
             wc -c < \"$p\" && head -c {limit} -- \"$p\"",
            Self::quoted(path)
        );
        let out = self.run(&script, None)?;
        let Some(newline) = out.iter().position(|&byte| byte == b'\n') else {
            return Err(io::Error::other("the remote read said nothing"));
        };
        let (size, bytes) = out.split_at(newline);
        let size = String::from_utf8_lossy(size).trim().parse();
        Ok(Contents {
            bytes: bytes.get(1..).unwrap_or_default().to_vec(),
            size: size.map_err(io::Error::other)?,
        })
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let script = format!(
            "p={}; if [ -e \"$p\" ]; then [ -f \"$p\" ] || exit {NOT_A_FILE}; \
             [ -w \"$p\" ] || exit {READ_ONLY}; fi; d=$(dirname -- \"$p\"); \
             mkdir -p -- \"$d\" && t=$(mktemp \"$d/.rig-write.XXXXXX\") || exit 1; \
             if [ -e \"$p\" ]; then cp -p -- \"$p\" \"$t\"; \
             else chmod \"$(printf %o $((0666 & ~$(umask))))\" \"$t\"; fi && \
             cat > \"$t\" && mv -f -- \"$t\" \"$p\" || {{ rm -f -- \"$t\"; exit 1; }}",
            Self::quoted(path)
        );
        self.run(&script, Some(bytes)).map(drop)
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        let script = format!(
            "p={}; [ -d \"$p\" ] && exit {A_DIRECTORY}; \
             [ -e \"$p\" ] || [ -L \"$p\" ] || exit {NOT_FOUND}; rm -f -- \"$p\"",
            Self::quoted(path)
        );
        self.run(&script, None).map(drop)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let script = format!(
            "f={}; t={}; [ -e \"$f\" ] || exit {NOT_FOUND}; \
             mkdir -p -- \"$(dirname -- \"$t\")\" && mv -f -- \"$f\" \"$t\"",
            Self::quoted(from),
            Self::quoted(to)
        );
        self.run(&script, None).map(drop)
    }

    /// Folds `..` lexically, against the remote directory when relative:
    /// it may be asked on a thread that must not wait, so it does not ask
    /// the remote machine, and links there are not followed. A path under
    /// `~` stays there, as the remote `sh` reads it.
    fn canonical(&self, path: &Path) -> PathBuf {
        let joined = if path.starts_with("~") {
            path.to_path_buf()
        } else {
            Path::new(&self.0.target.dir).join(path)
        };
        let mut out = PathBuf::new();
        for component in joined.components() {
            match component {
                Component::ParentDir => {
                    out.pop();
                }
                Component::CurDir => {}
                other => out.push(other),
            }
        }
        out
    }

    fn files(&self, root: &Path, glob: Option<&str>, all: bool) -> io::Result<Vec<PathBuf>> {
        let globs = glob
            .map(|glob| {
                OverrideBuilder::new(root)
                    .add(glob)
                    .and_then(|builder| builder.build())
                    .map_err(|error| {
                        io::Error::new(ErrorKind::InvalidInput, format!("bad glob: {error}"))
                    })
            })
            .transpose()?;
        let git = match all {
            true => "false",
            false => "git rev-parse --is-inside-work-tree >/dev/null 2>&1",
        };
        let script = format!(
            "r={}; [ -f \"$r\" ] && exit {ROOT_IS_A_FILE}; cd \"$r\" || exit {NOT_FOUND}; \
             if {git}; \
             then git ls-files -z -co --exclude-standard; \
             else find . -type f -print0; fi",
            Self::quoted(root)
        );
        let ran = self.0.run(&script, None)?;
        match ran.code {
            Some(ROOT_IS_A_FILE) => return Ok(vec![root.to_path_buf()]),
            Some(0) => {}
            Some(NOT_FOUND) => return Err(ErrorKind::NotFound.into()),
            _ => return Err(io::Error::other(ran.err.trim().to_owned())),
        }
        let mut files: Vec<PathBuf> = String::from_utf8_lossy(&ran.out)
            .split('\0')
            .map(|name| Path::new(name.trim_start_matches("./")))
            .filter(|name| !name.as_os_str().is_empty())
            .filter(|name| {
                all || name.components().all(|part| {
                    let part = part.as_os_str().as_encoded_bytes();
                    part != b"target" && !part.starts_with(b".")
                })
            })
            .map(|name| root.join(name))
            .filter(|path| {
                globs
                    .as_ref()
                    .is_none_or(|globs| globs.matched(path, false).is_whitelist())
            })
            .collect();
        files.sort();
        Ok(files)
    }

    /// Runs `program` there in `dir` (else the launcher's, else the remote
    /// directory), in the remote user's login environment with the
    /// launcher's variables, from a local `ssh` started under the
    /// launcher's policy. Without stdin it gets a terminal, so cancelling
    /// it hangs the remote process up; stderr then comes with stdout.
    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        how: Spawn,
    ) -> io::Result<Process> {
        self.0.ready()?;
        let tty = how.stdin.is_none();
        let dir = dir
            .or(launcher.cwd.as_deref())
            .unwrap_or(Path::new(&self.0.target.dir));
        let mut line = String::new();
        if tty {
            line.push_str("stty -onlcr 2>/dev/null; ");
        }
        line.push_str(&format!("cd {} && exec env", Self::quoted(dir)));
        for (name, value) in &launcher.env {
            let set = format!("{}={}", name.to_string_lossy(), value.to_string_lossy());
            line.push_str(&format!(" {}", quote(&set)));
        }
        line.push_str(" sh -lc 'exec \"$0\" \"$@\"'");
        for word in std::iter::once(program).chain(args.iter().map(OsString::as_os_str)) {
            line.push_str(&format!(" {}", quote(&word.to_string_lossy())));
        }
        let mut argv = self.0.argv(tty);
        argv.push(line);
        let Some((ssh, args)) = argv.split_first() else {
            return Err(io::Error::other("the `ssh` setting is empty"));
        };
        let mut launcher = launcher.clone();
        launcher.cwd = None;
        launcher.spawn(ssh, args, how)
    }
}
