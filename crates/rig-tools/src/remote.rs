//! Workspaces on other machines: [`Remote`], one machine reached through a
//! transport, and [`Hosts`], which sends each path written `HOST:PATH` to
//! that host's `Remote` and every other path to the workspace it wraps.
//!
//! A transport is a command that runs its last argument, a shell line, on
//! the other machine with the remote user's shell in their home, forwards
//! its stdin, keeps its stdout and stderr apart, ends with its exit status,
//! and ends with 255 when the transport itself failed: `ssh` does, and so
//! does `koh exec`. Every file operation is one short `sh` script run that
//! way, at most [`CALLS_PER_HOST`] at once, and every process a local
//! transport process running the command there. That command is stopped
//! there when the transport's stdin closes, which this side holds open
//! until the process ends or is stopped, so a timeout, an interrupt or a
//! lost connection leaves nothing running on the other machine.
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//! use rig_tools::process::Launcher;
//! use rig_tools::remote::{Hosts, Remote};
//! use rig_tools::workspace::{Local, Workspace};
//!
//! let bench = ["koh", "exec", "bench", "--sh"];
//! let dev = ["ssh", "-o", "BatchMode=yes", "dev@box"];
//! let hosts = Hosts::new(
//!     Arc::new(Local),
//!     [
//!         Remote::new("bench", bench, Launcher::default())?,
//!         Remote::new("box", dev, Launcher::default())?,
//!     ],
//! );
//! let text = hosts.read(Path::new("bench:~/strata/src/lib.rs"), 4096)?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::{self, ErrorKind, Write as _};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;

use ignore::overrides::OverrideBuilder;

use rig_core::tool::{ContentHash, content_hash};

use crate::process::{Launcher, Process, STOP_GRACE, Spawn};
use crate::workspace::{
    Contents, Excerpt, Listed, Listing, Observation, Stat, Workspace, excerpt_of, not_text,
    read_file, too_large,
};
use crate::{MAX_BYTES, MAX_FILE_BYTES, quote};

/// The exit status with which a transport says it failed, as `ssh` does.
pub const TRANSPORT_FAILED: i32 = 255;
/// How many files one script of [`Remote::read_files`] reads.
const FILES_PER_CALL: usize = 100;
/// How many file operations run on one host at once; the rest wait for a
/// turn. Each is a connection of its own, and a server may drop some of a
/// burst of new connections from one client, as one that takes only two
/// handshakes from a client at a time does.
pub const CALLS_PER_HOST: usize = 2;

/// The exit codes the scripts below say what went wrong with.
const NOT_FOUND: i32 = 2;
const NOT_A_FILE: i32 = 3;
const READ_ONLY: i32 = 4;
const A_DIRECTORY: i32 = 5;
const ROOT_IS_A_FILE: i32 = 6;

/// A shell function for the scripts below: whether the path `$1`, which
/// the shell does not see, is truly not there, since the nearest of its
/// directories that is there may be searched (or is no directory). One that
/// may not be searched hides what is in it, which is not absence.
const GONE: &str = "gone() { d=$1; while :; do d=$(dirname -- \"$d\"); \
    if [ -e \"$d\" ]; then [ ! -d \"$d\" ] || [ -x \"$d\" ]; return; fi; \
    case $d in /|.) return 0;; esac; done; }";

/// Another machine, reached through a transport (see the [module](self)
/// docs). Its paths are that machine's: absolute, or under `~`, its user's
/// home; a relative path is relative to that home, where the transport
/// starts. Paths are folded without asking the machine
/// ([`canonical`](Workspace::canonical)), so links there are not followed.
#[derive(Clone)]
pub struct Remote {
    name: String,
    transport: Vec<OsString>,
    /// How the transport starts for a file operation: in no directory of
    /// its own.
    launcher: Launcher,
    /// The file operations running there now, shared by its clones.
    running: Arc<(Mutex<usize>, Condvar)>,
}

/// Whether a script may run again when its transport failed: one that only
/// reads may, one that writes may not, as it may have done its work.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Again {
    Once,
    Never,
}

/// A turn to run a file operation, given back when dropped.
struct Turn<'a>(&'a (Mutex<usize>, Condvar));

impl<'a> Turn<'a> {
    /// Waits until fewer than [`CALLS_PER_HOST`] operations run.
    fn take(running: &'a (Mutex<usize>, Condvar)) -> Self {
        let (count, freed) = running;
        let mut count = count.lock().unwrap_or_else(PoisonError::into_inner);
        while *count >= CALLS_PER_HOST {
            count = freed.wait(count).unwrap_or_else(PoisonError::into_inner);
        }
        *count += 1;
        Self(running)
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let (count, freed) = self.0;
        let mut count = count.lock().unwrap_or_else(PoisonError::into_inner);
        *count = count.saturating_sub(1);
        freed.notify_one();
    }
}

/// What a remote script printed and how it ended.
struct Ran {
    code: Option<i32>,
    /// The signal that ended the transport, when one did.
    signal: Option<i32>,
    out: Vec<u8>,
    err: String,
}

impl Ran {
    /// Whether the transport failed or was killed, so the script may not
    /// have run there at all.
    fn dropped(&self) -> bool {
        self.code.is_none_or(|code| code == TRANSPORT_FAILED)
    }
}

impl fmt::Debug for Remote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Remote")
            .field("name", &self.name)
            .field("transport", &self.transport)
            .finish()
    }
}

impl Remote {
    /// The machine `name`, such as `bench`, reached through `transport`,
    /// such as `["ssh", "dev@box"]`, which file operations start under
    /// `launcher`'s policy (its wrapper and variables; not its directory).
    ///
    /// # Errors
    ///
    /// An [`InvalidInput`](ErrorKind::InvalidInput) error when `name` is
    /// empty or holds a `:`, `/` or space, or `transport` is empty.
    pub fn new(
        name: impl Into<String>,
        transport: impl IntoIterator<Item = impl Into<OsString>>,
        mut launcher: Launcher,
    ) -> io::Result<Self> {
        let name = name.into();
        let transport: Vec<OsString> = transport.into_iter().map(Into::into).collect();
        if name.is_empty() || name.contains([':', '/']) || name.contains(char::is_whitespace) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("`{name}` is no host name: it is empty or holds a `:`, `/` or space"),
            ));
        }
        if transport.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("the transport of {name} is empty"),
            ));
        }
        launcher.cwd = None;
        Ok(Self {
            name,
            transport,
            launcher,
            running: Arc::default(),
        })
    }

    /// Its name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The transport and `script` as its last argument, run with `sh`
    /// there whatever the remote user's shell is.
    fn command(&self, script: &str) -> io::Result<(OsString, Vec<OsString>)> {
        let Some((program, args)) = self.transport.split_first() else {
            return Err(io::Error::other(format!(
                "the transport of {} is empty",
                self.name
            )));
        };
        let mut args = args.to_vec();
        args.push(format!("sh -c {}", quote(script)).into());
        Ok((program.clone(), args))
    }

    /// Runs `script` there once, with `stdin`, when its turn comes, and
    /// says how it ended.
    fn once(&self, script: &str, stdin: Option<&[u8]>) -> io::Result<Ran> {
        let (program, args) = self.command(script)?;
        let _turn = Turn::take(&self.running);
        let mut command = self.launcher.command(&program);
        command
            .args(args)
            .stdin(match stdin {
                Some(_) => Stdio::piped(),
                None => Stdio::null(),
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| self.unstarted(&program, error))?;
        let pipe = child.stdin.take();
        let output = thread::scope(|scope| {
            if let (Some(mut pipe), Some(bytes)) = (pipe, stdin) {
                scope.spawn(move || pipe.write_all(bytes).ok());
            }
            child.wait_with_output()
        })?;
        #[cfg(unix)]
        let signal = std::os::unix::process::ExitStatusExt::signal(&output.status);
        #[cfg(not(unix))]
        let signal = None;
        Ok(Ran {
            code: output.status.code(),
            signal,
            out: output.stdout,
            err: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }

    /// Runs `script` there, failing when the transport failed or was
    /// killed, after trying again once when `again` allows.
    fn run(&self, script: &str, stdin: Option<&[u8]>, again: Again) -> io::Result<Ran> {
        let mut ran = self.once(script, stdin)?;
        if ran.dropped() && again == Again::Once {
            ran = self.once(script, stdin)?;
        }
        match ran.dropped() {
            true => Err(self.failed(&ran, again)),
            false => Ok(ran),
        }
    }

    /// Runs `script` there: an exit code the script names becomes its
    /// error, another an error with its stderr.
    fn checked(&self, script: &str, stdin: Option<&[u8]>, again: Again) -> io::Result<Vec<u8>> {
        let ran = self.run(script, stdin, again)?;
        self.outcome(ran, again)
    }

    /// What `ran` comes to: its output on exit 0, the error an exit code
    /// the scripts name stands for, or [`failed`](Self::failed).
    fn outcome(&self, ran: Ran, again: Again) -> io::Result<Vec<u8>> {
        match ran.code {
            Some(0) => Ok(ran.out),
            Some(NOT_FOUND) => Err(ErrorKind::NotFound.into()),
            Some(NOT_A_FILE) => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file",
            )),
            Some(READ_ONLY) => Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "the file is read-only; leave it unchanged or ask the user",
            )),
            Some(A_DIRECTORY) => Err(ErrorKind::IsADirectory.into()),
            _ => Err(self.failed(&ran, again)),
        }
    }

    /// What a call fails with when its script ended as `ran` did, with no
    /// exit status the script names: how it ended, what it said, and, for
    /// a script that writes, that it may have done its work. The transport
    /// failed when it ended with [`TRANSPORT_FAILED`] or by a signal, after
    /// the tries
    /// `again` allowed.
    fn failed(&self, ran: &Ran, again: Again) -> io::Error {
        let (kind, ended) = self.ended(ran);
        let after = match again {
            Again::Once if ran.dropped() => " (tried twice)",
            Again::Once => "",
            Again::Never => {
                "; the change may have been made or not, so read the file before trying again"
            }
        };
        io::Error::new(kind, format!("{ended}{after}"))
    }

    /// How a script that ended as `ran` did ended, and what it said.
    fn ended(&self, ran: &Ran) -> (ErrorKind, String) {
        let name = &self.name;
        let (kind, how) = match ran.code {
            Some(TRANSPORT_FAILED) => (
                ErrorKind::NotConnected,
                format!("the transport to {name} failed (exit status {TRANSPORT_FAILED})"),
            ),
            Some(code) => (
                ErrorKind::Other,
                format!("the operation on {name} ended with exit status {code}"),
            ),
            None => (
                ErrorKind::ConnectionAborted,
                match ran.signal {
                    #[cfg(unix)]
                    Some(signal) => format!(
                        "the transport to {name} was killed by {} (signal {signal}), which rig \
                         did not send",
                        crate::shell::sent(signal).0
                    ),
                    _ => format!("the transport to {name} was killed by a signal rig did not send"),
                },
            ),
        };
        let said = match ran.err.as_str() {
            "" => " and said nothing".to_owned(),
            err => format!(": {err}"),
        };
        (kind, format!("{how}{said}"))
    }

    /// Runs `script`, which writes `bytes` to the file `path`, once. When
    /// it ends with no exit status it names, such as a transport that
    /// dropped mid-write, the file is hashed there again (a read, so tried
    /// twice) and the error says whether it holds `bytes`: whether the
    /// write happened.
    fn written(&self, path: &Path, script: &str, bytes: &[u8]) -> io::Result<Vec<u8>> {
        let ran = self.once(script, Some(bytes))?;
        if matches!(
            ran.code,
            Some(0 | NOT_FOUND | NOT_A_FILE | READ_ONLY | A_DIRECTORY)
        ) {
            return self.outcome(ran, Again::Never);
        }
        let (kind, ended) = self.ended(&ran);
        let check = format!(
            "p={}; [ -f \"$p\" ] || exit {NOT_FOUND}; \
             h=$({{ sha256sum || shasum -a 256; }} < \"$p\" 2>/dev/null); echo \"${{h%% *}}\"",
            quote_path(path)
        );
        let checked = self.checked(&check, None, Again::Once);
        let sent = content_hash(bytes).0;
        let found = match checked.as_deref().map(|out| String::from_utf8_lossy(out)) {
            Ok(hash) if hash.trim() == sent => {
                "it holds what this write sent, so the write happened".to_owned()
            }
            Ok(hash) if !hash.trim().is_empty() => "it does not hold what this write sent, so \
                 the write did not happen; make the change again"
                .to_owned(),
            Err(error) if error.kind() == ErrorKind::NotFound => {
                "it is not there, so the write did not happen; make the change again".to_owned()
            }
            Ok(_) => "it could not be hashed there, so the write may or may not have \
                 happened; read the file before going on"
                .to_owned(),
            Err(error) => format!(
                "that failed too ({error}), so the write may or may not have happened; read \
                 the file before going on"
            ),
        };
        Err(io::Error::new(
            kind,
            format!("the write failed: {ended}. The file was checked again: {found}"),
        ))
    }

    /// What a call fails with when its transport could not start.
    fn unstarted(&self, program: &OsStr, error: io::Error) -> io::Error {
        io::Error::other(format!(
            "could not start `{}`, the transport to {}: {error}",
            program.to_string_lossy(),
            self.name
        ))
    }
}

impl Workspace for Remote {
    /// Its name.
    fn host(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        let script = format!(
            "{GONE}; p={}; if [ -f \"$p\" ]; then wc -c < \"$p\" || exit 1; \
             if [ -w \"$p\" ]; then echo w; else echo r; fi; \
             elif [ -d \"$p\" ]; then echo d; elif [ -e \"$p\" ]; then echo o; \
             elif gone \"$p\"; then echo m; else echo u; fi",
            quote_path(path)
        );
        let out = String::from_utf8_lossy(&self.checked(&script, None, Again::Once)?).into_owned();
        let mut words = out.split_whitespace();
        Ok(match (words.next(), words.next()) {
            (Some("d"), _) => Stat::Dir,
            (Some("o"), _) => Stat::Other,
            (Some("m"), _) => Stat::Missing,
            (Some("u"), _) => return Err(hidden()),
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
            quote_path(path)
        );
        let out = self.checked(&script, None, Again::Once)?;
        let Some((size, bytes)) = split_line(&out) else {
            return Err(io::Error::other("the remote read said nothing"));
        };
        Ok(Contents {
            bytes: bytes.to_vec(),
            size: size.trim().parse().map_err(io::Error::other)?,
        })
    }

    /// One script that sends the file's size, its line count, its hash and
    /// only the lines asked, or the whole file when it is at most
    /// [`MAX_BYTES`], all of a copy made aside first, so they agree when
    /// the file changes meanwhile. A machine with neither `sha256sum` nor
    /// `shasum` sends a larger file whole in a second call, to hash it here.
    fn excerpt(&self, path: &Path, first: usize, count: usize) -> io::Result<Excerpt> {
        let first = first.max(1);
        let last = first.saturating_add(count.max(1) - 1);
        let script = format!(
            "p={}; [ -f \"$p\" ] || {{ [ -e \"$p\" ] && exit {NOT_A_FILE}; exit {NOT_FOUND}; }}; \
             s=$(wc -c < \"$p\") || exit 1; [ $s -le {MAX_FILE_BYTES} ] || {{ echo $s; exit 0; }}; \
             t=$(mktemp) || exit 1; trap 'rm -f \"$t\"' EXIT; cat -- \"$p\" > \"$t\" || exit 1; \
             s=$(wc -c < \"$t\") && echo $s && wc -l < \"$t\" && tail -c 1 \"$t\" | wc -l && \
             {{ sha256sum || shasum -a 256 || echo -; }} < \"$t\" 2>/dev/null && \
             if [ $s -le {MAX_BYTES} ]; then cat \"$t\"; else sed -n '{first},{last}p;{last}q' \"$t\"; fi",
            quote_path(path)
        );
        let out = self.checked(&script, None, Again::Once)?;
        let mut rest = out.as_slice();
        let mut line = || {
            let (line, after) = split_line(rest)?;
            rest = after;
            Some(line.trim().to_owned())
        };
        let size: u64 = line()
            .and_then(|size| size.parse().ok())
            .ok_or_else(|| io::Error::other("the remote read said no size"))?;
        if size > MAX_FILE_BYTES {
            return Err(too_large(size));
        }
        let (Some(newlines), Some(ends_in_newline), Some(hash)) = (line(), line(), line()) else {
            return Err(io::Error::other("the remote read stopped short"));
        };
        let text = String::from_utf8(rest.to_vec()).map_err(|_| not_text());
        if size <= MAX_BYTES as u64 {
            return Ok(Excerpt::of(&text?, first, count));
        }
        let hash = hash.split_whitespace().next().unwrap_or_default();
        if hash.len() != 64 {
            return excerpt_of(self, path, first, count);
        }
        let newlines: usize = newlines.parse().map_err(io::Error::other)?;
        let unended = size > 0 && ends_in_newline == "0";
        Ok(Excerpt {
            text: text?,
            total: newlines + usize::from(unended),
            hash: ContentHash(hash.to_ascii_lowercase()),
            whole: None,
            endings: None,
        })
    }

    /// The files in scripts of 100 each, every file copied
    /// aside before it is sent, so one that changes meanwhile is sent whole
    /// as it was. Each file of a script that failed is
    /// [`Failed`](Observation::Failed) with its error.
    fn read_files(&self, paths: &[PathBuf], limit: u64) -> Vec<Observation<Vec<u8>>> {
        let mut read = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(FILES_PER_CALL) {
            let quoted: Vec<String> = chunk.iter().map(|path| quote_path(path)).collect();
            // Each file is its size and bytes, else a letter: `m` missing,
            // `x` excluded, `d` denied, `h` hidden by its directory, `f`
            // failed.
            let script = format!(
                "{GONE}; t=$(mktemp) || exit 1; trap 'rm -f \"$t\"' EXIT; for p in {}; do \
                 if [ -f \"$p\" ]; then if [ ! -r \"$p\" ]; then echo d; \
                 elif ! n=$(wc -c < \"$p\"); then echo f; elif [ $((n)) -gt {limit} ]; then echo x; \
                 elif cat -- \"$p\" > \"$t\" 2>/dev/null; then wc -c < \"$t\"; cat \"$t\"; \
                 else echo f; fi; elif [ -e \"$p\" ]; then echo x; elif gone \"$p\"; then echo m; \
                 else echo h; fi; done",
                quoted.join(" ")
            );
            let out = match self.run(&script, None, Again::Once) {
                Ok(ran) if ran.code == Some(0) => ran.out,
                Ok(ran) => {
                    let error = self.failed(&ran, Again::Once);
                    read.extend(chunk.iter().map(|_| failed_like(&error)));
                    continue;
                }
                Err(error) => {
                    read.extend(chunk.iter().map(|_| failed_like(&error)));
                    continue;
                }
            };
            let mut rest = out.as_slice();
            for _ in chunk {
                let Some((first, after)) = split_line(rest) else {
                    read.push(Observation::Failed(io::Error::other(
                        "the remote read stopped short",
                    )));
                    continue;
                };
                rest = after;
                read.push(match first.trim() {
                    "m" => Observation::Missing,
                    "x" => Observation::Excluded,
                    "d" => Observation::Failed(io::Error::new(
                        ErrorKind::PermissionDenied,
                        "it may not be read there",
                    )),
                    "h" => Observation::Failed(hidden()),
                    "f" => Observation::Failed(io::Error::other("it could not be read there")),
                    size => {
                        let file = size
                            .parse::<usize>()
                            .ok()
                            .and_then(|size| after.split_at_checked(size));
                        match file {
                            Some((bytes, after)) => {
                                rest = after;
                                Observation::Found(bytes.to_vec())
                            }
                            None => Observation::Failed(io::Error::other(
                                "the remote read said no size",
                            )),
                        }
                    }
                });
            }
        }
        read
    }

    /// [`replace`](Workspace::replace)s it, sending back nothing of what
    /// it held.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.replace(path, bytes, 0).map(drop)
    }

    /// Writes a temporary file beside it and moves it over the file, so a
    /// lost connection never leaves it half written, keeping an existing
    /// file's permissions; the copy the temporary file starts as is what
    /// is sent back, in the same call.
    fn replace(&self, path: &Path, bytes: &[u8], limit: u64) -> io::Result<io::Result<Contents>> {
        let script = format!(
            "p={}; if [ -e \"$p\" ]; then [ -f \"$p\" ] || exit {NOT_A_FILE}; \
             [ -w \"$p\" ] || exit {READ_ONLY}; fi; d=$(dirname -- \"$p\"); \
             mkdir -p -- \"$d\" && t=$(mktemp \"$d/.rig-write.XXXXXX\") || exit 1; \
             if [ -e \"$p\" ]; then cp -p -- \"$p\" \"$t\" && wc -c < \"$t\" && \
             head -c {limit} -- \"$t\"; \
             else chmod \"$(printf %o $((0666 & ~$(umask))))\" \"$t\"; fi && \
             cat > \"$t\" && mv -f -- \"$t\" \"$p\" || {{ rm -f -- \"$t\"; exit 1; }}",
            quote_path(path)
        );
        let out = self.written(path, &script, bytes)?;
        Ok(match split_line(&out) {
            None => Err(ErrorKind::NotFound.into()),
            Some((size, bytes)) => size
                .trim()
                .parse()
                .map(|size| Contents {
                    bytes: bytes.to_vec(),
                    size,
                })
                .map_err(io::Error::other),
        })
    }

    /// One script: a copy of the file is made beside it and hashed there,
    /// then filled with `bytes` and moved over the file when the hash is
    /// `held`, or sent back when it is not. A machine with neither
    /// `sha256sum` nor `shasum` always sends it back.
    fn write_if(
        &self,
        path: &Path,
        bytes: &[u8],
        held: &ContentHash,
    ) -> io::Result<Option<Contents>> {
        let script = format!(
            "p={}; [ -f \"$p\" ] || {{ [ -e \"$p\" ] && exit {NOT_A_FILE}; exit {NOT_FOUND}; }}; \
             [ -w \"$p\" ] || exit {READ_ONLY}; \
             t=$(mktemp \"$(dirname -- \"$p\")/.rig-write.XXXXXX\") || exit 1; \
             cp -p -- \"$p\" \"$t\" || {{ rm -f -- \"$t\"; exit 1; }}; \
             h=$({{ sha256sum || shasum -a 256; }} < \"$t\" 2>/dev/null); \
             if [ \"${{h%% *}}\" = {} ]; then \
             cat > \"$t\" && mv -f -- \"$t\" \"$p\" && echo = || {{ rm -f -- \"$t\"; exit 1; }}; \
             else wc -c < \"$t\" && head -c {MAX_FILE_BYTES} -- \"$t\"; s=$?; rm -f -- \"$t\"; exit $s; fi",
            quote_path(path),
            quote(&held.0)
        );
        let out = self.written(path, &script, bytes)?;
        let Some((first, rest)) = split_line(&out) else {
            return Err(io::Error::other("the remote write said nothing"));
        };
        if first == "=" {
            return Ok(None);
        }
        Ok(Some(Contents {
            bytes: rest.to_vec(),
            size: first.trim().parse().map_err(io::Error::other)?,
        }))
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        let script = format!(
            "p={}; [ -d \"$p\" ] && exit {A_DIRECTORY}; \
             [ -e \"$p\" ] || [ -L \"$p\" ] || exit {NOT_FOUND}; rm -f -- \"$p\"",
            quote_path(path)
        );
        self.checked(&script, None, Again::Never).map(drop)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let script = format!(
            "f={}; t={}; [ -e \"$f\" ] || exit {NOT_FOUND}; \
             mkdir -p -- \"$(dirname -- \"$t\")\" && mv -f -- \"$f\" \"$t\"",
            quote_path(from),
            quote_path(to)
        );
        self.checked(&script, None, Again::Never).map(drop)
    }

    /// A path starting with `~` stays whole: the remote `sh` reads it as
    /// the home directory there.
    fn join(&self, base: &Path, path: &Path) -> PathBuf {
        match path.starts_with("~") {
            true => path.to_path_buf(),
            false => base.join(path),
        }
    }

    /// Folds `.` and `..` without asking the machine, a relative path under
    /// `~`; a `..` above `~` stays, as the remote `sh` reads it.
    fn canonical(&self, path: &Path) -> PathBuf {
        let joined = match path.is_absolute() || path.starts_with("~") {
            true => path.to_path_buf(),
            false => Path::new("~").join(path),
        };
        let mut out = PathBuf::new();
        for component in joined.components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => match out.components().next_back() {
                    Some(Component::Normal(last)) if last != "~" => {
                        out.pop();
                    }
                    Some(Component::RootDir) => {}
                    _ => out.push(".."),
                },
                other => out.push(other),
            }
        }
        out
    }

    /// As git lists them there in a git work tree (unless `listing` is
    /// [`All`](Listing::All)), else as `find` does, then filtered by
    /// [`Listing::keeps`] and `glob`.
    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Listed> {
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
        let git = match listing {
            Listing::All => "false",
            Listing::Visible | Listing::Hidden => {
                "git rev-parse --is-inside-work-tree >/dev/null 2>&1"
            }
        };
        let script = format!(
            "{GONE}; r={}; [ -f \"$r\" ] && exit {ROOT_IS_A_FILE}; \
             [ -e \"$r\" ] || ! gone \"$r\" || exit {NOT_FOUND}; cd \"$r\" || exit 1; \
             if {git}; then git ls-files -z -co --exclude-standard; \
             else find . -type f -print0; fi",
            quote_path(root)
        );
        let ran = self.run(&script, None, Again::Once)?;
        match ran.code {
            Some(ROOT_IS_A_FILE) => {
                return Ok(Listed {
                    files: vec![root.to_path_buf()],
                    failed: Vec::new(),
                });
            }
            Some(0) => {}
            Some(NOT_FOUND) => return Err(ErrorKind::NotFound.into()),
            _ => return Err(self.failed(&ran, Again::Once)),
        }
        let mut files: Vec<PathBuf> = String::from_utf8_lossy(&ran.out)
            .split('\0')
            .map(|name| Path::new(name.trim_start_matches("./")))
            .filter(|name| !name.as_os_str().is_empty() && listing.keeps(name))
            .map(|name| root.join(name))
            .filter(|path| {
                globs
                    .as_ref()
                    .is_none_or(|globs| globs.matched(path, false).is_whitelist())
            })
            .collect();
        files.sort();
        Ok(Listed {
            files,
            failed: Vec::new(),
        })
    }

    /// Runs `program` there in `dir` (`None`: the remote home) with `sh -l`,
    /// so the remote user's login environment holds, and the launcher's
    /// variables, from a local transport started under the launcher's
    /// wrapper (the launcher's directory is this machine's, so it is not
    /// used). The process is marked [`through`](Process::through) this
    /// host: stopping it stops the local transport (`SIGTERM`, then
    /// `SIGKILL`), and whether that stops the command there is the
    /// transport's to say.
    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        how: Spawn,
    ) -> io::Result<Process> {
        let dir = dir.unwrap_or(Path::new("~"));
        let mut run = String::from("env");
        for (name, value) in &launcher.env {
            let set = format!("{}={}", name.to_string_lossy(), value.to_string_lossy());
            run.push_str(&format!(" {}", quote(&set)));
        }
        run.push_str(" sh -lc 'exec \"$0\" \"$@\"'");
        for word in std::iter::once(program).chain(args.iter().map(OsString::as_os_str)) {
            run.push_str(&format!(" {}", quote(&word.to_string_lossy())));
        }
        let cd = format!("cd {} || exit", quote_path(&self.canonical(dir)));
        let hold = how.stdin.is_none();
        let script = match hold {
            true => format!("{cd}; {}", guarded(&run)),
            false => format!("{cd}; exec {run}"),
        };
        let (transport, line) = self.command(&script)?;
        let mut launcher = launcher.clone();
        launcher.cwd = None;
        launcher
            .spawn_holding(&transport, line, how, hold)
            .map(|process| process.through(&self.name))
            .map_err(|error| self.unstarted(&transport, error))
    }
}

/// A workspace (the inner one, such as [`Local`](crate::workspace::Local))
/// whose paths written `HOST:PATH`, `HOST` the name of one of its
/// [`Remote`]s, are that host's `PATH`, as `scp` reads them:
/// `bench:~/strata/src/lib.rs`, `bench:/srv/app`. Every other path is the
/// inner workspace's. Every path it gives back for a remote file
/// ([`canonical`](Workspace::canonical), [`files`](Workspace::files)) keeps
/// the `HOST:` form, so the facts and footprints the tools make of them can
/// be passed back as they are; a relative path joined onto a `HOST:` base
/// stays there.
///
/// Its [`host`](Workspace::host) is the inner workspace's, as its project
/// is there; a call on a `HOST:` path waits for the transport.
#[derive(Clone, Debug)]
pub struct Hosts {
    inner: Arc<dyn Workspace>,
    hosts: BTreeMap<String, Remote>,
}

impl Hosts {
    /// `inner`, and `remotes` by their names; a later remote replaces an
    /// earlier one of the same name.
    pub fn new(inner: Arc<dyn Workspace>, remotes: impl IntoIterator<Item = Remote>) -> Self {
        let hosts = remotes
            .into_iter()
            .map(|remote| (remote.name.clone(), remote))
            .collect();
        Self { inner, hosts }
    }

    /// The remote `path` names and the path there: `~` for `HOST:` alone.
    pub fn route(&self, path: &Path) -> Option<(&Remote, PathBuf)> {
        let (host, rest) = path.to_str()?.split_once(':')?;
        let remote = self.hosts.get(host)?;
        Some((
            remote,
            PathBuf::from(if rest.is_empty() { "~" } else { rest }),
        ))
    }

    /// `path` on `remote` as this workspace names it: `HOST:PATH`.
    fn named(remote: &Remote, path: &Path) -> PathBuf {
        let mut named = OsString::from(&remote.name);
        named.push(":");
        named.push(path);
        PathBuf::from(named)
    }
}

impl Workspace for Hosts {
    fn host(&self) -> Option<&str> {
        self.inner.host()
    }

    /// A `HOST:` path stays whole; a path starting with `~` joined onto a
    /// `HOST:` base is in that host's home.
    fn join(&self, base: &Path, path: &Path) -> PathBuf {
        match (self.route(path), self.route(base)) {
            (Some(_), _) => path.to_path_buf(),
            (None, Some((remote, _))) if path.starts_with("~") => Self::named(remote, path),
            (None, _) => self.inner.join(base, path),
        }
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        match self.route(path) {
            Some((remote, path)) => remote.stat(&path),
            None => self.inner.stat(path),
        }
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        match self.route(path) {
            Some((remote, path)) => remote.read(&path, limit),
            None => self.inner.read(path, limit),
        }
    }

    /// All in one go on their host when every path is on one, else one by
    /// one.
    fn read_files(&self, paths: &[PathBuf], limit: u64) -> Vec<Observation<Vec<u8>>> {
        let first = paths.first().and_then(|path| self.route(path));
        let there = first.and_then(|(remote, _)| {
            let there: Option<Vec<PathBuf>> = paths
                .iter()
                .map(|path| {
                    let (other, there) = self.route(path)?;
                    (other.name == remote.name).then_some(there)
                })
                .collect();
            Some((remote, there?))
        });
        match there {
            Some((remote, there)) => remote.read_files(&there, limit),
            None => paths
                .iter()
                .map(|path| read_file(self, path, limit))
                .collect(),
        }
    }

    fn excerpt(&self, path: &Path, first: usize, count: usize) -> io::Result<Excerpt> {
        match self.route(path) {
            Some((remote, path)) => remote.excerpt(&path, first, count),
            None => self.inner.excerpt(path, first, count),
        }
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        match self.route(path) {
            Some((remote, path)) => remote.write(&path, bytes),
            None => self.inner.write(path, bytes),
        }
    }

    fn replace(&self, path: &Path, bytes: &[u8], limit: u64) -> io::Result<io::Result<Contents>> {
        match self.route(path) {
            Some((remote, path)) => remote.replace(&path, bytes, limit),
            None => self.inner.replace(path, bytes, limit),
        }
    }

    fn write_if(
        &self,
        path: &Path,
        bytes: &[u8],
        held: &ContentHash,
    ) -> io::Result<Option<Contents>> {
        match self.route(path) {
            Some((remote, path)) => remote.write_if(&path, bytes, held),
            None => self.inner.write_if(path, bytes, held),
        }
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        match self.route(path) {
            Some((remote, path)) => remote.remove(&path),
            None => self.inner.remove(path),
        }
    }

    /// Within one host, or within the inner workspace; a move between them
    /// is refused.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        match (self.route(from), self.route(to)) {
            (None, None) => self.inner.rename(from, to),
            (Some((one, from)), Some((other, to))) if one.name == other.name => {
                one.rename(&from, &to)
            }
            _ => Err(io::Error::new(
                ErrorKind::Unsupported,
                "a file is not moved between machines; read it and write it there instead",
            )),
        }
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        match self.route(path) {
            Some((remote, path)) => Self::named(remote, &remote.canonical(&path)),
            None => self.inner.canonical(path),
        }
    }

    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Listed> {
        match self.route(root) {
            Some((remote, root)) => {
                let listed = remote.files(&root, glob, listing)?;
                let named = |path: &Path| Self::named(remote, path);
                Ok(Listed {
                    files: listed.files.iter().map(|path| named(path)).collect(),
                    failed: (listed.failed.into_iter())
                        .map(|(path, error)| (named(&path), error))
                        .collect(),
                })
            }
            None => self.inner.files(root, glob, listing),
        }
    }

    /// On the host `dir` names, else in the inner workspace.
    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        how: Spawn,
    ) -> io::Result<Process> {
        match dir.and_then(|dir| self.route(dir)) {
            Some((remote, dir)) => remote.spawn(launcher, Some(&dir), program, args, how),
            None => self.inner.spawn(launcher, dir, program, args, how),
        }
    }

    /// A remote root is looked at every second, as the default does.
    fn watch(
        self: Arc<Self>,
        root: &Path,
        recursive: bool,
        listing: Listing,
    ) -> io::Result<crate::watch::Watch> {
        match self.route(root) {
            Some(_) => crate::watch::poll(self, root, recursive, listing),
            None => self.inner.clone().watch(root, recursive, listing),
        }
    }
}

/// `bytes` split at its first `\n`: the line before it, as text, and
/// what follows.
fn split_line(bytes: &[u8]) -> Option<(String, &[u8])> {
    let newline = bytes.iter().position(|&byte| byte == b'\n')?;
    let (line, rest) = bytes.split_at(newline);
    Some((
        String::from_utf8_lossy(line).into_owned(),
        rest.get(1..).unwrap_or_default(),
    ))
}

/// A script that runs the command line `run` there, with an empty stdin,
/// in a session of its own where `setsid` is found, and stops it when the
/// script's own stdin closes: the transport's, which this side holds open
/// with nothing written, so it closes when the transport goes, as when a
/// call times out, is interrupted or rig ends, or when the script is
/// signalled to hang up or stop. The stop asks the command's group (its
/// process alone without `setsid`) to end, and kills it [`STOP_GRACE`]
/// later. The script ends as the command does.
fn guarded(run: &str) -> String {
    let grace = STOP_GRACE.as_secs().max(1);
    format!(
        "exec 3<&0 </dev/null; \
         command -v setsid >/dev/null 2>&1 || setsid() {{ exec \"$@\"; }}; \
         setsid {run} 3<&- & c=$!; \
         stop() {{ kill -TERM -$c || kill -TERM $c; sleep {grace}; \
         kill -KILL -$c || kill -KILL $c; }} 2>/dev/null; \
         {{ read -r x <&3; stop; }} >/dev/null 2>&1 & w=$!; \
         trap 'kill $w; stop; exit 143' HUP INT TERM; \
         exec 3<&-; wait $c; s=$?; kill $w 2>/dev/null; exit $s"
    )
}

/// A remote path quoted for `sh`, a leading `~` kept as the home
/// directory.
fn quote_path(path: &Path) -> String {
    let path = path.to_string_lossy();
    match path.strip_prefix("~/") {
        Some(rest) => format!("\"$HOME\"/{}", quote(rest)),
        None if path == "~" => "\"$HOME\"".to_owned(),
        None => quote(&path),
    }
}

/// The error for a path in a directory that may not be searched, which
/// hides whether it is there.
fn hidden() -> io::Error {
    io::Error::new(
        ErrorKind::PermissionDenied,
        "a directory it is in may not be searched there",
    )
}

/// An error of the kind and message of `error`, for one more path it
/// stands for.
fn failed_like(error: &io::Error) -> Observation<Vec<u8>> {
    Observation::Failed(io::Error::new(error.kind(), error.to_string()))
}

#[cfg(test)]
mod tests;
