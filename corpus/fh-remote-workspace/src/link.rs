//! The SSH link to the remote workspace: one multiplexed connection
//! (OpenSSH's `ControlMaster`), the target it points at, whether it is up,
//! and the reconnect with backoff after it dropped.

use std::io::Write as _;
use std::process::Stdio;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use futures::channel::mpsc::UnboundedSender;
use rig_harness::prelude::Resource;
use rig_harness::rig_core::tool::{ToolErrorKind, ToolExecutionError};
use rig_tools::process::Launcher;

use crate::{RemoteSettings, Target};

/// The exit code with which `ssh` says the connection failed.
const SSH_FAILED: i32 = 255;
/// The first wait before a reconnect attempt; it doubles up to
/// [`MOST_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_millis(100);
/// The longest wait between reconnect attempts.
const MOST_BACKOFF: Duration = Duration::from_secs(5);

/// What a call that lost its connection to `host` fails with.
pub fn lost(host: &str) -> ToolExecutionError {
    ToolExecutionError::new(ToolErrorKind::Network, format!("connection to {host} lost"))
}

/// The link every remote call goes through, shared by its clones.
#[derive(Resource, Clone)]
pub struct Link(Arc<Shared>);

/// What the clones of a [`Link`] share.
struct Shared {
    /// The `ssh` command and how long a call waits for a reconnect.
    settings: RemoteSettings,
    /// Where the link points and whether it is up.
    state: Mutex<State>,
    /// Signalled when [`State`] changes.
    changed: Condvar,
    /// Where the link's news goes, for the app's notices.
    news: Mutex<Option<UnboundedSender<String>>>,
}

/// Where a [`Link`] points and whether it is up.
#[derive(Default)]
struct State {
    /// The remote workspace; `None` is local.
    target: Option<Target>,
    /// Whether calls go out now; `false` while reconnecting.
    up: bool,
    /// Counts target changes, so a reconnect for an older one stops.
    epoch: u64,
    /// Whether a reconnect of this epoch runs.
    reconnecting: bool,
}

/// What a remote command printed and how it ended.
pub struct Ran {
    /// Its exit code; `None` when killed by a signal.
    pub code: Option<i32>,
    /// Its stdout.
    pub out: String,
    /// Its stderr.
    pub err: String,
}

impl Link {
    /// A local link with `settings`, telling its news to `news`.
    pub fn new(settings: RemoteSettings, news: UnboundedSender<String>) -> Self {
        Self(Arc::new(Shared {
            settings,
            state: Mutex::default(),
            changed: Condvar::new(),
            news: Mutex::new(Some(news)),
        }))
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Points the link at `target`, or makes it local; up at once, since
    /// the connection is opened by the first call that needs it.
    pub fn set_target(&self, target: Option<Target>) {
        let mut state = self.state();
        state.target = target;
        state.up = true;
        state.epoch += 1;
        state.reconnecting = false;
        self.0.changed.notify_all();
    }

    /// The remote workspace once the link is up, waiting for a reconnect
    /// at most as long as the settings say.
    pub fn ready(&self) -> Result<Target, ToolExecutionError> {
        let deadline = Instant::now() + Duration::from_secs(self.0.settings.wait_secs);
        let mut state = self.state();
        loop {
            let Some(target) = state.target.clone() else {
                return Err(ToolExecutionError::other(
                    "the workspace is local again; call the tool again",
                ));
            };
            if state.up {
                return Ok(target);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(lost(&target.host));
            }
            state = match self.0.changed.wait_timeout(state, left) {
                Ok((state, _)) => state,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }

    /// `ssh` and its options for `target`, multiplexed over one connection
    /// per host, with a terminal for the remote command when `tty`, so it
    /// is hung up when the local end goes.
    pub fn argv(&self, target: &Target, tty: bool) -> Vec<String> {
        let mut argv = self.0.settings.ssh.clone();
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
        argv.push(target.host.clone());
        argv
    }

    /// Runs `script` with `sh` on the remote host, with `stdin`, once the
    /// link is up. A dropped connection fails it and starts a reconnect.
    pub fn exec(
        &self,
        launcher: &Launcher,
        script: &str,
        stdin: Option<&[u8]>,
    ) -> Result<Ran, ToolExecutionError> {
        let target = self.ready()?;
        let argv = self.argv(&target, false);
        let Some((program, args)) = argv.split_first() else {
            return Err(ToolExecutionError::other("the `ssh` setting is empty"));
        };
        let mut command = launcher.command(program);
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
        let mut child = command
            .spawn()
            .map_err(|error| ToolExecutionError::other(format!("could not run ssh: {error}")))?;
        if let (Some(mut pipe), Some(bytes)) = (child.stdin.take(), stdin) {
            pipe.write_all(bytes).ok();
        }
        let output = child
            .wait_with_output()
            .map_err(|error| ToolExecutionError::other(format!("ssh failed: {error}")))?;
        if output.status.code() == Some(SSH_FAILED) {
            self.lost(launcher, &target);
            return Err(lost(&target.host));
        }
        Ok(Ran {
            code: output.status.code(),
            out: String::from_utf8_lossy(&output.stdout).into_owned(),
            err: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Marks the link to `target` down and reconnects it in the background,
    /// unless that already runs.
    pub fn lost(&self, launcher: &Launcher, target: &Target) {
        let epoch = {
            let mut state = self.state();
            if state.target.as_ref() != Some(target) || state.reconnecting {
                return;
            }
            state.up = false;
            state.reconnecting = true;
            state.epoch
        };
        self.tell(format!("Connection to {} lost; reconnecting.", target.host));
        let (link, launcher, target) = (self.clone(), launcher.clone(), target.clone());
        let spawned = thread::Builder::new()
            .name("rig-remote-reconnect".to_owned())
            .spawn(move || link.reconnect(&launcher, &target, epoch));
        if spawned.is_err() {
            self.state().reconnecting = false;
        }
    }

    /// Tries `ssh host true` with a growing wait between tries, until it
    /// works or the target changed.
    fn reconnect(&self, launcher: &Launcher, target: &Target, epoch: u64) {
        let mut backoff = FIRST_BACKOFF;
        loop {
            thread::sleep(backoff);
            if self.state().epoch != epoch {
                return;
            }
            let argv = self.argv(target, false);
            let works = argv.split_first().is_some_and(|(program, args)| {
                let mut command = launcher.command(program);
                command.args(args).arg("true");
                command.stdin(Stdio::null()).stdout(Stdio::null());
                command.stderr(Stdio::null());
                command.status().is_ok_and(|status| status.success())
            });
            if works {
                let mut state = self.state();
                if state.epoch == epoch {
                    state.up = true;
                    state.reconnecting = false;
                    self.0.changed.notify_all();
                    drop(state);
                    self.tell(format!("Reconnected to {}.", target.host));
                }
                return;
            }
            backoff = (backoff * 2).min(MOST_BACKOFF);
        }
    }

    /// Tells the app `news` as a notice.
    fn tell(&self, news: String) {
        let sender = self.0.news.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(sender) = sender.as_ref() {
            sender.unbounded_send(news).ok();
        }
    }
}
