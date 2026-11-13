//! How the agent ended: one record the agent writes in its launcher's
//! [`RunDir`] as it exits, which the launcher reads with the agent's exit
//! status. The record says what the exit code alone cannot: which session
//! to switch to, or why the run was refused. A record whose code is not the
//! one the agent exited with, as when the agent was killed after it wrote
//! it, is not read: the exit status says how the agent ended instead.
//!
//! ```no_run
//! # fn main() -> std::io::Result<()> {
//! use rig_harness_protocol::RunDir;
//! use rig_harness_protocol::end::AgentEnd;
//!
//! let run = RunDir::at("run");
//! AgentEnd::Switch(None).write(&run)?;
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::io;
use std::num::NonZeroU8;
use std::process::ExitStatus;

use crate::{REFUSED_EXIT_CODE, RELOAD_EXIT_CODE, RunDir, SIGNAL_EXIT_CODE, SessionId, fs};

/// The name of the record in the run directory.
pub(crate) const FILE: &str = "end";

/// How the agent ended, and so what its launcher does next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEnd {
    /// The user quit.
    Quit,
    /// The launcher restarts the session on its staged build (`/reload`).
    Reload,
    /// The launcher runs another session next: the one named, else a new
    /// one.
    Switch(Option<SessionId>),
    /// The run was refused before it did its work, for the reason given as
    /// a sentence, which the agent said on stderr.
    Refused(String),
    /// A signal (SIGINT, SIGTERM, SIGHUP) ended the run, or the launcher
    /// that started it went away.
    Signal,
    /// The run failed with this exit code.
    Failed(FailCode),
}

impl AgentEnd {
    /// The exit code the agent exits with for this end.
    pub fn code(&self) -> u8 {
        match self {
            Self::Quit => 0,
            Self::Reload | Self::Switch(_) => RELOAD_EXIT_CODE,
            Self::Refused(_) => REFUSED_EXIT_CODE,
            Self::Signal => SIGNAL_EXIT_CODE,
            Self::Failed(code) => code.get(),
        }
    }

    /// The code the launcher exits with when it ends on this end: [`Self::code`],
    /// but 0 for a reload or a switch it does not follow, such as a one-shot
    /// run's `/new`, so a script never gets the reload code.
    pub fn launcher_code(&self) -> u8 {
        match self {
            Self::Reload | Self::Switch(_) => 0,
            end => end.code(),
        }
    }

    /// Writes this end as `run`'s record, whole or not at all: to a
    /// temporary file first, then renamed over the record.
    ///
    /// # Errors
    ///
    /// When the file cannot be written or renamed.
    pub fn write(&self, run: &RunDir) -> io::Result<()> {
        let written = run.path().join(format!("{FILE}.new"));
        fs::write(&written, self.to_string())?;
        fs::rename(&written, run.path().join(FILE))
    }

    /// How the agent of `run` that exited with `status` ended: its record
    /// when it wrote one for that exit code, else what the code says alone
    /// ([`Self::of_status`]). A record that cannot be read or does not read
    /// as an end is no record: the launcher must still clean up after the
    /// agent and put the terminal back.
    pub fn read(run: &RunDir, status: ExitStatus) -> Self {
        let recorded = fs::read_to_string(run.path().join(FILE))
            .ok()
            .and_then(|text| Self::parse(&text));
        match recorded {
            Some(end) if status.code() == Some(i32::from(end.code())) => end,
            _ => Self::of_status(status),
        }
    }

    /// How an agent that exited with `status` and left no record for it
    /// ended: 0 is a quit, [`SIGNAL_EXIT_CODE`] a signal, and anything else
    /// a failure, the reload code and a death by a signal included
    /// ([`FailCode::or_failure`]). Only a record asks to reload or switch.
    pub fn of_status(status: ExitStatus) -> Self {
        let code = status.code().map(|code| u8::try_from(code).unwrap_or(1));
        match code {
            Some(0) => Self::Quit,
            Some(SIGNAL_EXIT_CODE) => Self::Signal,
            code => Self::Failed(FailCode::or_failure(code.unwrap_or(1))),
        }
    }
}

/// The record's text: a word, then a session id or an exit code after a
/// space, or a refusal's reason on the lines after it.
impl fmt::Display for AgentEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quit => f.write_str("quit"),
            Self::Reload => f.write_str("reload"),
            Self::Switch(None) => f.write_str("switch"),
            Self::Switch(Some(session)) => write!(f, "switch {session}"),
            Self::Refused(why) => write!(f, "refused\n{why}"),
            Self::Signal => f.write_str("signal"),
            Self::Failed(code) => write!(f, "failed {}", code.get()),
        }
    }
}

impl AgentEnd {
    /// The end a record's `text` says, or `None` when it says none.
    fn parse(text: &str) -> Option<Self> {
        let (first, rest) = text.split_once('\n').unwrap_or((text, ""));
        let (word, value) = first.split_once(' ').unwrap_or((first, ""));
        match (word, value) {
            ("quit", "") => Some(Self::Quit),
            ("reload", "") => Some(Self::Reload),
            ("switch", "") => Some(Self::Switch(None)),
            ("switch", session) => Some(Self::Switch(Some(session.parse().ok()?))),
            ("refused", "") => Some(Self::Refused(rest.to_owned())),
            ("signal", "") => Some(Self::Signal),
            ("failed", code) => code.parse().ok().and_then(FailCode::new).map(Self::Failed),
            _ => None,
        }
    }
}

/// The exit code of a failed run: never 0, nor a code the launcher reads
/// as another end ([`RELOAD_EXIT_CODE`], [`SIGNAL_EXIT_CODE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FailCode(NonZeroU8);

impl FailCode {
    /// The plain failure, 1.
    pub const FAILURE: Self = Self(NonZeroU8::MIN);

    /// `code` as a failure's code, or `None` for 0 and the codes the
    /// launcher reads as another end.
    pub const fn new(code: u8) -> Option<Self> {
        match code {
            RELOAD_EXIT_CODE | SIGNAL_EXIT_CODE => None,
            code => match NonZeroU8::new(code) {
                Some(code) => Some(Self(code)),
                None => None,
            },
        }
    }

    /// `code` as a failure's code, or [`Self::FAILURE`] where
    /// [`Self::new`] refuses it, such as a child process's 75 passed on.
    pub fn or_failure(code: u8) -> Self {
        Self::new(code).unwrap_or(Self::FAILURE)
    }

    /// The exit code.
    pub const fn get(self) -> u8 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests;
