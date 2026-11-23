//! A stand-in agent for a launcher's tests: a POSIX shell script that says
//! it is ready, takes the screen and ends with the messages the agent
//! writes, so a test lists what the agent does as steps, one list for each
//! start. The script answers a request to describe itself with an empty
//! report, and counts its starts in `RIG_HOME/fake-agent-starts`.
//!
//! ```
//! use rig_harness_protocol::end::AgentEnd;
//! use rig_harness_protocol::fake_agent::{self, Step};
//!
//! // Reloads once, then quits.
//! let script = fake_agent::script(&[
//!     &[Step::Ready, Step::End(AgentEnd::Reload)],
//!     &[Step::Ready, Step::End(AgentEnd::Quit)],
//! ]);
//! assert!(script.starts_with("#!/bin/sh"));
//! ```

use std::fmt::Write as _;

use crate::end::{self, AgentEnd};
use crate::{READY_FILE, SCREEN_FILE, shell_word};

/// One thing the stand-in agent does.
#[derive(Clone, Debug)]
pub enum Step {
    /// Says the build started ([`RunDir::heard_ready`](crate::RunDir::heard_ready)).
    Ready,
    /// Says its front took the terminal's screen
    /// ([`RunDir::took_screen`](crate::RunDir::took_screen)).
    Screen,
    /// Shell commands of the test's own, in the agent's environment.
    Shell(String),
    /// Writes the record of this end without exiting, as an agent killed
    /// after it wrote it.
    Record(AgentEnd),
    /// Writes the record of this end and exits with its code.
    End(AgentEnd),
    /// Exits with this code and no record, as a crash or a plain exit does.
    Exit(u8),
}

/// The script of an agent whose n-th start runs the n-th list of `starts`,
/// and every later start the last list. A list that does not exit exits 0
/// with no record.
pub fn script(starts: &[&[Step]]) -> String {
    let mut script = String::from(
        "#!/bin/sh\n[ -n \"$RIG_DESCRIBE\" ] && exit 0\n\
         echo >> \"$RIG_HOME/fake-agent-starts\"\n\
         case $(( $(wc -l < \"$RIG_HOME/fake-agent-starts\") )) in\n",
    );
    let last = starts.len();
    for (start, steps) in (1..).zip(starts) {
        let pattern = if start == last {
            "*".to_owned()
        } else {
            start.to_string()
        };
        writeln!(script, "{pattern})\n{}  ;;", commands(steps)).ok();
    }
    script.push_str("esac\n");
    script
}

/// The shell commands that do `steps`, one line each, for a script of a
/// test's own.
pub fn commands(steps: &[Step]) -> String {
    steps
        .iter()
        .map(|step| format!("{}\n", step_script(step)))
        .collect()
}

/// The shell command that does `step`.
fn step_script(step: &Step) -> String {
    let say = |file: &str| format!(": > \"$RIG_RUN/{file}\"");
    let record = |end: &AgentEnd| {
        let file = end::FILE;
        format!(
            "printf %s {} > \"$RIG_RUN/{file}.new\" && mv \"$RIG_RUN/{file}.new\" \"$RIG_RUN/{file}\"",
            shell_word(&end.to_string())
        )
    };
    match step {
        Step::Ready => say(READY_FILE),
        Step::Screen => say(SCREEN_FILE),
        Step::Shell(commands) => commands.clone(),
        Step::Record(end) => record(end),
        Step::End(end) => format!("{}; exit {}", record(end), end.code()),
        Step::Exit(code) => format!("exit {code}"),
    }
}
