//! A helper as a live view in a fux pane: when this agent runs inside fux
//! (`TERM_PROGRAM=fux` with a `FUX_SOCKET`), `start_agent {view: true}`
//! runs `rig attach <id>` in a pane of its own, the way `fux help-agents`
//! starts a program, so the user can watch the helper and take it over.
//! Only an unrestricted session does: a sandboxed worker never gets fux's
//! socket.

use std::process::{Command, Stdio};

use rig_protocol::SessionId;

/// The `fux` arguments that open `id` in a pane named `name`, running
/// `rig`: each a separate argument, nothing interpolated into a shell.
pub fn fux_args(rig: &str, name: &str, id: &SessionId) -> Vec<String> {
    [
        "opts", "str", name, "name", "str", rig, "program", "str", "attach", "arg", "str", &id.0,
        "arg", "run",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

/// Opens the view, and says how it went, for the tool's answer.
pub fn open(name: &str, id: &SessionId) -> String {
    let in_fux = std::env::var("TERM_PROGRAM").is_ok_and(|t| t == "fux")
        && std::env::var_os("FUX_SOCKET").is_some();
    if !in_fux {
        return " (No live view: this agent doesn't run inside fux.)".to_owned();
    }
    let rig = std::env::var(rig_harness::harness_protocol::env::LAUNCHER)
        .unwrap_or_else(|_| "rig".to_owned());
    let fux = std::env::var("FUX_BIN").unwrap_or_else(|_| "fux".to_owned());
    let out = Command::new(fux)
        .args(fux_args(&rig, name, id))
        .stdin(Stdio::null())
        .output();
    match out {
        Ok(out) if out.status.success() => {
            let answer = String::from_utf8_lossy(&out.stdout);
            let pane = answer.split_whitespace().next().unwrap_or_default();
            format!(" It is open in fux pane {pane}.")
        }
        Ok(out) => format!(
            " (The fux view did not open: {})",
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        Err(error) => format!(" (The fux view did not open: {error})"),
    }
}
