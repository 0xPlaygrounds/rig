//! A helper as a live view the user can watch, through a command the user
//! configures: `[agents] view` in `settings.toml`, the argv of a program
//! that opens a window (a terminal tab, a multiplexer pane) running a
//! command. rig appends `rig attach <id>` to it, unless the argv places
//! that itself with `{rig}` and `{id}`; `{name}` is the helper's name. There
//! is no default, and only an unrestricted session opens a view: a
//! sandboxed worker never starts programs outside its sandbox.
//!
//! ```toml
//! [agents]
//! # A new terminal window running the view:
//! view = ["my-terminal", "--title", "{name}", "--"]
//! ```

use std::process::{Command, Stdio};

use rig_harness::prelude::*;
use rig_protocol::SessionId;
use serde::Deserialize;

/// The `[agents]` table of `settings.toml`.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct AgentsSettings {
    /// The command that opens a helper's live view; empty: none.
    pub view: Vec<String>,
}

/// The settings line a user adds to have views.
pub const VIEW_SETTING: &str =
    "[agents] view = [\"PROGRAM\", \"ARG\", …] in settings.toml (rig appends `rig attach <id>`)";

/// The argv of the view of helper `id` named `name`: `view` with its
/// placeholders filled, then `rig attach <id>` unless it placed `{rig}`.
/// `None` when no view command is set.
pub fn argv(view: &[String], rig: &str, name: &str, id: &SessionId) -> Option<Vec<String>> {
    let (program, _) = view.split_first()?;
    if program.is_empty() {
        return None;
    }
    let fill = |word: &String| {
        word.replace("{name}", name)
            .replace("{id}", &id.0)
            .replace("{rig}", rig)
    };
    let mut argv: Vec<String> = view.iter().map(fill).collect();
    if !view.iter().any(|w| w.contains("{rig}")) {
        argv.extend([rig.to_owned(), "attach".to_owned(), id.0.clone()]);
    }
    Some(argv)
}

/// Opens the view, and says how it went, for the tool's answer.
pub fn open(view: &[String], name: &str, id: &SessionId) -> String {
    let rig = std::env::var(rig_harness::harness_protocol::env::LAUNCHER)
        .unwrap_or_else(|_| "rig".to_owned());
    let Some(argv) = argv(view, &rig, name, id) else {
        return format!(" (No live view: no view command is set; add {VIEW_SETTING}.)");
    };
    let Some((program, args)) = argv.split_first() else {
        return String::new();
    };
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output();
    match out {
        Ok(out) if out.status.success() => {
            let said = String::from_utf8_lossy(&out.stdout);
            match said.lines().next().map(str::trim).filter(|l| !l.is_empty()) {
                Some(first) => format!(" Its live view is open ({first})."),
                None => " Its live view is open.".to_owned(),
            }
        }
        Ok(out) => format!(
            " (The live view did not open: {})",
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        Err(error) => format!(" (The live view did not open: {program}: {error})"),
    }
}
