//! GitHub Copilot CLI (`copilot`) as a model, over `copilot --acp --stdio`.
//! The program runs with a private `COPILOT_HOME` holding one agent
//! (`rig.agent.md`) whose body carries Rig's system prompt, `--available-tools
//! rig` to turn off all built-in tools while keeping Rig's MCP server, and
//! built-in MCPs, custom instructions, auto-update, ask-user, and telemetry
//! disabled.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use rig_harness::prelude::Processes;
use rig_harness::rig_core::completion::Effort;
use rig_harness::rig_tools::process::Process;
use serde_json::{Value, json};

use crate::acp::{Agent, Opening, SERVER};

#[cfg(test)]
mod tests;

/// The custom agent name written into `COPILOT_HOME/agents/`.
const AGENT: &str = "rig";

/// A model Copilot advertises in `session/new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Listed {
    pub id: String,
    pub name: String,
}

/// GitHub Copilot CLI.
pub(crate) struct Copilot;

/// Renders `rig.agent.md` with Rig's system prompt as its body.
fn agent_definition(system: &str) -> String {
    let mut text =
        format!("---\nname: {AGENT}\ndescription: A model for Rig, with Rig's tools.\n---\n");
    if !system.is_empty() {
        text.push_str(system);
        if !system.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// An effort level as `copilot --reasoning-effort` names it.
fn level(effort: Effort) -> &'static str {
    match effort {
        Effort::Minimal => "minimal",
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::XHigh => "xhigh",
        _ => "max",
    }
}

impl Agent for Copilot {
    fn name(&self) -> &'static str {
        "copilot"
    }

    fn prepare(&self, command: &mut Command, opening: &Opening) -> Result<(), String> {
        let copilot_home = opening.dir.join("copilot");
        let agents_dir = copilot_home.join("agents");
        std::fs::create_dir_all(&agents_dir)
            .and_then(|()| {
                std::fs::write(
                    agents_dir.join(format!("{AGENT}.agent.md")),
                    agent_definition(&opening.system),
                )
            })
            .map_err(|why| format!("could not write the session's files: {why}"))?;
        command
            .args([
                "--acp",
                "--stdio",
                "--disable-builtin-mcps",
                "--no-custom-instructions",
                "--no-auto-update",
                "--no-ask-user",
                "--no-bash-env",
                "--disable-client-telemetry",
                "--available-tools",
                SERVER,
                "--agent",
                AGENT,
                "--model",
                &opening.model,
            ])
            .env("COPILOT_HOME", &copilot_home)
            .env("COPILOT_AUTO_UPDATE", "false")
            .env("COPILOT_PLUGIN_DIR_ONLY", "true")
            .env("COPILOT_MCP_TOOL_CACHE", "false")
            .env("COPILOT_DISABLE_DESKTOP_NOTIFICATIONS", "1")
            .env("COPILOT_DISABLE_LOGIN_SHELL_ENV", "1");
        if let Some(effort) = opening.effort {
            command.args(["--reasoning-effort", level(effort)]);
        }
        Ok(())
    }

    /// The custom agent file carries Rig's system prompt.
    fn session_meta(&self, _: &Value, _: &str) -> Option<Value> {
        Some(json!({}))
    }

    fn choose(
        &self,
        session: &str,
        opened: &Value,
        opening: &Opening,
    ) -> Vec<(&'static str, Value)> {
        let current = opened
            .pointer("/models/currentModelId")
            .and_then(Value::as_str);
        if current == Some(opening.model.as_str()) {
            return Vec::new();
        }
        vec![(
            "session/set_model",
            json!({ "sessionId": session, "modelId": opening.model }),
        )]
    }

    fn sign_in(&self, _: &Value) -> Option<String> {
        None
    }

    fn login_hint(&self) -> &'static str {
        "copilot login"
    }

    fn note(&self, project: Option<&Path>) -> String {
        crate::harness::note(&format!("{SERVER}-read"), "", Some(project))
    }

    fn keep_text(&self, text: &str) -> bool {
        let trimmed = text.trim_start();
        !trimmed.starts_with("Info: Disabled tools:")
            && !trimmed.starts_with("Info: Unknown tool name in the tool allowlist:")
    }

    fn is_rig_tool(&self, tool_call: &Value) -> bool {
        tool_call
            .get("title")
            .and_then(Value::as_str)
            .is_some_and(is_rig_title)
    }

    fn is_rig_tool_with_title(&self, tool_call: &Value, recorded_title: Option<&str>) -> bool {
        recorded_title.is_some_and(is_rig_title) || self.is_rig_tool(tool_call)
    }
}

fn is_rig_title(title: &str) -> bool {
    title
        .strip_prefix(SERVER)
        .is_some_and(|rest| rest.starts_with('-') || rest.starts_with("__"))
}

/// Extracts deduplicated models from a Copilot `session/new` response.
pub(crate) fn models_from_session(result: &Value) -> Vec<Listed> {
    let Some(entries) = result
        .pointer("/models/availableModels")
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut listed = Vec::new();
    for entry in entries {
        let Some(id) = entry.get("modelId").and_then(Value::as_str) else {
            continue;
        };
        if id.is_empty() || !seen.insert(id.to_owned()) {
            continue;
        }
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .unwrap_or(id)
            .to_owned();
        listed.push(Listed {
            id: id.to_owned(),
            name,
        });
    }
    listed
}

/// Discovers Copilot models via a short-lived `copilot --acp --stdio` probe
/// (`initialize` + `session/new`) inside `scratch_root`.
pub(crate) fn discover_models(
    processes: &Processes,
    program: &Path,
    scratch_root: &Path,
    timeout: Duration,
) -> Vec<Listed> {
    let dir = scratch_root.join(format!(
        "rig-bridge-{}-copilot-list-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    let copilot_home = dir.join("copilot");
    if std::fs::create_dir_all(&copilot_home).is_err() {
        return Vec::new();
    }
    let result = probe_models(processes, program, &dir, &copilot_home, timeout);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn probe_models(
    processes: &Processes,
    program: &Path,
    dir: &Path,
    copilot_home: &Path,
    timeout: Duration,
) -> Vec<Listed> {
    let mut command = processes.launcher().command(program);
    command
        .current_dir(dir)
        .args([
            "--acp",
            "--stdio",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
            "--no-auto-update",
            "--no-ask-user",
            "--no-bash-env",
            "--disable-client-telemetry",
        ])
        .env("COPILOT_HOME", copilot_home)
        .env("COPILOT_AUTO_UPDATE", "false")
        .env("COPILOT_PLUGIN_DIR_ONLY", "true")
        .env("COPILOT_MCP_TOOL_CACHE", "false")
        .env("COPILOT_DISABLE_DESKTOP_NOTIFICATIONS", "1")
        .env("COPILOT_DISABLE_LOGIN_SHELL_ENV", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let Ok(mut process) = Process::start_with_stdio(command) else {
        return Vec::new();
    };
    let (Some(mut stdin), Some(stdout)) = (process.take_stdin(), process.take_stdout()) else {
        return Vec::new();
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let cwd = dir.to_string_lossy().into_owned();
    let reader = std::thread::spawn(move || {
        let init = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": { "readTextFile": false, "writeTextFile": false },
                    "terminal": false,
                },
                "clientInfo": { "name": "rig", "version": env!("CARGO_PKG_VERSION") },
            }
        });
        let new = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": cwd, "mcpServers": [] },
        });
        if writeln!(stdin, "{init}").is_err()
            || writeln!(stdin, "{new}").is_err()
            || stdin.flush().is_err()
        {
            return;
        }
        let mut lines = BufReader::new(stdout).lines();
        while let Some(Ok(line)) = lines.next() {
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if message.get("id").and_then(Value::as_u64) == Some(2) {
                let listed = message
                    .get("result")
                    .map(models_from_session)
                    .unwrap_or_default();
                let _ = tx.send(listed);
                return;
            }
        }
    });
    let deadline = Instant::now() + timeout;
    let listed = rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .unwrap_or_default();
    drop(process);
    let _ = reader.join();
    listed
}
