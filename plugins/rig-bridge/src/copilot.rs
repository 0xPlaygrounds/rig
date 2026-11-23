//! GitHub Copilot CLI (`copilot`) as a model, over `copilot --acp --stdio`.
//! The program runs with a private `COPILOT_HOME` holding one agent
//! (`rig.agent.md`) whose body carries Rig's system prompt, `--available-tools
//! rig` to turn off all built-in tools while keeping Rig's MCP server, and
//! built-in MCPs, custom instructions, auto-update, ask-user, and telemetry
//! disabled.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use rig_harness::prelude::Processes;
use rig_harness::rig_core::catalog::ReasoningSupport;
use rig_harness::rig_core::completion::Effort;
use serde_json::{Value, json};

use crate::acp::{Agent, Opening, SERVER, config_option, set_effort_request};
use crate::harness::{effort_wire, parse_effort};
use crate::rpc;
use crate::session::Dir;

#[cfg(test)]
mod tests;

/// The custom agent name written into `COPILOT_HOME/agents/`.
const AGENT: &str = "rig";

/// A model Copilot advertises in `session/new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Listed {
    pub id: String,
    pub name: String,
    pub reasoning: ReasoningSupport,
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

/// Applies the isolation flags and environment shared by Copilot sessions and
/// model-discovery probes.
fn configure_command(command: &mut Command, copilot_home: &Path) {
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
        ])
        .env("COPILOT_HOME", copilot_home)
        .env("COPILOT_AUTO_UPDATE", "false")
        .env("COPILOT_PLUGIN_DIR_ONLY", "true")
        .env("COPILOT_MCP_TOOL_CACHE", "false")
        .env("COPILOT_DISABLE_DESKTOP_NOTIFICATIONS", "1")
        .env("COPILOT_DISABLE_LOGIN_SHELL_ENV", "1");
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
        configure_command(command, &copilot_home);
        command.args([
            "--available-tools",
            SERVER,
            "--agent",
            AGENT,
            "--model",
            &opening.model,
        ]);
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
        let mut requests = Vec::new();
        let current = opened
            .pointer("/models/currentModelId")
            .and_then(Value::as_str);
        if current != Some(opening.model.as_str()) {
            requests.push((
                "session/set_model",
                json!({ "sessionId": session, "modelId": opening.model }),
            ));
        }
        // Apply startup effort via ACP config option (rather than CLI flag) so
        // `session/new` reports the model's unmodified default effort and models
        // without `reasoning_effort` skip it cleanly.
        if let Some(effort) = opening.effort
            && has_reasoning_option(opened)
        {
            requests.push(set_effort_request(session, effort_wire(effort, true)));
        }
        requests
    }

    fn effort_value(&self, effort: Effort) -> Option<&'static str> {
        Some(effort_wire(effort, true))
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

    fn is_rig_tool(&self, tool_call: &Value, recorded_title: Option<&str>) -> bool {
        recorded_title.is_some_and(is_rig_title)
            || tool_call
                .get("title")
                .and_then(Value::as_str)
                .is_some_and(is_rig_title)
    }
}

fn is_rig_title(title: &str) -> bool {
    title
        .strip_prefix(SERVER)
        .is_some_and(|rest| rest.starts_with('-') || rest.starts_with("__"))
}

fn has_reasoning_option(opened: &Value) -> bool {
    opened
        .get("configOptions")
        .and_then(Value::as_array)
        .is_none()
        || config_option(opened, "reasoning_effort").is_some()
}

/// Extracts reasoning effort levels advertised in `session/new` `configOptions`.
fn session_reasoning(result: &Value) -> ReasoningSupport {
    let Some(option) = config_option(result, "reasoning_effort") else {
        return ReasoningSupport::Unknown { default: None };
    };
    let mut levels = Vec::new();
    if let Some(items) = option.get("options").and_then(Value::as_array) {
        for item in items {
            if let Some(effort) = item
                .get("value")
                .and_then(Value::as_str)
                .and_then(parse_effort)
                && !levels.contains(&effort)
            {
                levels.push(effort);
            }
        }
    }
    let default = option
        .get("currentValue")
        .and_then(Value::as_str)
        .and_then(parse_effort);
    if levels.is_empty() {
        ReasoningSupport::Unknown { default }
    } else {
        ReasoningSupport::Listed {
            levels,
            budget: None,
            can_disable: false,
            default,
        }
    }
}

/// Extracts deduplicated models from a Copilot `session/new` response.
pub(crate) fn models_from_session(result: &Value) -> Vec<Listed> {
    let Some(entries) = result
        .pointer("/models/availableModels")
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let base_reasoning = session_reasoning(result);
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
        let reasoning = if id == "auto" {
            ReasoningSupport::Listed {
                levels: Vec::new(),
                budget: None,
                can_disable: false,
                default: None,
            }
        } else {
            base_reasoning.clone()
        };
        listed.push(Listed {
            id: id.to_owned(),
            name,
            reasoning,
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
    let Ok(dir) = Dir::create(scratch_root) else {
        return Vec::new();
    };
    let copilot_home = dir.path().join("copilot");
    if std::fs::create_dir_all(&copilot_home).is_err() {
        return Vec::new();
    }
    let mut command = processes.launcher().command(program);
    command.current_dir(dir.path());
    configure_command(&mut command, &copilot_home);
    let requests = vec![
        json!({
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
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": dir.path().to_string_lossy(), "mcpServers": [] },
        }),
    ];
    rpc::probe(command, requests, timeout, |msg| {
        msg.get("id").and_then(Value::as_u64) == Some(2)
    })
    .ok()
    .and_then(|msg| msg.get("result").map(models_from_session))
    .unwrap_or_default()
}
