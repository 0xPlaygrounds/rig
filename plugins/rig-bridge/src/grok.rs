//! Grok Build as a model, over `grok agent stdio`. The program runs with an
//! empty home of its own, so it finds no settings, rules, skills, hooks,
//! plugins or MCP servers of the user's or of another agent's, and with an
//! agent profile that leaves it the two tools it reaches MCP servers
//! through. Only its sign-in is the user's.

use std::path::{Path, PathBuf};
use std::process::Command;

use rig_harness::rig_core::completion::{Effort, Usage};
use serde_json::{Value, json};

use crate::acp::{Agent, Opening, SERVER};

#[cfg(test)]
mod tests;

/// The features of Grok's own that a model does not need, turned off.
const CONFIG: &str = r#"[cli]
auto_update = false

[features]
telemetry = false
feedback = false
lsp_tools = false
web_fetch = false
codebase_indexing = false
session_recap = false
session_search = false
image_gen = false
video_gen = false
ask_user_question = false
auto_wake = false

[memory]
enabled = false

[memory_v2]
enabled = false

[subagents]
enabled = false

[workflows]
enabled = false

[goal]
enabled = false

[managed_mcps]
enabled = false
"#;

/// The agent file Grok runs as: only the tools that list and call an MCP
/// server's tools, which Grok offers a model in place of the tools
/// themselves. Rig's system prompt is placed in the body as a fallback for
/// non-default model switches: `session/new`'s `systemPromptOverride`
/// replaces Grok's prompt cleanly on the default model (where `choose`
/// skips `session/set_model`), and when `session/set_model` must switch to
/// another model it rebuilds the prompt from this file's body instead of
/// dropping Rig's prompt.
fn profile(system: &str) -> String {
    let mut text = String::from(
        "---\n\
name: rig\n\
description: A model for Rig, with Rig's tools.\n\
tools: search_tool, use_tool\n\
mcpInheritance: none\n\
---\n",
    );
    if !system.is_empty() {
        text.push_str(system);
        if !system.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// The built-in tools the profile keeps.
const META_TOOLS: &[&str] = &["search_tool", "use_tool"];

/// Grok Build.
pub(crate) struct Grok;

/// Where the user's Grok sign-in file lives.
pub(crate) fn signed_in() -> Option<PathBuf> {
    std::env::var_os("GROK_AUTH_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let home = std::env::var_os("GROK_HOME")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    Some(
                        PathBuf::from(std::env::var_os("HOME").filter(|value| !value.is_empty())?)
                            .join(".grok"),
                    )
                })?;
            Some(home.join("auth.json"))
        })
}

impl Agent for Grok {
    fn name(&self) -> &'static str {
        "grok"
    }

    fn prepare(&self, command: &mut Command, opening: &Opening) -> Result<(), String> {
        // Where the user's sign-in is, before the home is replaced.
        let signed_in = signed_in();
        let home = opening.dir.join("home");
        let grok_home = home.join(".grok");
        let profile_path = opening.dir.join("rig.md");
        std::fs::create_dir_all(&grok_home)
            .and_then(|()| std::fs::write(grok_home.join("config.toml"), CONFIG))
            .and_then(|()| std::fs::write(&profile_path, profile(&opening.system)))
            .map_err(|why| format!("could not write the session's files: {why}"))?;
        command
            .env("HOME", &home)
            .env("GROK_HOME", &grok_home)
            .env("GROK_DISABLE_AUTOUPDATER", "1")
            .env("GROK_MEMORY", "0")
            .env("GROK_SUBAGENTS", "0")
            .env("GROK_WORKFLOWS", "0")
            .env("GROK_GOAL", "0")
            .env("GROK_STORAGE_MODE", "local")
            // Who a sign-in started from this process is for.
            .env("GROK_OAUTH2_REFERRER", "rig")
            // The sign-in answers, not an API key.
            .env_remove("XAI_API_KEY")
            .args(["agent", "--no-leader"]);
        if let Some(effort) = opening.effort {
            command.args(["--reasoning-effort", level(effort)]);
        }
        if let Some(signed_in) = signed_in {
            command.env("GROK_AUTH_PATH", signed_in);
        }
        command
            .arg("--agent-profile")
            .arg(&profile_path)
            .arg("stdio");
        Ok(())
    }

    fn session_meta(&self, _initialized: &Value, system: &str) -> Option<Value> {
        Some(json!({ "systemPromptOverride": system }))
    }

    fn choose(
        &self,
        session: &str,
        opened: &Value,
        opening: &Opening,
    ) -> Vec<(&'static str, Value)> {
        // `session/set_model` rebuilds the system prompt from `--agent-profile`,
        // discarding `systemPromptOverride` and prepending Grok's default prompt.
        // Skip it when `session/new` already opened on the requested model
        // (effort was already applied via `--reasoning-effort`).
        let current = opened
            .pointer("/models/currentModelId")
            .and_then(Value::as_str);
        if current == Some(opening.model.as_str()) {
            return Vec::new();
        }
        let mut chosen = json!({ "sessionId": session, "modelId": opening.model });
        if let (Some(effort), Some(chosen)) = (opening.effort, chosen.as_object_mut()) {
            chosen.insert(
                "_meta".to_owned(),
                json!({ "reasoningEffort": level(effort) }),
            );
        }
        vec![("session/set_model", chosen)]
    }

    /// As an editor extension Grok asks the client for each permission; as
    /// a generic client it refuses some itself without asking.
    fn initialize_meta(&self) -> Option<Value> {
        Some(json!({ "clientType": "extension" }))
    }

    fn prompt_meta(&self, prompt: &str) -> Option<Value> {
        Some(json!({ "promptId": prompt, "requestId": prompt }))
    }

    /// Grok says a prompt ended in notifications of its own, and may then
    /// never answer the request.
    fn ended(&self, method: &str, params: &Value, prompt: &str) -> Option<Value> {
        let text = |value: &Value, names: [&str; 2]| {
            let named = names.iter().find_map(|name| value.get(name)?.as_str());
            named.map(str::to_owned)
        };
        let (named, stopped) = if method.ends_with("x.ai/session/prompt_complete") {
            (
                text(params, ["promptId", "prompt_id"]),
                text(params, ["stopReason", "stop_reason"]),
            )
        } else if method.ends_with("x.ai/session_notification") || method == "_x.ai/session/update"
        {
            let update = params.get("update")?;
            let completed =
                update.get("sessionUpdate").and_then(Value::as_str) == Some("turn_completed");
            (
                text(update, ["prompt_id", "promptId"]).filter(|_| completed),
                text(update, ["stop_reason", "stopReason"]),
            )
        } else {
            return None;
        };
        (named.as_deref() == Some(prompt))
            .then(|| json!({ "stopReason": stopped.unwrap_or_else(|| "end_turn".to_owned()) }))
    }

    fn changes_effort(&self) -> bool {
        true
    }

    fn set_effort(&self, session: &str, effort: Effort) -> Option<(&'static str, Value)> {
        Some((
            "session/set_config_option",
            json!({
                "sessionId": session,
                "configId": "reasoning_effort",
                "value": level(effort),
            }),
        ))
    }

    fn sign_in(&self, initialized: &Value) -> Option<String> {
        let methods = initialized.get("authMethods")?.as_array()?;
        ["cached_token", "xai.api_key"]
            .into_iter()
            .find(|wanted| {
                methods
                    .iter()
                    .any(|method| method.get("id").and_then(Value::as_str) == Some(wanted))
            })
            .map(str::to_owned)
    }

    fn login_hint(&self) -> &'static str {
        "grok login"
    }

    fn note(&self, project: Option<&Path>) -> String {
        let how = ", found with `search_tool` and called with `use_tool`";
        crate::harness::note(&format!("{SERVER}__read"), how, Some(project))
    }

    /// A call of one of the two tools that reach MCP servers, of which the
    /// session has Rig's alone, and of no tool of another server.
    fn is_rig_tool(&self, tool_call: &Value, _: Option<&str>) -> bool {
        let built_in = tool_call
            .get("_meta")
            .and_then(|meta| meta.get("x.ai/tool"))
            .and_then(|tool| tool.get("name"))
            .and_then(Value::as_str);
        let named = tool_call
            .get("rawInput")
            .and_then(|input| input.get("tool_name"))
            .and_then(Value::as_str);
        let rigs = |name: &str| {
            name == "search_tool"
                || name
                    .strip_prefix(SERVER)
                    .is_some_and(|rest| rest.starts_with("__"))
        };
        built_in.is_some_and(|name| META_TOOLS.contains(&name)) && named.is_none_or(rigs)
    }

    /// Grok's own notification after each inference. Its input counter
    /// leaves out the tokens read from and written to the cache.
    fn counted(&self, method: &str, params: &Value) -> Option<Usage> {
        let update = params.get("update")?;
        let reported = method.ends_with("x.ai/session_notification")
            && update.get("sessionUpdate").and_then(Value::as_str) == Some(COUNTED);
        let counted = update.get("usage").filter(|_| reported)?;
        let count = |name: &str| counted.get(name).and_then(Value::as_u64);
        let (read, written) = (
            count("cache_read_input_tokens"),
            count("cache_creation_input_tokens"),
        );
        let mut usage = Usage::new();
        usage.input_tokens =
            count("input_tokens").map(|fresh| fresh + read.unwrap_or(0) + written.unwrap_or(0));
        usage.output_tokens = count("output_tokens");
        usage.cached_input_tokens = read;
        usage.cache_creation_input_tokens = written;
        usage.reasoning_tokens = count("reasoning_tokens");
        usage.total_tokens = usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(input, output)| input + output);
        Some(usage)
    }
}

/// The update in which Grok reports one inference's token counters.
const COUNTED: &str = "response_completed";

/// An effort as `--reasoning-effort` names it.
fn level(effort: Effort) -> &'static str {
    match effort {
        Effort::Minimal | Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        _ => "xhigh",
    }
}

/// The model ids `grok models` lists, the default first.
pub(crate) fn models(listing: &str) -> Vec<String> {
    let mut models: Vec<(bool, String)> = listing
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let (current, rest) = match line.split_at_checked(2)? {
                ("* ", rest) => (true, rest),
                ("- ", rest) => (false, rest),
                _ => return None,
            };
            let id = rest.split_whitespace().next()?;
            Some((current, id.to_owned()))
        })
        .collect();
    models.sort_by_key(|(current, _)| !current);
    models.into_iter().map(|(_, id)| id).collect()
}
