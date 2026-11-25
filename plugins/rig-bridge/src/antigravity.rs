//! Antigravity as a model, over `agy acp`. The program runs on a private
//! settings directory that holds one agent definition: no tools of its
//! own, none of its default prompt, rules, skills or other customizations,
//! Rig's system prompt as the agent's, and Rig's MCP server with every tool
//! listed. Only its sign-in is the user's.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::acp::{Agent, Opening, SERVER, Steering};
use crate::harness::{Part, link};

#[cfg(test)]
mod tests;

/// The agent the program is told to run as.
const AGENT: &str = "rig";

/// The directory under the settings directory that holds the sign-in.
const APP_DATA: &str = "antigravity-cli";

/// The file of the sign-in, in [`APP_DATA`].
const TOKEN: &str = "antigravity-oauth-token";

/// The program's log, in the session's directory.
const LOG: &str = "agy.log";

/// What the program logs when it could not load its agent and runs its
/// default one, with every tool, instead.
const FELL_BACK: &str = "falling back to default";

/// How long the program waits for one of Rig's tools, in seconds. Its own
/// limit is three minutes.
const TOOL_TIMEOUT_SECS: u64 = 86_400;

/// Antigravity.
pub(crate) struct Antigravity;

/// The agent definition: the system prompt under a header that turns
/// everything of Antigravity's own off and lists each of Rig's tools, which
/// a custom agent is otherwise given only a generic caller for.
fn definition(opening: &Opening) -> String {
    let mut definition = format!(
        "---\nname: {AGENT}\ndescription: A model for Rig, with Rig's tools.\nmainAgent: true\n\
         subagent: false\ntools: []\nexcludeDefaultComponents: true\n\
         inheritCustomizations: false\nmcpServers:\n  - name: {SERVER}\n"
    );
    if !opening.tools.is_empty() {
        definition.push_str("    tools:\n");
    }
    for tool in &opening.tools {
        // A JSON string is a YAML string.
        definition.push_str(&format!("      {}: {{eager: true}}\n", json!(tool)));
    }
    definition.push_str("---\n");
    definition.push_str(&opening.system);
    definition.push('\n');
    definition
}

/// The settings directory of the session.
fn settings(opening: &Opening) -> PathBuf {
    opening.dir.join("gemini")
}

/// Where the user's sign-in is kept.
pub(crate) fn signed_in() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME")?);
    Some(home.join(".gemini").join(APP_DATA).join(TOKEN))
}

impl Agent for Antigravity {
    fn name(&self) -> &'static str {
        "antigravity"
    }

    fn prepare(&self, command: &mut Command, opening: &Opening) -> Result<(), String> {
        let settings = settings(opening);
        let config = settings.join("config");
        let app_data = settings.join(APP_DATA);
        let servers = json!({ "mcpServers": { SERVER: {
            "url": opening.mcp_url,
            "headers": { "Authorization": format!("Bearer {}", opening.mcp_token) },
            "timeoutSeconds": TOOL_TIMEOUT_SECS,
        } } });
        let written = std::fs::create_dir_all(config.join("agents"))
            .and_then(|()| std::fs::create_dir_all(&app_data))
            .and_then(|()| {
                let agent = config.join("agents").join(format!("{AGENT}.md"));
                std::fs::write(agent, definition(opening))
            })
            .and_then(|()| std::fs::write(config.join("mcp_config.json"), servers.to_string()));
        written.map_err(|why| format!("could not write the session's files: {why}"))?;
        // The sign-in is shared, so a token the program renews is renewed
        // for the user's own sessions too.
        if let Some(token) = signed_in().filter(|token| token.is_file()) {
            link(&token, &app_data.join(TOKEN))
                .map_err(|why| format!("could not reach the sign-in: {why}"))?;
        }
        command
            .arg("acp")
            .arg("--gemini-dir")
            .arg(&settings)
            .args(["--app-data-dir", APP_DATA])
            .args(["--agent", AGENT])
            .arg("--disable-telemetry")
            .arg("--log-file")
            .arg(opening.dir.join(LOG))
            .env("AGY_CLI_DISABLE_AUTO_UPDATE", "1")
            // The sign-in answers, not an API key.
            .env_remove("GEMINI_API_KEY")
            .env_remove("GOOGLE_API_KEY");
        Ok(())
    }

    /// The agent definition is the system prompt.
    fn session_meta(&self, _: &Value, _: &str) -> Option<Value> {
        Some(json!({}))
    }

    /// The model's id names its effort.
    fn choose(&self, session: &str, _: &Value, opening: &Opening) -> Vec<(&'static str, Value)> {
        vec![(
            "session/set_model",
            json!({ "sessionId": session, "modelId": opening.model }),
        )]
    }

    /// A sign-in the program would start opens a browser and waits.
    fn sign_in(&self, _: &Value) -> Option<String> {
        None
    }

    fn login_hint(&self) -> &'static str {
        "agy"
    }

    fn note(&self, _: Option<&Path>) -> String {
        crate::harness::note(&format!("mcp_{SERVER}_read"), "", None)
    }

    /// The agent definition names each tool as the program starts.
    fn relists(&self) -> bool {
        false
    }

    /// A session keeps the tools of the agent definition it was made with:
    /// started again under another, its model reaches a new tool only
    /// through `call_mcp_tool`, which is not Rig's to allow.
    fn retools_saved(&self) -> bool {
        false
    }

    fn is_rig_tool(&self, tool_call: &Value, _: Option<&str>) -> bool {
        let name = tool_call.get("name").and_then(Value::as_str);
        name.is_some_and(|name| name.starts_with(&format!("mcp_{SERVER}_")))
    }

    /// The server is in the settings directory's MCP file, where its tool
    /// timeout can be set.
    fn serves_in_session(&self) -> bool {
        false
    }

    /// The program runs its default agent, silently, when it cannot load
    /// the one it was given.
    fn opened(&self, opening: &Opening) -> Result<(), String> {
        match std::fs::read_to_string(opening.dir.join(LOG)) {
            Ok(log) if !log.contains(FELL_BACK) => Ok(()),
            Ok(_) => Err("it did not load the agent Rig gave it".to_owned()),
            Err(why) => Err(format!(
                "its log, which says which agent it runs, is missing: {why}"
            )),
        }
    }

    fn input_leaves_out_cache(&self) -> bool {
        true
    }

    /// From 1.5.0 the program takes the next prompt while it answers one,
    /// as a message for that one. Earlier ones answer it afterwards.
    fn steer(&self, started: &Value, _: &str, _: &[Part]) -> Option<Steering> {
        let version = started.pointer("/agentInfo/version")?.as_str()?;
        let mut numbers = version.split('.').map(|number| {
            let digits = number.bytes().take_while(u8::is_ascii_digit).count();
            number.get(..digits)?.parse::<u32>().ok()
        });
        let (major, minor) = (numbers.next()??, numbers.next()??);
        ((major, minor) >= (1, 5)).then_some(Steering::Prompt)
    }
}

/// The `(id, display_name)` pairs `agy models` lists: two tab-separated columns.
pub(crate) fn models(listing: &str) -> Vec<(String, String)> {
    listing
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .filter_map(|(id, name)| {
            let id = id.trim();
            if id.is_empty() || id.contains(char::is_whitespace) {
                return None;
            }
            let name = name.trim();
            let display = if name.is_empty() { id } else { name };
            Some((id.to_owned(), display.to_owned()))
        })
        .collect()
}
