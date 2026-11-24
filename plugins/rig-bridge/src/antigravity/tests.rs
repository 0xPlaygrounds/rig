use serde_json::json;

use super::{Antigravity, definition, models};
use crate::acp::{Agent, Opening, Steering};

fn opening(tools: &[&str]) -> Opening {
    Opening {
        dir: "/session".into(),
        system: "Be brief.".to_owned(),
        model: "gemini-4-neon-high".to_owned(),
        effort: None,
        mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
        mcp_token: "secret".to_owned(),
        tools: tools.iter().map(|tool| (*tool).to_owned()).collect(),
    }
}

#[test]
fn the_agent_has_rig_s_prompt_and_tools_and_nothing_of_antigravity_s() {
    let definition = definition(&opening(&["read", "run_steel"]));

    assert_eq!(
        definition,
        "---\nname: rig\ndescription: A model for Rig, with Rig's tools.\nmainAgent: true\n\
         subagent: false\ntools: []\nexcludeDefaultComponents: true\n\
         inheritCustomizations: false\nmcpServers:\n  - name: rig\n    tools:\n      \
         \"read\": {eager: true}\n      \"run_steel\": {eager: true}\n---\nBe brief.\n",
    );
}

#[test]
fn the_listed_models_are_the_id_and_display_name_columns() {
    let listing = "Fetching available models...\ngemini-4-neon-high\tGemini 4 Neon (High)\n\
                   claude-opus-5-5-low\tClaude Opus 5.5 (Low)\nError: something else\n";

    assert_eq!(
        models(listing),
        [
            (
                "gemini-4-neon-high".to_owned(),
                "Gemini 4 Neon (High)".to_owned()
            ),
            (
                "claude-opus-5-5-low".to_owned(),
                "Claude Opus 5.5 (Low)".to_owned()
            ),
        ]
    );
}

#[test]
fn only_a_tool_of_rig_s_server_is_a_rig_tool() {
    let ours = json!({ "name": "mcp_rig_read", "title": "Run mcp_rig_read" });
    let native = json!({ "name": "run_command", "title": "Run command" });
    let other = json!({ "name": "mcp_rigged_read" });

    let judged = [ours, native, other].map(|call| Antigravity.is_rig_tool(&call, None));

    assert_eq!(judged, [true, false, false]);
}

#[test]
fn a_prompt_steers_the_running_one_from_1_5_0() {
    let steers = |version: &str| {
        let started = json!({ "agentInfo": { "name": "Antigravity", "version": version } });
        matches!(
            Antigravity.steer(&started, "s1", &[]),
            Some(Steering::Prompt)
        )
    };

    assert!(steers("1.5.0"));
    assert!(steers("1.5.15"));
    assert!(steers("2.0.1-beta"));
    // The 1.4 program answers the next prompt after the running one.
    assert!(!steers("1.4.9"));
    assert!(!steers("soon"));
    assert!(Antigravity.steer(&json!({}), "s1", &[]).is_none());
}
