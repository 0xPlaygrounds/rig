use serde_json::json;

use super::{Copilot, Listed, agent_definition, models_from_session};
use crate::acp::Agent;

#[test]
fn agent_definition_places_system_prompt_in_body() {
    let rendered = agent_definition("You are Rig.");
    assert!(rendered.starts_with("---\nname: rig\n"));
    assert!(rendered.ends_with("---\nYou are Rig.\n"));
}

#[test]
fn models_from_session_deduplicates_entries() {
    let response = json!({
        "models": {
            "availableModels": [
                { "modelId": "auto", "name": "Auto" },
                { "modelId": "auto", "name": "Auto" },
                { "modelId": "gpt-6.1-terra", "name": "GPT-6.1 Terra" },
                { "modelId": "grok-4.7", "name": "" }
            ]
        }
    });
    assert_eq!(
        models_from_session(&response),
        vec![
            Listed {
                id: "auto".to_owned(),
                name: "Auto".to_owned()
            },
            Listed {
                id: "gpt-6.1-terra".to_owned(),
                name: "GPT-6.1 Terra".to_owned()
            },
            Listed {
                id: "grok-4.7".to_owned(),
                name: "grok-4.7".to_owned()
            },
        ]
    );
}

#[test]
fn copilot_filters_disabled_tools_banner_chunks() {
    assert!(!Copilot.keep_text("Info: Disabled tools: bash, edit"));
    assert!(!Copilot.keep_text("Info: Unknown tool name in the tool allowlist: \"rig\""));
    assert!(Copilot.keep_text("OK."));
}

#[test]
fn copilot_accepts_only_rig_prefixed_tool_titles() {
    let stripped = json!({ "title": "read" });
    let prefixed = json!({ "title": "rig-read" });
    let other = json!({ "title": "bash" });

    assert!(Copilot.is_rig_tool_with_title(&stripped, Some("rig-read")));
    assert!(!Copilot.is_rig_tool_with_title(&stripped, Some("bash")));
    assert!(!Copilot.is_rig_tool_with_title(&stripped, None));
    assert!(Copilot.is_rig_tool_with_title(&prefixed, None));
    assert!(!Copilot.is_rig_tool_with_title(&other, None));
}
