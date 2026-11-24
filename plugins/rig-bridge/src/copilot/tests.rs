use rig_harness::rig_core::catalog::ReasoningSupport;
use rig_harness::rig_core::completion::Effort;
use serde_json::json;

use super::{Copilot, Listed, agent_definition, models_from_session};
use crate::acp::{Agent, Opening};

#[test]
fn agent_definition_places_system_prompt_in_body() {
    let rendered = agent_definition("You are Rig.");
    assert!(rendered.starts_with("---\nname: rig\n"));
    assert!(rendered.ends_with("---\nYou are Rig.\n"));
}

#[test]
fn models_from_session_deduplicates_entries_and_parses_reasoning_options() {
    let response = json!({
        "configOptions": [
            {
                "id": "reasoning_effort",
                "currentValue": "medium",
                "options": [
                    { "value": "low" },
                    { "value": "medium" },
                    { "value": "high" },
                    { "value": "xhigh" },
                    { "value": "max" }
                ]
            }
        ],
        "models": {
            "availableModels": [
                { "modelId": "auto", "name": "Auto" },
                { "modelId": "auto", "name": "Auto" },
                { "modelId": "gpt-6.1-terra", "name": "GPT-6.1 Terra" },
                { "modelId": "grok-4.7", "name": "" }
            ]
        }
    });
    let expected_reasoning = ReasoningSupport::Listed {
        levels: vec![
            Effort::Low,
            Effort::Medium,
            Effort::High,
            Effort::XHigh,
            Effort::Max,
        ],
        budget: None,
        can_disable: false,
        default: Some(Effort::Medium),
    };
    assert_eq!(
        models_from_session(&response),
        vec![
            Listed {
                id: "auto".to_owned(),
                name: "Auto".to_owned(),
                reasoning: ReasoningSupport::Listed {
                    levels: Vec::new(),
                    budget: None,
                    can_disable: false,
                    default: None,
                },
            },
            Listed {
                id: "gpt-6.1-terra".to_owned(),
                name: "GPT-6.1 Terra".to_owned(),
                reasoning: expected_reasoning.clone(),
            },
            Listed {
                id: "grok-4.7".to_owned(),
                name: "grok-4.7".to_owned(),
                reasoning: expected_reasoning,
            },
        ]
    );
}

#[test]
fn copilot_choose_applies_effort_only_when_session_advertises_reasoning_effort() {
    let opening = Opening {
        dir: "/work".into(),
        resume: None,
        system: String::new(),
        model: "gpt-6.1-terra".to_owned(),
        effort: Some(Effort::High),
        mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
        mcp_token: "tok".to_owned(),
        tools: Vec::new(),
    };
    let supported = json!({
        "models": { "currentModelId": "gpt-6.1-terra" },
        "configOptions": [{ "id": "reasoning_effort", "currentValue": "medium" }]
    });
    assert_eq!(
        Copilot.choose("s1", &supported, &opening),
        vec![(
            "session/set_config_option",
            json!({
                "sessionId": "s1",
                "configId": "reasoning_effort",
                "value": "high",
            }),
        )]
    );

    let unsupported = json!({
        "models": { "currentModelId": "gpt-6.1-terra" },
        "configOptions": [{ "id": "mode", "currentValue": "agent" }]
    });
    assert!(Copilot.choose("s1", &unsupported, &opening).is_empty());
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

    assert!(Copilot.is_rig_tool(&stripped, Some("rig-read")));
    assert!(!Copilot.is_rig_tool(&stripped, Some("bash")));
    assert!(!Copilot.is_rig_tool(&stripped, None));
    assert!(Copilot.is_rig_tool(&prefixed, None));
    assert!(!Copilot.is_rig_tool(&other, None));
    assert!(Copilot.cumulative_usage());
}
