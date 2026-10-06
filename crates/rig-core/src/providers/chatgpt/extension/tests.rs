//! ChatGPT's options serialize to their one section, writing no leaf the
//! request or its generation options own, and its extras read a reply
//! built in the test. The recorded reply is read in the Responses wire's
//! tests.

use super::*;
use crate::completion::ProviderOptions;
use crate::providers::openai::extension::CyberAccess;
use serde_json::json;

#[test]
fn a_fully_set_section_serializes_under_the_responses_route() {
    let options = ChatGptOptions::default()
        .prompt_cache_key("conversation-1")
        .client_metadata("originator", "rig")
        .access_programs(AccessPrograms::cyber(CyberAccess::Standard));
    let entry = ProviderOptions::new()
        .with::<ChatGpt>(&options)
        .expect("the options are sections");
    let sections = serde_json::to_value(entry.get::<ChatGpt>()).expect("the sections serialize");
    assert_eq!(
        sections,
        json!({"openai.responses": {
            "prompt_cache_key": "conversation-1",
            "client_metadata": {"originator": "rig"},
            "access_programs": {"cyber": "standard"}
        }})
    );
    for reserved in [
        "store",
        "model",
        "input",
        "instructions",
        "reasoning",
        "service_tier",
        "text",
    ] {
        assert!(
            sections["openai.responses"].get(reserved).is_none(),
            "{reserved}"
        );
    }
}

/// A terminal response with a message item reads its phase.
#[test]
fn extras_read_a_message_phase() {
    let extras = ChatGptExtras::from_reply(
        &Api::from_static("openai.responses"),
        &json!({
            "service_tier": "priority",
            "output": [
                {"type": "reasoning", "id": "rs_1"},
                {"type": "message", "id": "msg_1", "phase": "commentary"}
            ]
        }),
    )
    .expect("the view reads");
    assert_eq!(extras.service_tier.as_deref(), Some("priority"));
    assert_eq!(
        extras.phases,
        Some(vec![ItemPhase {
            id: "msg_1".to_owned(),
            phase: Some("commentary".to_owned()),
        }])
    );
}
