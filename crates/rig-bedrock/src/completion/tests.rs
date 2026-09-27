//! Converse request rules for Claude models that reject forced tool choice.
//! These pin what Rig refuses to send before any traffic, so no recorded
//! reply can witness them.

use super::*;
use rig_core::completion::ToolDefinition;

fn forced(choice: ToolChoice) -> CompletionRequest {
    let mut request = CompletionRequest::new("extract");
    request.tools = vec![ToolDefinition {
        name: "submit".to_owned(),
        description: "Submit the result.".to_owned(),
        parameters: serde_json::json!({"type": "object", "properties": {}}),
    }];
    request.tool_choice = Some(choice);
    request
}

#[test]
fn new_claude_constants_use_the_model_card_profile_ids() {
    assert_eq!(ANTHROPIC_CLAUDE_OPUS_5_5, "us.anthropic.claude-opus-5-5");
    assert_eq!(ANTHROPIC_CLAUDE_FABLE_5_1, "us.anthropic.claude-fable-5-1");
}

#[test]
fn forced_tool_choice_fails_to_encode_for_every_listed_id() {
    for model in REJECTS_FORCED_TOOL_CHOICE {
        let wire = Converse::new(*model);
        assert!(
            !wire
                .describe()
                .capabilities
                .completion
                .accepts_forced_tool_choice,
            "{model}"
        );
        for choice in [
            ToolChoice::Required,
            ToolChoice::Specific {
                function_names: vec!["submit".to_owned()],
            },
        ] {
            for mode in [Mode::Unary, Mode::Streaming] {
                let Err(error) = wire.encode(forced(choice.clone()), mode) else {
                    panic!("{model} must not be sent a forced tool choice");
                };
                let message = ProviderError::from(error).to_string();
                assert!(message.contains(model), "{message}");
                assert!(message.contains("ToolChoice::Auto"), "{message}");
            }
        }
        assert!(wire.encode(forced(ToolChoice::Auto), Mode::Unary).is_ok());
    }
}

#[test]
fn a_per_request_model_override_is_checked_too() {
    let wire = Converse::new(ANTHROPIC_CLAUDE_OPUS_5);
    let mut request = forced(ToolChoice::Required);
    request.model = Some(ANTHROPIC_CLAUDE_OPUS_5_5.to_owned());
    assert!(wire.encode(request, Mode::Unary).is_err());
}

#[test]
fn prefix_sharing_models_keep_forced_tool_choice() {
    for model in [
        ANTHROPIC_CLAUDE_OPUS_5,
        "anthropic.claude-opus-5",
        "global.anthropic.claude-opus-5",
        "us.anthropic.claude-fable-5",
    ] {
        let wire = Converse::new(model);
        assert!(
            wire.describe()
                .capabilities
                .completion
                .accepts_forced_tool_choice
        );
        assert!(
            wire.encode(forced(ToolChoice::Required), Mode::Unary)
                .is_ok()
        );
    }
}

#[test]
fn a_raw_forced_tool_choice_in_additional_params_fails_too() {
    let mut request = forced(ToolChoice::Auto);
    request.tool_choice = None;
    request.additional_params = Some(serde_json::json!({"tool_choice": {"type": "any"}}));
    assert!(
        Converse::new(ANTHROPIC_CLAUDE_OPUS_5_5)
            .encode(request.clone(), Mode::Unary)
            .is_err()
    );
    assert!(
        Converse::new(ANTHROPIC_CLAUDE_OPUS_5)
            .encode(request, Mode::Unary)
            .is_ok()
    );
}
