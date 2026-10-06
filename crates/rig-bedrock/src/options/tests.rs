use rig_core::completion::{
    CacheRetention, CompletionRequest, Effort, GenerationOptions, Message, OnUnsupported, Reasoning,
};
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::{Value, json};

use crate::completion::{
    AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_HAIKU_4_5, ANTHROPIC_CLAUDE_OPUS_4_5,
    ANTHROPIC_CLAUDE_SONNET_5, Converse, LLAMA_3_8B_INSTRUCT,
};

fn sent(model: &str, request: CompletionRequest) -> Result<Value, ProviderError> {
    let wire = Converse::new(model);
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    Ok(serde_json::to_value(&encoded.body)?)
}

fn with(options: GenerationOptions) -> CompletionRequest {
    CompletionRequest::new("hi")
        .max_tokens(4096)
        .options(options)
}

fn refused(result: Result<Value, ProviderError>) -> Option<&'static str> {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => Some(option.option),
        _ => None,
    }
}

#[test]
fn reasoning_goes_in_the_models_own_fields() {
    let body = sent(
        ANTHROPIC_CLAUDE_SONNET_5,
        with(GenerationOptions::default().reasoning(Effort::Low)).additional_params(json!({
            "anthropic_beta": ["interleaved-thinking-2025-05-14"],
            "thinking": {"display": "summarized"},
        })),
    )
    .expect("Sonnet 5 takes an effort");
    assert_eq!(
        body["additionalModelRequestFields"],
        json!({
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "low"},
            "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        })
    );
    let body = sent(
        ANTHROPIC_CLAUDE_HAIKU_4_5,
        with(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 2048 })),
    )
    .expect("Haiku 4.5 takes a budget");
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );
    let body = sent(
        ANTHROPIC_CLAUDE_OPUS_4_5,
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("Opus 4.5 does not think unless asked");
    assert!(body.get("additionalModelRequestFields").is_none(), "{body}");
    for (model, reasoning) in [
        (ANTHROPIC_CLAUDE_HAIKU_4_5, Reasoning::Effort(Effort::High)),
        (
            ANTHROPIC_CLAUDE_SONNET_5,
            Reasoning::Budget { tokens: 2048 },
        ),
        (ANTHROPIC_CLAUDE_SONNET_5, Reasoning::Effort(Effort::Max)),
        (LLAMA_3_8B_INSTRUCT, Reasoning::Effort(Effort::Low)),
    ] {
        assert_eq!(
            refused(sent(
                model,
                with(GenerationOptions::default().reasoning(reasoning))
            )),
            Some("reasoning"),
            "{model}: {reasoning:?}"
        );
    }
}

#[test]
fn checkpoints_follow_the_cache_retention() {
    let request = |cache| {
        CompletionRequest::from(vec![Message::system("Be brief."), Message::user("hi")])
            .options(GenerationOptions::default().cache(cache))
    };
    let body = sent(ANTHROPIC_CLAUDE_SONNET_5, request(CacheRetention::Long)).expect("1h");
    assert_eq!(
        body["system"][1],
        json!({"cachePoint": {"type": "default", "ttl": "1h"}})
    );
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"cachePoint": {"type": "default", "ttl": "1h"}})
    );
    let body = sent(AMAZON_NOVA_LITE, request(CacheRetention::Short)).expect("Nova caches");
    assert_eq!(
        body["system"][1],
        json!({"cachePoint": {"type": "default"}})
    );
    assert_eq!(
        refused(sent(LLAMA_3_8B_INSTRUCT, request(CacheRetention::Short))),
        Some("cache")
    );
}

/// Supersedes #1833: a checkpoint after a reasoning turn is never skipped
/// in silence. The default policy refuses it; `Ignore` places the system
/// checkpoint and skips the message one with a warning.
#[test]
fn a_message_checkpoint_after_reasoning_is_reported_not_skipped() {
    let reasoning = rig_core::message::AssistantContent::reasoning("thinking it over")
        .with_native(json!({"reasoningContent": {"reasoningText": {
            "text": "thinking it over", "signature": "sig"}}}));
    let history = vec![
        Message::system("Be brief."),
        Message::user("hi"),
        Message::Assistant(rig_core::message::AssistantMessage::new(vec![
            reasoning,
            rig_core::message::AssistantContent::text("hello"),
        ])),
        Message::user("again"),
    ];
    let request = |policy| {
        CompletionRequest::from(history.clone()).options(
            GenerationOptions::default()
                .cache(CacheRetention::Short)
                .on_unsupported(policy),
        )
    };
    let wire = Converse::new(ANTHROPIC_CLAUDE_SONNET_5);
    let prepared = |policy| {
        let request = Completion::prepare(request(policy), &wire.describe())?;
        let encoded = wire.encode(request, Mode::Unary)?;
        Ok::<_, ProviderError>(serde_json::to_value(&encoded.body)?)
    };
    let has_reasoning = |body: &Value| {
        body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|message| message["content"].as_array().into_iter().flatten())
            .any(|block| block.get("reasoningContent").is_some())
    };
    match prepared(OnUnsupported::Error) {
        Err(ProviderError::UnsupportedOption(option)) => assert_eq!(option.option, "cache"),
        // A history whose reasoning does not replay has no reasoning turn.
        Ok(body) => assert!(!has_reasoning(&body), "{body}"),
        Err(other) => panic!("{other}"),
    }
    let capture = rig_core::test_utils::TraceCapture::default();
    let body =
        tracing::subscriber::with_default(capture.subscriber(), || prepared(OnUnsupported::Ignore))
            .expect("Ignore encodes the rest");
    assert_eq!(
        body["system"][1],
        json!({"cachePoint": {"type": "default"}})
    );
    if has_reasoning(&body) {
        let last = body["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .cloned()
            .unwrap_or_default();
        assert!(
            !last["content"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|block| block.get("cachePoint").is_some()),
            "{body}"
        );
        assert_eq!(
            capture
                .warnings()
                .iter()
                .filter(|warning| warning.contains("option=cache"))
                .count(),
            1
        );
    }
}
