use rig::message::AssistantContent;
use rig::streaming::Item;
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::openai::wire::{DEEPSEEK, OpenAIConfig};

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn deepseek_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "deepseek",
        spec,
        "https://api.deepseek.com",
    )
    .await;
    let bound = OpenAIConfig::with_key(&DEEPSEEK, cassette.api_key("DEEPSEEK_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, bound)
}

pub(super) async fn with_deepseek_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "deepseek", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_deepseek_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, bound) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s: the shared model-listing fetch
/// must classify a rejected listing with provider, path and status context
/// (rig#2079), and only a real rejection proves it.
pub(super) async fn with_deepseek_cassette_bogus_key_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "deepseek",
        spec,
        "https://api.deepseek.com",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let bound = OpenAIConfig::with_key(&DEEPSEEK, "sk-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}

/// Everything one raw DeepSeek stream produced, in arrival order.
///
/// The shared `collect_raw_stream_observation` drops the terminal record and
/// the block ordering; both are the subject of these matrices, so the matrix
/// cells collect their own.
pub(super) struct RawStreamOutcome {
    pub(super) text: String,
    pub(super) tool_calls: Vec<rig::message::ToolCall>,
    pub(super) reasoning: String,
    /// `"reasoning"`, `"text"`, `"tool_call"` in the order the stream emitted
    /// them, deltas collapsed into their first occurrence's kind.
    pub(super) order: Vec<&'static str>,
    pub(super) final_record: Option<rig::completion::CompletionResponse>,
    pub(super) errors: Vec<String>,
}

impl RawStreamOutcome {
    pub(super) fn finish_reason(&self) -> Option<rig::completion::FinishReason> {
        self.final_record
            .as_ref()
            .and_then(|record| record.finish_reason().clone())
    }

    pub(super) fn tool_call_names(&self) -> Vec<&str> {
        self.tool_calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect()
    }
}

pub(super) async fn collect_raw_stream_outcome(
    mut stream: rig::streaming::CompletionStream,
) -> RawStreamOutcome {
    use futures::StreamExt as _;
    use rig::streaming::StreamEvent;

    let mut outcome = RawStreamOutcome {
        text: String::new(),
        tool_calls: Vec::new(),
        reasoning: String::new(),
        order: Vec::new(),
        final_record: None,
        errors: Vec::new(),
    };

    fn note(order: &mut Vec<&'static str>, kind: &'static str) {
        if order.last() != Some(&kind) {
            order.push(kind);
        }
    }

    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => {
                outcome.text.push_str(&text);
                note(&mut outcome.order, "text");
            }
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => {
                outcome.tool_calls.push(tool_call);
                note(&mut outcome.order, "tool_call");
            }
            Ok(Item::Event(StreamEvent::Arguments { .. })) => {}
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            })) => {
                outcome.reasoning.push_str(&reasoning.text);
                note(&mut outcome.order, "reasoning");
            }
            Ok(Item::Event(StreamEvent::Reasoning {
                text: reasoning, ..
            })) => {
                outcome.reasoning.push_str(&reasoning);
                note(&mut outcome.order, "reasoning");
            }
            Ok(
                Item::Unknown(_)
                | Item::Event(StreamEvent::Start { .. })
                | Item::Event(StreamEvent::End { .. }),
            ) => {}
            Err(error) => outcome.errors.push(error.to_string()),
        }
    }
    if let Ok(record) = stream.finish().await {
        outcome.final_record = Some(record);
    }

    outcome
}
