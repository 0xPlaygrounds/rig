//! Every agent surface sends through the model, so the request boundary
//! rejects an empty turn before the provider sees it: on an awaited run, a
//! streamed run and an extraction alike.

use futures::StreamExt;
use serde_json::json;

use crate::{
    agent::{AgentBuilder, AgentHook, DispatchAction, DispatchEvent, HookContext},
    completion::Message,
    extractor::ExtractorBuilder,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
};

/// Clears the history of every completion it sees: the one way an agent
/// request reaches the model with no messages.
struct EmptyHistory;

impl AgentHook for EmptyHistory {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.kind {
            rig_core::effect::EffectKind::Completion { request, stream } => {
                let mut request = request.clone();
                request.chat_history.clear();
                DispatchAction::Patch(rig_core::effect::EffectKind::Completion {
                    request,
                    stream: *stream,
                })
            }
            _ => DispatchAction::Proceed,
        }
    }
}

fn parsed(message: serde_json::Value) -> Message {
    serde_json::from_value(message).expect("an empty list parses")
}

/// A prompt and history carrying each empty piece, whether the history is
/// cleared on dispatch, and the text the rejection names.
fn cases() -> Vec<(Message, Vec<Message>, bool, &'static str)> {
    vec![
        (
            Message::user("hello"),
            Vec::new(),
            true,
            "request has an empty chat history",
        ),
        (
            parsed(json!({"role": "user", "content": []})),
            Vec::new(),
            false,
            "user message at index 0 has no content",
        ),
        (
            Message::user("hello"),
            vec![parsed(
                json!({"role": "assistant", "id": null, "content": []}),
            )],
            false,
            "assistant message at index 0 has no content",
        ),
    ]
}

#[tokio::test]
async fn a_streamed_run_rejects_each_empty_piece_before_the_provider() {
    for (prompt, history, clear, expected) in cases() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("unreachable"),
            MockStreamEvent::final_response(crate::completion::Usage::default()),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let runner = agent.prompt(prompt).history(history);
        let runner = if clear {
            runner.add_hook(EmptyHistory)
        } else {
            runner
        };
        let mut stream = runner.stream();
        let mut failure = None;
        while let Some(item) = stream.next().await {
            if let Err(error) = item {
                failure = Some(error);
            }
        }
        let error = failure.expect("the stream fails");
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(model.request_count(), 0, "{expected}");
    }
}

#[tokio::test]
async fn an_extraction_rejects_empty_input_before_the_provider() {
    let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
    let extractor = ExtractorBuilder::<serde_json::Value>::new(model.clone()).build();
    let error = extractor
        .extract(parsed(json!({"role": "user", "content": []})))
        .await
        .expect_err("the extraction fails");
    assert!(error.to_string().contains("has no content"), "{error}");
    assert_eq!(model.request_count(), 0);
}
