//! The completion stream contract, checked over a replayed stream: every
//! part of the response starts once and first, its fragments concatenate to
//! its finished text or argument JSON, its end is the part, and the stream
//! finishes with the response.

use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::error::ProviderError;
use rig_core::message::AssistantContent;
use rig_core::streaming::{CompletionStream, Item, PartKind, StreamEvent, Transcript};

/// One part of the response, as the stream delivered it.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveredPart {
    /// The kind its `Start` named.
    pub kind: PartKind,
    /// Its fragments, concatenated.
    pub text: String,
    /// Its `End`.
    pub part: AssistantContent,
}

/// What a stream delivered: its items, then the response it finished with.
#[derive(Debug)]
pub struct Updates {
    /// Every item, in order.
    pub items: Vec<Item<StreamEvent>>,
    /// The response, or the error that ended the stream.
    pub response: Result<CompletionResponse, ProviderError>,
}

/// Every item of `stream`, read to its end, and its response.
pub async fn collect_updates(mut stream: CompletionStream) -> Updates {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(item) => items.push(item),
            Err(_) => break,
        }
    }
    Updates {
        items,
        response: stream.finish().await,
    }
}

/// Check the part contract over `updates` against the response the stream
/// finished with, and return that response with each part as delivered.
/// Panics on any violation or on an error.
pub fn assert_update_contract(updates: &Updates) -> (CompletionResponse, Vec<DeliveredPart>) {
    let done = updates
        .response
        .as_ref()
        .unwrap_or_else(|error| panic!("the stream finishes: {error:?}"));
    let transcript =
        Transcript::parse(serde_json::to_value(&updates.items).expect("stream items serialize"))
            .unwrap_or_else(|error| panic!("the items are in the writer's order: {error}"));
    let mut delivered = Vec::new();
    for (index, expected) in done.choice.iter().enumerate() {
        let own: Vec<&StreamEvent> = transcript
            .events()
            .filter(|event| event.part().index() == index)
            .collect();
        let Some(StreamEvent::Start { kind, .. }) = own.first() else {
            panic!("part {index} starts first: {own:#?}");
        };
        let text: String = own
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Text { text, .. } | StreamEvent::Reasoning { text, .. } => {
                    Some(text.as_str())
                }
                StreamEvent::Arguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        let Some(StreamEvent::End { content: part, .. }) = own.last() else {
            panic!("part {index} ends: {own:#?}");
        };
        assert_eq!(part, expected, "part {index}'s end is choice[{index}]");
        match expected {
            AssistantContent::ToolCall(call) => assert_eq!(
                serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .as_ref(),
                Some(&call.function.arguments),
                "part {index}'s arguments"
            ),
            _ => assert_eq!(text, finished_text(expected), "part {index}'s fragments"),
        }
        delivered.push(DeliveredPart {
            kind: *kind,
            text,
            part: part.clone(),
        });
    }
    for event in transcript.events() {
        assert!(
            event.part().index() < done.choice.len(),
            "{event:?} names a part of choice"
        );
    }
    (done.clone(), delivered)
}

/// The text a finished part's fragments concatenate to.
fn finished_text(part: &AssistantContent) -> String {
    match part {
        AssistantContent::Text(text) => text.text.clone(),
        AssistantContent::Reasoning(reasoning) => reasoning.text.clone(),
        AssistantContent::ToolCall(_)
        | AssistantContent::Image(_)
        | AssistantContent::Opaque(_) => String::new(),
    }
}
