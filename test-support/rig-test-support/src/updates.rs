//! The `updates()` contract, checked over a replayed stream: every part of
//! the response starts once and first, its deltas concatenate to its
//! finished text or argument JSON, its last end is the part, and `Done`
//! comes last.

use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::error::ErrorReport;
use rig_core::message::{AssistantContent, ReasoningContent};
use rig_core::streaming::{CompletionStream, PartKind, Update};

/// One part of the response, as the updates delivered it.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveredPart {
    /// The kind its `Start` named.
    pub kind: PartKind,
    /// Its deltas, concatenated.
    pub text: String,
    /// Its last `End`.
    pub part: AssistantContent,
}

/// Every update of `stream`, read to its end.
pub async fn collect_updates(stream: &mut CompletionStream) -> Vec<Result<Update, ErrorReport>> {
    stream.updates().collect().await
}

/// Check the index contract over `updates` against the response `Done`
/// carries, and return that response with each part as delivered. Panics
/// on any violation or on an error item.
pub fn assert_update_contract(
    updates: &[Result<Update, ErrorReport>],
) -> (CompletionResponse, Vec<DeliveredPart>) {
    let updates: Vec<&Update> = updates
        .iter()
        .map(|update| {
            update
                .as_ref()
                .unwrap_or_else(|error| panic!("an error item: {error:?}"))
        })
        .collect();
    let Some(Update::Done(done)) = updates.last() else {
        panic!("the updates end with Done: {updates:#?}");
    };
    let mut delivered = Vec::new();
    for (index, expected) in done.choice.iter().enumerate() {
        let own: Vec<&Update> = updates
            .iter()
            .copied()
            .filter(|update| match update {
                Update::Start { index: at, .. }
                | Update::Delta { index: at, .. }
                | Update::End { index: at, .. } => *at == index,
                _ => false,
            })
            .collect();
        let Some(Update::Start { part: kind, .. }) = own.first() else {
            panic!("part {index} starts first: {own:#?}");
        };
        assert_eq!(
            own.iter()
                .filter(|update| matches!(update, Update::Start { .. }))
                .count(),
            1,
            "part {index} starts once"
        );
        let text: String = own
            .iter()
            .filter_map(|update| match update {
                Update::Delta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let Some(part) = own.iter().rev().find_map(|update| match update {
            Update::End { part, .. } => Some(part.clone()),
            _ => None,
        }) else {
            panic!("part {index} ends: {own:#?}");
        };
        assert_eq!(
            &part, expected,
            "part {index}'s last end is choice[{index}]"
        );
        assert_eq!(text, finished_text(expected), "part {index}'s deltas");
        delivered.push(DeliveredPart {
            kind: kind.clone(),
            text,
            part,
        });
    }
    for update in &updates {
        if let Update::Start { index, .. }
        | Update::Delta { index, .. }
        | Update::End { index, .. } = update
        {
            assert!(
                *index < done.choice.len(),
                "{update:?} names a part of choice"
            );
        }
    }
    (done.clone(), delivered)
}

/// The text a finished part's deltas concatenate to.
fn finished_text(part: &AssistantContent) -> String {
    match part {
        AssistantContent::Text(text) => text.text.clone(),
        AssistantContent::Reasoning(reasoning) => reasoning
            .content
            .iter()
            .map(|content| match content {
                ReasoningContent::Text { text, .. } => text.as_str(),
                ReasoningContent::Summary(summary) => summary.as_str(),
                ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => "",
            })
            .collect(),
        AssistantContent::ToolCall(call) => call.function.arguments.to_string(),
        AssistantContent::Image(_) => String::new(),
    }
}
