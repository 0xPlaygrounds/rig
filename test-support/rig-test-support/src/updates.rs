//! The `updates()` contract, checked over a replayed stream: every part of
//! the response starts once and first, its deltas concatenate to its
//! finished text or argument JSON, its last end is the part, and exactly
//! one of `Done` and `Failed` comes last. A failed stream keeps the
//! contract for every part that ended, against `Failed`'s `partial`.

use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::error::RigError;
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
pub async fn collect_updates(stream: &mut CompletionStream) -> Vec<Update> {
    stream.updates().collect().await
}

/// Exactly one terminal update, and it is the last.
fn assert_one_terminal(updates: &[Update]) {
    let terminals = updates
        .iter()
        .filter(|update| matches!(update, Update::Done(_) | Update::Failed { .. }))
        .count();
    assert_eq!(terminals, 1, "exactly one terminal update: {updates:#?}");
}

/// Check the index contract over `updates` against the response `Done`
/// carries, and return that response with each part as delivered. Panics
/// on any violation, or when the stream failed.
pub fn assert_update_contract(updates: &[Update]) -> (CompletionResponse, Vec<DeliveredPart>) {
    assert_one_terminal(updates);
    let Some(Update::Done(done)) = updates.last() else {
        panic!("the updates end with Done: {updates:#?}");
    };
    let delivered = done
        .choice
        .iter()
        .enumerate()
        .map(|(index, expected)| delivered_part(updates, index, expected))
        .collect();
    for update in updates {
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

/// Check a failed stream's updates: `Failed` is the one terminal update,
/// every part that ended keeps the index contract, and the parts that
/// ended, in index order, are `partial`'s `choice`. Returns the error,
/// `partial`, and each part that ended as delivered. Panics on any
/// violation, or when the stream completed.
pub fn assert_failed_update_contract(
    updates: &[Update],
) -> (RigError, CompletionResponse, Vec<DeliveredPart>) {
    assert_one_terminal(updates);
    let Some(Update::Failed { error, partial }) = updates.last() else {
        panic!("the updates end with Failed: {updates:#?}");
    };
    let ended: std::collections::BTreeSet<usize> = updates
        .iter()
        .filter_map(|update| match update {
            Update::End { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(
        ended.len(),
        partial.choice.len(),
        "the parts that ended are partial's choice: {updates:#?}"
    );
    let delivered = ended
        .iter()
        .zip(&partial.choice)
        .map(|(index, expected)| delivered_part(updates, *index, expected))
        .collect();
    (error.clone(), partial.clone(), delivered)
}

/// The part at `index` as `updates` delivered it, checked against
/// `expected`: it starts first and once, its deltas concatenate to its
/// finished text, and its last end is `expected`.
fn delivered_part(updates: &[Update], index: usize, expected: &AssistantContent) -> DeliveredPart {
    let own: Vec<&Update> = updates
        .iter()
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
    DeliveredPart {
        kind: kind.clone(),
        text,
        part,
    }
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
