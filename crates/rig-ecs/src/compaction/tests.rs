use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::providers::registry::ProviderId;

use super::{CompactReason, Compacted, CompactionPolicy, KEEP_RECENT};

fn spec(window: u32) -> Option<ModelSpec> {
    ProviderId::catalog("ollama")
        .map(|vendor| ModelSpec::new(vendor, "test").with_context_window(window))
}

/// Where `reason` cuts `messages` for a model with `window` tokens.
fn cut(messages: &[Message], window: u32, reason: &CompactReason) -> Option<usize> {
    let policy = CompactionPolicy::default();
    spec(window).and_then(|spec| Compacted::default().cut(messages, &policy, &spec, reason))
}

/// A conversation of turns, each a question and a long answer.
fn turns(count: usize) -> Vec<Message> {
    let answer = "word ".repeat(2_000);
    (0..count)
        .flat_map(|turn| {
            [
                Message::user(format!("question {turn}")),
                Message::assistant(answer.clone()),
            ]
        })
        .collect()
}

#[test]
fn an_asked_compaction_summarizes_all_but_the_newest_reply() {
    let messages = turns(4);
    let asked = CompactReason::Asked {
        focus: String::new(),
    };
    assert_eq!(cut(&messages, 200_000, &asked), Some(messages.len() - 1));
}

#[test]
fn an_automatic_compaction_keeps_the_recent_work() {
    let policy = CompactionPolicy::default();
    let messages = turns(20);
    let overflow = cut(&messages, 200_000, &CompactReason::Overflow);
    assert!(
        overflow.is_some_and(|at| at < messages.len() - 2),
        "{overflow:?}"
    );
    let keep = |window| spec(window).map(|spec| policy.keep(&CompactReason::Threshold, &spec));
    assert_eq!(keep(200_000), Some(KEEP_RECENT));
    assert_eq!(keep(40_000), Some(10_000));
}

#[test]
fn a_short_conversation_is_compacted_only_when_it_must_be() {
    let messages = turns(2);
    assert!(spec(200_000).is_some());
    assert_eq!(cut(&messages, 200_000, &CompactReason::Threshold), None);
    assert_eq!(
        cut(&messages, 200_000, &CompactReason::Overflow),
        Some(messages.len() - 1)
    );
}
