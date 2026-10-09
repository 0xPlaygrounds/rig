use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::providers::registry::ProviderId;

use super::{CompactReason, Compacted, CompactionPolicy, KEEP_RECENT};

fn spec(window: u32) -> ModelSpec {
    let vendor = ProviderId::catalog("ollama").expect("a catalog vendor");
    ModelSpec::new(vendor, "test").with_context_window(window)
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
    let policy = CompactionPolicy::default();
    let messages = turns(4);
    let asked = CompactReason::Asked {
        focus: String::new(),
    };
    let cut = Compacted::default().cut(&messages, &policy, &spec(200_000), &asked);
    assert_eq!(cut, Some(messages.len() - 1));
}

#[test]
fn an_automatic_compaction_keeps_the_recent_work() {
    let policy = CompactionPolicy::default();
    let messages = turns(20);
    let cut = Compacted::default()
        .cut(&messages, &policy, &spec(200_000), &CompactReason::Overflow)
        .expect("a cut");
    assert!(cut < messages.len() - 2, "{cut}");
    assert_eq!(
        policy.keep(&CompactReason::Threshold, &spec(200_000)),
        KEEP_RECENT
    );
    assert_eq!(
        policy.keep(&CompactReason::Threshold, &spec(40_000)),
        10_000
    );
}

#[test]
fn a_short_conversation_is_compacted_only_when_it_must_be() {
    let policy = CompactionPolicy::default();
    let messages = turns(2);
    let compacted = Compacted::default();
    let threshold = compacted.cut(
        &messages,
        &policy,
        &spec(200_000),
        &CompactReason::Threshold,
    );
    assert_eq!(threshold, None);
    let overflow = compacted.cut(&messages, &policy, &spec(200_000), &CompactReason::Overflow);
    assert_eq!(overflow, Some(messages.len() - 1));
}
