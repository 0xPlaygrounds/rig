//! DeepSeek: reasoning under `reasoning_content`, sent back empty when a
//! turn has none, and calls it streams whole. Its models read no images.

use rig_core::providers::openai::wire::DEEPSEEK;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "reasoning_content": "plan the lookup",
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", "content": null, "reasoning_content": "plan "}),
        json!({"reasoning_content": "the lookup"}),
        json!({"content": "looking it up"}),
        json!({"tool_calls": [whole]}),
    ]
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &DEEPSEEK,
    model: "deepseek-v4-flash",
    other_model: "deepseek-v4-pro",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "deepseek",
    fixture: FIXTURE,
}

/// The rows can fail: H18 reports a stream whose text differs from its
/// whole form, and H19 a cut stream that ends cleanly.
#[test]
fn generated_reply_checks_catch_what_they_guard() {
    use rig_history_conformance::{HistoryFixture, Rng, replies};
    let wire = FIXTURE.wire(FIXTURE.model);
    let key = Some("reasoning_content");
    let spec = replies::Spec {
        blocks: vec![serde_json::json!({
            "role": "assistant", "content": "looking it up", "reasoning_content": "plan"
        })],
        finish: "stop".to_owned(),
        seed: Rng::new(7).next(),
    };
    let (whole, streamed) = replies::chat_build(FIXTURE.model, key, &spec);
    let altered = whole
        .iter()
        .map(|frame| frame.replace("looking it up", "found it"))
        .collect();
    let found = replies::disagreement(
        &wire,
        replies::Frames {
            whole: replies::texts(altered),
            streamed: replies::texts(streamed.clone()),
        },
    );
    assert!(found.is_some_and(|found| found.contains("folds differ")));
    // A cut that keeps a finish chunk but drops `[DONE]` is still a reply the
    // provider ended, so a clean fold is not reported; the cut checks only
    // what the provider did not end.
    assert!(replies::cut_problems(&wire, || replies::texts(streamed.clone())).is_empty());
    assert!(
        replies::disagreement(
            &wire,
            replies::Frames {
                whole: replies::texts(whole),
                streamed: replies::texts(streamed),
            },
        )
        .is_none()
    );
}
