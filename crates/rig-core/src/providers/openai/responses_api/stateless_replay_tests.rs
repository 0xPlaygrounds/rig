//! rig#2269: items a stateless Responses client must send back unchanged.
//!
//! - `compaction` items round-trip verbatim on both the output and the input
//!   side of the wire;
//! - an output message's `phase` survives the trip through rig history and
//!   is re-sent on the assistant input item, never leaked onto a text block;

use super::*;
use crate::completion;
use crate::message::{self, Text};
use serde_json::json;

#[test]
fn compaction_output_item_round_trips_verbatim() {
    let wire = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes",
        "status": "completed",
        "future_field": {"nested": [1, 2, 3]}
    });
    let output: Output = serde_json::from_value(wire.clone()).expect("compaction decodes");
    let Output::Compaction(fields) = &output else {
        panic!("expected Output::Compaction, got {output:?}");
    };
    assert_eq!(fields.get("id"), Some(&json!("cmp_123")));
    assert!(
        fields.get("type").is_none(),
        "the tag must not be duplicated inside the payload"
    );

    let back = serde_json::to_value(&output).expect("compaction re-serializes");
    assert_eq!(back, wire);
}

#[test]
fn compaction_input_item_round_trips_verbatim() {
    let wire = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes"
    });
    let item: InputItem = serde_json::from_value(wire.clone()).expect("compaction input decodes");
    assert!(matches!(item.input, InputContent::Compaction(_)));
    let back = serde_json::to_value(&item).expect("compaction input re-serializes");
    assert_eq!(back, wire);
}

/// The exact window `/responses/compact` returns — regular items around an
/// opaque compaction item — decodes as `output[]` without dropping anything.
#[test]
fn compacted_window_decodes_with_every_item_typed() {
    let output: Vec<Output> = serde_json::from_value(json!([
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "..."},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
         "arguments": "{}", "status": "completed"}
    ]))
    .expect("window decodes");
    assert!(matches!(output[0], Output::Compaction(_)));
    assert!(matches!(output[1], Output::Message(_)));
    assert!(matches!(output[2], Output::FunctionCall(_)));
}

#[test]
fn output_message_phase_decodes_and_is_absent_by_default() {
    let with: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [], "phase": "final_answer"
    }))
    .expect("decodes");
    assert_eq!(with.phase.as_deref(), Some("final_answer"));
    let without: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed", "content": []
    }))
    .expect("decodes");
    assert_eq!(without.phase, None);
    let back = serde_json::to_value(&without).expect("serializes");
    assert!(
        back.get("phase").is_none(),
        "absent phase must not serialize as null"
    );
}

/// `Output::Message` → rig history → assistant input item: `phase` arrives on
/// the item and not on the text block.
#[test]
fn phase_survives_history_and_is_resent_on_the_assistant_item() {
    let output = Output::Message(OutputMessage {
        id: "msg_1".to_string(),
        role: OutputRole::Assistant,
        status: ResponseStatus::Completed,
        content: vec![AssistantContent::OutputText(OutputText::new("the answer"))],
        phase: Some("final_answer".to_string()),
    });
    let content = super::tests::folded_choice(vec![output]);
    let history = completion::Message::Assistant {
        id: Some("msg_1".to_string()),
        content,
    };

    let items = replayed(history);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["phase"], "final_answer");
    assert_eq!(items[0]["id"], "msg_1");
    assert_no_block_carries_message_fields(&items);
}

/// A message without a phase replays exactly as before: no `phase` key.
#[test]
fn history_without_phase_replays_without_the_key() {
    let history = completion::Message::Assistant {
        id: Some("msg_1".to_string()),
        content: vec![message::AssistantContent::Text(Text::new("plain"))],
    };
    let items = Vec::<InputItem>::try_from(history).expect("history converts");
    let wire = serde_json::to_value(&items[0]).expect("item serializes");
    assert!(wire.get("phase").is_none(), "{wire}");
}

/// The assistant turn a unary reply with `output` folds into, as history
/// holds it: the fold's choice under the fold's message id.
fn history_of(output: serde_json::Value) -> completion::Message {
    history_of_from("openai", output)
}

/// [`history_of`], for the Responses dialect `provider`.
fn history_of_from(provider: &str, output: serde_json::Value) -> completion::Message {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "error": null, "incomplete_details": null, "instructions": null,
        "max_output_tokens": null, "model": "gpt-5.3-codex", "usage": null,
        "output": output,
    }))
    .expect("reply decodes");
    wire::fold_body(provider, response)
        .expect("the body folds")
        .message()
        .expect("the reply has content")
}

fn message_item(id: &str, phase: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message", "id": id, "role": "assistant", "status": "completed",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

/// The serialized input items `history` replays as, replaying OpenAI's own
/// reasoning.
fn replayed(history: completion::Message) -> Vec<serde_json::Value> {
    super::input_items(history, &["openai".into()])
        .expect("history converts")
        .iter()
        .map(|item| serde_json::to_value(item).expect("item serializes"))
        .collect()
}

/// `(type, id, phase)` of each replayed item, in order.
fn shape(items: &[serde_json::Value]) -> Vec<(String, String, String)> {
    let field = |item: &serde_json::Value, key: &str| {
        item.get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("-")
            .to_owned()
    };
    items
        .iter()
        .map(|item| (field(item, "type"), field(item, "id"), field(item, "phase")))
        .collect()
}

fn assert_no_block_carries_message_fields(items: &[serde_json::Value]) {
    for item in items {
        for block in item["content"].as_array().into_iter().flatten() {
            assert!(
                block.get("phase").is_none() && block.get("message_id").is_none(),
                "a message field leaked onto a content block: {block}"
            );
        }
    }
}

/// A reply with a commentary message and a final answer replays as two
/// message items, each under its own id with its own `phase`, in the order
/// the reply stated them, after the reasoning. No id repeats.
#[test]
fn several_message_items_replay_each_with_its_own_phase_and_id() {
    let history = history_of(json!([
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
        message_item("msg_1", "commentary", "Let me think."),
        message_item("msg_2", "final_answer", "Apple."),
    ]));
    let items = replayed(history);
    assert_eq!(
        shape(&items),
        [
            ("reasoning".into(), "rs_1".into(), "-".into()),
            ("message".into(), "msg_1".into(), "commentary".into()),
            ("message".into(), "msg_2".into(), "final_answer".into()),
        ]
    );
    assert_eq!(items[1]["content"][0]["text"], "Let me think.");
    assert_eq!(items[2]["content"][0]["text"], "Apple.");
    assert_no_block_carries_message_fields(&items);
}

/// A commentary message stated before a function call replays before it,
/// with its `phase`, and the call keeps its ids.
#[test]
fn a_commentary_message_replays_before_its_function_call() {
    let history = history_of(json!([
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
        message_item("msg_1", "commentary", "Checking the weather."),
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
         "arguments": "{\"city\":\"Paris\"}", "status": "completed"},
    ]));
    let items = replayed(history);
    assert_eq!(
        shape(&items),
        [
            ("reasoning".into(), "rs_1".into(), "-".into()),
            ("message".into(), "msg_1".into(), "commentary".into()),
            ("function_call".into(), "fc_1".into(), "-".into()),
        ]
    );
    assert_eq!(items[2]["call_id"], "call_1");
}

/// Canonical text joins the assistant message's own item, unless a
/// replayed item already uses that id: no input item id repeats.
#[test]
fn canonical_text_never_repeats_a_replayed_item_id() {
    let decoded = history_of(json!([
        message_item("msg_1", "commentary", "one"),
        message_item("msg_2", "final_answer", "two"),
    ]));
    let completion::Message::Assistant { id, mut content } = decoded else {
        panic!("expected an assistant turn");
    };
    assert_eq!(id.as_deref(), Some("msg_2"));
    content.push(message::AssistantContent::Text(Text::new("three")));
    let items = replayed(completion::Message::Assistant { id, content });
    assert_eq!(
        shape(&items),
        [
            ("message".into(), "msg_1".into(), "commentary".into()),
            ("message".into(), "msg_2".into(), "final_answer".into()),
            ("message".into(), "-".into(), "-".into()),
        ]
    );
    assert_eq!(items[2]["content"], "three");
    assert_no_block_carries_message_fields(&items);
}

/// The staleness rule: a decoded text block edited afterwards replays
/// canonically, and its message item, which no longer says what the block
/// says, is not sent.
#[test]
fn an_edited_text_block_replays_canonically() {
    let decoded = history_of(json!([message_item(
        "msg_1",
        "commentary",
        "Let me think."
    )]));
    let completion::Message::Assistant { id, mut content } = decoded else {
        panic!("expected an assistant turn");
    };
    let Some(message::AssistantContent::Text(text)) = content.first_mut() else {
        panic!("expected a text block");
    };
    assert!(text.native.is_some(), "the phase rides the text's item");
    text.text = "An edited thought.".to_owned();
    let items = replayed(completion::Message::Assistant { id, content });
    assert_eq!(
        items,
        [json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "An edited thought."}],
        })]
    );
}

/// `phase` is an assistant field: a provider item on a user text block
/// never reaches the user item, and a system message has no seat for it.
#[test]
fn phase_never_rides_user_or_system_messages() {
    let phased = message::Sealed::new(
        message::Issuer::from("openai"),
        message::NativeItem::new(DIALECT, message_item("msg_1", "final_answer", "hi")),
    );
    let user = completion::Message::User {
        content: vec![message::UserContent::Text(Text {
            native: Some(phased),
            ..Text::new("hi")
        })],
    };
    let system = completion::Message::system("be brief");
    for history in [user, system] {
        for item in replayed(history) {
            assert!(item.get("phase").is_none(), "{item}");
            assert!(item.get("id").is_none(), "{item}");
            assert_no_block_carries_message_fields(std::slice::from_ref(&item));
        }
    }
}

/// Every Responses dialect keeps `phase` on replay; none strips it.
/// Unit-level because this is request shaping per dialect. OpenAI and
/// ChatGPT accept it in recorded follow-ups; probes outside the corpus found
/// Copilot and OpenRouter accept it and answer with their own, and xAI
/// accepts and ignores it.
#[test]
fn every_responses_dialect_resends_phase() {
    use crate::providers::openai::OpenAIConfig;
    use crate::providers::openai::wire::{Dialect, OPENAI, OPENROUTER};
    let dialects: [(&str, &Dialect); 5] = [
        ("openai", &OPENAI),
        ("xai", &crate::providers::xai::DIALECT),
        ("chatgpt", &crate::providers::chatgpt::DIALECT),
        ("copilot", &crate::providers::copilot::wire::DIALECT),
        ("openrouter", &OPENROUTER),
    ];
    for (name, dialect) in dialects {
        // Each dialect replays its own turn.
        let history = history_of_from(
            name,
            json!([
                message_item("msg_1", "commentary", "Let me think."),
                message_item("msg_2", "final_answer", "Apple."),
            ]),
        );
        let wire = OpenAIConfig::with_key(dialect, "dummy-key").responses("gpt-5.3-codex");
        let request = completion::CompletionRequest::new("One more fruit?")
            .messages([completion::Message::user("Three fruits?"), history.clone()]);
        let request = wire
            .responses_request(request, vec![crate::message::Issuer::from(name)], false)
            .expect("request converts");
        let items: Vec<serde_json::Value> = request
            .input
            .iter()
            .map(|item| serde_json::to_value(item).expect("item serializes"))
            .collect();
        let assistant: Vec<(String, String, String)> = shape(&items)
            .into_iter()
            .filter(|(_, id, _)| id.starts_with("msg_"))
            .collect();
        assert_eq!(
            assistant,
            [
                ("message".into(), "msg_1".into(), "commentary".into()),
                ("message".into(), "msg_2".into(), "final_answer".into()),
            ],
            "{name}"
        );
    }
}
