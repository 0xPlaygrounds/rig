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

/// The key [`ResponsesText`] is stored under, which earlier releases used too.
const OPENAI_RESPONSES_EXTRAS_KEY: &str = <ResponsesText as message::Extension>::KEY;

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

    let items = Vec::<InputItem>::try_from(history).expect("history converts");
    assert_eq!(items.len(), 1);
    let InputContent::Message(Message::Assistant {
        phase, content, id, ..
    }) = &items[0].input
    else {
        panic!("expected an assistant input item, got {:?}", items[0].input);
    };
    assert_eq!(id, "msg_1");
    assert_eq!(phase.as_deref(), Some("final_answer"));

    // Never on the block: the flatten would put it beside `text`.
    let block = serde_json::to_value(&content[0]).expect("block serializes");
    assert!(
        block.get("phase").is_none(),
        "phase leaked onto the text block: {block}"
    );

    let wire = serde_json::to_value(&items[0]).expect("item serializes");
    assert_eq!(wire["phase"], "final_answer");
    assert_eq!(wire["id"], "msg_1");
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
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "error": null, "incomplete_details": null, "instructions": null,
        "max_output_tokens": null, "model": "gpt-5.3-codex", "usage": null,
        "output": output,
    }))
    .expect("reply decodes");
    wire::fold_body("openai", response)
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

/// Text blocks naming one message item join it, wherever they sit; a block
/// naming none joins the assistant message's own item. No input item id
/// repeats.
#[test]
fn text_blocks_of_one_item_join_it_and_no_id_repeats() {
    let block = |text: &str, extras: serde_json::Value| {
        message::AssistantContent::Text(Text {
            text: text.to_owned(),
            additional_params: message::AdditionalParams::from_entries(Some((
                OPENAI_RESPONSES_EXTRAS_KEY,
                extras,
            ))),
        })
    };
    let history = completion::Message::Assistant {
        id: Some("msg_2".to_owned()),
        content: vec![
            block("one", json!({"message_id": "msg_1", "phase": "commentary"})),
            block(
                "two",
                json!({"message_id": "msg_2", "phase": "final_answer"}),
            ),
            block(
                "three",
                json!({"message_id": "msg_1", "phase": "commentary"}),
            ),
            message::AssistantContent::Text(Text::new("four")),
        ],
    };
    let items = replayed(history);
    assert_eq!(
        shape(&items),
        [
            ("message".into(), "msg_1".into(), "commentary".into()),
            ("message".into(), "msg_2".into(), "final_answer".into()),
        ]
    );
    let texts = |item: &serde_json::Value| -> Vec<String> {
        item["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|block| block["text"].as_str().map(str::to_owned))
            .collect()
    };
    assert_eq!(texts(&items[0]), ["one", "three"]);
    assert_eq!(texts(&items[1]), ["two", "four"]);
    assert_no_block_carries_message_fields(&items);
}

/// Text without any message id replays id-less. With a `phase` it takes the
/// output-message form, which carries one, and no content-part extras;
/// without one it keeps the plain input-message form.
#[test]
fn idless_text_replays_its_phase_without_an_id() {
    let history = completion::Message::Assistant {
        id: None,
        content: vec![
            message::AssistantContent::Text(Text {
                text: "Let me think.".to_owned(),
                additional_params: message::AdditionalParams::from_entries(Some((
                    OPENAI_RESPONSES_EXTRAS_KEY,
                    json!({"phase": "commentary", "annotations": [{"type": "url_citation"}]}),
                ))),
            }),
            message::AssistantContent::Text(Text::new("Apple.")),
        ],
    };
    let items = replayed(history);
    assert_eq!(
        items,
        [
            json!({
                "type": "message", "role": "assistant", "status": "completed",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Let me think."}],
            }),
            json!({"type": "message", "role": "assistant", "content": "Apple."}),
        ]
    );
}

/// `phase` is an assistant field: extras on a user text block never reach
/// the user item, and a system message has no seat for it.
#[test]
fn phase_never_rides_user_or_system_messages() {
    let phased = message::AdditionalParams::from_entries(Some((
        OPENAI_RESPONSES_EXTRAS_KEY,
        json!({"phase": "final_answer", "message_id": "msg_1"}),
    )));
    let user = completion::Message::User {
        content: vec![message::UserContent::Text(Text {
            text: "hi".to_owned(),
            additional_params: phased,
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
    let history = history_of(json!([
        message_item("msg_1", "commentary", "Let me think."),
        message_item("msg_2", "final_answer", "Apple."),
    ]));
    for (name, dialect) in dialects {
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

/// Items Rig has no canonical form for, as the API documents them, plus one
/// no release has seen yet. The compaction item is the recorded cell's.
fn unmodelled_items() -> Vec<serde_json::Value> {
    vec![
        json!({"encrypted_content": "encrypted_content_REDACTED_1", "id": "cmp_REDACTED_1",
               "status": "completed", "type": "compaction"}),
        json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
               "action": {"type": "search", "query": "rig"}}),
        json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c1",
               "name": "grammar", "input": "x = 1", "status": "completed"}),
        json!({"type": "item_from_the_future", "id": "fut_1", "payload": [1.5, null]}),
    ]
}

/// Loss 2: a compaction item, which OpenAI documents as must-replay, used
/// to leave the turn as a stream-only unknown item. It is now part of
/// history, as is every other unmodelled item, and each replays verbatim in
/// its place among the turn's items.
#[test]
fn unmodelled_items_are_kept_and_replay_in_place() {
    let mut output = unmodelled_items();
    output.insert(2, message_item("msg_1", "final_answer", "ACK-1"));
    let history = history_of(json!(output));
    let completion::Message::Assistant { content, .. } = &history else {
        panic!("an assistant turn");
    };
    assert_eq!(content.len(), output.len());
    assert!(matches!(content[0], message::AssistantContent::Opaque(_)));

    let items = replayed(history);
    assert_eq!(items.len(), output.len());
    for (index, item) in unmodelled_items().into_iter().enumerate() {
        let at = if index < 2 { index } else { index + 1 };
        assert_eq!(items[at], item, "item {index} replays verbatim in place");
    }
    assert_eq!(
        shape(&items[2..3]),
        [("message".into(), "msg_1".into(), "final_answer".into())]
    );
}

/// The cross-dialect rule on this wire: items another issuer sealed are
/// left out, and so are opaque items holding only another wire's types.
#[test]
fn opaque_items_replay_only_to_their_issuer_and_wire() {
    let mut history = history_of(json!([
        unmodelled_items()[0].clone(),
        message_item("msg_1", "final_answer", "ACK-1"),
    ]));
    let completion::Message::Assistant { content, .. } = &mut history else {
        panic!("an assistant turn");
    };
    content.push(message::AssistantContent::Opaque(message::Sealed::new(
        "openai",
        message::Opaque::of(&crate::providers::anthropic::completion::AnthropicBlock(
            json!({"type": "container_upload", "file_id": "file_01"}),
        ))
        .expect("serializes"),
    )));
    let types = |issuer: &'static str| -> Vec<String> {
        super::input_items(history.clone(), &[issuer.into()])
            .expect("history converts")
            .iter()
            .map(|item| serde_json::to_value(item).expect("item serializes"))
            .map(|item| item["type"].as_str().unwrap_or("-").to_owned())
            .collect()
    };
    assert_eq!(types("openai"), ["compaction", "message"]);
    assert_eq!(types("xai"), ["message"]);
}

/// The staleness rule: the block's annotations describe its text, so an
/// edit drops them; `phase` and the item id are identity and survive.
#[test]
fn annotations_go_stale_when_their_text_is_edited_and_phase_survives() {
    let annotated = json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "Rust is fast.",
                     "annotations": [{"type": "url_citation", "start_index": 0,
                                      "end_index": 4, "url": "https://a.example",
                                      "title": "A"}]}],
    });
    let history = history_of(json!([annotated]));
    let items = replayed(history.clone());
    assert_eq!(
        items[0]["content"][0]["annotations"],
        annotated["content"][0]["annotations"]
    );

    let completion::Message::Assistant { id, mut content } = history else {
        panic!("an assistant turn");
    };
    let Some(message::AssistantContent::Text(text)) = content.first_mut() else {
        panic!("a text block");
    };
    text.text = "Rust is slow.".to_owned();
    let items = replayed(completion::Message::Assistant { id, content });
    assert_eq!(
        items[0],
        json!({"type": "message", "role": "assistant", "id": "msg_1",
               "status": "completed", "phase": "final_answer",
               "content": [{"type": "output_text", "text": "Rust is slow."}]})
    );
}
