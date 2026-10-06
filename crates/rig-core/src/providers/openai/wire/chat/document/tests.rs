//! The `chat.completion` a stream of hand-built chunks rebuilds. The
//! recorded pairs are checked whole in `test_utils::raw_parity`; these cover
//! the folding rules one at a time, and the dialects no pair covers.

use serde_json::{Value, json};

use super::super::Chat;
use crate::providers::openai::wire::{
    DEEPSEEK, Dialect, MIRA, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY,
};
use crate::wire::document::Reassemble;
use crate::wire::{Wire, WireFrame};

fn chat(dialect: &'static Dialect) -> Chat {
    OpenAIConfig::new("key").with_dialect(dialect).chat("model")
}

/// The document `frames`, each one event's data, rebuild on `wire`.
fn rebuilt(wire: &Chat, frames: &[Value]) -> Value {
    let mut document = wire.reassembler();
    for frame in frames {
        let data = match frame {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        document.absorb(&WireFrame::Text(data));
    }
    document.finish()
}

/// A chunk of reply `chatcmpl-1` whose one choice carries `delta`, with
/// `extra` choice fields.
fn chunk(delta: Value, extra: Value) -> Value {
    let mut choice = json!({"index": 0, "delta": delta});
    if let (Some(choice), Value::Object(extra)) = (choice.as_object_mut(), extra) {
        choice.extend(extra);
    }
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": [choice]
    })
}

#[test]
fn text_appends_and_the_envelope_keeps_its_last_value() {
    let mut first = chunk(json!({"role": "assistant", "content": ""}), json!({}));
    first["obfuscation"] = json!("pad");
    first["system_fingerprint"] = json!(null);
    let mut last = chunk(json!({"content": "ng"}), json!({"finish_reason": "stop"}));
    last["system_fingerprint"] = json!("fp_1");
    last["p"] = json!("abcdef");
    let usage = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "obfuscation": "pad"
    });
    let document = rebuilt(
        &chat(&OPENAI),
        &[
            first,
            chunk(json!({"content": "po"}), json!({"finish_reason": null})),
            last,
            usage,
            json!("[DONE]"),
        ],
    );
    assert_eq!(
        document,
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "system_fingerprint": "fp_1",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })
    );
}

#[test]
fn tool_calls_merge_by_index_and_close_as_the_unary_body_states_them() {
    let opened = |index: u64, id: &str, name: &str| {
        json!({"index": index, "id": id, "type": "function",
            "function": {"name": name, "arguments": ""}})
    };
    let more =
        |index: u64, arguments: &str| json!({"index": index, "function": {"arguments": arguments}});
    let document = rebuilt(
        &chat(&OPENAI),
        &[
            chunk(
                json!({"role": "assistant", "content": null, "tool_calls": [opened(0, "call_a", "ping")]}),
                json!({}),
            ),
            chunk(json!({"tool_calls": [more(0, "{\"x\":")]}), json!({})),
            chunk(
                json!({"tool_calls": [opened(1, "call_b", "pong")]}),
                json!({}),
            ),
            chunk(
                json!({"tool_calls": [more(0, "1}"), more(1, "{}")]}),
                json!({}),
            ),
            chunk(json!({}), json!({"finish_reason": "tool_calls"})),
        ],
    );
    assert_eq!(
        document.pointer("/choices/0/message"),
        Some(&json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_a", "type": "function",
                    "function": {"name": "ping", "arguments": "{\"x\":1}"}},
                {"id": "call_b", "type": "function",
                    "function": {"name": "pong", "arguments": "{}"}}
            ]
        }))
    );
    assert_eq!(
        document.pointer("/choices/0/finish_reason"),
        Some(&json!("tool_calls"))
    );
}

/// The calls the message of `frames`, each a chunk's `tool_calls`, rebuilds.
fn rebuilt_calls(frames: &[Value]) -> Value {
    let chunks: Vec<Value> = frames
        .iter()
        .map(|calls| chunk(json!({"tool_calls": calls}), json!({})))
        .collect();
    rebuilt(&chat(&OPENAI), &chunks)
        .pointer("/choices/0/message/tool_calls")
        .cloned()
        .unwrap_or_default()
}

/// A whole function call fragment.
fn call(index: Option<u64>, id: &str, name: &str, arguments: &str) -> Value {
    let mut call = json!({"id": id, "type": "function",
        "function": {"name": name, "arguments": arguments}});
    if let Some(index) = index {
        call["index"] = json!(index);
    }
    call
}

#[test]
fn a_new_id_under_a_held_index_starts_a_new_call_once_the_held_one_is_whole() {
    let calls = rebuilt_calls(&[
        json!([call(Some(0), "c3", "weather", "{\"city\":\"Oslo\"}")]),
        json!([call(Some(0), "d4", "weather", "{\"city\":\"Bern\"}")]),
    ]);
    assert_eq!(
        calls,
        json!([
            call(None, "c3", "weather", "{\"city\":\"Oslo\"}"),
            call(None, "d4", "weather", "{\"city\":\"Bern\"}")
        ])
    );
}

#[test]
fn a_new_id_under_an_index_continues_a_call_whose_arguments_are_incomplete() {
    // Some providers (GLM) send a fresh id, and no name, with every chunk of
    // one call.
    let calls = rebuilt_calls(&[
        json!([call(Some(0), "a", "weather", "{\"city\":")]),
        json!([{"index": 0, "id": "b", "function": {"arguments": "\"Oslo\"}"}}]),
    ]);
    assert_eq!(
        calls,
        json!([call(None, "b", "weather", "{\"city\":\"Oslo\"}")])
    );
}

#[test]
fn index_less_fragments_with_new_ids_are_new_calls() {
    let calls = rebuilt_calls(&[
        json!([call(None, "a1", "weather", "{\"city\":")]),
        json!([{"function": {"arguments": "\"Paris\"}"}}]),
        json!([call(None, "b2", "weather", "{\"city\":\"Rome\"}")]),
        json!([{"id": "a1", "function": {"arguments": ""}}]),
    ]);
    assert_eq!(
        calls,
        json!([
            call(None, "a1", "weather", "{\"city\":\"Paris\"}"),
            call(None, "b2", "weather", "{\"city\":\"Rome\"}")
        ])
    );
}

#[test]
fn an_index_less_fragment_with_no_id_starts_a_call_after_a_whole_one() {
    let calls = rebuilt_calls(&[
        json!([{"type": "function", "function": {"name": "weather", "arguments": "{}"}}]),
        json!([{"type": "function", "function": {"name": "time", "arguments": "{}"}}]),
    ]);
    assert_eq!(
        calls,
        json!([
            {"type": "function", "function": {"name": "weather", "arguments": "{}"}},
            {"type": "function", "function": {"name": "time", "arguments": "{}"}}
        ])
    );
}

#[test]
fn every_choice_is_kept_by_its_index() {
    let second = |delta: Value, extra: Value| {
        let mut frame = chunk(delta, extra);
        frame["choices"][0]["index"] = json!(1);
        frame
    };
    let document = rebuilt(
        &chat(&OPENAI),
        &[
            chunk(json!({"content": "a"}), json!({})),
            second(json!({"content": "b"}), json!({})),
            second(json!({"content": "b"}), json!({"finish_reason": "stop"})),
            chunk(json!({"content": "a"}), json!({"finish_reason": "length"})),
        ],
    );
    assert_eq!(
        document["choices"],
        json!([
            {"index": 0, "message": {"content": "aa"}, "finish_reason": "length"},
            {"index": 1, "message": {"content": "bb"}, "finish_reason": "stop"}
        ])
    );
}

#[test]
fn reasoning_text_appends_and_reasoning_details_merge_as_the_decoder_merges_them() {
    let detail = |text: &str, signature: Value| {
        json!([{"type": "reasoning.text", "index": 0, "format": "anthropic-claude-v1",
            "text": text, "signature": signature}])
    };
    let document = rebuilt(
        &chat(&OPENROUTER),
        &[
            chunk(
                json!({"role": "assistant", "content": "", "reasoning": "Think",
                    "reasoning_details": detail("Think", Value::Null)}),
                json!({}),
            ),
            chunk(
                json!({"content": "", "reasoning": "ing.",
                    "reasoning_details": detail("ing.", json!("sig"))}),
                json!({}),
            ),
            chunk(
                json!({"content": "pong"}),
                json!({"finish_reason": "stop", "native_finish_reason": "end_turn"}),
            ),
        ],
    );
    assert_eq!(
        document.pointer("/choices/0"),
        Some(&json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "pong",
                "reasoning": "Thinking.",
                "reasoning_details": [{"type": "reasoning.text", "index": 0,
                    "format": "anthropic-claude-v1", "text": "Thinking.", "signature": "sig"}]
            },
            "finish_reason": "stop",
            "native_finish_reason": "end_turn"
        }))
    );
}

#[test]
fn logprobs_annotations_and_audio_append() {
    let token = |text: &str| json!({"token": text, "logprob": -0.1, "top_logprobs": []});
    let citation = |url: &str| {
        json!({"type": "url_citation", "url_citation": {"url": url, "title": "t",
            "start_index": 0, "end_index": 1}})
    };
    let document = rebuilt(
        &chat(&OPENAI),
        &[
            chunk(
                json!({"content": "a", "audio": {"id": "audio_1", "data": "AA", "transcript": "a"}}),
                json!({"logprobs": {"content": [token("a")], "refusal": null}}),
            ),
            chunk(
                json!({"content": "b", "annotations": [citation("https://a")],
                    "audio": {"data": "BB", "transcript": "b", "expires_at": 9}}),
                json!({"logprobs": {"content": [token("b")]}}),
            ),
            chunk(
                json!({"annotations": [citation("https://b")]}),
                json!({"logprobs": null, "finish_reason": "stop"}),
            ),
        ],
    );
    assert_eq!(
        document.pointer("/choices/0/message"),
        Some(&json!({
            "content": "ab",
            "audio": {"id": "audio_1", "data": "AABB", "transcript": "ab", "expires_at": 9},
            "annotations": [citation("https://a"), citation("https://b")]
        }))
    );
    assert_eq!(
        document.pointer("/choices/0/logprobs"),
        Some(&json!({"content": [token("a"), token("b")], "refusal": null}))
    );
}

#[test]
fn a_whole_frame_is_kept_as_sent() {
    let whole = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_a", "type": "function",
                "function": {"name": "ping", "arguments": "{}"}}]
        }}]
    });
    assert_eq!(
        rebuilt(&chat(&OPENAI), &[whole.clone(), json!("[DONE]")]),
        whole
    );
}

/// Perplexity streams the message so far beside each delta: the choice is
/// kept as its last chunk sent it, as its unary body states it.
#[test]
fn a_choice_carrying_its_message_beside_its_delta_is_kept_as_sent() {
    let choice = |delta: &str, message: &str, finish: Value| {
        json!({"index": 0, "delta": {"role": "assistant", "content": delta},
            "message": {"role": "assistant", "content": message}, "finish_reason": finish})
    };
    let frame = |choice: Value, object: &str| json!({"id": "1", "object": object, "model": "sonar", "choices": [choice]});
    let document = rebuilt(
        &chat(&PERPLEXITY),
        &[
            frame(choice("po", "po", Value::Null), "chat.completion.chunk"),
            frame(choice("ng", "pong", Value::Null), "chat.completion.chunk"),
            frame(choice("", "pong", json!("stop")), "chat.completion.done"),
        ],
    );
    assert_eq!(
        document,
        json!({"id": "1", "object": "chat.completion", "model": "sonar", "choices": [
            choice("", "pong", json!("stop"))
        ]})
    );
}

#[test]
fn a_bare_string_or_a_stream_of_nothing_has_no_document() {
    assert_eq!(
        rebuilt(&chat(&MIRA), &[json!("\"the whole answer\"")]),
        Value::Null
    );
    assert_eq!(rebuilt(&chat(&OPENAI), &[json!("[DONE]")]), Value::Null);
    assert_eq!(rebuilt(&chat(&OPENAI), &[json!("not json")]), Value::Null);
}

/// The in-band error envelope is the decoder's failure, not part of the
/// document: the reply's `raw` is what arrived before it.
#[test]
fn an_in_band_error_is_not_part_of_the_document() {
    let failure = json!({"error": {"code": 502, "message": "upstream died"}});
    let document = rebuilt(
        &chat(&OPENROUTER),
        &[chunk(json!({"content": "po"}), json!({})), failure],
    );
    assert_eq!(
        document.pointer("/choices/0/message/content"),
        Some(&json!("po"))
    );
    assert_eq!(document.pointer("/choices/0/finish_reason"), None);
    assert_eq!(document.get("error"), None);
}

/// DeepSeek streams its reasoning as `reasoning_content` and its cache
/// counters inside the last chunk's `usage`, which land where its unary body
/// has them.
#[test]
fn deepseek_reasoning_and_cache_counters_land_where_the_unary_body_has_them() {
    let usage = json!({
        "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
        "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2,
        "completion_tokens_details": {"reasoning_tokens": 3}
    });
    let mut last = chunk(json!({"content": ""}), json!({"finish_reason": "stop"}));
    last["usage"] = usage.clone();
    let document = rebuilt(
        &chat(&DEEPSEEK),
        &[
            chunk(
                json!({"role": "assistant", "content": null, "reasoning_content": "Hm"}),
                json!({"logprobs": null, "finish_reason": null}),
            ),
            chunk(
                json!({"content": null, "reasoning_content": "m."}),
                json!({}),
            ),
            chunk(
                json!({"content": "pong", "reasoning_content": null}),
                json!({}),
            ),
            last,
        ],
    );
    assert_eq!(
        document.pointer("/choices/0/message"),
        Some(&json!({"role": "assistant", "content": "pong", "reasoning_content": "Hmm."}))
    );
    assert_eq!(document.pointer("/choices/0/logprobs"), Some(&Value::Null));
    assert_eq!(document.get("usage"), Some(&usage));
}
