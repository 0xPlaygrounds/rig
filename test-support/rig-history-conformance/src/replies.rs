//! Generated replies for rows H18 and H19: per-family generators of wire
//! replies (each a [`Spec`]: wire-JSON blocks, a finish, and a seed for the
//! stream's restatement choices), built into a whole and a streamed form,
//! with the checks the rows make and a shrinker over the spec.
//!
//! H18 folds both forms and requires the same turn and stop. H19 cuts the
//! stream after every frame and requires a cut the provider did not end to
//! fail and hold no provider item.

use std::panic::{AssertUnwindSafe, catch_unwind};

use serde_json::{Value, json};

use rig_core::completion::CompletionRequest;
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Wire};

use crate::decode;
pub use crate::generated::Rng;

/// A generated reply: its blocks as wire JSON, its finish, and a seed the
/// builder draws stream splits and other restatement choices from.
#[derive(Clone, Debug)]
pub struct Spec {
    /// The reply's blocks (for Chat, one element: the assistant message).
    pub blocks: Vec<Value>,
    /// The wire's finish or status.
    pub finish: String,
    /// The seed of the restatement choices.
    pub seed: u64,
}

/// `frames`, JSON texts, as the frames a text wire reads.
pub fn texts(frames: Vec<String>) -> Vec<rig_core::wire::WireFrame> {
    frames
        .into_iter()
        .map(rig_core::wire::WireFrame::Text)
        .collect()
}

/// `frames`, JSON documents, as the frames a text wire reads.
pub fn values(frames: Vec<Value>) -> Vec<rig_core::wire::WireFrame> {
    frames
        .into_iter()
        .map(|frame| rig_core::wire::WireFrame::Text(frame.to_string()))
        .collect()
}

/// A spec as a wire's frames: the whole reply and its stream.
pub struct Frames<F> {
    /// The whole reply.
    pub whole: Vec<F>,
    /// The streamed reply.
    pub streamed: Vec<F>,
}

/// `run`'s result, or its panic message.
pub(crate) fn quiet<T>(run: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(run)).map_err(|panic| {
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
            .unwrap_or_else(|| "panic".to_owned())
    })
}

fn first_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(200)
        .collect()
}

/// A turn as JSON with rig-issued ids numbered and fingerprints left out,
/// as `assert_restated_agrees` compares.
fn numbered(mut turn: Value) -> Value {
    fn walk(value: &mut Value, next: &mut usize) {
        match value {
            Value::Object(fields) => {
                fields.shift_remove("fingerprint");
                if let Some(local) = fields.get_mut("local") {
                    *local = json!(*next);
                    *next += 1;
                }
                fields.values_mut().for_each(|value| walk(value, next));
            }
            Value::Array(values) => values.iter_mut().for_each(|value| walk(value, next)),
            _ => {}
        }
    }
    walk(&mut turn, &mut 0);
    turn
}

/// How the whole and the streamed fold of `frames` differ on `wire`, if
/// they do (row H18): their turns or stops, or one mode failing alone.
pub fn disagreement<W: Wire<Op = Completion>>(
    wire: &W,
    frames: Frames<W::Frame>,
) -> Option<String> {
    let request = CompletionRequest::new("restate");
    let Frames { whole, streamed } = frames;
    let unary = quiet(|| decode(wire, &request, Mode::Unary, whole));
    let stream = quiet(|| decode(wire, &request, Mode::Streaming, streamed));
    let outcome = |result: &Result<
        Result<rig_core::completion::CompletionResponse, rig_core::error::ProviderError>,
        String,
    >| match result {
        Ok(Ok(_)) => "ok".to_owned(),
        Ok(Err(error)) => format!("error: {}", first_line(&error.to_string())),
        Err(panic) => format!("panic: {}", first_line(panic)),
    };
    match (&unary, &stream) {
        (Ok(Ok(unary)), Ok(Ok(stream))) => {
            let (whole, streamed) = (
                numbered(json!(unary.message())),
                numbered(json!(stream.message())),
            );
            if whole != streamed {
                return Some(format!(
                    "the folds differ\nwhole:    {whole}\nstreamed: {streamed}"
                ));
            }
            (unary.stop() != stream.stop()).then(|| {
                format!(
                    "the stops differ: whole {:?}, streamed {:?}",
                    unary.stop(),
                    stream.stop()
                )
            })
        }
        (unary, stream) => {
            let (unary, stream) = (outcome(unary), outcome(stream));
            let panicked = unary.starts_with("panic") || stream.starts_with("panic");
            ((unary == "ok") != (stream == "ok") || panicked)
                .then(|| format!("whole {unary}, streamed {stream}"))
        }
    }
}

/// The smallest spec that still `fails`: blocks dropped one at a time, then,
/// for a Chat spec (one message), its fields.
pub fn shrink(spec: Spec, fails: impl Fn(&Spec) -> bool) -> Spec {
    let mut spec = spec;
    let mut at = 0;
    while at < spec.blocks.len() {
        let mut candidate = spec.clone();
        candidate.blocks.remove(at);
        if fails(&candidate) {
            spec = candidate;
        } else {
            at += 1;
        }
    }
    while let Some(candidate) = chat_fields(&spec)
        .into_iter()
        .find(|candidate| fails(candidate))
    {
        spec = candidate;
    }
    spec
}

pub const TEXTS: &[&str] = &["a", "looking it up", "ünï ✓ \"q\"", "line\nbreak", "x y z"];

pub fn text(rng: &mut Rng) -> String {
    rng.pick(TEXTS).to_owned()
}

pub fn maybe_blank(rng: &mut Rng) -> String {
    if rng.chance(10) {
        rng.pick(&["", " ", "\n"]).to_owned()
    } else {
        text(rng)
    }
}

pub fn args(rng: &mut Rng) -> Value {
    match rng.below(5) {
        0 => json!({}),
        1 => json!({"q": "rig"}),
        2 => json!({"limit": 20.0, "scale": 0.5, "n": 3}),
        3 => json!({"nested": {"a": [1, {"b": null}]}, "s": "ü\"x"}),
        _ => json!({"big": 12_345_678_901_234_567_u64}),
    }
}

// ------------------------------------------------------------ Anthropic

/// Messages content blocks; `hosted` adds redacted thinking and server
/// tools, `signature` is what thinking carries.
pub fn anthropic_spec(rng: &mut Rng, hosted: bool, signature: &str) -> Spec {
    let mut blocks = Vec::new();
    let mut calls = 0;
    for _ in 0..rng.range(0, 5) {
        match rng.below(8) {
            0 | 1 => blocks.push(
                json!({"type": "thinking", "thinking": maybe_blank(rng), "signature": signature}),
            ),
            2 if hosted => blocks.push(json!({"type": "redacted_thinking", "data": "opaque"})),
            3 if hosted => {
                let id = format!("srvtoolu_{}", blocks.len());
                blocks.push(json!({"type": "server_tool_use", "id": id, "name": "web_search", "input": {"query": "rig"}}));
                blocks.push(json!({"type": "web_search_tool_result", "tool_use_id": id, "content": [{"type": "web_search_result", "url": "https://rig.rs", "title": "Rig", "encrypted_content": "enc"}]}));
            }
            4 | 5 => {
                calls += 1;
                blocks.push(json!({"type": "tool_use", "id": format!("toolu_{calls}"), "name": "lookup", "input": args(rng)}));
            }
            6 if rng.chance(30) => {
                blocks.push(json!({"type": "x_rig_invented", "id": "x_1", "payload": {"n": 1}}))
            }
            _ => {
                let mut block = json!({"type": "text", "text": maybe_blank(rng)});
                if hosted && rng.chance(20) {
                    block["citations"] = json!([{"type": "web_search_result_location", "url": "https://rig.rs", "title": "Rig", "encrypted_index": "ei", "cited_text": "rig"}]);
                }
                blocks.push(block);
            }
        }
    }
    let finish = rng
        .pick(&[
            "end_turn",
            "tool_use",
            "max_tokens",
            "pause_turn",
            "refusal",
            "stop_sequence",
            "model_context_window_exceeded",
        ])
        .to_owned();
    Spec {
        blocks,
        finish,
        seed: 0,
    }
}

fn anthropic_message(model: &str, content: Vec<Value>, stop: Value) -> Value {
    json!({
        "type": "message", "id": "msg_1", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 5}
    })
}

/// The whole message and its event stream, as JSON texts.
pub fn anthropic_build(model: &str, spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut rng = Rng::new(spec.seed);
    let whole = anthropic_message(model, spec.blocks.clone(), json!(spec.finish)).to_string();
    let mut events = vec![
        json!({"type": "message_start", "message": anthropic_message(model, vec![], Value::Null)}),
    ];
    if rng.chance(30) {
        events.push(json!({"type": "ping"}));
    }
    for (index, block) in spec.blocks.iter().enumerate() {
        let mut start = block.clone();
        let mut deltas = Vec::new();
        if let Some(citations) = block.get("citations").and_then(Value::as_array) {
            start["citations"] = json!([]);
            for citation in citations {
                deltas.push(json!({"type": "citations_delta", "citation": citation}));
            }
        }
        for (key, delta) in [
            ("text", "text_delta"),
            ("thinking", "thinking_delta"),
            ("signature", "signature_delta"),
        ] {
            if let Some(text) = block.get(key).and_then(Value::as_str) {
                start[key] = json!("");
                if !text.is_empty() {
                    let pieces = if key == "signature" {
                        vec![text.to_owned()]
                    } else {
                        rng.split(text)
                    };
                    for piece in pieces {
                        let mut fields = serde_json::Map::new();
                        fields.insert("type".to_owned(), json!(delta));
                        fields.insert(key.to_owned(), json!(piece));
                        deltas.push(Value::Object(fields));
                    }
                }
            }
        }
        if let Some(input) = block.get("input") {
            start["input"] = json!({});
            let empty = input.as_object().is_some_and(serde_json::Map::is_empty) && rng.chance(50);
            if (block["type"] == "tool_use" || block["type"] == "server_tool_use") && !empty {
                let text = input.to_string();
                let mut pieces = rng.split(&text);
                if rng.chance(20) {
                    pieces.insert(0, String::new());
                }
                for piece in pieces {
                    deltas.push(json!({"type": "input_json_delta", "partial_json": piece}));
                }
            }
        }
        events.push(json!({"type": "content_block_start", "index": index, "content_block": start}));
        events.extend(
            deltas.into_iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ),
        );
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    events.push(json!({"type": "message_delta", "delta": {"stop_reason": spec.finish, "stop_sequence": null}, "usage": {"output_tokens": 5}}));
    if rng.chance(70) {
        events.push(json!({"type": "message_stop"}));
    }
    (
        vec![whole],
        events.into_iter().map(|e| e.to_string()).collect(),
    )
}

// ------------------------------------------------------------ Responses

pub fn responses_spec(rng: &mut Rng) -> Spec {
    let mut blocks = Vec::new();
    for k in 0..rng.range(0, 5) {
        match rng.below(7) {
            0 | 1 => {
                let summary: Vec<Value> = (0..rng.range(0, 2))
                    .map(|_| json!({"type": "summary_text", "text": text(rng)}))
                    .collect();
                let mut item = json!({"type": "reasoning", "id": format!("rs_{k}"), "summary": summary});
                if rng.chance(80) {
                    item["encrypted_content"] = json!(format!("ciphertext-{k}"));
                }
                blocks.push(item);
            }
            2 | 3 => blocks.push(json!({"type": "function_call", "id": format!("fc_{k}"), "call_id": format!("call_{k}"), "name": "lookup", "arguments": args(rng).to_string(), "status": "completed"})),
            4 => blocks.push(json!({"type": "web_search_call", "id": format!("ws_{k}"), "status": "completed", "action": {"type": "search", "query": "rig"}})),
            5 if rng.chance(30) => blocks.push(json!({"type": "x_rig_invented", "id": format!("x_{k}"), "payload": {"n": 1}})),
            5 if rng.chance(50) => blocks.push(json!({"type": "custom_tool_call", "id": format!("ctc_{k}"), "call_id": format!("call_{k}"), "name": "lookup", "input": text(rng), "status": "completed"})),
            5 => blocks.push(json!({"type": "reasoning", "id": format!("rs_{k}"), "summary": [], "content": [{"type": "reasoning_text", "text": text(rng)}]})),
            _ => {
                let mut item = json!({"type": "message", "id": format!("msg_{k}"), "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": maybe_blank(rng), "annotations": [], "logprobs": []}]});
                if rng.chance(30) {
                    item["phase"] = json!(rng.pick(&["commentary", "final_answer"]));
                }
                blocks.push(item);
            }
        }
    }
    let finish = if rng.chance(80) {
        "completed"
    } else {
        "incomplete"
    }
    .to_owned();
    Spec {
        blocks,
        finish,
        seed: 0,
    }
}

fn responses_response(output: &[Value], finish: &str) -> Value {
    let mut response = json!({
        "id": "resp_1", "object": "response", "created_at": 1_700_000_000, "model": "the-model",
        "status": finish, "output": output,
        "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 2}, "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 30},
    });
    if finish == "incomplete" {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    response
}

pub fn responses_build(spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut rng = Rng::new(spec.seed);
    let whole = responses_response(&spec.blocks, &spec.finish).to_string();
    let mut events = vec![
        json!({"type": "response.created", "response": {"id": "resp_1", "object": "response", "status": "in_progress", "output": []}}),
    ];
    if rng.chance(50) {
        events.push(json!({"type": "response.in_progress", "response": {"id": "resp_1", "object": "response", "status": "in_progress", "output": []}}));
    }
    // 0: standard; 1: no output_index on any event; 2: some items only in
    // the terminal response; 3: calls with no argument deltas.
    let variant = match rng.below(100) {
        0..=59 => 0,
        60..=74 => 1,
        75..=89 => 2,
        _ => 3,
    };
    let start = events.len();
    for (index, item) in spec.blocks.iter().enumerate() {
        if variant == 2 && rng.chance(35) {
            continue;
        }
        let id = item.get("id").cloned().unwrap_or(Value::Null);
        let mut added = item.clone();
        let kind = item["type"].as_str().unwrap_or_default().to_owned();
        match kind.as_str() {
            "message" => {
                added["content"] = json!([]);
                added["status"] = json!("in_progress");
            }
            "reasoning" => {
                added["summary"] = json!([]);
                if added.get("content").is_some() {
                    added["content"] = json!([]);
                }
                if let Some(fields) = added.as_object_mut() {
                    fields.shift_remove("encrypted_content");
                }
            }
            "function_call" => {
                added["arguments"] = json!("");
                added["status"] = json!("in_progress");
            }
            "custom_tool_call" => {
                added["input"] = json!("");
                added["status"] = json!("in_progress");
            }
            _ => added["status"] = json!("in_progress"),
        }
        events.push(
            json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        );
        match kind.as_str() {
            "message" => {
                let part = &item["content"][0];
                let text = part["text"].as_str().unwrap_or_default();
                events.push(json!({"type": "response.content_part.added", "output_index": index, "item_id": id, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []}}));
                for piece in rng.split(text) {
                    if !piece.is_empty() {
                        events.push(json!({"type": "response.output_text.delta", "output_index": index, "item_id": id, "content_index": 0, "delta": piece}));
                    }
                }
                events.push(json!({"type": "response.output_text.done", "output_index": index, "item_id": id, "content_index": 0, "text": text}));
                events.push(json!({"type": "response.content_part.done", "output_index": index, "item_id": id, "content_index": 0, "part": part}));
            }
            "reasoning" => {
                for (content_index, part) in
                    item["content"].as_array().into_iter().flatten().enumerate()
                {
                    let text = part["text"].as_str().unwrap_or_default();
                    for piece in rng.split(text) {
                        events.push(json!({"type": "response.reasoning_text.delta", "output_index": index, "item_id": id, "content_index": content_index, "delta": piece}));
                    }
                    events.push(json!({"type": "response.reasoning_text.done", "output_index": index, "item_id": id, "content_index": content_index, "text": text}));
                }
                for (summary_index, summary) in
                    item["summary"].as_array().into_iter().flatten().enumerate()
                {
                    let text = summary["text"].as_str().unwrap_or_default();
                    events.push(json!({"type": "response.reasoning_summary_part.added", "output_index": index, "item_id": id, "summary_index": summary_index,
                        "part": {"type": "summary_text", "text": ""}}));
                    for piece in rng.split(text) {
                        events.push(json!({"type": "response.reasoning_summary_text.delta", "output_index": index, "item_id": id, "summary_index": summary_index, "delta": piece}));
                    }
                    events.push(json!({"type": "response.reasoning_summary_text.done", "output_index": index, "item_id": id, "summary_index": summary_index, "text": text}));
                    events.push(json!({"type": "response.reasoning_summary_part.done", "output_index": index, "item_id": id, "summary_index": summary_index, "part": summary}));
                }
            }
            "function_call" if variant != 3 => {
                let arguments = item["arguments"].as_str().unwrap_or_default();
                for piece in rng.split(arguments) {
                    events.push(json!({"type": "response.function_call_arguments.delta", "output_index": index, "item_id": id, "delta": piece}));
                }
                events.push(json!({"type": "response.function_call_arguments.done", "output_index": index, "item_id": id, "arguments": arguments}));
            }
            "custom_tool_call" if variant != 3 => {
                let input = item["input"].as_str().unwrap_or_default();
                for piece in rng.split(input) {
                    events.push(json!({"type": "response.custom_tool_call_input.delta", "output_index": index, "item_id": id, "delta": piece}));
                }
                events.push(json!({"type": "response.custom_tool_call_input.done", "output_index": index, "item_id": id, "input": input}));
            }
            _ => {}
        }
        events.push(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        );
    }
    if variant == 1 {
        for event in &mut events[start..] {
            if let Some(fields) = event.as_object_mut() {
                fields.shift_remove("output_index");
            }
        }
    }
    let terminal = if spec.finish == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    events.push(
        json!({"type": terminal, "response": responses_response(&spec.blocks, &spec.finish)}),
    );
    let events: Vec<String> = events
        .into_iter()
        .enumerate()
        .map(|(n, mut e)| {
            e["sequence_number"] = json!(n);
            e.to_string()
        })
        .collect();
    (vec![whole], events)
}

// ------------------------------------------------------------ Chat

/// What a Chat dialect's replies carry beyond content, reasoning and calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatShapes {
    /// OpenRouter's `reasoning_details`: a signed text entry with the
    /// reasoning, a ciphertext ahead of the answer, an entry signing a call,
    /// and a bare signature after the answer.
    pub details: bool,
    /// Mistral's content parts: thinking parts around the text, and
    /// refusals.
    pub thinking_parts: bool,
}

/// The block list is one element: the assistant message. `key` is the
/// dialect's reasoning field.
pub fn chat_spec(rng: &mut Rng, key: Option<&str>, shapes: ChatShapes) -> Spec {
    let mut message = json!({"role": "assistant"});
    message["content"] = match rng.below(10) {
        0 => Value::Null,
        1 => json!(""),
        _ => json!(text(rng)),
    };
    if shapes.thinking_parts && rng.chance(50) {
        let thinking = |rng: &mut Rng| json!({"type": "thinking", "thinking": [{"type": "text", "text": text(rng)}]});
        let mut parts = Vec::new();
        if rng.chance(70) {
            parts.push(thinking(rng));
        }
        parts.push(json!({"type": "text", "text": text(rng)}));
        if rng.chance(15) {
            parts.push(json!({"type": "refusal", "refusal": "no"}));
        }
        if rng.chance(30) {
            parts.push(thinking(rng));
        }
        message["content"] = json!(parts);
    }
    if let Some(key) = key.filter(|_| rng.chance(60)) {
        message[key] = json!(text(rng));
    }
    let calls = rng.range(0, 3);
    let idless = rng.chance(10);
    if calls > 0 {
        let calls: Vec<Value> = (0..calls)
            .map(|i| {
                let arguments = if rng.chance(10) { String::new() } else { args(rng).to_string() };
                let mut call = json!({"id": format!("abcDEF{i:03}"), "type": "function", "function": {"name": "lookup", "arguments": arguments}});
                if idless && let Some(fields) = call.as_object_mut() {
                    fields.shift_remove("id");
                }
                call
            })
            .collect();
        message["tool_calls"] = json!(calls);
    }
    if shapes.details && rng.chance(60) {
        let mut details = Vec::new();
        let reasoning = key.and_then(|key| message.get(key)).and_then(Value::as_str);
        if let Some(reasoning) = reasoning.filter(|_| rng.chance(60)) {
            details.push(
                json!({"type": "reasoning.text", "text": reasoning, "signature": "EsYF",
                "format": "anthropic-claude-v1", "index": 0}),
            );
        }
        if rng.chance(30) {
            details.push(
                json!({"type": "reasoning.encrypted", "data": "enc", "id": "rs_1",
                "format": "openai-responses-v1", "index": 0}),
            );
        }
        if calls > 0 && !idless && rng.chance(30) {
            let signed = rng.below(calls);
            details.push(json!({"type": "reasoning.encrypted", "data": "gsig",
                "id": format!("abcDEF{signed:03}"), "format": "google-gemini-v1", "index": 0}));
        }
        if details.is_empty() || rng.chance(25) {
            details.push(json!({"type": "reasoning.text", "signature": "AY89",
                "format": "google-gemini-v1", "index": 0}));
        }
        message["reasoning_details"] = json!(details);
    }
    let finish = if calls > 0 && rng.chance(85) {
        "tool_calls".to_owned()
    } else {
        rng.pick(&["stop", "stop", "length"]).to_owned()
    };
    Spec {
        blocks: vec![message],
        finish,
        seed: 0,
    }
}

/// Shrinking drops the whole message; this one shrinks fields instead.
pub fn chat_fields(spec: &Spec) -> Vec<Spec> {
    let mut out = Vec::new();
    if spec.blocks.len() != 1 || spec.blocks[0].get("role").is_none() {
        return out;
    }
    if let Some(Value::Object(fields)) = spec.blocks.first() {
        for key in fields.keys().filter(|k| *k != "role") {
            let mut smaller = spec.clone();
            if let Some(Value::Object(f)) = smaller.blocks.first_mut() {
                f.shift_remove(key);
            }
            out.push(smaller);
        }
    }
    out
}

pub fn chat_build(model: &str, key: Option<&str>, spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut rng = Rng::new(spec.seed);
    let message = spec
        .blocks
        .first()
        .cloned()
        .unwrap_or_else(|| json!({"role": "assistant", "content": null}));
    let usage = json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15});
    let whole = json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": spec.finish, "message": message}], "usage": usage}).to_string();
    let first_content = if rng.chance(50) {
        Value::Null
    } else {
        json!("")
    };
    let mut deltas = vec![json!({"role": "assistant", "content": first_content})];
    // A stream sends each detail where it belongs: a signed text entry with
    // the reasoning it signs, a ciphertext ahead of the answer, an entry
    // signing a call with that call, and a bare signature last.
    let details: Vec<Value> = message
        .get("reasoning_details")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let of = |kind: &str, signs: bool| -> Vec<Value> {
        details
            .iter()
            .filter(|detail| detail["type"] == kind && detail.get("text").is_some() != signs)
            .cloned()
            .collect()
    };
    let signed_text = of("reasoning.text", false);
    let bare = of("reasoning.text", true);
    let leading: Vec<Value> = details
        .iter()
        .filter(|detail| detail["type"] == "reasoning.encrypted" && detail["id"] == "rs_1")
        .cloned()
        .collect();
    let signing: Vec<Value> = details
        .iter()
        .filter(|detail| detail["type"] == "reasoning.encrypted" && detail["id"] != "rs_1")
        .cloned()
        .collect();
    if let Some(key) = key
        && let Some(text) = message.get(key).and_then(Value::as_str)
    {
        let pieces = rng.split(text);
        let last = pieces.len() - 1;
        for (at, piece) in pieces.into_iter().enumerate() {
            let mut delta = json!({key: piece});
            if let Some(detail) = signed_text.first() {
                let mut detail = detail.clone();
                detail["text"] = json!(piece);
                if at != last
                    && let Some(fields) = detail.as_object_mut()
                {
                    fields.shift_remove("signature");
                }
                delta["reasoning_details"] = json!([detail]);
            }
            deltas.push(delta);
        }
    }
    if !leading.is_empty() {
        deltas.push(json!({"reasoning_details": leading}));
    }
    match message.get("content") {
        Some(Value::String(text)) => {
            for piece in rng.split(text) {
                if !piece.is_empty() {
                    deltas.push(json!({"content": piece}));
                }
            }
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                match part["text"].as_str().filter(|_| part["type"] == "text") {
                    Some(text) => {
                        for piece in rng.split(text) {
                            deltas.push(json!({"content": piece}));
                        }
                    }
                    None => deltas.push(json!({"content": [part]})),
                }
            }
        }
        _ => {}
    }
    let whole_calls = rng.chance(30);
    let indexless = whole_calls && rng.chance(30);
    for (i, call) in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let arguments = call["function"]["arguments"].as_str().unwrap_or_default();
        let signs: Vec<Value> = signing
            .iter()
            .filter(|detail| call.get("id") == detail.get("id"))
            .cloned()
            .collect();
        let with_signs = |mut delta: Value| {
            if !signs.is_empty() {
                delta["reasoning_details"] = json!(signs);
            }
            delta
        };
        if whole_calls {
            let mut call = call.clone();
            if !indexless {
                call["index"] = json!(i);
            }
            deltas.push(with_signs(json!({"tool_calls": [call]})));
        } else {
            let mut opening = json!({"index": i, "type": "function", "function": {"name": "lookup", "arguments": ""}});
            if let Some(id) = call.get("id") {
                opening["id"] = id.clone();
            }
            deltas.push(with_signs(json!({"tool_calls": [opening]})));
            for piece in rng.split(arguments) {
                deltas
                    .push(json!({"tool_calls": [{"index": i, "function": {"arguments": piece}}]}));
            }
        }
    }
    if !bare.is_empty() {
        deltas.push(json!({"reasoning_details": bare}));
    }
    let chunk = |delta: Value, finish: Value| {
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        .to_string()
    };
    let finish_alone = rng.chance(60);
    let count = deltas.len();
    let mut frames: Vec<String> = deltas
        .into_iter()
        .enumerate()
        .map(|(at, delta)| {
            chunk(
                delta,
                if !finish_alone && at + 1 == count {
                    json!(spec.finish)
                } else {
                    Value::Null
                },
            )
        })
        .collect();
    if finish_alone {
        frames.push(chunk(json!({}), json!(spec.finish)));
    }
    frames.push(json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage}).to_string());
    frames.push("[DONE]".to_owned());
    (vec![whole], frames)
}

// ------------------------------------------------------------ Gemini

const SIGS: &[&str] = &["c2lnLTE=", "c2lnLTI=", "c2lnLTM="];

pub fn gemini_spec(rng: &mut Rng, ids: bool) -> Spec {
    let mut blocks: Vec<Value> = Vec::new();
    let mut last = "";
    for k in 0..rng.range(0, 5) {
        let kind = rng.pick(&["thought", "text", "call", "call", "image", "code", "result"]);
        if (kind == "thought" || kind == "text") && kind == last {
            continue;
        }
        last = kind;
        let mut part = match kind {
            "thought" => json!({"text": text(rng), "thought": true}),
            "text" => json!({"text": text(rng)}),
            "call" => {
                let mut call = if rng.chance(10) {
                    json!({"name": "lookup"})
                } else {
                    json!({"name": "lookup", "args": args(rng)})
                };
                if ids && rng.chance(60) {
                    call["id"] = json!(format!("call_{k}"));
                }
                json!({"functionCall": call})
            }
            "image" => json!({"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}),
            "code" => json!({"executableCode": {"language": "PYTHON", "code": "print(1)"}}),
            _ => json!({"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}}),
        };
        if matches!(kind, "thought" | "text" | "call") && rng.chance(50) {
            part["thoughtSignature"] = json!(rng.pick(SIGS));
        }
        blocks.push(part);
    }
    let finish = rng
        .pick(&[
            "STOP",
            "STOP",
            "STOP",
            "MAX_TOKENS",
            "SAFETY",
            "MALFORMED_FUNCTION_CALL",
        ])
        .to_owned();
    Spec {
        blocks,
        finish,
        seed: 0,
    }
}

pub fn gemini_document(parts: Value, finish: Option<&str>, model: &str) -> Value {
    let mut candidate = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let Some(finish) = finish {
        candidate["finishReason"] = json!(finish);
    }
    json!({
        "candidates": [candidate],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": model,
        "responseId": "resp_1",
    })
}

/// The whole document and the stream's chunks. `signature_only` allows a
/// text's signature to come in a trailing empty-text part, as Gemini 3
/// streams it.
pub fn gemini_build(model: &str, spec: &Spec, signature_only: bool) -> (Vec<Value>, Vec<Value>) {
    let mut rng = Rng::new(spec.seed);
    let whole = gemini_document(Value::Array(spec.blocks.clone()), Some(&spec.finish), model);
    let mut chunks: Vec<Vec<Value>> = Vec::new();
    for part in &spec.blocks {
        match part.get("text").and_then(Value::as_str) {
            Some(text) => {
                let pieces = rng.split(text);
                let last = pieces.len() - 1;
                let alone =
                    signature_only && part.get("thoughtSignature").is_some() && rng.chance(40);
                for (i, piece) in pieces.into_iter().enumerate() {
                    let mut piece_part = part.clone();
                    piece_part["text"] = json!(piece);
                    if (i != last || alone)
                        && let Some(fields) = piece_part.as_object_mut()
                    {
                        fields.shift_remove("thoughtSignature");
                    }
                    chunks.push(vec![piece_part]);
                }
                if alone {
                    let mut sig = json!({"text": "", "thoughtSignature": part["thoughtSignature"]});
                    if part.get("thought").is_some() {
                        sig["thought"] = json!(true);
                    }
                    chunks.push(vec![sig]);
                }
            }
            None => {
                if rng.chance(30) && !chunks.is_empty() {
                    let last = chunks.len() - 1;
                    let joinable = chunks[last].iter().all(|p| p.get("text").is_none());
                    if joinable {
                        chunks[last].push(part.clone());
                        continue;
                    }
                }
                chunks.push(vec![part.clone()]);
            }
        }
    }
    let finish_alone = chunks.is_empty() || rng.chance(30);
    let count = chunks.len();
    let mut docs: Vec<Value> = chunks
        .into_iter()
        .enumerate()
        .map(|(at, parts)| {
            gemini_document(
                Value::Array(parts),
                (!finish_alone && at + 1 == count).then_some(spec.finish.as_str()),
                model,
            )
        })
        .collect();
    if finish_alone {
        let mut doc = gemini_document(json!([]), Some(&spec.finish), model);
        if let Some(candidate) = doc["candidates"][0].as_object_mut() {
            candidate.shift_remove("content");
        }
        docs.push(doc);
    }
    (vec![whole], docs)
}

// ------------------------------------------------------------ Converse

pub fn converse_spec(rng: &mut Rng, signature: Option<&str>) -> Spec {
    let mut blocks = Vec::new();
    for k in 0..rng.range(0, 5) {
        match rng.below(7) {
            0 | 1 => {
                let mut reasoning = json!({"text": text(rng)});
                if let Some(signature) = signature {
                    reasoning["signature"] = json!(format!("{signature}-{k}"));
                }
                blocks.push(json!({"reasoningContent": {"reasoningText": reasoning}}));
            }
            2 if signature.is_some() => blocks.push(json!({"reasoningContent": {"redactedContent": "AGNpcGhlcnRleHQA"}})),
            3 | 4 => blocks.push(json!({"toolUse": {"toolUseId": format!("tooluse_{k}"), "name": "lookup", "input": args(rng)}})),
            5 if rng.chance(30) => blocks.push(json!({"citationsContent": {
                "content": [{"text": text(rng)}],
                "citations": [{"title": "hours", "location": {"documentChar": {"documentIndex": 0, "start": 0, "end": 3}}}]}})),
            _ => blocks.push(json!({"text": text(rng)})),
        }
    }
    let finish = rng
        .pick(&[
            "end_turn",
            "tool_use",
            "max_tokens",
            "stop_sequence",
            "guardrail_intervened",
        ])
        .to_owned();
    Spec {
        blocks,
        finish,
        seed: 0,
    }
}

/// The whole document and the stream events, each `{"<type>": payload}`.
pub fn converse_build(spec: &Spec) -> (Value, Vec<Value>) {
    let mut rng = Rng::new(spec.seed);
    let usage = json!({ "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 });
    let whole = json!({
        "output": { "message": { "role": "assistant", "content": spec.blocks } },
        "stopReason": spec.finish, "usage": usage, "metrics": { "latencyMs": 5 },
    });
    let delta = |index: usize, delta: Value| json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } });
    let mut events = vec![json!({ "messageStart": { "role": "assistant" } })];
    for (index, block) in spec.blocks.iter().enumerate() {
        let (kind, body) = block
            .as_object()
            .and_then(|b| b.iter().next())
            .expect("a Converse block is one key");
        match kind.as_str() {
            "text" => {
                for piece in rng.split(body.as_str().unwrap_or_default()) {
                    events.push(delta(index, json!({ "text": piece })));
                }
            }
            "reasoningContent" => {
                if let Some(redacted) = body.get("redactedContent") {
                    events.push(delta(
                        index,
                        json!({ "reasoningContent": { "redactedContent": redacted } }),
                    ));
                } else {
                    let reasoning = &body["reasoningText"];
                    for piece in rng.split(reasoning["text"].as_str().unwrap_or_default()) {
                        events.push(delta(
                            index,
                            json!({ "reasoningContent": { "text": piece } }),
                        ));
                    }
                    if let Some(signature) = reasoning.get("signature") {
                        events.push(delta(
                            index,
                            json!({ "reasoningContent": { "signature": signature } }),
                        ));
                    }
                }
            }
            "citationsContent" => {
                for part in body["content"].as_array().into_iter().flatten() {
                    events.push(delta(index, json!({ "text": part["text"] })));
                }
                for citation in body["citations"].as_array().into_iter().flatten() {
                    events.push(delta(index, json!({ "citation": citation })));
                }
            }
            "toolUse" => {
                let mut opened = body.clone();
                if let Value::Object(fields) = &mut opened {
                    fields.shift_remove("input");
                }
                events.push(json!({ "contentBlockStart": { "contentBlockIndex": index, "start": { "toolUse": opened } } }));
                for piece in rng.split(&body["input"].to_string()) {
                    events.push(delta(index, json!({ "toolUse": { "input": piece } })));
                }
            }
            _ => events.push(
                json!({ "contentBlockStart": { "contentBlockIndex": index, "start": block } }),
            ),
        }
        events.push(json!({ "contentBlockStop": { "contentBlockIndex": index } }));
    }
    events.push(json!({ "messageStop": { "stopReason": spec.finish } }));
    events.push(json!({ "metadata": { "usage": usage, "metrics": { "latencyMs": 5 } } }));
    (whole, events)
}

// ------------------------------------------------------------ Interactions

pub fn interactions_spec(rng: &mut Rng) -> Spec {
    let mut blocks = Vec::new();
    let mut calls = false;
    for k in 0..rng.range(0, 5) {
        match rng.below(6) {
            0 | 1 => blocks.push(json!({"type": "thought", "signature": rng.pick(SIGS), "summary": [{"type": "text", "text": text(rng)}]})),
            2 | 3 => {
                calls = true;
                blocks.push(json!({"type": "function_call", "id": format!("call_{k}"), "name": "lookup", "arguments": args(rng)}));
            }
            4 if rng.chance(40) => blocks.push(json!({"type": "model_output", "content": [
                {"type": "text", "text": text(rng)}, {"type": "image", "data": "aW1n", "mime_type": "image/png"}]})),
            _ => blocks.push(json!({"type": "model_output", "content": [{"type": "text", "text": text(rng)}]})),
        }
    }
    let finish = if calls && rng.chance(80) {
        "requires_action"
    } else {
        rng.pick(&["completed", "completed", "incomplete", "failed"])
    }
    .to_owned();
    Spec {
        blocks,
        finish,
        seed: 0,
    }
}

pub fn interactions_build(model: &str, spec: &Spec) -> (Vec<Value>, Vec<Value>) {
    let mut rng = Rng::new(spec.seed);
    let usage = json!({"total_input_tokens": 10, "total_output_tokens": 5, "total_thought_tokens": 2, "total_tokens": 17});
    let whole = json!({"id": "int_1", "model": model, "object": "interaction", "status": spec.finish, "steps": spec.blocks, "usage": usage});
    let mut events = vec![json!({"event_type": "interaction.created",
        "interaction": {"id": "int_1", "model": model, "object": "interaction", "status": "in_progress"}})];
    for (index, step) in spec.blocks.iter().enumerate() {
        let start = |step: Value| json!({"event_type": "step.start", "index": index, "step": step});
        let delta =
            |delta: Value| json!({"event_type": "step.delta", "index": index, "delta": delta});
        match step["type"].as_str().unwrap_or_default() {
            "thought" => {
                events.push(start(json!({"type": "thought"})));
                for summary in step["summary"].as_array().into_iter().flatten() {
                    events.push(delta(
                        json!({"type": "thought_summary", "content": summary}),
                    ));
                }
                events.push(delta(
                    json!({"type": "thought_signature", "signature": step["signature"]}),
                ));
            }
            "model_output" => {
                let mut opening = step.clone();
                if let Some(fields) = opening.as_object_mut() {
                    fields.shift_remove("content");
                }
                events.push(start(opening));
                for item in step["content"].as_array().into_iter().flatten() {
                    match item["text"].as_str() {
                        Some(text) => {
                            for piece in rng.split(text) {
                                events.push(delta(json!({"type": "text", "text": piece})));
                            }
                        }
                        None => events.push(delta(item.clone())),
                    }
                }
            }
            "function_call" => {
                let mut opening = step.clone();
                opening["arguments"] = json!({});
                events.push(start(opening));
                for piece in rng.split(&step["arguments"].to_string()) {
                    events.push(delta(
                        json!({"type": "arguments_delta", "arguments": piece}),
                    ));
                }
            }
            _ => events.push(start(step.clone())),
        }
        events.push(json!({"event_type": "step.stop", "index": index}));
    }
    events.push(json!({"event_type": "interaction.completed",
        "interaction": {"id": "int_1", "model": model, "object": "interaction", "status": spec.finish, "usage": usage}}));
    (vec![whole], events)
}
