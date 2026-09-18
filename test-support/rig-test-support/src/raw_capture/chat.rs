//! The chat-completions format contract.
//!
//! Every provider speaking the OpenAI chat-completions wire reports the same
//! fields under the same names, so "the normalized view reproduces the
//! recorded reply" and "the terminal record reproduces the recorded terminal
//! frame" are one operation per format rather than one per provider. What is
//! *not* shared: which fixture frame counts as the terminal one (dialects
//! disagree — see [`recorded_sole_usage_frame`] against
//! [`recorded_agreeing_usage_frames`]), what a dialect adds beside the
//! OpenAI-compatible fields, and whether it contracts a transport id header.
//! Those stay at the call site.

use rig_core::completion::{CompletionResponse, FinishReason};
use rig_core::providers::openai;
use rig_core::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};
use rig_core::streaming::StreamFinal;
use serde::Deserialize as _;
use serde_json::Value;

use crate::support::{assert_matches_recorded_token, assistant_text};

/// The provider-native terminal record a chat-completions stream's
/// [`StreamFinal::raw`] holds, over the dialect-tolerant accounting.
pub type Terminal = StreamingCompletionResponse<ChatUsage>;

/// The finish reason a recorded chat-completions body reports.
///
/// Mapped by hand from the wire word, so the expectation is independent of
/// the decoder that produced the normalized reason it is compared against.
pub fn recorded_chat_finish_reason(body: &Value) -> FinishReason {
    match body["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        other => panic!("recorded turn should finish on stop or length, got {other:?}"),
    }
}

/// The finish reason a chat-completions `finish_reason` word names.
///
/// The same hand-written mapping as [`recorded_chat_finish_reason`], for the
/// cells that read the word off a provider-native typed view rather than out
/// of the fixture.
pub fn native_finish_reason(word: &str) -> FinishReason {
    match word {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        other => panic!("unexpected native finish reason {other:?}"),
    }
}

/// The normalized response's fields, checked against the recorded reply bytes
/// that produced them.
///
/// The transport request id is not here: whether a dialect contracts an id
/// header, sends none, or reports one only sometimes are three different
/// contracts, so each cell states its own with
/// [`assert_contracted_request_id`](super::assert_contracted_request_id) or
/// [`assert_no_request_id`](super::assert_no_request_id).
pub fn assert_reproduces_body(
    response: &CompletionResponse,
    provider: &str,
    body: &Value,
    context: &str,
) {
    assert_eq!(response.provider, provider, "{context}: provider");
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_eq!(
        response.model.as_deref(),
        body["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        response.finish_reason(),
        Some(recorded_chat_finish_reason(body)),
        "{context}: finish reason"
    );
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            body["usage"]["prompt_tokens"].as_u64(),
            body["usage"]["completion_tokens"].as_u64(),
            body["usage"]["total_tokens"].as_u64(),
        ),
        "{context}: usage"
    );
    assert_eq!(
        assistant_text(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("recorded content"),
        "{context}: choice text"
    );
}

/// The provider-native view of the captured reply, beside the normalized
/// fields the decoder mapped from it.
///
/// `native` is read out of the response's own `raw` by the cell, which is why
/// this pins the mapping one decoder performed rather than comparing the
/// normalized response with a second copy of the same mapping. Dialects
/// wrapping the shared type (Venice) pass their inner
/// [`openai::CompletionResponse`].
pub fn assert_native_matches_normalized(
    response: &CompletionResponse,
    native: &openai::CompletionResponse,
    context: &str,
) {
    assert_eq!(
        response.response_id.as_deref(),
        Some(native.id.as_str()),
        "{context}: native response id"
    );
    assert_eq!(
        response.model.as_deref(),
        Some(native.model.as_str()),
        "{context}: native model"
    );
    let native_choice = native
        .choices
        .first()
        .expect("the reply carries at least one choice");
    assert_eq!(
        response.finish_reason(),
        Some(native_finish_reason(native_choice.finish_reason.as_str())),
        "{context}: the normalized reason is the native one"
    );
    let native_usage = native.usage.as_ref().expect("the reply reports usage");
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            Some(native_usage.prompt_tokens as u64),
            native_usage.completion_tokens.map(|tokens| tokens as u64),
            Some(native_usage.total_tokens as u64),
        ),
        "{context}: the normalized counters are the native ones"
    );
}

/// The normalized terminal record's fields, checked against the recorded
/// terminal frame that produced them.
///
/// As with [`assert_reproduces_body`], the transport id contract is the
/// cell's to state.
pub fn assert_terminal_reproduces_frame(
    terminal: &StreamFinal,
    provider: &str,
    frame: &Value,
    context: &str,
) {
    assert_eq!(terminal.provider, provider, "{context}: provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        frame["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_eq!(
        terminal.model.as_deref(),
        frame["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        (
            terminal.usage.input_tokens,
            terminal.usage.output_tokens,
            terminal.usage.total_tokens
        ),
        (
            frame["usage"]["prompt_tokens"].as_u64(),
            frame["usage"]["completion_tokens"].as_u64(),
            frame["usage"]["total_tokens"].as_u64(),
        ),
        "{context}: usage"
    );
}

/// The terminal record `raw` holds, read back and required to be exactly the
/// decoder's record serialized.
///
/// A streamed reply is many frames and no single one of them is the answer,
/// so what rides along is the record the decoder reassembled — which makes an
/// exact typed round trip the right claim here, unlike the blocking path
/// where `raw` is the reply *document*. The returned typed record is the
/// cell's handle on whatever its dialect keeps beside the shared fields.
pub fn assert_terminal_round_trips(terminal: &StreamFinal) -> Terminal {
    let raw = &terminal.raw;
    let typed = Terminal::deserialize(raw)
        .expect("raw is the chat-completions terminal record, serialized");
    assert_eq!(
        serde_json::to_value(&typed).expect("typed serializes"),
        *raw,
        "the captured value is the typed terminal serialized, nothing more"
    );
    assert_eq!(typed.response_id, terminal.response_id, "response id");
    assert_eq!(typed.model, terminal.model, "model");
    assert_eq!(typed.finish_reason, terminal.finish_reason, "finish reason");
    let usage = typed
        .usage
        .as_ref()
        .expect("the terminal record carries the reply's accounting");
    assert_eq!(
        usage.to_normalized(),
        terminal.usage,
        "the normalized usage is that accounting, normalized"
    );
    assert_eq!(
        typed.provider_request_id, None,
        "the transport id is stamped on the normalized terminal, not the native record"
    );
    typed
}

/// The one usage-bearing SSE frame of a recorded chat-completions stream.
///
/// The premise of a terminal-record matrix is that the wire reported its
/// accounting exactly once, on the stream's last data frame; both halves are
/// asserted here rather than assumed.
pub fn recorded_sole_usage_frame(provider: &str, scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(provider, scenario);
    let mut with_usage = frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| !frame["usage"].is_null());
    let (index, terminal) = with_usage
        .next()
        .expect("the recorded stream must carry usage on its terminal frame");
    assert!(
        with_usage.next().is_none(),
        "usage must be reported on exactly one (terminal) frame"
    );
    assert_eq!(
        index + 1,
        frames.len(),
        "the usage-bearing frame must be the stream's last data frame"
    );
    terminal.clone()
}

/// Every recorded SSE frame of a single-interaction stream, beside its last
/// frame, which must carry usage.
///
/// A third premise, distinct from both [`recorded_sole_usage_frame`] (which
/// also requires that no earlier frame reported usage) and
/// [`recorded_agreeing_usage_frames`]: the dialects using this one repeat
/// envelope fields across every chunk, so the cells need all the frames as
/// well as the terminal one.
pub fn recorded_frames_with_terminal(provider: &str, scenario: &str) -> (Vec<Value>, Value) {
    assert_eq!(
        crate::cassettes::recorded_interaction_bodies(provider, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let frames = crate::cassettes::recorded_sse_json_frames(provider, scenario);
    let terminal = frames
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream should carry frames"));
    assert!(
        terminal.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded stream must end with a usage-bearing frame — \
         without it the terminal record carries no usage and this cell proves nothing"
    );
    (frames, terminal)
}

/// The value every recorded chunk agrees on for one envelope key.
///
/// The premise behind a cell about an accumulated envelope field: the wire
/// repeated it, and the terminal record can only be pinned against it if the
/// repetitions agree.
pub fn recorded_envelope_field(frames: &[Value], key: &str, scenario: &str) -> Value {
    let mut values = frames.iter().filter_map(|frame| frame.get(key)).cloned();
    let first = values
        .next()
        .unwrap_or_else(|| panic!("{scenario}: recorded chunks must carry `{key}`"));
    assert!(
        values.all(|value| value == first),
        "{scenario}: every recorded chunk must agree on `{key}`"
    );
    first
}

/// The last usage-bearing SSE frame of a recorded stream whose dialect
/// repeats its accounting across closing frames.
///
/// A different premise from [`recorded_sole_usage_frame`], and deliberately
/// not a relaxation of it: every usage-bearing frame must agree on the counts
/// and none of them may be a content frame, so "the accounting the terminal
/// record reports" stays knowable from the bytes even though more than one
/// frame carried it.
pub fn recorded_agreeing_usage_frames(provider: &str, scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(provider, scenario);
    let with_usage: Vec<&Value> = frames
        .iter()
        .filter(|frame| !frame["usage"].is_null())
        .collect();
    let terminal = (*with_usage
        .last()
        .expect("the recorded stream must carry usage on its terminal frame"))
    .clone();
    for frame in &with_usage {
        assert_eq!(
            frame["usage"], terminal["usage"],
            "every usage-bearing frame reports the same counts"
        );
        assert!(
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .is_none_or(str::is_empty),
            "usage must not ride on a content frame: {frame}"
        );
    }
    terminal
}
