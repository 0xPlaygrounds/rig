//! The history conformance suite: one set of rows every completion wire
//! runs, so each replay invariant is checked on every wire instead of being
//! patched once where a bug surfaced. A wire supplies a [`HistoryFixture`]
//! (its target for a model, replies of each [`Shape`], and its request body
//! as JSON); [`history_conformance_suite!`](crate::history_conformance_suite)
//! expands one test per row, and the workspace registry fails when a wire in
//! [`HISTORY_WIRES`] has no suite.
//!
//! Each row's function documents its invariant. The audit findings each row
//! closes are listed in [`ROWS`]; a finding is closed only by a row or by a
//! test named beside it.
//!
//! ```ignore
//! mod anthropic_history {
//!     rig_core::history_conformance_suite! {
//!         wire: "anthropic",
//!         fixture: super::AnthropicHistory,
//!     }
//! }
//! ```

use std::sync::Mutex;

use serde_json::Value;

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::{EncodeError, ProviderError};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Origin, StopReason, Text, ToolCall,
    ToolFunction, ToolName, ToolResult, ToolResultContent, UserContent,
};
use crate::operation::Completion;
use crate::wire::{Call, Mode, Operation, Reply, Shared, Wire};

/// Every completion wire and dialect that must run the suite, by the name
/// its `history_conformance_suite!` invocation gives.
pub const HISTORY_WIRES: &[&str] = &[
    "mock",
    "anthropic",
    "anthropic_moonshot",
    "openai_responses",
    "chatgpt",
    "copilot",
    "xai",
    "openai_chat",
    "deepseek",
    "openrouter",
    "mistral",
    "groq",
    "perplexity",
    "cohere",
    "ollama",
    "gemini_rest",
    "gemini_interactions",
    "vertexai",
    "gemini_grpc",
    "bedrock_claude",
    "bedrock_nova",
    "candle",
];

/// Each row, and the audit findings (`audit/AUDIT.md`) it closes.
pub const ROWS: &[(&str, &str)] = &[
    (
        "h01_same_model_verbatim",
        "#1315 (Anthropic native before unsigned-thinking rule), Vertex #2026 same-model half, \
         Responses NEW edited-reasoning orphan (identity half)",
    ),
    (
        "h02_cross_model_canonical",
        "chatA NEW-4 (Mistral id collisions), anthropic NEW cross-model assistant images, \
         Gemini #2026 rebuilt calls",
    ),
    (
        "h03_stream_equals_whole",
        "#2647 class (whole and streamed folds disagree), chatA NEW-5",
    ),
    (
        "h04_unknown_kept",
        "#807, #1835, chatB NEW Mistral thinking, gemini NEW mixed Interactions output, \
         core NEW Bedrock server_tool_use and citations, chatA NEW-3 custom calls",
    ),
    (
        "h05_field_ablation",
        "#2668, #1426, #1176, #1512, #2194, #2591, #1984, #2475, #2509, #2510, gemini NEW \
         missing args",
    ),
    (
        "h06_malformed_arguments",
        "#1085, #2359, #2447, #2554, core NEW null arguments",
    ),
    (
        "h07_failed_turns",
        "core #1559, anthropic NEW partial turns, gemini NEW non-success finishes, chatB NEW \
         unknown finish reasons",
    ),
    ("h08_pairing", "#2560, chatA NEW-4, core NEW is_error"),
    (
        "h09_capability_downgrades",
        "#305, #2143, #2380, chatA NEW-1, NEW-2, NEW-6, chatB NEW placeholder, anthropic NEW \
         image capability",
    ),
    ("h10_persistence", "core NEW fingerprint key order"),
    (
        "h11_order",
        "#2647, Responses NEW terminal-only order, chatA NEW-5",
    ),
    (
        "h12_edits",
        "Responses NEW edited reasoning orphans its pair, Responses NEW added snapshot",
    ),
    (
        "h13_rollback",
        "core #1559, core NEW rollback origin, chatB NEW rollback",
    ),
    (
        "h14_model_identity",
        "gemini NEW resume origin model, gemini NEW Vertex model override",
    ),
];

/// The reply shapes a fixture supplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Every block kind the wire has, each with its provider item: text,
    /// reasoning with its signature, a tool call, and an image or a hosted
    /// item where the wire has them.
    Rich,
    /// `[reasoning, text, reasoning, tool call]`, the #2647 shape.
    Interleaved,
    /// An item of the invented type `x_rig_invented`, and a known item
    /// carrying the invented field `x_rig_field`.
    Unknown,
}

/// The JSON-field ablation of a whole reply ([`h05_field_ablation`]).
pub struct Ablation<F> {
    /// The whole reply document.
    pub document: Value,
    /// JSON pointers of the fields a reply cannot do without: the ones a
    /// block or the finish is built from. `*` matches any array index or
    /// object key.
    pub required: &'static [&'static str],
    /// The frames carrying `document`.
    pub frames: fn(Value) -> Vec<F>,
}

/// How a documented finish ends a turn ([`h07_failed_turns`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// A success: `Stop`, `Length` or `ToolUse`.
    Success,
    /// A failure: `Error` or `Aborted`.
    Failure,
}

/// One wire's side of the suite.
pub trait HistoryFixture {
    /// The wire.
    type Wire: Wire<Op = Completion>;

    /// The wire addressing `model`.
    fn wire(&self, model: &str) -> Self::Wire;

    /// The model the replies come from.
    fn model(&self) -> &'static str;

    /// Another model on the same wire.
    fn other_model(&self) -> &'static str;

    /// A model on this wire that reads no images, when it has one.
    fn text_only_model(&self) -> Option<&'static str> {
        None
    }

    /// The request `request` sends on `wire` in `mode`, as JSON. `request`
    /// is prepared already, as the driver prepares it; the body includes
    /// the model it addresses wherever the wire sends it.
    fn body(
        &self,
        wire: &Self::Wire,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError>;

    /// A reply of `shape` in `mode`, or `None` when the wire cannot express
    /// it. [`Shape::Rich`] is required in both modes.
    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<<Self::Wire as Wire>::Frame>>;

    /// A whole reply with one tool call whose argument text is `arguments`,
    /// or `None` when the wire cannot carry that text.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<<Self::Wire as Wire>::Frame>>;

    /// Every finish the wire documents, as a whole reply, and how it ends.
    fn finishes(&self) -> Vec<(&'static str, Vec<<Self::Wire as Wire>::Frame>, Ending)>;

    /// The field ablation of a whole reply, when the wire's replies are
    /// JSON documents.
    fn ablation(&self) -> Option<Ablation<<Self::Wire as Wire>::Frame>> {
        None
    }

    /// Whether the wire keeps provider items. A local runtime keeps none.
    fn keeps_natives(&self) -> bool {
        true
    }

    /// What same-model replay sends for `turn`: by default each current
    /// block native and each replaying opaque item, verbatim. A
    /// message-shaped wire returns the projection its rebuild sends.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        turn.content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Opaque(opaque) if opaque.replay => Some(opaque.item.clone()),
                block => block.native_item().cloned(),
            })
            .collect()
    }
}

/// The response `frames` fold into on `wire`, as a reply to `request`.
///
/// # Panics
///
/// Never; decode failures are returned.
pub fn decode<W: Wire<Op = Completion>>(
    wire: &W,
    request: &CompletionRequest,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> Result<CompletionResponse, ProviderError> {
    let describe = wire.describe();
    let fold = Completion::fold(request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let fed = crate::driver::feed(&mut wire.decoder(), &shared, frames);
    crate::driver::settle(shared, fed, reply_of(describe.name)).outcome
}

/// What a reply cut off after `frames` leaves: the partial response a
/// consumer reads, with no provider end unless the frames carried it.
pub fn partial<W: Wire<Op = Completion>>(
    wire: &W,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> (CompletionResponse, bool) {
    let describe = wire.describe();
    let request = CompletionRequest::new("restate");
    let fold = Completion::fold(&request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let mut decoder = wire.decoder();
    let mut failure = None;
    for frame in frames {
        match crate::driver::step(&mut decoder, &shared, frame, None) {
            Ok(crate::wire::Flow::More) => {}
            Ok(crate::wire::Flow::Ended(_)) => break,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    drop(decoder);
    let mut shared = shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while let Some(item) = shared.take() {
        if let Err(error) = item {
            failure.get_or_insert(error);
        }
    }
    let ended = shared.end.is_some();
    let response = shared.fold.partial(
        shared.end.as_ref(),
        &reply_of(describe.name),
        failure.as_ref(),
    );
    (response, ended)
}

fn reply_of(provider: &str) -> Reply {
    Reply {
        provider: provider.to_owned(),
        raw: Value::Null,
        provider_request_id: None,
    }
}

/// The body `history` sends to `model` on the fixture's wire, prepared as
/// the driver prepares it.
pub fn sent<F: HistoryFixture>(
    fixture: &F,
    model: &str,
    history: Vec<Message>,
    mode: Mode,
) -> Result<Value, String> {
    let wire = fixture.wire(model);
    let mut request = CompletionRequest::new("next");
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let prepared = Completion::prepare(request, &wire.describe()).map_err(|e| e.to_string())?;
    fixture
        .body(&wire, prepared, mode)
        .map_err(|error| error.to_string())
}

/// The turn `shape` decodes to in `mode`.
///
/// # Panics
///
/// When the fixture has no such reply for a required shape, or it fails to
/// decode.
pub fn turn_of<F: HistoryFixture>(
    fixture: &F,
    shape: Shape,
    mode: Mode,
) -> Option<AssistantMessage> {
    let frames = fixture.reply(shape, mode);
    if shape == Shape::Rich {
        assert!(
            frames.is_some(),
            "every wire supplies the rich reply in {mode:?}"
        );
    }
    let frames = frames?;
    let wire = fixture.wire(fixture.model());
    let response = decode(&wire, &CompletionRequest::new("restate"), mode, frames)
        .unwrap_or_else(|error| panic!("the {shape:?} reply decodes in {mode:?}: {error}"));
    match response.message() {
        Some(Message::Assistant(turn)) => Some(turn),
        other => panic!("the {shape:?} reply is an assistant turn: {other:?}"),
    }
}

/// The JSON body and the path of an HTTP wire's encoded request, as one
/// document: the body's fields, with the request path under `"$path"`, so
/// a model a wire sends in its path is part of what [`HistoryFixture::body`]
/// returns.
///
/// # Errors
///
/// When the body is multipart or is not JSON.
pub fn http_body(encoded: &crate::wire::Encoded) -> Result<Value, EncodeError> {
    let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
        return Err(EncodeError::request(
            "the request body is multipart, not JSON",
        ));
    };
    let mut body: Value = serde_json::from_slice(bytes)?;
    if let Value::Object(fields) = &mut body {
        fields.insert(
            "$path".to_owned(),
            Value::String(encoded.request.uri().path().to_owned()),
        );
    }
    Ok(body)
}

/// Whether `item` appears in `document` as a whole subtree.
pub fn contains_subtree(document: &Value, item: &Value) -> bool {
    position_of(document, item).is_some()
}

/// The position of `item` in a depth-first walk of `document`.
fn position_of(document: &Value, item: &Value) -> Option<usize> {
    fn walk(value: &Value, item: &Value, at: &mut usize) -> Option<usize> {
        if value == item {
            return Some(*at);
        }
        *at += 1;
        match value {
            Value::Object(fields) => fields.values().find_map(|value| walk(value, item, at)),
            Value::Array(values) => values.iter().find_map(|value| walk(value, item, at)),
            _ => None,
        }
    }
    walk(document, item, &mut 0)
}

const MODES: [Mode; 2] = [Mode::Unary, Mode::Streaming];

/// H1. A turn the same model produced replays every provider item it holds
/// exactly: the fixture's [`HistoryFixture::replayed`] items all appear in
/// the next request.
pub fn h01_same_model_verbatim<F: HistoryFixture>(fixture: &F) {
    for mode in MODES {
        let Some(turn) = turn_of(fixture, Shape::Rich, mode) else {
            continue;
        };
        let expected = fixture.replayed(&turn);
        if fixture.keeps_natives() {
            assert!(
                !expected.is_empty(),
                "{mode:?}: the rich reply keeps provider items: {turn:?}"
            );
        }
        let body = sent(
            fixture,
            fixture.model(),
            vec![Message::user("q"), Message::Assistant(turn.clone())],
            mode,
        )
        .unwrap_or_else(|error| panic!("{mode:?}: the history encodes: {error}"));
        for item in expected {
            assert!(
                contains_subtree(&body, &item),
                "{mode:?}: the same model gets {item} verbatim in {body}"
            );
        }
    }
}

/// H2. Another model gets canonical fields only: no provider item, no
/// opaque item, reasoning as text, and call ids its wire accepts, with
/// each result following its call.
pub fn h02_cross_model_canonical<F: HistoryFixture>(fixture: &F) {
    for mode in MODES {
        let Some(mut turn) = turn_of(fixture, Shape::Rich, mode) else {
            continue;
        };
        // Two calls whose ids normalize alike stay distinct.
        for id in ["call|a-1", "call|a_1"] {
            turn.content.push(AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire(id),
                ToolFunction::new(name("lookup"), serde_json::json!({"q": id})),
            )));
        }
        let results: Vec<UserContent> = turn
            .tool_calls()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")])))
            .collect();
        let history = vec![
            Message::user("q"),
            Message::Assistant(turn.clone()),
            Message::User { content: results },
        ];
        let target = fixture.wire(fixture.other_model());
        let describe = target.describe();
        let replay = describe
            .replay
            .expect("every completion wire names a replay target");
        let adapted = crate::completion::history::adapt_for_model(
            &history,
            replay,
            Some(fixture.other_model()),
            false,
        );
        let mut calls = Vec::new();
        for message in &adapted {
            match message {
                Message::Assistant(turn) => {
                    for block in &turn.content {
                        assert!(
                            block.native_item().is_none()
                                && !matches!(
                                    block,
                                    AssistantContent::Opaque(_) | AssistantContent::Reasoning(_)
                                ),
                            "{mode:?}: another model gets canonical blocks only: {block:?}"
                        );
                    }
                    calls.extend(turn.tool_calls().map(|call| call.id.clone()));
                }
                Message::User { content } => {
                    for part in content {
                        if let UserContent::ToolResult(result) = part {
                            assert!(
                                calls.contains(&result.call),
                                "{mode:?}: result {:?} follows a call of {calls:?}",
                                result.call
                            );
                        }
                    }
                }
                Message::System { .. } => {}
            }
        }
        let distinct: std::collections::HashSet<_> = calls.iter().collect();
        assert_eq!(
            distinct.len(),
            calls.len(),
            "{mode:?}: call ids stay distinct"
        );
        let reasoning: Vec<&str> = turn
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Reasoning(reasoning)
                    if !reasoning.redacted && !reasoning.text.trim().is_empty() =>
                {
                    Some(reasoning.text.as_str())
                }
                _ => None,
            })
            .collect();
        let body = sent(fixture, fixture.other_model(), history[..2].to_vec(), mode)
            .unwrap_or_else(|error| panic!("{mode:?}: another model's history encodes: {error}"));
        let text = body.to_string();
        for reasoning in reasoning {
            let escaped = serde_json::to_string(reasoning).unwrap_or_default();
            assert!(
                text.contains(escaped.trim_matches('"')),
                "{mode:?}: reasoning reaches another model as text: {body}"
            );
        }
    }
}

/// H3. A whole reply and its restatement as a stream fold into the same
/// turn, for the rich and the interleaved shapes.
pub fn h03_stream_equals_whole<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    for shape in [Shape::Rich, Shape::Interleaved, Shape::Unknown] {
        let (Some(whole), Some(streamed)) = (
            fixture.reply(shape, Mode::Unary),
            fixture.reply(shape, Mode::Streaming),
        ) else {
            assert!(shape != Shape::Rich, "every wire supplies the rich reply");
            continue;
        };
        crate::test_utils::history::assert_restated_agrees(&wire, whole, streamed);
    }
}

/// H4. An invented item type and an invented field survive decode and
/// same-model replay, and never fail the reply.
pub fn h04_unknown_kept<F: HistoryFixture>(fixture: &F) {
    for mode in MODES {
        let Some(turn) = turn_of(fixture, Shape::Unknown, mode) else {
            eprintln!("skip: the wire has no unknown-item shape in {mode:?}");
            continue;
        };
        let body = sent(
            fixture,
            fixture.model(),
            vec![Message::user("q"), Message::Assistant(turn)],
            mode,
        )
        .unwrap_or_else(|error| panic!("{mode:?}: the history encodes: {error}"));
        let text = body.to_string();
        for invented in ["x_rig_invented", "x_rig_field"] {
            assert!(
                text.contains(invented),
                "{mode:?}: `{invented}` replays to the same model: {body}"
            );
        }
    }
}

/// H5. Removing any one field no block or the finish is built from still
/// decodes the whole reply.
pub fn h05_field_ablation<F: HistoryFixture>(fixture: &F) {
    let Some(ablation) = fixture.ablation() else {
        eprintln!("skip: the wire's replies are not JSON documents");
        return;
    };
    let wire = fixture.wire(fixture.model());
    let request = CompletionRequest::new("restate");
    decode(
        &wire,
        &request,
        Mode::Unary,
        (ablation.frames)(ablation.document.clone()),
    )
    .unwrap_or_else(|error| panic!("the whole reply decodes: {error}"));
    for pointer in pointers(&ablation.document) {
        if ablation
            .required
            .iter()
            .any(|required| matches_pointer(required, &pointer))
        {
            continue;
        }
        let mut document = ablation.document.clone();
        remove_pointer(&mut document, &pointer);
        if let Err(error) = decode(&wire, &request, Mode::Unary, (ablation.frames)(document)) {
            panic!("the reply without `{pointer}` still decodes: {error}");
        }
    }
}

/// Every JSON pointer of a field in `document`.
fn pointers(document: &Value) -> Vec<String> {
    fn walk(value: &Value, at: &str, out: &mut Vec<String>) {
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    let path = format!("{at}/{}", key.replace('~', "~0").replace('/', "~1"));
                    out.push(path.clone());
                    walk(value, &path, out);
                }
            }
            Value::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    walk(value, &format!("{at}/{index}"), out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(document, "", &mut out);
    out
}

fn matches_pointer(pattern: &str, pointer: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let pointer: Vec<&str> = pointer.split('/').collect();
    pattern.len() == pointer.len()
        && pattern
            .iter()
            .zip(&pointer)
            .all(|(pattern, part)| *pattern == "*" || pattern == part)
}

fn remove_pointer(document: &mut Value, pointer: &str) {
    let Some((parent, key)) = pointer.rsplit_once('/') else {
        return;
    };
    let key = key.replace("~1", "/").replace("~0", "~");
    if let Some(Value::Object(fields)) = document.pointer_mut(parent) {
        fields.shift_remove(&key);
    }
}

/// H6. Arguments that are not a JSON object never fail the reply: the call
/// keeps an object, and the text it arrived as when it was not one.
pub fn h06_malformed_arguments<F: HistoryFixture>(fixture: &F) {
    let cases: [(&str, Value, bool); 6] = [
        (r#"{"a": 1}"#, serde_json::json!({"a": 1}), false),
        (r#"{"a": "b"#, serde_json::json!({"a": "b"}), true),
        ("not json", serde_json::json!({}), true),
        ("null", serde_json::json!({}), false),
        (r#""{\"a\":1}""#, serde_json::json!({"a": 1}), false),
        ("[1]", serde_json::json!({}), true),
    ];
    let wire = fixture.wire(fixture.model());
    for mode in MODES {
        for (arguments, expected, invalid) in &cases {
            let Some(frames) = fixture.call_reply(arguments, mode) else {
                eprintln!("skip: the wire cannot carry `{arguments}` in {mode:?}");
                continue;
            };
            let response = decode(&wire, &CompletionRequest::new("restate"), mode, frames)
                .unwrap_or_else(|error| {
                    panic!("{mode:?}: arguments `{arguments}` do not fail the reply: {error}")
                });
            let calls: Vec<&ToolCall> = response.tool_calls().collect();
            let [call] = calls.as_slice() else {
                panic!(
                    "{mode:?}: the call to `{arguments}` is kept: {:?}",
                    response.choice
                );
            };
            assert_eq!(
                call.function.arguments_value(),
                *expected,
                "{mode:?}: `{arguments}`"
            );
            assert_eq!(
                call.function.invalid_arguments.is_some(),
                *invalid,
                "{mode:?}: `{arguments}` keeps its text only when it is not an object"
            );
        }
    }
}

/// H7. Every documented finish ends a turn as documented, a reply cut off
/// before its end is a failed turn, and the adapter skips failed turns.
pub fn h07_failed_turns<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    let finishes = fixture.finishes();
    assert!(!finishes.is_empty(), "the wire documents its finishes");
    for (name, frames, ending) in finishes {
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            frames,
        )
        .unwrap_or_else(|error| panic!("the `{name}` reply decodes: {error}"));
        let failed = response.stop().is_failure();
        assert_eq!(
            failed,
            ending == Ending::Failure,
            "`{name}` ends as {:?}",
            response.stop()
        );
    }
    let Some(count) = fixture
        .reply(Shape::Rich, Mode::Streaming)
        .map(|frames| frames.len())
    else {
        return;
    };
    for cut in 1..count {
        let frames = fixture
            .reply(Shape::Rich, Mode::Streaming)
            .unwrap_or_default();
        let (response, ended) = partial(&wire, Mode::Streaming, frames.into_iter().take(cut));
        if ended {
            continue;
        }
        let Some(Message::Assistant(turn)) = response.message() else {
            continue;
        };
        assert!(
            turn.stop.as_ref().is_some_and(StopReason::is_failure),
            "a reply cut after {cut} frames is a failed turn: {:?}",
            turn.stop
        );
        let history = vec![
            Message::user("q"),
            Message::Assistant(turn),
            Message::user("next"),
        ];
        let describe = wire.describe();
        let replay = describe.replay.expect("a replay target");
        let adapted = crate::completion::history::adapt(&history, replay);
        assert!(
            adapted
                .iter()
                .all(|message| !matches!(message, Message::Assistant(_))),
            "a failed turn is never replayed: {adapted:?}"
        );
    }
}

/// H8. Every call is answered once: an unanswered call gets an error
/// result, a result no call asked for is dropped, and the history encodes.
pub fn h08_pairing<F: HistoryFixture>(fixture: &F) {
    let other = Origin::new("other.api", "other", "other-model");
    let turn = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("call_answered"),
                ToolFunction::new(name("lookup"), serde_json::json!({})),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("call_unanswered"),
                ToolFunction::new(name("lookup"), serde_json::json!({})),
            )),
        ],
        origin: Some(other),
        stop: Some(StopReason::ToolUse),
        native: None,
    };
    let answered = turn.content.first().and_then(|block| match block {
        AssistantContent::ToolCall(call) => Some(call.result(vec![ToolResultContent::text("ok")])),
        _ => None,
    });
    let orphan = ToolResult {
        call: CallId::from_wire("call_gone"),
        name: name("lookup"),
        content: vec![ToolResultContent::text("stale")],
        is_error: false,
    };
    let history = vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::User {
            content: answered
                .into_iter()
                .chain([orphan])
                .map(UserContent::ToolResult)
                .collect(),
        },
    ];
    let wire = fixture.wire(fixture.model());
    let describe = wire.describe();
    let replay = describe.replay.expect("a replay target");
    let adapted = crate::completion::history::adapt(&history, replay);
    let results: Vec<&ToolResult> = adapted
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content.iter().collect::<Vec<_>>(),
            Message::System { .. } | Message::Assistant(_) => Vec::new(),
        })
        .filter_map(|part| match part {
            UserContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(
        results.len(),
        2,
        "one result per call, the orphan gone: {results:?}"
    );
    assert!(
        results.iter().any(|result| result.is_error
            && result.content.first().and_then(ToolResultContent::as_text)
                == Some(crate::completion::history::NO_RESULT_PROVIDED)),
        "the unanswered call gets an error result: {results:?}"
    );
    for mode in MODES {
        sent(fixture, fixture.model(), history[1..].to_vec(), mode)
            .unwrap_or_else(|error| panic!("{mode:?}: the paired history encodes: {error}"));
    }
}

/// H9. Content the model does not read is downgraded by the adapter, so
/// the encoder never refuses a history: images in every role, tool calls
/// and results, and blank system messages.
pub fn h09_capability_downgrades<F: HistoryFixture>(fixture: &F) {
    let image = Image {
        data: crate::message::DocumentSourceKind::base64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        ),
        media_type: Some(crate::message::ImageMediaType::PNG),
        ..Image::default()
    };
    let call = ToolCall::new(
        CallId::from_wire("call_shot"),
        ToolFunction::new(name("screenshot"), serde_json::json!({})),
    );
    let history = vec![
        Message::system(""),
        Message::User {
            content: vec![UserContent::text("look"), UserContent::Image(image.clone())],
        },
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(image.clone()),
                AssistantContent::Text(Text::new("taking a shot")),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("other.api", "other", "other-model")),
            stop: Some(StopReason::ToolUse),
            native: None,
        }),
        Message::User {
            content: vec![UserContent::ToolResult(call.result(vec![
                ToolResultContent::text("here"),
                ToolResultContent::Image(image),
            ]))],
        },
    ];
    let models = std::iter::once(fixture.model()).chain(fixture.text_only_model());
    for model in models {
        for mode in MODES {
            sent(fixture, model, history.clone(), mode).unwrap_or_else(|error| {
                panic!(
                    "{model} in {mode:?}: the adapter leaves nothing the encoder refuses: {error}"
                )
            });
        }
    }
}

/// H10. A turn stored and loaded again, by serde and by a store that sorts
/// object keys, replays to the same model exactly as before.
pub fn h10_persistence<F: HistoryFixture>(fixture: &F) {
    fn sorted(value: Value) -> Value {
        match value {
            Value::Object(fields) => {
                let mut fields: Vec<_> = fields.into_iter().collect();
                fields.sort_by(|left, right| left.0.cmp(&right.0));
                Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| (key, sorted(value)))
                        .collect(),
                )
            }
            Value::Array(values) => Value::Array(values.into_iter().map(sorted).collect()),
            value => value,
        }
    }
    for mode in MODES {
        let Some(turn) = turn_of(fixture, Shape::Rich, mode) else {
            continue;
        };
        let expected = fixture.replayed(&turn);
        let stored = serde_json::to_value(Message::Assistant(turn)).expect("a turn serializes");
        for loaded in [stored.clone(), sorted(stored)] {
            let loaded: Message = serde_json::from_value(loaded).expect("a stored turn loads");
            let Message::Assistant(turn) = &loaded else {
                panic!("an assistant turn loads as one");
            };
            let replayed = fixture.replayed(turn);
            assert_eq!(
                replayed.len(),
                expected.len(),
                "{mode:?}: a stored turn keeps every provider item current"
            );
            let body = sent(
                fixture,
                fixture.model(),
                vec![Message::user("q"), loaded],
                mode,
            )
            .unwrap_or_else(|error| panic!("{mode:?}: a stored turn encodes: {error}"));
            for item in replayed {
                assert!(
                    contains_subtree(&body, &item),
                    "{mode:?}: a stored turn replays {item}"
                );
            }
        }
    }
}

/// H11. `[reasoning, text, reasoning, call]` keeps its order in the turn
/// and in what the same model is sent.
pub fn h11_order<F: HistoryFixture>(fixture: &F) {
    for mode in MODES {
        let Some(turn) = turn_of(fixture, Shape::Interleaved, mode) else {
            eprintln!("skip: the wire has no interleaved shape in {mode:?}");
            continue;
        };
        let kinds: Vec<&str> = turn
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(_) => "text",
                AssistantContent::Reasoning(_) => "reasoning",
                AssistantContent::ToolCall(_) => "call",
                AssistantContent::Image(_) => "image",
                AssistantContent::Opaque(_) => "opaque",
            })
            .collect();
        assert_eq!(
            kinds,
            ["reasoning", "text", "reasoning", "call"],
            "{mode:?}: the blocks keep the provider's order"
        );
        let body = sent(
            fixture,
            fixture.model(),
            vec![Message::user("q"), Message::Assistant(turn.clone())],
            mode,
        )
        .unwrap_or_else(|error| panic!("{mode:?}: the history encodes: {error}"));
        let positions: Vec<usize> = fixture
            .replayed(&turn)
            .iter()
            .filter_map(|item| position_of(&body, item))
            .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{mode:?}: the same model gets the items in order: {positions:?} in {body}"
        );
    }
}

/// H12. Editing one block makes only that block canonical: its stale item
/// is not sent, and every sibling still sends its own.
pub fn h12_edits<F: HistoryFixture>(fixture: &F) {
    for mode in MODES {
        let Some(mut turn) = turn_of(fixture, Shape::Rich, mode) else {
            continue;
        };
        let Some(at) = turn.content.iter().position(
            |block| matches!(block, AssistantContent::Text(text) if !text.text.is_empty()),
        ) else {
            continue;
        };
        let stale = turn
            .content
            .get(at)
            .and_then(AssistantContent::native_item)
            .cloned();
        if let Some(AssistantContent::Text(text)) = turn.content.get_mut(at) {
            text.text.push_str(" (edited)");
        }
        let body = sent(
            fixture,
            fixture.model(),
            vec![Message::user("q"), Message::Assistant(turn.clone())],
            mode,
        )
        .unwrap_or_else(|error| panic!("{mode:?}: the edited history encodes: {error}"));
        assert!(
            body.to_string().contains(" (edited)"),
            "{mode:?}: the edit is sent: {body}"
        );
        if let Some(stale) = stale {
            assert!(
                !contains_subtree(&body, &stale),
                "{mode:?}: the edited block's stale item is not sent: {body}"
            );
        }
        for item in fixture.replayed(&turn) {
            assert!(
                contains_subtree(&body, &item),
                "{mode:?}: a sibling of the edit still sends {item}"
            );
        }
    }
}

/// H13. A turn a runtime rolls back (`CompletionResponse::continued`) keeps
/// its origin and the provider items of the blocks that closed, so it
/// replays to the same model.
pub fn h13_rollback<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    let Some(frames) = fixture.reply(Shape::Rich, Mode::Streaming) else {
        return;
    };
    let response = decode(
        &wire,
        &CompletionRequest::new("restate"),
        Mode::Streaming,
        frames,
    )
    .unwrap_or_else(|error| panic!("the rich reply decodes: {error}"));
    let turn = response.continued(response.choice.clone());
    let origin = turn
        .origin
        .as_ref()
        .expect("a rolled-back turn keeps its origin");
    assert!(!origin.model.is_empty(), "the origin names its model");
    let expected = fixture.replayed(&turn);
    let body = sent(
        fixture,
        fixture.model(),
        vec![Message::user("q"), Message::Assistant(turn)],
        Mode::Streaming,
    )
    .unwrap_or_else(|error| panic!("the rolled-back history encodes: {error}"));
    for item in expected {
        assert!(
            contains_subtree(&body, &item),
            "a rolled-back turn replays {item}"
        );
    }
}

/// H14. A request's model override reaches the encoder and the turn's
/// origin, and a turn's origin always names a model.
pub fn h14_model_identity<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    let other = fixture.other_model();
    for mode in MODES {
        let Some(frames) = fixture.reply(Shape::Rich, mode) else {
            continue;
        };
        let request = CompletionRequest::new("restate").model(other);
        let response = decode(&wire, &request, mode, frames)
            .unwrap_or_else(|error| panic!("{mode:?}: the reply decodes: {error}"));
        assert_eq!(
            response.origin.model, other,
            "{mode:?}: the origin names the override"
        );
        let prepared = Completion::prepare(request, &wire.describe())
            .unwrap_or_else(|error| panic!("{mode:?}: the request prepares: {error}"));
        assert_eq!(prepared.model.as_deref(), Some(other));
        let body = fixture
            .body(&wire, prepared, mode)
            .unwrap_or_else(|error| panic!("{mode:?}: the request encodes: {error}"));
        assert!(
            body.to_string().contains(other),
            "{mode:?}: the override reaches the request: {body}"
        );
    }
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).unwrap_or_else(|_| panic!("`{name}` is a tool name"))
}

/// Expand the history conformance suite for one wire: one test per row of
/// [`ROWS`], and the `HISTORY_WIRE` constant the registry links.
///
/// `wire` is the wire's name in [`HISTORY_WIRES`]; `fixture` is an
/// expression producing its [`HistoryFixture`].
#[macro_export]
macro_rules! history_conformance_suite {
    (wire: $wire:literal, fixture: $fixture:expr $(,)?) => {
        /// The wire this suite covers, for the workspace registry.
        pub const HISTORY_WIRE: &str = $wire;

        $crate::__history_conformance_rows! {
            $fixture;
            h01_same_model_verbatim,
            h02_cross_model_canonical,
            h03_stream_equals_whole,
            h04_unknown_kept,
            h05_field_ablation,
            h06_malformed_arguments,
            h07_failed_turns,
            h08_pairing,
            h09_capability_downgrades,
            h10_persistence,
            h11_order,
            h12_edits,
            h13_rollback,
            h14_model_identity,
        }

        #[test]
        fn suite_runs_every_row() {
            let emitted = [
                "h01_same_model_verbatim",
                "h02_cross_model_canonical",
                "h03_stream_equals_whole",
                "h04_unknown_kept",
                "h05_field_ablation",
                "h06_malformed_arguments",
                "h07_failed_turns",
                "h08_pairing",
                "h09_capability_downgrades",
                "h10_persistence",
                "h11_order",
                "h12_edits",
                "h13_rollback",
                "h14_model_identity",
            ];
            let rows: Vec<&str> = $crate::test_utils::history_conformance::ROWS
                .iter()
                .map(|(row, _)| *row)
                .collect();
            assert_eq!(emitted.as_slice(), rows.as_slice());
            assert!(
                $crate::test_utils::history_conformance::HISTORY_WIRES.contains(&$wire),
                "`{}` is in HISTORY_WIRES",
                $wire
            );
        }
    };
}

/// One test per row of [`history_conformance_suite!`](crate::history_conformance_suite).
#[doc(hidden)]
#[macro_export]
macro_rules! __history_conformance_rows {
    ($fixture:expr; $($row:ident),+ $(,)?) => {
        $(
            #[test]
            fn $row() {
                $crate::test_utils::history_conformance::$row(&$fixture);
            }
        )+
    };
}
