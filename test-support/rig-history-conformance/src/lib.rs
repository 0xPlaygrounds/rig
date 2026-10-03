//! The history conformance suite: one set of rows every completion wire
//! runs, so each replay invariant is checked on every wire instead of being
//! patched once where a bug surfaced. A wire supplies a [`HistoryFixture`]
//! (its target for a model, replies of each [`Shape`], and its request body
//! as JSON); [`history_conformance_suite!`]
//! expands one test per row, and the workspace registry fails when a wire in
//! [`HISTORY_WIRES`] has no suite.
//!
//! Each row's function documents its invariant. The audit findings each row
//! closes are listed in [`ROWS`], and those a named test closes in
//! [`TESTS`]; a finding is closed only by one of them.
//!
//! ```ignore
//! mod anthropic_history {
//!     rig_history_conformance::history_conformance_suite! {
//!         wire: "anthropic",
//!         fixture: super::AnthropicHistory,
//!     }
//! }
//! ```

// The rows are test assertions: a failed one panics with what it found.
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use serde_json::Value;

use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::error::EncodeError;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Origin, StopReason, Text, ToolCall,
    ToolFunction, ToolName, ToolResult, ToolResultContent, UserContent,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};

pub use rig_core::test_utils::history_conformance::{decode, partial};

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
    "azure",
    "hyperbolic",
    "mira",
    "together",
    "huggingface",
    "llamacpp",
    "venice",
    "doubleword",
    "zai",
    "minimax",
    "moonshot",
    "xiaomimimo",
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
         missing args, review: typed parts fail a whole reply (null and retyped fields)",
    ),
    (
        "h06_malformed_arguments",
        "#1085, #2359, #2447, #2554, core NEW null arguments",
    ),
    (
        "h07_failed_turns",
        "core #1559, anthropic NEW partial turns, gemini NEW non-success finishes, chatB NEW \
         unknown finish reasons, review: Chat finish `end`, Responses response.incomplete",
    ),
    ("h08_pairing", "#2560, chatA NEW-4, core NEW is_error"),
    (
        "h09_capability_downgrades",
        "#305, #2143, #2380, chatA NEW-1, NEW-2, NEW-6, chatB NEW placeholder (accepts_images \
         dead, with the Moonshot, Z.AI, MiniMax and MiMo text-only models), anthropic NEW image \
         capability (Moonshot's Messages wire), review: encoders refuse canonical content on \
         every wire",
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
        "core #1559, core NEW rollback origin, chatB NEW rollback, review: a cut Chat, Cohere \
         or Ollama reply loses its reasoning",
    ),
    (
        "h14_model_identity",
        "gemini NEW resume origin model, gemini NEW Vertex model override",
    ),
];

/// Each audit finding a named test closes rather than a row: the file that
/// holds it, the test function, and the findings. The workspace registry
/// fails when a named test no longer exists.
pub const TESTS: &[(&str, &str, &str)] = &[
    (
        "crates/rig-core/src/completion/request/tests.rs",
        "documents_join_the_first_user_message_so_roles_alternate",
        "#1179",
    ),
    (
        "crates/rig-agent/src/agent/runner/entry_tests.rs",
        "resume_appends_its_messages_without_loading",
        "#2244",
    ),
    (
        "crates/rig-bedrock/src/streaming/tests.rs",
        "a_streams_raw_is_bedrocks_json",
        "#2311",
    ),
    (
        "crates/rig-bedrock/src/types/assistant_content/tests.rs",
        "claude_behind_an_application_profile_keeps_its_signatures",
        "core NEW Bedrock application inference profile",
    ),
    (
        "crates/rig-bedrock/src/types/completion_request/tests.rs",
        "a_later_system_message_stays_in_place",
        "core NEW Bedrock system hoisting",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/history_tests.rs",
        "response_only_fields_never_go_back",
        "chatB NEW response-only fields replayed, #1835",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "a_call_added_but_never_done_fails_the_turn",
        "Responses NEW call added but never done, review: a never-done call loses its id",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "the_container_survives_a_dropped_block_and_an_edited_call",
        "anthropic NEW container, review: container lost when content changes",
    ),
    (
        "crates/rig-core/src/completion/message/native/tests.rs",
        "a_store_that_writes_whole_numbers_as_integers_keeps_the_item_current",
        "review: fingerprint number form",
    ),
    (
        "crates/rig-core/src/operation/completion/tests.rs",
        "a_reused_call_id_is_renamed_and_loses_its_item",
        "review: duplicate call id fails the reply",
    ),
    (
        "crates/rig-core/src/serve/handler/tests.rs",
        "a_written_reply_names_its_origin_before_its_first_item",
        "review: Reply::written sends no origin",
    ),
    (
        "crates/rig-core/src/completion/history/tests.rs",
        "results_come_before_the_text_of_the_message_they_merge_into",
        "review: text before tool_result after a merge",
    ),
    (
        "crates/rig-core/src/completion/history/tests.rs",
        "media_the_encoder_cannot_carry_becomes_a_placeholder_or_its_text",
        "review: encoders refuse canonical content",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "an_edited_call_keeps_its_caller",
        "review: an edited Anthropic call loses its caller",
    ),
    (
        "crates/rig-core/src/providers/anthropic/streaming/tests.rs",
        "an_empty_reply_folds_the_same_whole_or_streamed",
        "review: Anthropic empty replies differ whole and streamed",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "tool_results_lead_a_merged_user_message",
        "review: Anthropic text before tool_result",
    ),
    (
        "crates/rig-cassette/tests/providers/anthropic/cassette/malformed_tool_args_matrix.rs",
        "streaming_malformed_call_is_answered_with_an_error",
        "review: hand-derived malformed-arguments cassette",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "reasoning_done_without_ciphertext_keeps_its_item_in_a_stream_without_indices",
        "review: Responses reasoning loses its item without output_index",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "a_terminal_only_item_does_not_repeat_text_streamed_without_indices",
        "review: Responses terminal-only items duplicate text",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "a_done_item_s_text_replaces_the_text_its_deltas_streamed",
        "review: Responses block text and native disagree",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "an_item_without_a_type_never_replays",
        "review: Responses item without a type replays",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/streaming/tests.rs",
        "response_incomplete_is_incomplete_whatever_its_status_says",
        "review: response.incomplete without status is Stop",
    ),
    (
        "crates/rig-gemini-grpc/src/completion/tests.rs",
        "a_url_tool_result_image_follows_the_results_on_gemini_3",
        "review: gRPC URL tool-result image",
    ),
    (
        "crates/rig-gemini-grpc/src/streaming/tests.rs",
        "a_part_of_an_undeclared_kind_never_replays_without_data",
        "review: gRPC part with no data replays",
    ),
    (
        "crates/rig-bedrock/src/types/completion_request/tests.rs",
        "a_hosted_use_replays_only_with_its_result",
        "review: Bedrock server_tool_use replays without its result",
    ),
    (
        "crates/rig-bedrock/src/types/completion_request/tests.rs",
        "only_nova_and_claude_get_a_result_status",
        "review: Bedrock status sent to every family",
    ),
    (
        "crates/rig-bedrock/tests/history_conformance.rs",
        "a_reply_the_sdk_cannot_read_decodes_from_its_json",
        "review: Bedrock whole reply read strictly",
    ),
    (
        "crates/rig-candle/src/protocol/tests.rs",
        "every_renderer_takes_any_media_the_adapter_hands_over",
        "review: Candle refuses media",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/history_tests.rs",
        "review_text_only_chat_dialect_models_get_placeholders",
        "chatB NEW accepts_images dead (STILL_POSSIBLE)",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/tests.rs",
        "the_done_sentinel_without_a_finish_fails",
        "review: Chat [DONE] without finish_reason is Stop",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/history_tests.rs",
        "an_answers_audio_transcript_is_text",
        "review: an OpenAI audio transcript is never text",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/history_tests.rs",
        "a_mistyped_part_never_fails_a_reply",
        "review: Chat index null, numeric id, object content",
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

/// `history`, which ends in a user message, as the adapter shapes it for
/// `model` on the fixture's wire, read from the request the driver
/// prepares.
fn adapted<F: HistoryFixture>(fixture: &F, model: &str, history: &[Message]) -> Vec<Message> {
    let wire = fixture.wire(model);
    let mut request = CompletionRequest::new("next").model(model);
    request.chat_history = history.to_vec();
    Completion::prepare(request, &wire.describe())
        .unwrap_or_else(|error| panic!("the history prepares for {model}: {error}"))
        .chat_history
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
pub fn http_body(encoded: &rig_core::wire::Encoded) -> Result<Value, EncodeError> {
    let rig_core::wire::Body::Bytes(bytes) = encoded.request.body() else {
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
        let adapted = adapted(fixture, fixture.other_model(), &history);
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
        rig_core::test_utils::history::assert_restated_agrees(&wire, whole, streamed);
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

/// H5. Removing any one field no block or the finish is built from, making
/// it `null`, or giving it a value of another type still decodes the whole
/// reply.
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
        let Some(value) = ablation.document.pointer(&pointer) else {
            continue;
        };
        for (change, replacement) in [("null", Value::Null), ("retyped", retyped(value))] {
            let mut document = ablation.document.clone();
            if let Some(field) = document.pointer_mut(&pointer) {
                *field = replacement;
            }
            if let Err(error) = decode(&wire, &request, Mode::Unary, (ablation.frames)(document)) {
                panic!("the reply with `{pointer}` {change} still decodes: {error}");
            }
        }
    }
}

/// A value of another JSON type than `value`.
fn retyped(value: &Value) -> Value {
    match value {
        Value::String(_) => Value::from(7),
        Value::Number(_) => Value::from("7"),
        Value::Bool(_) => Value::from("true"),
        Value::Array(_) => serde_json::json!({}),
        Value::Object(_) => serde_json::json!([]),
        Value::Null => Value::from(false),
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
        let adapted = rig_core::completion::history::adapt(&history, replay);
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
    let adapted = rig_core::completion::history::adapt(&history, replay);
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
    if replay.accepts(fixture.model()).tools {
        assert_eq!(
            results.len(),
            2,
            "one result per call, the orphan gone: {results:?}"
        );
        assert!(
            results.iter().any(|result| result.is_error
                && result.content.first().and_then(ToolResultContent::as_text)
                    == Some(rig_core::completion::history::NO_RESULT_PROVIDED)),
            "the unanswered call gets an error result: {results:?}"
        );
    } else {
        // A model without tools reads calls and results as text, so nothing
        // is left to pair.
        assert!(
            results.is_empty(),
            "a model without tools gets no results: {results:?}"
        );
    }
    for mode in MODES {
        sent(fixture, fixture.model(), history[1..].to_vec(), mode)
            .unwrap_or_else(|error| panic!("{mode:?}: the paired history encodes: {error}"));
    }
}

/// H9. The adapter downgrades whatever the model does not read or the wire
/// cannot carry in its form, so the encoder never refuses a history and
/// never drops a part the adapter kept: images, audio, video and documents
/// in every source form and role, tool calls and results, and blank system
/// messages. A text-only model gets no image at all.
pub fn h09_capability_downgrades<F: HistoryFixture>(fixture: &F) {
    let history = media_history();
    let models = std::iter::once(fixture.model()).chain(fixture.text_only_model());
    for model in models {
        let wire = fixture.wire(model);
        let describe = wire.describe();
        let replay = describe.replay.expect("a replay target");
        let adapted = adapted(fixture, model, &history);
        if Some(model) == fixture.text_only_model() {
            assert!(
                !replay.accepts(model).user_images,
                "{model} is text-only, so its wire says it reads no images"
            );
            assert!(
                payloads(&adapted).iter().all(|(kind, _)| *kind != "image"),
                "{model} gets no image: {adapted:?}"
            );
        }
        for mode in MODES {
            let body = sent(fixture, model, history.clone(), mode).unwrap_or_else(|error| {
                panic!(
                    "{model} in {mode:?}: the adapter leaves nothing the encoder refuses: {error}"
                )
            });
            let text = body.to_string();
            for (kind, needles) in payloads(&adapted) {
                assert!(
                    needles.iter().any(|needle| {
                        let escaped = serde_json::to_string(needle).unwrap_or_default();
                        text.contains(needle.as_str()) || text.contains(escaped.trim_matches('"'))
                    }),
                    "{model} in {mode:?}: the {kind} the adapter kept is sent ({needles:?}): {body}"
                );
            }
        }
    }
}

/// A history holding every canonical media form in every role.
fn media_history() -> Vec<Message> {
    use rig_core::message::{
        Audio, AudioMediaType, Document, DocumentMediaType, DocumentSourceKind as Source,
        ImageMediaType, Video, VideoMediaType,
    };
    let image = |data: Source, media_type: Option<ImageMediaType>| Image {
        data,
        media_type,
        ..Image::default()
    };
    let document = |data: Source, media_type: Option<DocumentMediaType>| Document {
        data,
        media_type,
        additional_params: None,
    };
    let call = ToolCall::new(
        CallId::from_wire("call_shot"),
        ToolFunction::new(name("screenshot"), serde_json::json!({})),
    );
    vec![
        Message::system(""),
        Message::User {
            content: vec![
                UserContent::text("look"),
                UserContent::Image(image(Source::base64(MATRIX_PNG), Some(ImageMediaType::PNG))),
                UserContent::Image(image(Source::base64("R0lGODlhAQABAIAAAP///wAAACw="), None)),
                UserContent::Image(image(
                    Source::url("https://example.com/rig-matrix.png"),
                    None,
                )),
                UserContent::Image(image(Source::file_id("file-rig-matrix-image"), None)),
                UserContent::Image(image(
                    Source::Raw(vec![
                        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F',
                    ]),
                    None,
                )),
                UserContent::Image(image(Source::string("rig-matrix-string-image"), None)),
                UserContent::Image(image(Source::Unknown, Some(ImageMediaType::PNG))),
                UserContent::Audio(Audio {
                    data: Source::base64("SUQzBAAAAAAAI1RTU0UAAAAP"),
                    media_type: Some(AudioMediaType::MP3),
                }),
                UserContent::Audio(Audio {
                    data: Source::url("https://example.com/rig-matrix.mp3"),
                    media_type: Some(AudioMediaType::MP3),
                }),
                UserContent::Video(Video {
                    data: Source::url("https://example.com/rig-matrix.mp4"),
                    media_type: Some(VideoMediaType::MP4),
                    additional_params: None,
                }),
                UserContent::Video(Video {
                    data: Source::base64("AAAAIGZ0eXBpc29tAAACAA"),
                    media_type: Some(VideoMediaType::MP4),
                    additional_params: None,
                }),
                UserContent::Document(document(
                    Source::base64("JVBERi0xLjQKcmlnLW1hdHJpeA=="),
                    Some(DocumentMediaType::PDF),
                )),
                UserContent::Document(document(
                    Source::url("https://example.com/rig-matrix.pdf"),
                    Some(DocumentMediaType::PDF),
                )),
                UserContent::Document(document(
                    Source::string("rig matrix plain document"),
                    Some(DocumentMediaType::TXT),
                )),
                UserContent::Document(document(
                    Source::base64("cmlnLG1hdHJpeAoxLDIK"),
                    Some(DocumentMediaType::CSV),
                )),
                UserContent::Document(document(
                    Source::string("# rig matrix markdown"),
                    Some(DocumentMediaType::MARKDOWN),
                )),
                UserContent::Document(document(Source::file_id("file-rig-matrix-document"), None)),
                UserContent::Document(document(Source::base64("cmlnIG1hdHJpeCB1bnR5cGVk"), None)),
            ],
        },
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(image(
                    Source::url("https://example.com/rig-matrix-assistant.png"),
                    Some(ImageMediaType::PNG),
                )),
                AssistantContent::Text(Text::new("taking a shot")),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("other.api", "other", "other-model")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(call.result(vec![
                ToolResultContent::text("here"),
                ToolResultContent::Image(image(
                    Source::base64(MATRIX_WEBP),
                    Some(ImageMediaType::WEBP),
                )),
                ToolResultContent::Image(image(
                    Source::url("https://example.com/rig-matrix-tool.png"),
                    Some(ImageMediaType::PNG),
                )),
            ]))],
        },
    ]
}

/// A 1x1 PNG.
const MATRIX_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// A 1x1 WEBP.
const MATRIX_WEBP: &str = "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA=";

/// Each media part `history` still holds, by kind, with the strings one of
/// which its encoding contains: its URL, file id, string, or the head of
/// its data (and a text document's decoded text).
fn payloads(history: &[Message]) -> Vec<(&'static str, Vec<String>)> {
    use base64::Engine as _;
    use rig_core::message::DocumentSourceKind as Source;
    fn needles(source: &Source) -> Vec<String> {
        match source {
            Source::Url(text) | Source::FileId(text) | Source::String(text) => {
                vec![text.clone()]
            }
            Source::Base64(data) => {
                let mut needles = vec![data.chars().take(16).collect()];
                if let Some(text) = base64::prelude::BASE64_STANDARD
                    .decode(data.as_bytes())
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
                {
                    needles.push(text);
                }
                needles
            }
            Source::Raw(_) | Source::Unknown => vec![String::from("<no payload>")],
        }
    }
    let mut found = Vec::new();
    for message in history {
        match message {
            Message::User { content } => {
                for part in content {
                    match part {
                        UserContent::Image(image) => found.push(("image", needles(&image.data))),
                        UserContent::Audio(audio) => found.push(("audio", needles(&audio.data))),
                        UserContent::Video(video) => found.push(("video", needles(&video.data))),
                        UserContent::Document(document) => {
                            found.push(("document", needles(&document.data)));
                        }
                        UserContent::ToolResult(result) => {
                            for part in &result.content {
                                if let ToolResultContent::Image(image) = part {
                                    found.push(("image", needles(&image.data)));
                                }
                            }
                        }
                        UserContent::Text(_) => {}
                    }
                }
            }
            Message::Assistant(turn) => {
                for block in &turn.content {
                    if let AssistantContent::Image(image) = block {
                        found.push(("image", needles(&image.data)));
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    found
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

/// H13. A turn a runtime rolls back (`CompletionResponse::continued`), at
/// its end or cut off just before its finish, keeps its origin and the
/// provider items of the blocks that closed, so it replays to the same
/// model. A block closes once the next one starts, so a cut turn loses at
/// most its last block.
pub fn h13_rollback<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    let Some(frames) = fixture.reply(Shape::Rich, Mode::Streaming) else {
        return;
    };
    let count = frames.len();
    let whole = decode(
        &wire,
        &CompletionRequest::new("restate"),
        Mode::Streaming,
        frames,
    )
    .unwrap_or_else(|error| panic!("the rich reply decodes: {error}"));
    let blocks = whole.choice.len();
    rolled_back(fixture, &whole, "at its end");
    let cut_at = |cut: usize| {
        let frames = fixture
            .reply(Shape::Rich, Mode::Streaming)
            .unwrap_or_default();
        partial(&wire, Mode::Streaming, frames.into_iter().take(cut))
    };
    let Some((cut, cut_off)) = (1..count)
        .rev()
        .map(|cut| (cut, cut_at(cut)))
        .find_map(|(cut, (response, ended))| (!ended).then_some((cut, response)))
    else {
        return;
    };
    let closed = blocks.saturating_sub(1);
    let kept: Vec<AssistantContent> = cut_off
        .choice
        .iter()
        .take(closed)
        .map(AssistantContent::canonical)
        .collect();
    let expected: Vec<AssistantContent> = whole
        .choice
        .iter()
        .take(closed)
        .map(AssistantContent::canonical)
        .collect();
    assert_eq!(
        kept, expected,
        "a reply cut after {cut} frames keeps every block but the last"
    );
    rolled_back(fixture, &cut_off, "cut before its finish");
}

/// Roll `response` back as a runtime does and check it replays.
fn rolled_back<F: HistoryFixture>(fixture: &F, response: &CompletionResponse, when: &str) {
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
    .unwrap_or_else(|error| panic!("the turn rolled back {when} encodes: {error}"));
    for item in expected {
        assert!(
            contains_subtree(&body, &item),
            "a turn rolled back {when} replays {item}"
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
            let rows: Vec<&str> = $crate::ROWS.iter().map(|(row, _)| *row).collect();
            assert_eq!(emitted.as_slice(), rows.as_slice());
            assert!(
                $crate::HISTORY_WIRES.contains(&$wire),
                "`{}` is in HISTORY_WIRES",
                $wire
            );
        }
    };
}

/// One test per row of [`history_conformance_suite!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __history_conformance_rows {
    ($fixture:expr; $($row:ident),+ $(,)?) => {
        $(
            #[test]
            fn $row() {
                $crate::$row(&$fixture);
            }
        )+
    };
}
