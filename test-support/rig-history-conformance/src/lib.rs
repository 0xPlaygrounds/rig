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

mod generated;
pub mod replies;

pub use generated::Rng;

/// Every completion wire and dialect that must run the suite, by the name
/// its `history_conformance_suite!` invocation gives.
pub const HISTORY_WIRES: &[&str] = &[
    "mock",
    "anthropic",
    "anthropic_moonshot",
    "anthropic_zai",
    "anthropic_minimax",
    "anthropic_xiaomimimo",
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
    "cohere_native",
    "ollama",
    "ollama_native",
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
    (
        "h10_persistence",
        "core NEW fingerprint key order, round 5: generated finding 7 (integers beyond 2^53)",
    ),
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
         or Ollama reply loses its reasoning, round 5: a cut turn keeps no provider item",
    ),
    (
        "h14_model_identity",
        "gemini NEW resume origin model, gemini NEW Vertex model override",
    ),
    (
        "h15_item_round_trip",
        "round 4: Anthropic A4 and A5, Responses NEW-C, Gemini NEW-3 (stored items the wire \
         would refuse)",
    ),
    (
        "h16_call_identity",
        "#2655, round 4: Chat NEW-1 and NEW-2 (merged parallel calls), Mistral tool-0, Gemini \
         3 id-less call, Responses NEW-B and NEW-E, Anthropic A1, stored duplicate ids, rename \
         scope",
    ),
    (
        "h17_generated_histories",
        "round 4: fuzz F1 (duplicate ids), F2 (fingerprint number form), role alternation, \
         round 5: generated findings 1 (windows), 5 (store bodies, explicit skips, reached \
         checks) and 6 (hosted pairs, late and split results, own streamed and generated \
         turns, adjacency)",
    ),
    (
        "h18_generated_stream_equals_whole",
        "round 6: fuzz R6-A (terminal-only Responses items reordered), R6-B (index-less \
         id-less Chat calls merged), round 5: generated finding 2 (OpenRouter details and \
         Mistral parts in the Chat generator)",
    ),
    (
        "h19_generated_cuts",
        "round 6: fuzz (e) (a cut stream never replays as a success), #2647 cut paths, \
         round 5: generated findings 3 and 4 (continued and delivered turns keep no item)",
    ),
];

/// Each audit finding a named test closes rather than a row: the file that
/// holds it, the test function, and the findings. The workspace registry
/// fails when a named test no longer exists.
pub const TESTS: &[(&str, &str, &str)] = &[
    (
        "crates/rig-bedrock/tests/history_conformance.rs",
        "an_orphan_first_result_still_leaves_a_user_message_first",
        "round 5: generated finding 1 (a Bedrock request opens with the assistant)",
    ),
    (
        "crates/rig-core/src/providers/openai/wire/chat/history_tests.rs",
        "reasoning_details_fold_alike_whole_and_streamed_wherever_they_arrive",
        "round 5: generated finding 2 (OpenRouter details after the content)",
    ),
    (
        "crates/rig-core/src/providers/openai/responses_api/history_tests.rs",
        "a_call_cut_off_after_a_missing_output_index_goes_back_without_its_item",
        "round 5: generated finding 3 (a Responses cut at a missing output_index)",
    ),
    (
        "tests/core/history_conformance/deepseek.rs",
        "generated_reply_checks_catch_what_they_guard",
        "round 5: generated finding 4 (H19's checks can fail)",
    ),
    (
        "crates/rig-gemini-grpc/src/completion/tests.rs",
        "a_user_video_keeps_its_video_metadata",
        "#2658, round 4: gemini NEW-1 (gRPC refuses videoMetadata)",
    ),
    (
        "crates/rig-gemini-grpc/src/completion/tests.rs",
        "the_rest_request_transcodes_in_full",
        "#2658 (gRPC drops the tool choice, generation config and hosted tools)",
    ),
    (
        "crates/rig-gemini-grpc/src/completion/tests.rs",
        "a_signature_only_part_replays_its_signature",
        "round 4: gemini NEW-3 on gRPC",
    ),
    (
        "crates/rig-core/src/providers/gemini/streaming/tests.rs",
        "a_signature_only_part_joins_the_text_before_it",
        "round 4: gemini NEW-3 (a signature-only part never replays)",
    ),
    (
        "crates/rig-core/src/providers/gemini/interactions_api/history_tests.rs",
        "a_multi_part_result_reaches_gemini_2_as_one_string",
        "#2143, round 4: gemini NEW-2 (Interactions multi-part result on Gemini 2)",
    ),
    (
        "crates/rig-core/src/providers/gemini/interactions_api/history_tests.rs",
        "a_stored_continuation_keeps_its_results_without_declared_tools",
        "round 4 core: tools withdrawn turned a stored interaction's results into text",
    ),
    (
        "crates/rig-core/src/providers/gemini/interactions_api/history_tests.rs",
        "every_gemini_wire_classifies_a_model_alike",
        "round 4 architecture: REST and Interactions classify aliases apart",
    ),
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
        "crates/rig-bedrock/src/request/tests.rs",
        "an_edited_block_is_rebuilt_for_the_family",
        "core NEW Bedrock application inference profile",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
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
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "an_idless_call_replays_under_the_id_its_result_gets",
        "#2655, Responses NEW-B neighbour (an Anthropic tool_use without an id)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "a_binding_model_replays_its_thinking_and_asks_for_drop_block",
        "#2703 (Opus 5.5 thinking bound to its tools)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "unsigned_thinking_keeps_its_item_only_where_the_dialect_takes_it",
        "round 4: Anthropic A4 (unsigned thinking replayed as thinking)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "a_kept_tool_use_item_always_states_an_object_input",
        "round 4: Anthropic A5 (a tool_use item without input)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "a_server_tool_result_never_replays_without_its_use",
        "round 4: Anthropic A3 (an orphaned server-tool result)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/completion/tests.rs",
        "a_blank_text_block_keeps_no_item_and_is_never_sent",
        "round 4: Anthropic blank text with a current item (encoder drop deleted)",
    ),
    (
        "crates/rig-core/src/providers/anthropic/streaming/tests.rs",
        "a_stop_reason_states_every_open_block_complete",
        "round 4: Anthropic A6 (a signature lost without content_block_stop)",
    ),
    (
        "tests/core/history_conformance_registry.rs",
        "every_messages_dialect_has_a_history_suite",
        "round 4: Anthropic A8 (no Messages suite for Z.AI, MiniMax and MiMo)",
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
        "a_done_item_that_contradicts_its_deltas_states_the_block",
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
        "crates/rig-bedrock/src/request/tests.rs",
        "a_hosted_use_replays_only_with_its_result",
        "review: Bedrock server_tool_use replays without its result",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "only_nova_and_claude_get_a_result_status",
        "review: Bedrock status sent to every family",
    ),
    (
        "crates/rig-bedrock/tests/history_conformance.rs",
        "a_reply_the_sdk_cannot_read_decodes_from_its_json",
        "review: Bedrock whole reply read strictly",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "a_tool_history_without_tools_is_text",
        "round 4: Bedrock NEW-R4-toolconfig",
    ),
    (
        "crates/rig-bedrock/src/streaming/tests.rs",
        "a_stream_without_a_stop_reason_fails",
        "round 4: core NEW-R4-finish (Bedrock metadata without messageStop)",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "a_same_model_assistant_image_is_not_sent",
        "round 4: NEW-bedrock-assistant-image",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "duplicate_stored_ids_reach_converse_distinct",
        "round 4: NEW-dup-stored-ids on Converse",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "documents_are_named_by_content_and_land_in_the_first_user_message",
        "round 4: #43, #652, #404, #405 (Bedrock document placement and names)",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "cache_points_follow_what_the_request_sends",
        "round 4: #1673 (cache point after a reasoning turn)",
    ),
    (
        "crates/rig-bedrock/src/request/tests.rs",
        "a_stored_item_goes_back_with_whole_numbers",
        "round 4: Bedrock H10 (a store writes whole numbers as floats)",
    ),
    (
        "crates/rig-bedrock/tests/history_conformance.rs",
        "the_transport_sends_the_encoded_body",
        "round 4: Bedrock request built as JSON, sent as encoded",
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

/// How two streamed calls are told apart ([`HistoryFixture::calls_reply`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallShape {
    /// Fragments carry no index.
    Indexless,
    /// Fragments carry `index: null`.
    NullIndex,
    /// Both calls use index 0.
    ReusedIndex,
    /// A whole reply whose calls all state one index.
    WholeList,
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

    /// `block`'s provider item decoded alone, as the wire's whole reply of
    /// one item, for row H15; `None` when the wire has no such decode.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let _ = block;
        None
    }

    /// Two streamed calls to `weather`, ids `a1` and `b2`, arguments
    /// `{"city":"Paris"}` and `{"city":"Rome"}`, told apart as `shape`
    /// says (row H16); `None` when the wire cannot express the shape.
    fn calls_reply(
        &self,
        shape: CallShape,
        mode: Mode,
    ) -> Option<Vec<<Self::Wire as Wire>::Frame>> {
        let _ = (shape, mode);
        None
    }

    /// A reply that ends cleanly with no content (row H3).
    fn empty_reply(&self, mode: Mode) -> Option<Vec<<Self::Wire as Wire>::Frame>> {
        let _ = mode;
        None
    }

    /// The JSON pointer of the finish reason in the [`Self::ablation`]
    /// document (row H5): without it the reply must fail.
    fn finish_reason_pointer(&self) -> Option<&'static str> {
        None
    }

    /// A frame carrying an in-band error, which fails a stream (row H7).
    fn error_frame(&self) -> Option<<Self::Wire as Wire>::Frame> {
        None
    }

    /// Whether the wire requires user and assistant messages to alternate
    /// (Bedrock), checked on generated histories (H17).
    fn strict_roles(&self) -> bool {
        false
    }

    /// `tool` as the provider's own JSON, for `additional_params.tools`, or
    /// `None` when the wire takes no raw tools there; H17 then declares
    /// them as usual.
    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        let _ = tool;
        None
    }

    /// A generated reply for rows H18 and H19, drawn from `rng`, or `None`
    /// when the wire has no reply generator (the mock).
    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        let _ = rng;
        None
    }

    /// `spec` as this wire's frames, whole and streamed.
    fn reply_frames(
        &self,
        spec: &replies::Spec,
    ) -> Option<replies::Frames<<Self::Wire as Wire>::Frame>> {
        let _ = spec;
        None
    }

    /// Whether the provider combines consecutive messages of one role
    /// itself (Anthropic documents it), so H17 checks neither alternation
    /// nor adjacent user messages.
    fn combines_same_role(&self) -> bool {
        false
    }

    /// Whether the wire sends a `none` tool choice beside the request's
    /// tools, so a `ToolChoice::None` request keeps its tool history (row
    /// H8). Converse has no such choice, so Bedrock gets the history as
    /// text.
    fn sends_tool_choice_none(&self) -> bool {
        true
    }

    /// Whether the wire keeps provider items. A local runtime keeps none.
    fn keeps_natives(&self) -> bool {
        true
    }

    /// Why the wire's bodies carry no tool call or result the walker can
    /// find, so rows skip their pairing and adjacency checks and say so;
    /// `None` when they carry them.
    fn no_tool_calls(&self) -> Option<&'static str> {
        None
    }

    /// Why a stream on the wire has no point to cut before its end, so H19
    /// skips and says so; `None` when it has.
    fn no_cuts(&self) -> Option<&'static str> {
        None
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
    sent_as(fixture, model, history, mode, generated::Tools::Declared)
}

/// [`sent`], with the request's tools declared as `tools` says.
fn sent_as<F: HistoryFixture>(
    fixture: &F,
    model: &str,
    history: Vec<Message>,
    mode: Mode,
    tools: generated::Tools,
) -> Result<Value, String> {
    let wire = fixture.wire(model);
    let mut request = CompletionRequest::new("next");
    match tools {
        generated::Tools::Declared => request.tools = tools_of(&history),
        generated::Tools::ChoiceNone => {
            request.tools = tools_of(&history);
            request.tool_choice = Some(rig_core::message::ToolChoice::None);
        }
        generated::Tools::Raw => {
            let declared = tools_of(&history);
            let raw: Option<Vec<Value>> =
                declared.iter().map(|tool| fixture.raw_tool(tool)).collect();
            match raw {
                Some(raw) if !raw.is_empty() => {
                    request.additional_params = Some(serde_json::json!({ "tools": raw }));
                }
                _ => request.tools = declared,
            }
        }
    }
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let prepared = Completion::prepare(request, &wire.describe()).map_err(|e| e.to_string())?;
    fixture
        .body(&wire, prepared, mode)
        .map_err(|error| error.to_string())
}

/// A definition for every tool `history` calls or answers, as a request
/// that continues a tool loop declares them.
pub fn tools_of(history: &[Message]) -> Vec<rig_core::completion::ToolDefinition> {
    let mut names: Vec<ToolName> = Vec::new();
    for message in history {
        let found: Vec<ToolName> = match message {
            Message::Assistant(turn) => turn
                .tool_calls()
                .map(|call| call.function.name.clone())
                .collect(),
            Message::User { content } => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::ToolResult(result) => Some(result.name.clone()),
                    _ => None,
                })
                .collect(),
            Message::System { .. } => Vec::new(),
        };
        for name in found {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
        .into_iter()
        .map(|name| rig_core::completion::ToolDefinition {
            name,
            description: "a tool the history calls".to_owned(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        })
        .collect()
}

/// `history`, which ends in a user message, as the adapter shapes it for
/// `model` on the fixture's wire, read from the request the driver
/// prepares.
fn adapted<F: HistoryFixture>(fixture: &F, model: &str, history: &[Message]) -> Vec<Message> {
    let wire = fixture.wire(model);
    let mut request = CompletionRequest::new("next").model(model);
    request.tools = tools_of(history);
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
        if same(value, item) {
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

/// JSON equality with numbers compared by value, so `0` and `0.0` match:
/// a store or an SDK may write either.
fn same(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left.as_f64() == right.as_f64(),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| same(value, other)))
        }
        (left, right) => left == right,
    }
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
    // An empty reply is a success in both modes, decided once by the fold.
    if let (Some(whole), Some(streamed)) = (
        fixture.empty_reply(Mode::Unary),
        fixture.empty_reply(Mode::Streaming),
    ) {
        for (mode, frames) in [(Mode::Unary, whole), (Mode::Streaming, streamed)] {
            let response = decode(&wire, &CompletionRequest::new("restate"), mode, frames)
                .unwrap_or_else(|error| panic!("{mode:?}: an empty reply decodes: {error}"));
            assert!(
                !response.stop().is_failure(),
                "{mode:?}: an empty reply that ended cleanly is a success: {:?}",
                response.stop()
            );
        }
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
    if let Some(pointer) = fixture.finish_reason_pointer() {
        let mut document = ablation.document.clone();
        remove_pointer(&mut document, pointer);
        let failed = decode(&wire, &request, Mode::Unary, (ablation.frames)(document))
            .map_or(true, |response| response.stop().is_failure());
        assert!(
            failed,
            "a reply without its finish reason `{pointer}` failed"
        );
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
    if let (Some(error), Some(mut frames)) = (
        fixture.error_frame(),
        fixture.reply(Shape::Rich, Mode::Streaming),
    ) {
        frames.truncate(frames.len().saturating_sub(1).max(1));
        frames.push(error);
        let (response, _) = partial(&wire, Mode::Streaming, frames);
        assert!(
            response.stop().is_failure(),
            "an in-band error fails the turn: {:?}",
            response.stop()
        );
    }
}

/// H8. Every call is answered once: an unanswered call gets an error
/// result, a result no call asked for is dropped, and the history encodes.
pub fn h08_pairing<F: HistoryFixture>(fixture: &F) {
    let other = Origin::new("other.api", "other", "other-model");
    let turn = AssistantMessage::new(vec![
        AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_answered"),
            ToolFunction::new(name("lookup"), serde_json::json!({})),
        )),
        AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_unanswered"),
            ToolFunction::new(name("lookup"), serde_json::json!({})),
        )),
    ])
    .with_origin(other)
    .with_stop(StopReason::ToolUse);
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
    // A request that lets the model call no tools gets calls and results as
    // text, so a wire that needs tool definitions for them never sees one.
    for mode in MODES {
        let wire = fixture.wire(fixture.model());
        let mut request = CompletionRequest::new("next");
        request.chat_history = history[1..].to_vec();
        request.chat_history.push(Message::user("next"));
        let prepared = Completion::prepare(request, &wire.describe())
            .unwrap_or_else(|error| panic!("{mode:?}: the history prepares: {error}"));
        assert!(
            prepared.chat_history.iter().all(|message| match message {
                Message::Assistant(turn) => turn.tool_calls().next().is_none(),
                Message::User { content } => content
                    .iter()
                    .all(|part| !matches!(part, UserContent::ToolResult(_))),
                Message::System { .. } => true,
            }),
            "{mode:?}: no tools, so no calls: {:?}",
            prepared.chat_history
        );
        fixture
            .body(&wire, prepared, mode)
            .unwrap_or_else(|error| panic!("{mode:?}: the history without tools encodes: {error}"));
    }
    // `ToolChoice::None` with the tools declared keeps calls and results
    // where the wire sends its own `none`, so switching the choice never
    // rewrites the prompt prefix; elsewhere they go as text.
    let keeps = replay.accepts(fixture.model()).tools && fixture.sends_tool_choice_none();
    for mode in MODES {
        let wire = fixture.wire(fixture.model());
        let mut request = CompletionRequest::new("next");
        request.chat_history = history[1..].to_vec();
        request.tools = tools_of(&request.chat_history);
        request.tool_choice = Some(rig_core::message::ToolChoice::None);
        let prepared = Completion::prepare(request, &wire.describe())
            .unwrap_or_else(|error| panic!("{mode:?}: the history prepares: {error}"));
        let calls = prepared
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::Assistant(turn) => Some(turn.tool_calls().count()),
                _ => None,
            })
            .sum::<usize>();
        let results = prepared
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(
                    content
                        .iter()
                        .filter(|part| matches!(part, UserContent::ToolResult(_)))
                        .count(),
                ),
                _ => None,
            })
            .sum::<usize>();
        let expected = if keeps { (2, 2) } else { (0, 0) };
        assert_eq!(
            (calls, results),
            expected,
            "{mode:?}: ToolChoice::None keeps tool history: {keeps}: {:?}",
            prepared.chat_history
        );
        fixture.body(&wire, prepared, mode).unwrap_or_else(|error| {
            panic!("{mode:?}: the history with ToolChoice::None encodes: {error}")
        });
    }
    // A system message that arrives while a call waits is held until it is
    // answered, then sent before the user's own text.
    let held = vec![
        Message::user("q"),
        history[1].clone(),
        Message::system("steer"),
        history[2].clone(),
    ];
    for mode in MODES {
        sent(fixture, fixture.model(), held.clone(), mode)
            .unwrap_or_else(|error| panic!("{mode:?}: a held system message encodes: {error}"));
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
    let listing = ToolCall::new(
        CallId::from_wire("call_list"),
        ToolFunction::new(name("list"), serde_json::json!({})),
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
        Message::Assistant(
            AssistantMessage::new(vec![
                AssistantContent::Image(image(
                    Source::url("https://example.com/rig-matrix-assistant.png"),
                    Some(ImageMediaType::PNG),
                )),
                AssistantContent::Text(Text::new("taking a shot")),
                AssistantContent::ToolCall(call.clone()),
            ])
            .with_origin(Origin::new("other.api", "other", "other-model"))
            .with_stop(StopReason::ToolUse),
        ),
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
        Message::Assistant(
            AssistantMessage::new(vec![AssistantContent::ToolCall(listing.clone())])
                .with_origin(Origin::new("other.api", "other", "other-model"))
                .with_stop(StopReason::ToolUse),
        ),
        Message::User {
            content: vec![UserContent::ToolResult(listing.result(vec![
                ToolResultContent::text("first part"),
                ToolResultContent::text("second part"),
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
/// object keys, replays to the same model exactly as before. A store that
/// reads numbers as JavaScript does cannot hold an integer beyond 2^53, so
/// it changes a call's arguments: the request then carries the value the
/// store holds, never the stale item.
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
        for loaded in [
            stored.clone(),
            sorted(stored.clone()),
            numbers(stored.clone(), Numbers::ThroughF64),
            numbers(stored, Numbers::WholeAsFloat),
        ] {
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
    let wire = fixture.wire(fixture.model());
    for mode in MODES {
        let Some(frames) = fixture.call_reply(r#"{"big":12345678901234567}"#, mode) else {
            continue;
        };
        let response = decode(&wire, &CompletionRequest::new("restate"), mode, frames)
            .unwrap_or_else(|error| panic!("{mode:?}: a call with a big integer decodes: {error}"));
        let Some(turn) = response.message() else {
            panic!("{mode:?}: the call is a turn");
        };
        let stored = serde_json::to_value(turn).expect("a turn serializes");
        let loaded: Message = serde_json::from_value(numbers(stored, Numbers::ThroughF64))
            .expect("a rewritten turn loads");
        let Message::Assistant(loaded) = loaded else {
            panic!("an assistant turn loads as one");
        };
        let results = loaded
            .tool_calls()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")])))
            .collect();
        let history = vec![
            Message::user("q"),
            Message::Assistant(loaded),
            Message::User { content: results },
        ];
        let body = sent(fixture, fixture.model(), history, mode)
            .unwrap_or_else(|error| panic!("{mode:?}: a call a store changed encodes: {error}"));
        let text = body.to_string();
        assert!(
            !text.contains("2345678901234567") && text.contains("2345678901234568"),
            "{mode:?}: a call a store changed goes out with the value the store holds: {body}"
        );
    }
}

/// How a store rewrites numbers ([`h10_persistence`]).
#[derive(Clone, Copy, Debug)]
enum Numbers {
    /// Every number passes through an f64, as JavaScript and Mongo read it.
    ThroughF64,
    /// Every integer is written as a float.
    WholeAsFloat,
}

/// `value` with every number rewritten as `how` says.
fn numbers(value: Value, how: Numbers) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, numbers(value, how)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| numbers(value, how))
                .collect(),
        ),
        Value::Number(number) => match how {
            Numbers::ThroughF64 => number
                .as_f64()
                .and_then(serde_json::Number::from_f64)
                .map_or(Value::Number(number), Value::Number),
            Numbers::WholeAsFloat if number.is_f64() => Value::Number(number),
            Numbers::WholeAsFloat => number
                .as_f64()
                .and_then(serde_json::Number::from_f64)
                .map_or(Value::Number(number), Value::Number),
        },
        value => value,
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
                _ => "other",
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
    // Every other kind the wire has: an edited reasoning block or call is
    // rebuilt, its stale item never sent.
    for mode in MODES {
        let Some(turn) = turn_of(fixture, Shape::Rich, mode) else {
            continue;
        };
        for at in 0..turn.content.len() {
            let mut edited = turn.clone();
            let stale = edited.content[at].native_item().cloned();
            let changed = match &mut edited.content[at] {
                AssistantContent::Reasoning(reasoning) if !reasoning.redacted => {
                    reasoning.text.push_str(" (edited)");
                    true
                }
                AssistantContent::ToolCall(call) => {
                    call.function
                        .arguments
                        .insert("x_rig_edit".to_owned(), Value::from(true));
                    true
                }
                _ => false,
            };
            let Some(stale) = stale.filter(|_| changed) else {
                continue;
            };
            let body = sent(
                fixture,
                fixture.model(),
                vec![Message::user("q"), Message::Assistant(edited)],
                mode,
            )
            .unwrap_or_else(|error| panic!("{mode:?}: an edit at {at} encodes: {error}"));
            assert!(
                !contains_subtree(&body, &stale),
                "{mode:?}: block {at}'s stale item is not sent: {body}"
            );
        }
    }
}

/// H13. A turn a runtime rolls back (`CompletionResponse::continued`) keeps
/// its origin. At its end it keeps the provider items of its blocks, so it
/// replays to the same model. Cut off before its finish it keeps every
/// block but the last, and no provider item at all, so it replays
/// canonically; a block closes once the next one starts. A turn the
/// provider ended keeps an item that needs the one after it only with it.
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
    let kept: Vec<AssistantContent> = cut_off.choice.iter().take(closed).map(canonical).collect();
    let expected: Vec<AssistantContent> = whole.choice.iter().take(closed).map(canonical).collect();
    assert_eq!(
        kept, expected,
        "a reply cut after {cut} frames keeps every block but the last"
    );
    rolled_back(fixture, &cut_off, "cut before its finish");
    let describe = wire.describe();
    let target = describe.replay.expect("a replay target");
    for cut in 1..count {
        let (response, ended) = cut_at(cut);
        let turn = response.continued(response.choice.clone());
        if !ended {
            assert!(
                turn.content.iter().all(|block| !holds_item(block)),
                "a reply cut after {cut} frames keeps a provider item: {:?}",
                turn.content
            );
            continue;
        }
        let items: Vec<Option<Value>> = turn
            .content
            .iter()
            .map(|block| block.native_item().cloned())
            .collect();
        for (at, item) in items.iter().enumerate() {
            if let Some(item) = item
                && target.needs_next(item)
            {
                assert!(
                    items.get(at + 1).is_some_and(Option::is_some),
                    "a reply ended after {cut} frames keeps {item} without the item it needs"
                );
            }
        }
    }
}

/// `block` with no provider item, an opaque item no longer replaying.
fn canonical(block: &AssistantContent) -> AssistantContent {
    match block {
        AssistantContent::Opaque(opaque) => AssistantContent::Opaque(rig_core::message::Opaque {
            replay: false,
            ..opaque.clone()
        }),
        block => block.canonical(),
    }
}

/// Whether `block` holds a provider item that replays: a native, or an
/// opaque item that replays.
fn holds_item(block: &AssistantContent) -> bool {
    match block {
        AssistantContent::Opaque(opaque) => opaque.replay,
        block => block.native_item().is_some(),
    }
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
    // On a wire that binds items to the request's tools, a turn made under
    // other tools replays as if from another model.
    let describe = wire.describe();
    let target = describe.replay.expect("a replay target");
    if target.binds_context(fixture.model()) {
        let Some(frames) = fixture.reply(Shape::Rich, Mode::Unary) else {
            return;
        };
        let mut made = CompletionRequest::new("restate");
        made.tools = tools_of(&[]);
        let turn = match decode(&wire, &made, Mode::Unary, frames)
            .unwrap_or_else(|error| panic!("the reply decodes: {error}"))
            .message()
        {
            Some(Message::Assistant(turn)) => turn,
            other => panic!("an assistant turn: {other:?}"),
        };
        let mut request = CompletionRequest::new("next");
        request.tools = vec![rig_core::completion::ToolDefinition {
            name: name("x_rig_new_tool"),
            description: "a tool added since".to_owned(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        request.chat_history = vec![
            Message::user("q"),
            Message::Assistant(turn.clone()),
            Message::user("next"),
        ];
        let prepared = Completion::prepare(request, &wire.describe())
            .unwrap_or_else(|error| panic!("the request prepares: {error}"));
        let body = fixture
            .body(&wire, prepared, Mode::Unary)
            .unwrap_or_else(|error| panic!("the request encodes: {error}"));
        for item in fixture.replayed(&turn) {
            assert!(
                !contains_subtree(&body, &item),
                "a turn made under other tools does not replay {item}: {body}"
            );
        }
    }
}

/// H15. Every provider item a block keeps is one the wire takes back: the
/// item decoded alone, as the wire's whole reply, gives back the block.
pub fn h15_item_round_trip<F: HistoryFixture>(fixture: &F) {
    let mut checked = 0;
    for mode in MODES {
        for shape in [Shape::Rich, Shape::Interleaved] {
            let Some(turn) = turn_of(fixture, shape, mode) else {
                continue;
            };
            for block in turn
                .content
                .iter()
                .filter(|block| block.native_item().is_some())
            {
                let Some(decoded) = fixture.decode_item(block) else {
                    continue;
                };
                checked += 1;
                assert_eq!(
                    decoded.canonical(),
                    block.canonical(),
                    "{mode:?}: the stored item {:?} decodes back to its block",
                    block.native_item()
                );
            }
        }
    }
    if checked == 0 {
        eprintln!("skip: the wire decodes no single item");
    }
}

/// H16. Calls are told apart and keep their identity: streamed calls with
/// no index, a `null` one or a reused one, and a whole list whose calls
/// state one index, are two calls; and a history whose calls share ids, in
/// one turn and across turns, sends each call once with one result under
/// the same wire id.
pub fn h16_call_identity<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    for shape in [
        CallShape::Indexless,
        CallShape::NullIndex,
        CallShape::ReusedIndex,
        CallShape::WholeList,
    ] {
        for mode in MODES {
            let Some(frames) = fixture.calls_reply(shape, mode) else {
                continue;
            };
            let response = decode(&wire, &CompletionRequest::new("restate"), mode, frames)
                .unwrap_or_else(|error| panic!("{shape:?} in {mode:?} decodes: {error}"));
            let calls: Vec<(String, Value)> = response
                .tool_calls()
                .map(|call| (call.id.wire().into_owned(), call.function.arguments_value()))
                .collect();
            assert_eq!(
                calls,
                [
                    ("a1".to_owned(), serde_json::json!({"city": "Paris"})),
                    ("b2".to_owned(), serde_json::json!({"city": "Rome"})),
                ],
                "{shape:?} in {mode:?}: two calls"
            );
        }
    }
    let call = |id: &str| {
        AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(name("lookup"), serde_json::json!({"q": id})),
        ))
    };
    let foreign = |content| {
        Message::Assistant(
            AssistantMessage::new(content)
                .with_origin(Origin::new("other.api", "other", "other-model"))
                .with_stop(StopReason::ToolUse),
        )
    };
    let answer = |id: &str, text: &str| Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(id),
            name: name("lookup"),
            content: vec![ToolResultContent::text(text)],
            is_error: false,
        })],
    };
    let local = ToolCall::new(
        CallId::Local(rig_core::message::LocalCallId::new()),
        ToolFunction::new(name("lookup"), serde_json::json!({})),
    );
    let history = vec![
        Message::user("q"),
        foreign(vec![call("dup"), call("dup")]),
        Message::User {
            content: vec![answer("dup", "one"), answer("dup", "two")]
                .into_iter()
                .flat_map(|message| match message {
                    Message::User { content } => content,
                    _ => Vec::new(),
                })
                .collect(),
        },
        foreign(vec![call("dup")]),
        answer("dup", "three"),
        foreign(vec![AssistantContent::ToolCall(local.clone())]),
        Message::User {
            content: vec![UserContent::ToolResult(
                local.result(vec![ToolResultContent::text("four")]),
            )],
        },
    ];
    for mode in MODES {
        let body = sent(fixture, fixture.model(), history.clone(), mode)
            .unwrap_or_else(|error| panic!("{mode:?}: shared call ids encode: {error}"));
        if let Some(reason) = fixture.no_tool_calls() {
            eprintln!("skip pairing: {reason}");
            continue;
        }
        let problems = generated::pairing(&body);
        assert!(problems.is_empty(), "{mode:?}: {problems:?} in {body}");
    }
}

/// The number of generated cases a row runs: 256, or `RIG_HISTORY_CASES`.
fn cases() -> u64 {
    std::env::var("RIG_HISTORY_CASES")
        .ok()
        .and_then(|cases| cases.parse().ok())
        .unwrap_or(256)
}

/// How often a row reached each of its checks, so a check that never ran
/// fails the row instead of passing it.
#[derive(Default)]
struct Reached(std::cell::RefCell<std::collections::BTreeMap<&'static str, usize>>);

impl Reached {
    fn hit(&self, check: &'static str) {
        self.add(check, 1);
    }

    fn add(&self, check: &'static str, count: usize) {
        *self.0.borrow_mut().entry(check).or_default() += count;
    }

    /// Assert every one of `checks` ran at least once.
    fn assert(&self, row: &str, checks: &[&'static str]) {
        let reached = self.0.borrow();
        for check in checks {
            assert!(
                reached.get(check).is_some_and(|count| *count > 0),
                "{row} never reached its `{check}` check: {reached:?}"
            );
        }
    }
}

/// What breaks the wire's message rules in `body`: a call without exactly
/// one result right after it, or a result without a call (unless the
/// fixture says the wire sends no tool calls); a first message that is not
/// the user's on a wire that requires it; and roles that do not alternate
/// where the wire requires it, or two user messages in a row where it
/// joins them.
fn message_rules<F: HistoryFixture>(fixture: &F, body: &Value, reached: &Reached) -> Vec<String> {
    let mut problems = Vec::new();
    if fixture.no_tool_calls().is_none() {
        for event in generated::events(body) {
            reached.hit(match event {
                generated::Event::Call(_) => "calls",
                generated::Event::Result(_) => "results",
            });
        }
        problems.extend(generated::pairing(body));
        problems.extend(generated::adjacency(body));
    }
    let starts_with_user = fixture
        .wire(fixture.model())
        .describe()
        .replay
        .is_some_and(|target| target.starts_with_user());
    if starts_with_user {
        reached.hit("first role");
        problems.extend(generated::first_is_user(body));
    }
    if fixture.strict_roles() {
        reached.hit("alternation");
        problems.extend(generated::alternation(body));
    } else if !fixture.combines_same_role() {
        problems.extend(generated::adjacent_users(body));
    }
    problems
}

/// The checks [`message_rules`] reaches on the fixture's wire.
fn rule_checks<F: HistoryFixture>(fixture: &F) -> Vec<&'static str> {
    let mut checks = Vec::new();
    match fixture.no_tool_calls() {
        Some(reason) => eprintln!("skip pairing and adjacency: {reason}"),
        None => checks.extend(["calls", "results"]),
    }
    if fixture
        .wire(fixture.model())
        .describe()
        .replay
        .is_some_and(|target| target.starts_with_user())
    {
        checks.push("first role");
    }
    if fixture.strict_roles() {
        checks.push("alternation");
    }
    checks
}

/// The fixture's own turns: every reply shape in both modes, and generated
/// replies whole and streamed, whatever their stop.
fn own_turns<F: HistoryFixture>(fixture: &F) -> Vec<AssistantMessage> {
    let wire = fixture.wire(fixture.model());
    let request = CompletionRequest::new("restate");
    let mut frames = Vec::new();
    for shape in [Shape::Rich, Shape::Interleaved, Shape::Unknown] {
        for mode in MODES {
            frames.extend(fixture.reply(shape, mode).map(|frames| (mode, frames)));
        }
    }
    let mut rng = Rng::new(0x0e5e_ed00);
    for _ in 0..32 {
        let Some(mut spec) = fixture.reply_spec(&mut rng) else {
            break;
        };
        spec.seed = rng.next();
        if let Some(replies::Frames { whole, streamed }) = fixture.reply_frames(&spec) {
            frames.push((Mode::Unary, whole));
            frames.push((Mode::Streaming, streamed));
        }
    }
    frames
        .into_iter()
        .filter_map(
            |(mode, frames)| match decode(&wire, &request, mode, frames) {
                Ok(response) => match response.message() {
                    Some(Message::Assistant(turn)) => Some(turn),
                    _ => None,
                },
                Err(_) => None,
            },
        )
        .collect()
}

/// Whether `value` holds an integer beyond 2^53, which a store that reads
/// numbers as JavaScript does cannot hold (row H10 checks that case).
fn holds_big_integer(value: &Value) -> bool {
    const LIMIT: u64 = 1 << 53;
    match value {
        Value::Number(number) => {
            number.as_u64().is_some_and(|n| n > LIMIT)
                || number.as_i64().is_some_and(|n| n.unsigned_abs() > LIMIT)
        }
        Value::String(text) => serde_json::from_str::<Value>(text)
            .is_ok_and(|inner| inner.is_object() && holds_big_integer(&inner)),
        Value::Array(values) => values.iter().any(holds_big_integer),
        Value::Object(fields) => fields.values().any(holds_big_integer),
        _ => false,
    }
}

/// `value` with every whole float written as an integer, in text too, so
/// two bodies compare by the numbers they carry: a store that writes `3` as
/// `3.0` changes canonical arguments a rebuild then spells `3.0`.
fn whole_numbers(value: Value) -> Value {
    fn text(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut at = 0;
        while at < chars.len() {
            let whole = chars[at] == '.'
                && at > 0
                && chars[at - 1].is_ascii_digit()
                && chars.get(at + 1) == Some(&'0')
                && !chars
                    .get(at + 2)
                    .is_some_and(|next| next.is_ascii_digit() || matches!(next, 'e' | 'E'));
            if whole {
                at += 2;
                continue;
            }
            out.push(chars[at]);
            at += 1;
        }
        out
    }
    match value {
        Value::String(value) => Value::String(text(&value)),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, whole_numbers(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(whole_numbers).collect()),
        value => value,
    }
}

/// `history` stored and loaded again by a store that rewrites numbers as
/// `how` says.
fn reloaded(history: &[Message], how: Numbers) -> Vec<Message> {
    let stored = serde_json::to_value(history).expect("a history serializes");
    serde_json::from_value(numbers(stored, how)).expect("a rewritten history loads")
}

/// H17. Generated histories (`generated::history`) of the fixture's own
/// turns (every reply shape and generated replies, whole and streamed) and
/// other models' turns: every one encodes; every call in the body has
/// exactly one result, right after it in the wire's shape, and no result
/// lacks a call; roles alternate on a wire that requires it, and a wire
/// that requires a leading user message gets one. A store that rewrites
/// numbers sends the same body after a reload, and a stored own turn
/// replays the same items. The seed is fixed; `RIG_HISTORY_CASES` raises
/// the case count (256 by default). Each check asserts it ran, unless the
/// fixture states why the wire has nothing for it.
pub fn h17_generated_histories<F: HistoryFixture>(fixture: &F) {
    let own = own_turns(fixture);
    let reached = Reached::default();
    let problems = |history: &[Message], mode: Mode, tools: generated::Tools| -> Vec<String> {
        let body = match sent_as(fixture, fixture.model(), history.to_vec(), mode, tools) {
            Ok(body) => body,
            Err(error) => return vec![format!("it does not encode: {error}")],
        };
        reached.hit("encoded");
        let mut problems = message_rules(fixture, &body, &reached);
        let stored = serde_json::to_value(history).expect("a history serializes");
        if holds_big_integer(&stored) {
            return problems;
        }
        for how in [Numbers::ThroughF64, Numbers::WholeAsFloat] {
            match sent_as(
                fixture,
                fixture.model(),
                reloaded(history, how),
                mode,
                tools,
            ) {
                Ok(again) if same(&whole_numbers(body.clone()), &whole_numbers(again.clone())) => {
                    reached.hit("store");
                }
                Ok(again) => problems.push(format!(
                    "a store that rewrites numbers ({how:?}) sends another body: {again}"
                )),
                Err(error) => problems.push(format!(
                    "a store that rewrites numbers ({how:?}) leaves it unencodable: {error}"
                )),
            }
        }
        problems
    };
    for case in 0..cases() {
        let mut rng = generated::Rng::new(0x5eed_0000 + case);
        let history = generated::history(&mut rng, &own);
        let tools = generated::tools(&mut rng);
        for mode in MODES {
            let found = problems(&history, mode, tools);
            if !found.is_empty() {
                let small = generated::shrink(history.clone(), |history| {
                    !problems(history, mode, tools).is_empty()
                });
                panic!(
                    "case {case} in {mode:?} with tools {tools:?}: {found:?}\nhistory: {}\nshrunk: {}\nproblems there: {:?}",
                    generated::render(&history),
                    generated::render(&small),
                    problems(&small, mode, tools)
                );
            }
        }
    }
    for turn in &own {
        let expected = fixture.replayed(turn);
        for how in [Numbers::ThroughF64, Numbers::WholeAsFloat] {
            let message = Message::Assistant(turn.clone());
            if holds_big_integer(&serde_json::to_value(&message).expect("a turn serializes")) {
                continue;
            }
            let Some(Message::Assistant(loaded)) = reloaded(&[message], how).pop() else {
                panic!("an assistant turn loads as one");
            };
            let replayed = fixture.replayed(&loaded);
            reached.add("stored items", replayed.len());
            assert!(
                replayed.len() == expected.len()
                    && replayed.iter().zip(&expected).all(|(left, right)| {
                        same(&whole_numbers(left.clone()), &whole_numbers(right.clone()))
                    }),
                "a store that rewrites numbers ({how:?}) keeps every provider item current\nbefore: {expected:?}\nafter:  {replayed:?}"
            );
        }
    }
    let mut checks = vec!["encoded", "store"];
    checks.extend(rule_checks(fixture));
    if fixture.keeps_natives() {
        checks.push("stored items");
    }
    reached.assert("H17", &checks);
}

/// H18: on generated replies ([`HistoryFixture::reply_spec`]), the stream
/// folds to the turn and stop its whole form folds to, and neither mode
/// fails or panics alone.
pub fn h18_generated_stream_equals_whole<F: HistoryFixture>(fixture: &F) {
    let wire = fixture.wire(fixture.model());
    let reached = Reached::default();
    let ran = generated_replies(fixture, |spec| {
        let Some(frames) = fixture.reply_frames(spec) else {
            return Vec::new();
        };
        reached.hit("compared");
        replies::disagreement(&wire, frames).into_iter().collect()
    });
    if ran {
        reached.assert("H18", &["compared"]);
    }
}

/// H19: on generated replies, a stream cut after any frame before the
/// provider's end folds to a failed turn that holds no provider item, and no
/// cut panics. The turns runtimes continue from a cut hold no provider item
/// either, and each encodes by the wire's message rules with its calls
/// answered: `CompletionResponse::continued` over the partial reply, as
/// rig-agent rolls a turn back, and `streaming::delivered` over the items
/// the consumer took, up to each call that ended and in all, as rig-ecs
/// does.
pub fn h19_generated_cuts<F: HistoryFixture>(fixture: &F) {
    if let Some(reason) = fixture.no_cuts() {
        eprintln!("skip H19: {reason}");
        return;
    }
    let reached = Reached::default();
    let ran = generated_replies(fixture, |spec| {
        if fixture.reply_frames(spec).is_none() {
            return Vec::new();
        }
        let streamed = || {
            fixture
                .reply_frames(spec)
                .map(|frames| frames.streamed)
                .unwrap_or_default()
        };
        cut_problems(fixture, &streamed, &reached)
    });
    if ran {
        let mut checks = vec!["cuts", "continued"];
        checks.extend(rule_checks(fixture));
        reached.assert("H19", &checks);
    }
}

/// What is wrong with the cuts of `streamed` ([`h19_generated_cuts`]).
fn cut_problems<F: HistoryFixture>(
    fixture: &F,
    streamed: &dyn Fn() -> Vec<<F::Wire as Wire>::Frame>,
    reached: &Reached,
) -> Vec<String> {
    let wire = fixture.wire(fixture.model());
    let mut problems = Vec::new();
    let count = streamed().len();
    for cut in 1..count {
        let frames = streamed().into_iter().take(cut);
        let (response, ended, items) = match replies::quiet(|| {
            rig_core::test_utils::history_conformance::cut(&wire, Mode::Streaming, frames)
        }) {
            Ok(folded) => folded,
            Err(panic) => {
                problems.push(format!("the cut after {cut} frames panics: {panic}"));
                continue;
            }
        };
        if ended {
            continue;
        }
        reached.hit("cuts");
        if let Some(Message::Assistant(turn)) = response.message()
            && !turn.stop.as_ref().is_some_and(StopReason::is_failure)
        {
            problems.push(format!(
                "the cut after {cut} of {count} frames is a successful turn ({:?}): {}",
                turn.stop,
                serde_json::json!(turn)
            ));
        }
        let mut turns = vec![(
            "continued".to_owned(),
            response.continued(response.choice.clone()),
        )];
        let origin = Some(response.origin.clone());
        let ended_calls = items
            .iter()
            .enumerate()
            .filter_map(|(at, item)| match item {
                rig_core::streaming::Item::Event(rig_core::streaming::StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                }) => Some(at + 1),
                _ => None,
            });
        for taken in ended_calls.chain([items.len()]) {
            let content = rig_core::streaming::delivered(&items[..taken]);
            turns.push((
                format!("delivered after {taken} items"),
                AssistantMessage::rolled_back(origin.clone(), content),
            ));
        }
        for (how, turn) in turns {
            if turn.content.is_empty() {
                continue;
            }
            reached.hit("continued");
            problems.extend(
                continued_problems(fixture, &turn, reached)
                    .into_iter()
                    .map(|problem| format!("the cut after {cut} frames, {how}: {problem}")),
            );
        }
    }
    problems
}

/// What is wrong with `turn`, a turn a runtime continues from a reply cut
/// before its end ([`h19_generated_cuts`]): a block that holds a provider
/// item, a history with its calls answered that does not encode, or a body
/// that breaks the wire's message rules.
pub fn continuation_problems<F: HistoryFixture>(
    fixture: &F,
    turn: &AssistantMessage,
) -> Vec<String> {
    continued_problems(fixture, turn, &Reached::default())
}

fn continued_problems<F: HistoryFixture>(
    fixture: &F,
    turn: &AssistantMessage,
    reached: &Reached,
) -> Vec<String> {
    let mut problems: Vec<String> = turn
        .content
        .iter()
        .filter(|block| holds_item(block))
        .map(|block| {
            format!(
                "a block keeps its provider item: {}",
                serde_json::json!(block)
            )
        })
        .collect();
    let results: Vec<UserContent> = turn
        .tool_calls()
        .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("done")])))
        .collect();
    let mut history = vec![Message::user("q"), Message::Assistant(turn.clone())];
    if !results.is_empty() {
        history.push(Message::User { content: results });
    }
    match sent(fixture, fixture.model(), history, Mode::Streaming) {
        Ok(body) => problems.extend(message_rules(fixture, &body, reached)),
        Err(error) => problems.push(format!("it does not encode: {error}")),
    }
    problems
}

/// Run `problems` over the fixture's generated replies, shrinking and
/// reporting the first spec that has any. False when the fixture has no
/// reply generator, which the row then states.
fn generated_replies<F: HistoryFixture>(
    fixture: &F,
    problems: impl Fn(&replies::Spec) -> Vec<String>,
) -> bool {
    for case in 0..cases() {
        let mut rng = Rng::new(0x5eed_d0e5 + case);
        let Some(mut spec) = fixture.reply_spec(&mut rng) else {
            eprintln!("skip: the wire has no reply generator");
            return false;
        };
        spec.seed = rng.next();
        let found = problems(&spec);
        if !found.is_empty() {
            let small = replies::shrink(spec.clone(), |spec| !problems(spec).is_empty());
            panic!(
                "case {case}: {found:?}\nspec: {} finish {} seed {}\nshrunk: {} finish {} seed {}\nproblems there: {:?}",
                Value::Array(spec.blocks.clone()),
                spec.finish,
                spec.seed,
                Value::Array(small.blocks.clone()),
                small.finish,
                small.seed,
                problems(&small)
            );
        }
    }
    true
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
            h15_item_round_trip,
            h16_call_identity,
            h17_generated_histories,
            h18_generated_stream_equals_whole,
            h19_generated_cuts,
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
                "h15_item_round_trip",
                "h16_call_identity",
                "h17_generated_histories",
                "h18_generated_stream_equals_whole",
                "h19_generated_cuts",
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
