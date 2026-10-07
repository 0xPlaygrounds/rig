//! Scripted stream faults for the runner fault matrix.
//!
//! A fault cell serves a real provider adapter a stream that ends badly and
//! asserts what the runtime does with it: the error it surfaces, the record
//! it keeps, the history it commits, the tools it runs. The bodies are cut
//! from committed recordings ([`recorded_sse_frames`]) or, where a wire only
//! records single terminal frames, assembled from frames a real capture or
//! the adapter's own unit tests pin; every synthetic frame is a labelled
//! constant beside its cell. The scripted transport replaces the cassette
//! proxy for those cells, so their request boundary is pinned by the
//! recording's owning test, not here. Setup failures replay the committed
//! error recordings through the ordinary cassette wrappers.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bytes::Bytes;
use futures::StreamExt;

use rig_agent::agent::MultiTurnStreamItem;

use rig_agent::agent::StreamingResult;

use rig_agent::completion::PromptError;

use rig_core::effect::EffectFamily;

use rig_core::error::ErrorKind;

use rig_core::error::ErrorReport;

use rig_core::observe::Action;

use rig_core::observe::AdapterEvent;

use rig_core::observe::ObservationLog;

use rig_core::streaming::{Item, StreamEvent};

use rig_agent::test_utils::SequencedStreamingHttpClient;

use rig_core::tool::Tool;

use rig_core::tool::ToolContext;

use rig_cassette::effect_log::EffectLog;

use crate::{
    goldens::families,
    support::{MathError, OperationArgs, Subtract},
};

/// The recorded SSE frames of one interaction of a scenario: the response
/// body split at its blank-line delimiters, in wire order.
pub fn recorded_sse_frames(provider: &str, scenario: &str, interaction: usize) -> Vec<String> {
    let recorded = crate::cassettes::recorded_statuses_and_bodies(provider, scenario);
    let (status, body) = recorded
        .get(interaction)
        .unwrap_or_else(|| panic!("{provider}/{scenario} has no interaction {interaction}"));
    assert_eq!(
        *status, 200,
        "{provider}/{scenario}[{interaction}] is not a successful stream"
    );
    let frames = sse_frames(body);
    assert!(
        frames.len() > 1,
        "{provider}/{scenario}[{interaction}] streams one frame; nothing can be cut from it"
    );
    frames
}

/// A body's SSE frames, without their delimiters. A wire that delimits
/// with CRLF (Gemini) is read like one that delimits with LF; the frames
/// are re-emitted with LF, which every adapter reads.
pub fn sse_frames(body: &str) -> Vec<String> {
    body.replace("\r\n", "\n")
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

/// The frames before the first one `is_terminal` accepts: a recording cut
/// short of its ending. At least one frame must precede the ending, or the
/// cut would not be a truncation of anything.
pub fn frames_before(frames: &[String], is_terminal: impl Fn(&str) -> bool) -> Vec<String> {
    let end = frames
        .iter()
        .position(|frame| is_terminal(frame))
        .expect("the recording carries the terminal frame to cut before");
    assert!(end > 0, "no frame precedes the terminal");
    frames[..end].to_vec()
}

/// The `data:` payload of an SSE frame, parsed.
pub fn frame_data(frame: &str) -> serde_json::Value {
    let data = frame
        .lines()
        .find_map(|line| {
            line.strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
        })
        .unwrap_or_else(|| panic!("frame carries no data line: {frame}"));
    serde_json::from_str(data).unwrap_or_else(|error| panic!("frame data is JSON: {error}: {data}"))
}

/// A wire family's SSE shape: where a recorded stream's terminal sits,
/// what its text deltas carry, and the in-band error frame the adapter's
/// unit tests pin for it. The failure rows cut a wire's own #2501
/// recording with these; the #2490 cells read the same shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SseShape {
    /// OpenAI Chat Completions and every OpenAI-compatible wire (DeepSeek,
    /// Doubleword, Venice): `data: {"choices":[{"delta":…,
    /// "finish_reason":null}]}` frames, a `finish_reason` frame, a usage
    /// frame, `data: [DONE]`.
    Chat,
    /// OpenAI Responses: `event: response.*` frames, `response.completed`
    /// the terminal.
    Responses,
    /// Gemini REST `streamGenerateContent`: `data: {"candidates":…}`
    /// frames, the last with a `finishReason`.
    Gemini,
}

impl SseShape {
    /// Whether `frame` is the recorded turn's terminal, or follows it:
    /// nothing at or after it belongs in a cut.
    fn is_terminal(self, frame: &str) -> bool {
        match self {
            // A `finish_reason` other than `null` closes the choice; the
            // usage frame and `[DONE]` follow it.
            Self::Chat => {
                frame.contains(r#""finish_reason":""#) || frame.starts_with("data: [DONE]")
            }
            Self::Responses => frame.starts_with("event: response.completed"),
            Self::Gemini => frame.contains("finishReason"),
        }
    }

    /// The frames of a recorded text turn up to, not including, its
    /// text's completion: content deltas, then nothing.
    pub fn text_prefix(self, frames: &[String]) -> Vec<String> {
        let prefix = match self {
            Self::Responses => frames_before(frames, |frame| {
                frame.starts_with("event: response.output_text.done")
            }),
            Self::Chat | Self::Gemini => frames_before(frames, |frame| self.is_terminal(frame)),
        };
        assert!(
            !self.delta_text(&prefix).is_empty(),
            "the recording streams text before its end: {prefix:?}"
        );
        prefix
    }

    /// The frames of a recorded tool-call turn up to, not including, its
    /// terminal: the whole call, then nothing.
    pub fn tool_prefix(self, frames: &[String]) -> Vec<String> {
        let prefix = frames_before(frames, |frame| self.is_terminal(frame));
        let carries_call = match self {
            Self::Chat => prefix.iter().any(|frame| frame.contains("tool_calls")),
            Self::Responses => prefix
                .iter()
                .any(|frame| frame.starts_with("event: response.output_item.done")),
            Self::Gemini => prefix.iter().any(|frame| frame.contains("functionCall")),
        };
        assert!(carries_call, "the cut keeps the completed call: {prefix:?}");
        prefix
    }

    /// The text the deltas of `frames` carry.
    pub fn delta_text(self, frames: &[String]) -> String {
        frames
            .iter()
            .filter(|frame| !frame.starts_with("data: [DONE]"))
            .filter_map(|frame| match self {
                Self::Chat => frame_data(frame)["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned),
                Self::Responses => frame
                    .starts_with("event: response.output_text.delta")
                    .then(|| frame_data(frame)["delta"].as_str().map(str::to_owned))
                    .flatten(),
                Self::Gemini => frame_data(frame)["candidates"][0]["content"]["parts"]
                    .as_array()
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| part["text"].as_str())
                            .collect::<String>()
                    }),
            })
            .collect()
    }

    /// An in-band error frame after content, the shape the adapter's unit
    /// tests pin: the OpenAI-compatible envelope under a 200
    /// (`streaming_in_band_error_envelope_preserves_full_payload`), the
    /// Responses `error` event (`streaming_error_event_preserves_full_payload`),
    /// Gemini's error envelope in band
    /// (`in_band_http_errors_match_unary_classification`).
    pub fn error_frame(self) -> String {
        match self {
            Self::Chat => {
                r#"data: {"error":{"message":"upstream unavailable","type":"server_error"}}"#
                    .to_owned()
            }
            Self::Responses => RESPONSES_ERROR_EVENT.to_owned(),
            Self::Gemini => format!("data: {GEMINI_IN_BAND_ERROR}"),
        }
    }
}

/// The frames of `error_frames`' fault: the cut text turn, then the in-band
/// error frame.
impl SseShape {
    /// The text prefix of the recorded turn followed by the wire's in-band
    /// error frame (`error_frame`).
    pub fn error_frames(self, frames: &[String]) -> Vec<String> {
        let mut out = self.text_prefix(frames);
        out.push(self.error_frame());
        out
    }

    /// The finish the recorded turn stops with, and the one a filtered
    /// turn stops with: OpenAI's `content_filter`, Gemini's `SAFETY`
    /// (`map_google_finish_reason`), both `FinishReason::ContentFilter`.
    const fn stop_finish(self) -> &'static str {
        match self {
            Self::Chat => r#""finish_reason":"stop""#,
            Self::Gemini => r#""finishReason":"STOP""#,
            Self::Responses => "",
        }
    }

    const fn filtered_finish(self) -> &'static str {
        match self {
            Self::Chat => r#""finish_reason":"content_filter""#,
            Self::Gemini => r#""finishReason":"SAFETY""#,
            Self::Responses => "",
        }
    }

    /// The recorded text turn with its finish rewritten to the filtered
    /// one, the text kept or dropped: `content_filter` on the Chat shape,
    /// `SAFETY` on Gemini's, `response.incomplete` with
    /// `incomplete_details.reason: content_filter` on the Responses shape
    /// (`map_finish_reason`; `completion_response_incomplete_reports_the_truncation_reason`).
    pub fn filtered(self, frames: &[String], with_text: bool) -> Vec<String> {
        if self == Self::Responses {
            return responses_filtered(frames, with_text);
        }
        let finish = frames
            .iter()
            .position(|frame| frame.contains(self.stop_finish()))
            .expect("the recorded turn stops");
        let mut out: Vec<String> = if with_text {
            frames[..finish].to_vec()
        } else {
            frames[..finish]
                .iter()
                .filter(|frame| self.delta_text(std::slice::from_ref(frame)).is_empty())
                .cloned()
                .collect()
        };
        // Every finish frame from the first on: a wire may send its finish
        // twice (Doubleword), and the adapter's terminal is the last it saw.
        out.extend(
            frames[finish..]
                .iter()
                .map(|frame| frame.replace(self.stop_finish(), self.filtered_finish())),
        );
        out
    }

    /// A refusal in the wire's own shape: the Chat refusal turn the
    /// adapter's unit tests pin ([`CHAT_REFUSAL_FRAMES`]), the recorded
    /// Responses text turn rewritten to `refusal` parts and deltas
    /// (`refusal_content_part_frames_are_no_ops_and_refusal_text_streams`),
    /// Gemini's blocked prompt ([`GEMINI_BLOCKED_FRAME`]). The Chat and
    /// Responses refusals stream as the answer; Gemini's fails the run.
    pub fn refusal(self, frames: &[String]) -> Vec<String> {
        match self {
            Self::Chat => CHAT_REFUSAL_FRAMES
                .iter()
                .map(|frame| format!("data: {frame}"))
                .chain(std::iter::once("data: [DONE]".to_owned()))
                .collect(),
            Self::Responses => responses_refusal(frames),
            Self::Gemini => vec![format!("data: {GEMINI_BLOCKED_FRAME}")],
        }
    }
}

/// The refusal turn the Chat adapter's unit tests pin
/// (`refusal_only_stream_delivers_the_refusal_text`): `content` held at
/// `null` for the whole turn, the refusal on its own key, a clean `stop`.
pub const CHAT_REFUSAL_FRAMES: [&str; 5] = [
    r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"refusal":""},"finish_reason":null}]}"#,
    r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":"I'm sorry"},"finish_reason":null}]}"#,
    r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":", I can't help."},"finish_reason":null}]}"#,
    r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}"#,
];
/// The text [`CHAT_REFUSAL_FRAMES`] carries.
pub const CHAT_REFUSAL_TEXT: &str = "I'm sorry, I can't help.";
/// The whole body Gemini answered a refused prompt with: one feedback
/// chunk, no candidates, then the stream closes (#2475; the rigcoder
/// capture of 2026-09-08).
pub const GEMINI_BLOCKED_FRAME: &str = r#"{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":795,"totalTokenCount":795}}"#;

/// A recorded Responses frame's event name and data, rewritten by `edit`
/// and reassembled.
fn rewrite(frame: &str, edit: impl Fn(&mut serde_json::Value)) -> String {
    let mut data = frame_data(frame);
    edit(&mut data);
    let event = data["type"].as_str().expect("an event type").to_owned();
    format!("event: {event}\ndata: {data}")
}

/// A content part turned into the refusal shape.
fn refuse_part(part: &mut serde_json::Value) {
    if part["type"] == "output_text" {
        part["type"] = "refusal".into();
        let text = part["text"].take();
        let object = part.as_object_mut().expect("a part");
        object.shift_remove("text");
        object.shift_remove("annotations");
        object.shift_remove("logprobs");
        part["refusal"] = text;
    }
}

fn refuse_item(item: &mut serde_json::Value) {
    if let Some(content) = item["content"].as_array_mut() {
        content.iter_mut().for_each(refuse_part);
    }
}

/// The recorded Responses text turn as a refusal turn: `response.refusal.delta`
/// deltas, `refusal` content parts on the item and in the terminal. Every
/// other frame is the recording's.
fn responses_refusal(frames: &[String]) -> Vec<String> {
    frames
        .iter()
        .map(|frame| {
            rewrite(frame, |data| {
                match data["type"].as_str().expect("an event type") {
                    "response.output_text.delta" => data["type"] = "response.refusal.delta".into(),
                    "response.output_text.done" => {
                        data["type"] = "response.refusal.done".into();
                        let text = data["text"].take();
                        let object = data.as_object_mut().expect("an event");
                        object.shift_remove("text");
                        object.shift_remove("logprobs");
                        data["refusal"] = text;
                    }
                    "response.content_part.added" | "response.content_part.done" => {
                        refuse_part(&mut data["part"]);
                    }
                    "response.output_item.done" | "response.output_item.added" => {
                        refuse_item(&mut data["item"]);
                    }
                    "response.completed" => {
                        if let Some(output) = data["response"]["output"].as_array_mut() {
                            output.iter_mut().for_each(refuse_item);
                        }
                    }
                    _ => {}
                }
            })
        })
        .collect()
}

/// The recorded Responses text turn ended by `response.incomplete` with
/// `content_filter`, the message item kept or dropped.
fn responses_filtered(frames: &[String], with_text: bool) -> Vec<String> {
    frames
        .iter()
        .filter(|frame| {
            with_text || {
                let data = frame_data(frame);
                let kind = data["type"].as_str().expect("an event type");
                !(kind.starts_with("response.output_text")
                    || kind.starts_with("response.content_part")
                    || (kind.starts_with("response.output_item")
                        && data["item"]["type"] == "message"))
            }
        })
        .map(|frame| {
            rewrite(frame, |data| {
                if data["type"] == "response.completed" {
                    data["type"] = "response.incomplete".into();
                    data["response"]["status"] = "incomplete".into();
                    data["response"]["incomplete_details"] =
                        serde_json::json!({ "reason": "content_filter" });
                    if !with_text && let Some(output) = data["response"]["output"].as_array_mut() {
                        output.retain(|item| item["type"] != "message");
                    }
                }
            })
        })
        .collect()
}

/// The Responses wire's `error` event, the shape the adapter's unit tests
/// pin (`streaming_error_event_preserves_full_payload`).
pub const RESPONSES_ERROR_EVENT: &str = r#"event: error
data: {"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
/// Gemini's error envelope in band under HTTP 200: the envelope Gemini
/// returned for an overloaded model in the rigcoder capture of 2026-09-08,
/// in the frame position the adapter's unit tests pin
/// (`in_band_http_errors_match_unary_classification`).
pub const GEMINI_IN_BAND_ERROR: &str = r#"{"error":{"code":503,"message":"The model is overloaded. Please try again later.","status":"UNAVAILABLE"}}"#;

/// The wire bytes of `frames`, each followed by its delimiter.
pub fn sse_bytes(frames: &[String]) -> Bytes {
    let mut body = frames.join("\n\n");
    body.push_str("\n\n");
    Bytes::from(body)
}

/// A transport that answers its one streaming request with `chunks`, then
/// EOF. A second request fails: every cell scripts exactly one exchange.
pub fn scripted(chunks: Vec<Bytes>) -> SequencedStreamingHttpClient {
    SequencedStreamingHttpClient::new(chunks.into_iter().map(Ok).collect())
}

/// What the runner's stream delivered before it ended.
#[derive(Debug, Default)]
pub struct Drained {
    /// The text deltas, concatenated.
    pub text: String,
    /// Provider terminal records forwarded to the consumer.
    pub terminals: usize,
    /// Committed model tool calls.
    pub tool_calls: usize,
    /// Final responses: a successful run yields exactly one.
    pub finals: usize,
    /// Every error item, as a report.
    pub errors: Vec<ErrorReport>,
}

/// Drain a runner stream to EOF.
pub async fn drain(stream: &mut StreamingResult) -> Drained {
    let mut drained = Drained::default();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            }))) => drained.text.push_str(&text),
            Ok(MultiTurnStreamItem::CompletionCall(_)) => drained.terminals += 1,
            Ok(MultiTurnStreamItem::ToolCall { .. }) => drained.tool_calls += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => drained.finals += 1,
            Ok(_) => {}
            Err(error) => drained.errors.push(report_of(&error)),
        }
    }
    drained
}

/// The report a runner stream error carries.
pub fn report_of(error: &PromptError) -> ErrorReport {
    match error {
        PromptError::Provider(error) => ErrorReport::from(error),
        PromptError::Report(report) => report.clone(),
        PromptError::Cancelled { reason, .. } => {
            ErrorReport::new(ErrorKind::Cancelled, reason.clone())
        }
        other => panic!("a provider-shaped failure, not {other:?}"),
    }
}

/// The error the one completion record holds. A fault cell records exactly
/// one completion and nothing after it: no tool, no memory, no retry.
pub fn sole_failed_completion(log: &EffectLog) -> &ErrorReport {
    assert_eq!(
        families(log),
        [EffectFamily::Completion],
        "one completion record and no effect after the fault"
    );
    log.records[0]
        .outcome
        .as_ref()
        .expect_err("the completion record holds the fault")
}

/// The recorded stream error items of the one completion, as
/// `(position, report)`.
pub fn recorded_stream_errors(log: &EffectLog) -> Vec<(usize, ErrorReport)> {
    log.header
        .stream_errors
        .values()
        .flatten()
        .map(|error| (error.item, error.error.clone()))
        .collect()
}

/// A setup failure: the provider refused the request before any frame.
/// Both runtimes surface the recorded status and body under the wire's own
/// classification (`kind`), record the one completion as that failure and
/// stream nothing.
pub fn assert_setup_failure(report: &ErrorReport, kind: ErrorKind, status: u16) {
    assert_eq!(report.kind, kind, "{report:?}");
    assert_eq!(report.http_status, Some(status), "{report:?}");
    let body = report
        .provider_response_body()
        .expect("the provider's body travels with the failure");
    let body: serde_json::Value =
        serde_json::from_str(body).expect("the recorded error body is JSON");
    assert!(
        body.get("error").is_some(),
        "the provider's error envelope is preserved: {body}"
    );
}

/// Counts a tool's executions.
#[derive(Clone, Default)]
pub struct Invocations(Arc<AtomicUsize>);

impl Invocations {
    /// Return the number of observed tool invocations.
    pub fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// `subtract`, counting its executions: a fault cell's proof that a call
/// the model issued before the fault never ran.
#[derive(Clone)]
pub struct CountedSubtract(pub Invocations);

impl Tool for CountedSubtract {
    const NAME: &'static str = Subtract::NAME;
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        Subtract.description()
    }

    fn parameters(&self) -> serde_json::Value {
        Subtract.parameters()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.0.fetch_add(1, Ordering::SeqCst);
        Subtract.call(context, args).await
    }
}

/// Every program ending the witness saw, by code.
pub fn endings(log: &ObservationLog) -> Vec<String> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Ended { ending } => Some(ending.code.clone()),
            _ => None,
        })
        .collect()
}

/// Every provider-boundary fact, in order.
pub fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect()
}

/// Every truncation the bus witnessed: items delivered and error items.
pub fn truncations(log: &ObservationLog) -> Vec<(usize, usize)> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::StreamTruncated {
                delivered, errors, ..
            } => Some((*delivered, errors.len())),
            _ => None,
        })
        .collect()
}

/// The whole trace, serialized: what an analysis sink would persist.
pub fn trace_json(log: &ObservationLog) -> String {
    serde_json::to_string(&log.trace()).expect("the trace serializes")
}
