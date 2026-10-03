//! The Gemini GenerateContent REST wire's history suite: unary
//! `generateContent` bodies and `streamGenerateContent` SSE chunks, each
//! part a block whose provider item is the part as Gemini sent it.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::AssistantContent;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::gemini::completion::GenerateContent;
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{Ablation, CallShape, Ending, HistoryFixture, Shape, http_body};
use serde_json::{Value, json};

/// Every `finishReason` Gemini and Vertex AI document that ends a turn as a
/// failure: content filters, recitation, unsupported language, image
/// failures, tool-protocol failures and the unused zero value.
pub const FAILURE_FINISHES: &[&str] = &[
    "FINISH_REASON_UNSPECIFIED",
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "OTHER",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_OTHER",
    "NO_IMAGE",
    "IMAGE_RECITATION",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "MISSING_THOUGHT_SIGNATURE",
    "MALFORMED_RESPONSE",
    "ESCALATION",
    "PUP_LIMITED_DISABLED",
    "MODEL_ARMOR",
];

/// Two calls to `weather`, `a1` and `b2`, as `functionCall` parts.
pub fn weather_calls() -> [Value; 2] {
    [
        json!({ "functionCall": { "name": "weather", "args": { "city": "Paris" }, "id": "a1" } }),
        json!({ "functionCall": { "name": "weather", "args": { "city": "Rome" }, "id": "b2" } }),
    ]
}

pub struct GeminiRestHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply document holding `parts`, ending on `finish` when it is set.
fn chunk(parts: Value, finish: Option<&str>) -> Value {
    let mut candidate = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let (Some(finish), Some(candidate)) = (finish, candidate.as_object_mut()) {
        candidate.insert("finishReason".to_owned(), json!(finish));
    }
    json!({
        "candidates": [candidate],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": MODEL,
        "responseId": "resp_1",
    })
}

fn frames(documents: impl IntoIterator<Item = Value>) -> Vec<WireFrame> {
    documents
        .into_iter()
        .map(|document| WireFrame::Text(document.to_string()))
        .collect()
}

/// The parts of `shape`, whole, and the same parts split into the chunks a
/// stream sends them in.
fn parts(shape: Shape) -> (Value, Vec<Value>) {
    match shape {
        Shape::Rich => (
            json!([
                { "text": "plan the lookup", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ=" },
                { "text": "looking it up" },
                { "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } },
                { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                {
                    "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" },
                    "thoughtSignature": "c2lnLWNhbGw="
                },
            ]),
            vec![
                json!([{ "text": "plan the ", "thought": true }]),
                json!([{ "text": "lookup", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ=" }]),
                json!([{ "text": "looking " }]),
                json!([{ "text": "it up" }]),
                json!([
                    { "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } },
                    { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                ]),
                json!([{
                    "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" },
                    "thoughtSignature": "c2lnLWNhbGw="
                }]),
            ],
        ),
        Shape::Interleaved => (
            json!([
                { "text": "first", "thought": true },
                { "text": "between" },
                { "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" },
                { "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" } },
            ]),
            vec![
                json!([{ "text": "first", "thought": true }]),
                json!([{ "text": "between" }]),
                json!([{ "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" }]),
                json!([{ "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" } }]),
            ],
        ),
        Shape::Unknown => (
            json!([
                { "text": "noted", "x_rig_field": true },
                { "x_rig_invented": { "id": "x_1" } },
            ]),
            vec![
                json!([{ "text": "noted", "x_rig_field": true }]),
                json!([{ "x_rig_invented": { "id": "x_1" } }]),
            ],
        ),
    }
}

fn reply_of(shape: Shape, mode: Mode) -> Vec<WireFrame> {
    let (whole, streamed) = parts(shape);
    match mode {
        Mode::Unary => frames([chunk(whole, Some("STOP"))]),
        Mode::Streaming => {
            let last = streamed.len().saturating_sub(1);
            frames(
                streamed
                    .into_iter()
                    .enumerate()
                    .map(|(at, parts)| chunk(parts, (at == last).then_some("STOP"))),
            )
        }
    }
}

impl HistoryFixture for GeminiRestHistory {
    type Wire = GenerateContent;

    fn wire(&self, model: &str) -> GenerateContent {
        GenerateContent::new(GeminiConfig::new("test-key"), model)
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "gemini-2.5-flash"
    }

    fn body(
        &self,
        wire: &GenerateContent,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply_of(shape, mode))
    }

    /// Gemini sends `args` as JSON, so only argument text that is JSON can
    /// arrive; truncated or non-JSON text cannot be expressed on this wire.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<WireFrame>> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        Some(frames([chunk(
            json!([{ "functionCall": { "name": "lookup", "args": args } }]),
            Some("STOP"),
        )]))
    }

    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        let reply = |finish: &str| frames([chunk(json!([{ "text": "partial" }]), Some(finish))]);
        [("STOP", Ending::Success), ("MAX_TOKENS", Ending::Success)]
            .into_iter()
            .chain(
                FAILURE_FINISHES
                    .iter()
                    .map(|finish| (*finish, Ending::Failure)),
            )
            .chain([("A_REASON_FROM_TOMORROW", Ending::Failure)])
            .map(|(finish, ending)| (finish, reply(finish), ending))
            .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: chunk(parts(Shape::Rich).0, Some("STOP")),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/finishReason",
            ],
            frames: |document| frames([document]),
        })
    }

    /// A part is its own whole reply.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let part = block.native_item()?.clone();
        let wire = self.wire(MODEL);
        let response = rig_core::test_utils::history::decode(
            &wire,
            Mode::Unary,
            frames([chunk(json!([part]), Some("STOP"))]),
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Gemini sends whole calls: one reply listing both, or one chunk each.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let [first, second] = weather_calls();
        match (shape, mode) {
            (CallShape::WholeList, _) => {
                Some(frames([chunk(json!([first, second]), Some("STOP"))]))
            }
            (CallShape::Indexless, Mode::Streaming) => Some(frames([
                chunk(json!([first]), None),
                chunk(json!([second]), Some("STOP")),
            ])),
            _ => None,
        }
    }

    fn empty_reply(&self, _mode: Mode) -> Option<Vec<WireFrame>> {
        Some(frames([chunk(json!([]), Some("STOP"))]))
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/candidates/0/finishReason")
    }

    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(
            json!({ "error": { "code": 503, "message": "overloaded", "status": "UNAVAILABLE" } })
                .to_string(),
        ))
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "gemini_rest",
    fixture: GeminiRestHistory,
}
