//! The history conformance suite for the Gemini gRPC wire. Replies are
//! protobuf messages built from the REST JSON the REST suite uses, so both
//! wires are checked against the same documents; the request body is the
//! protobuf request's REST JSON.

#![allow(clippy::expect_used)]

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::test_utils::history_conformance::{Ablation, Ending, HistoryFixture, Shape};
use rig_core::wire::{Mode, Wire};
use rig_gemini_grpc::completion::GenerateContent;
use rig_gemini_grpc::proto::{self, GenerateContentResponse};
use rig_gemini_grpc::rest::{from_rest, to_rest};
use serde_json::{Value, json};

struct GrpcHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply holding `parts`, ending on `finish` when it is set, as the
/// protobuf message its REST JSON describes.
fn chunk(parts: Value, finish: Option<Value>) -> Value {
    let mut candidate = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let (Some(finish), Some(candidate)) = (finish, candidate.as_object_mut()) {
        candidate.insert("finishReason".to_owned(), finish);
    }
    json!({
        "candidates": [candidate],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": MODEL,
        "responseId": "resp_1",
    })
}

fn frame(document: Value) -> GenerateContentResponse {
    from_rest(document).expect("the document is a GenerateContentResponse")
}

fn stop() -> Option<Value> {
    Some(json!("STOP"))
}

/// The parts of `shape`, whole, and split into the chunks a stream sends.
fn parts(shape: Shape) -> Option<(Value, Vec<Value>)> {
    Some(match shape {
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
        // Protobuf decoding drops fields and part kinds its schema does not
        // declare, so an invented item or field never reaches the decoder.
        Shape::Unknown => return None,
    })
}

impl HistoryFixture for GrpcHistory {
    type Wire = GenerateContent;

    fn wire(&self, model: &str) -> GenerateContent {
        GenerateContent::new(model)
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "gemini-2.5-flash"
    }

    /// The protobuf request as its REST JSON; its `model` names the model.
    fn body(
        &self,
        wire: &GenerateContent,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        Ok(to_rest(&wire.encode(request, mode)?)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let (whole, streamed) = parts(shape)?;
        Some(match mode {
            Mode::Unary => vec![frame(chunk(whole, stop()))],
            Mode::Streaming => {
                let last = streamed.len().saturating_sub(1);
                streamed
                    .into_iter()
                    .enumerate()
                    .map(|(at, parts)| frame(chunk(parts, if at == last { stop() } else { None })))
                    .collect()
            }
        })
    }

    /// `args` is a `google.protobuf.Struct`: only an object, or no
    /// arguments at all, can arrive.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let call = match serde_json::from_str::<Value>(arguments).ok()? {
            Value::Null => json!({ "name": "lookup" }),
            args @ Value::Object(_) => json!({ "name": "lookup", "args": args }),
            _ => return None,
        };
        Some(vec![frame(chunk(
            json!([{ "functionCall": call }]),
            stop(),
        ))])
    }

    /// Every finish reason the proto lists, and a number it does not,
    /// which is how a reason newer than the proto arrives.
    fn finishes(&self) -> Vec<(&'static str, Vec<GenerateContentResponse>, Ending)> {
        let reply =
            |finish: Value| vec![frame(chunk(json!([{ "text": "partial" }]), Some(finish)))];
        let mut finishes = vec![
            ("STOP", reply(json!("STOP")), Ending::Success),
            ("MAX_TOKENS", reply(json!("MAX_TOKENS")), Ending::Success),
        ];
        for name in [
            "SAFETY",
            "RECITATION",
            "OTHER",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
        ] {
            assert!(
                proto::candidate::FinishReason::from_str_name(name).is_some(),
                "the proto lists {name}"
            );
            finishes.push((name, reply(json!(name)), Ending::Failure));
        }
        finishes.push(("an unlisted number", reply(json!(99)), Ending::Failure));
        finishes
    }

    fn ablation(&self) -> Option<Ablation<GenerateContentResponse>> {
        Some(Ablation {
            document: chunk(parts(Shape::Rich)?.0, stop()),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/finishReason",
            ],
            frames: |document| vec![frame(document)],
        })
    }
}

rig_core::history_conformance_suite! {
    wire: "gemini_grpc",
    fixture: GrpcHistory,
}
