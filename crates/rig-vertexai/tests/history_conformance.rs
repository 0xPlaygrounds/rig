//! The history conformance suite for the Vertex AI `GenerateContent` wire.
//! Replies are SDK messages read from the REST JSON Vertex AI sends; the
//! request body is the SDK request's JSON, whose `model` names the model.

#![allow(clippy::expect_used)]

use google_cloud_aiplatform_v1::model::GenerateContentResponse;
use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::test_utils::history_conformance::{Ablation, Ending, HistoryFixture, Shape};
use rig_core::wire::{Mode, Wire};
use rig_vertexai::completion::GenerateContent;
use serde_json::{Value, json};

struct VertexHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply holding `parts`, ending on `finish`.
fn reply_document(parts: Value, finish: &str) -> Value {
    json!({
        "candidates": [{
            "content": { "role": "model", "parts": parts },
            "finishReason": finish,
            "index": 0,
        }],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": MODEL,
        "responseId": "resp_1",
    })
}

fn frame(document: Value) -> GenerateContentResponse {
    serde_json::from_value(document).expect("the document is a GenerateContentResponse")
}

fn parts(shape: Shape) -> Value {
    match shape {
        Shape::Rich => json!([
            { "text": "plan the lookup", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ=" },
            { "text": "looking it up" },
            { "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } },
            { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
            {
                "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" },
                "thoughtSignature": "c2lnLWNhbGw="
            },
        ]),
        Shape::Interleaved => json!([
            { "text": "first", "thought": true },
            { "text": "between" },
            { "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" },
            { "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" } },
        ]),
        // The SDK keeps the fields it does not model, so both survive.
        Shape::Unknown => json!([
            { "text": "noted", "x_rig_field": true },
            { "x_rig_invented": { "id": "x_1" } },
        ]),
    }
}

impl HistoryFixture for VertexHistory {
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

    fn body(
        &self,
        wire: &GenerateContent,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        Ok(serde_json::to_value(wire.encode(request, mode)?)?)
    }

    /// Vertex AI has no streaming RPC here: a streamed call re-emits the
    /// unary reply, so both modes read the same whole reply.
    fn reply(&self, shape: Shape, _mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        Some(vec![frame(reply_document(parts(shape), "STOP"))])
    }

    /// `args` is a `google.protobuf.Struct`: only an object, or no
    /// arguments at all, can arrive.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let call = match serde_json::from_str::<Value>(arguments).ok()? {
            Value::Null => json!({ "name": "lookup" }),
            args @ Value::Object(_) => json!({ "name": "lookup", "args": args }),
            _ => return None,
        };
        Some(vec![frame(reply_document(
            json!([{ "functionCall": call }]),
            "STOP",
        ))])
    }

    /// Every finish reason Vertex AI documents, and one it does not.
    /// `FINISH_REASON_UNSPECIFIED` is the proto default, which the SDK
    /// leaves out, so on this wire it reads as no finish at all.
    fn finishes(&self) -> Vec<(&'static str, Vec<GenerateContentResponse>, Ending)> {
        let reply = |finish: &str| {
            vec![frame(reply_document(
                json!([{ "text": "partial" }]),
                finish,
            ))]
        };
        let success = ["STOP", "MAX_TOKENS"];
        let failure = [
            "SAFETY",
            "RECITATION",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "MODEL_ARMOR",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
            "IMAGE_OTHER",
            "UNEXPECTED_TOOL_CALL",
            "NO_IMAGE",
            "A_REASON_FROM_TOMORROW",
        ];
        success
            .into_iter()
            .map(|finish| (finish, reply(finish), Ending::Success))
            .chain(
                failure
                    .into_iter()
                    .map(|finish| (finish, reply(finish), Ending::Failure)),
            )
            .collect()
    }

    fn ablation(&self) -> Option<Ablation<GenerateContentResponse>> {
        Some(Ablation {
            document: reply_document(parts(Shape::Rich), "STOP"),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/finishReason",
            ],
            frames: |mut document| {
                fitted(&mut document, &reply_document(parts(Shape::Rich), "STOP"));
                vec![frame(document)]
            },
        })
    }
}

/// `value` with each field and element whose JSON type differs from the
/// same one in `typed` left out. A reply frame is a typed message, so a
/// field of another type never arrives; it arrives unset.
fn fitted(value: &mut Value, typed: &Value) {
    let fits = |value: &mut Value, typed: Option<&Value>| match typed {
        Some(typed) if std::mem::discriminant(value) != std::mem::discriminant(typed) => false,
        Some(typed) => {
            fitted(value, typed);
            true
        }
        None => true,
    };
    match (value, typed) {
        (Value::Object(fields), Value::Object(types)) => {
            fields.retain(|key, field| fits(field, types.get(key)));
        }
        (Value::Array(values), Value::Array(types)) => {
            let mut types = types.iter();
            values.retain_mut(|value| fits(value, types.next()));
        }
        _ => {}
    }
}

rig_core::history_conformance_suite! {
    wire: "vertexai",
    fixture: VertexHistory,
}
