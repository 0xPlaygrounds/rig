//! The history conformance suite for the Vertex AI `GenerateContent` wire.
//! Replies are SDK messages read from the REST JSON Vertex AI sends; the
//! request body is the SDK request's JSON, whose `model` names the model.

#![allow(clippy::expect_used)]

use google_cloud_aiplatform_v1::model::GenerateContentResponse;
use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::gemini::streaming::GenerateContentChunk;
use rig_core::wire::{Mode, Wire};
use rig_history_conformance::{Ablation, CallShape, Ending, HistoryFixture, Rng, Shape, replies};
use rig_vertexai::completion::{GenerateContent, rest_chunk};
use serde_json::{Value, json};

struct VertexHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply holding `parts`, ending on `finish` when it is set.
fn chunk(parts: Value, finish: Option<&str>) -> Value {
    replies::gemini_document(parts, finish, MODEL)
}

/// `document` as the transport hands it over: read into the SDK's reply,
/// then transcoded back to its REST chunk.
fn frame(document: Value) -> GenerateContentChunk {
    let response: GenerateContentResponse =
        serde_json::from_value(document).expect("the document is a GenerateContentResponse");
    rest_chunk(&response).expect("the reply transcodes")
}

/// The parts of `shape`, whole, and split into the chunks a stream sends.
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
        // The SDK keeps the fields it does not model, so both survive.
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

impl HistoryFixture for VertexHistory {
    type Wire = GenerateContent;

    fn wire(&self, model: &str) -> GenerateContent {
        GenerateContent::new(model)
    }

    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        Some(serde_json::json!({"functionDeclarations": [{
            "name": tool.name.as_str(), "description": tool.description,
            "parametersJsonSchema": tool.parameters,
        }]}))
    }

    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        Some(replies::gemini_spec(rng, true))
    }

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<GenerateContentChunk>> {
        let (whole, streamed) = replies::gemini_build(MODEL, spec, true);
        Some(replies::Frames {
            whole: whole.into_iter().map(frame).collect(),
            streamed: streamed.into_iter().map(frame).collect(),
        })
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

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<GenerateContentChunk>> {
        let (whole, streamed) = parts(shape);
        Some(match mode {
            Mode::Unary => vec![frame(chunk(whole, Some("STOP")))],
            Mode::Streaming => {
                let last = streamed.len().saturating_sub(1);
                streamed
                    .into_iter()
                    .enumerate()
                    .map(|(at, parts)| frame(chunk(parts, (at == last).then_some("STOP"))))
                    .collect()
            }
        })
    }

    /// `args` is a `google.protobuf.Struct`: only an object, or no
    /// arguments at all, can arrive.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<GenerateContentChunk>> {
        let call = match serde_json::from_str::<Value>(arguments).ok()? {
            Value::Null => json!({ "name": "lookup" }),
            args @ Value::Object(_) => json!({ "name": "lookup", "args": args }),
            _ => return None,
        };
        Some(vec![frame(chunk(
            json!([{ "functionCall": call }]),
            Some("STOP"),
        ))])
    }

    /// Every finish reason Vertex AI documents, and one it does not.
    /// `FINISH_REASON_UNSPECIFIED` is the proto default, which the SDK
    /// leaves out, so on this wire it reads as no finish at all.
    fn finishes(&self) -> Vec<(&'static str, Vec<GenerateContentChunk>, Ending)> {
        let reply = |finish: &str| vec![frame(chunk(json!([{ "text": "partial" }]), Some(finish)))];
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

    fn ablation(&self) -> Option<Ablation<GenerateContentChunk>> {
        Some(Ablation {
            document: chunk(parts(Shape::Rich).0, Some("STOP")),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/finishReason",
            ],
            frames: |mut document| {
                fitted(&mut document, &chunk(parts(Shape::Rich).0, Some("STOP")));
                vec![frame(document)]
            },
        })
    }

    /// A part is its own whole reply.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let part = block.native_item()?.clone();
        let response = rig_core::test_utils::history::decode(
            &self.wire(MODEL),
            Mode::Unary,
            [frame(chunk(json!([part]), Some("STOP")))],
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Gemini sends whole calls: one reply listing both, or one chunk each.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<GenerateContentChunk>> {
        let first = json!({ "functionCall": { "name": "weather", "args": { "city": "Paris" }, "id": "a1" } });
        let second = json!({ "functionCall": { "name": "weather", "args": { "city": "Rome" }, "id": "b2" } });
        match (shape, mode) {
            (CallShape::WholeList, _) => {
                Some(vec![frame(chunk(json!([first, second]), Some("STOP")))])
            }
            (CallShape::Indexless, Mode::Streaming) => Some(vec![
                frame(chunk(json!([first]), None)),
                frame(chunk(json!([second]), Some("STOP"))),
            ]),
            _ => None,
        }
    }

    fn empty_reply(&self, _mode: Mode) -> Option<Vec<GenerateContentChunk>> {
        Some(vec![frame(chunk(json!([]), Some("STOP")))])
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/candidates/0/finishReason")
    }

    /// Each provider part as the SDK sends it: its JSON read into the SDK's
    /// `Part`, which spells enums by number.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        let sdk = |item: &Value| {
            serde_json::from_value::<google_cloud_aiplatform_v1::model::Part>(item.clone())
                .ok()
                .and_then(|part| serde_json::to_value(part).ok())
        };
        turn.content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Opaque(opaque) if opaque.replay => sdk(&opaque.item),
                AssistantContent::Opaque(_) => None,
                block => block.native_item().and_then(sdk),
            })
            .collect()
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

rig_history_conformance::history_conformance_suite! {
    wire: "vertexai",
    fixture: VertexHistory,
}
