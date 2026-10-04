//! The decoder of GenerateContent replies, a whole `generateContent` body or
//! a stream of `streamGenerateContent` chunks, read as REST JSON. The REST,
//! Vertex AI and gRPC wires share it: each part becomes a block whose
//! provider item is the part as Gemini sent it.
//!
//! ```
//! use rig_core::providers::gemini::streaming::GenerateContentDecoder;
//!
//! let decoder = GenerateContentDecoder::default();
//! # let _ = decoder;
//! ```

use serde_json::{Map, Value};

use super::completion::blocked_prompt_error;
use super::completion::{map_google_finish_reason, usage_of};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, DocumentSourceKind, Image, MediaType, MimeType, ToolName};
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};

/// The recognizability markers of a `streamGenerateContent` chunk: every
/// genuine frame carries `candidates`, `usageMetadata` and/or
/// `promptFeedback` (a blocked prompt's only chunk may carry nothing but
/// the feedback), and the service's in-band abort carries only `error`. A
/// frame with any of them is a chunk. A valid ID-only frame is recognized
/// separately as metadata; other JSON is `Unknown`.
const RECOGNIZABLE_CHUNK_KEYS: &[&str] =
    &["candidates", "usageMetadata", "promptFeedback", "error"];

/// A GenerateContent reply document in REST JSON: the whole
/// `generateContent` body or one `streamGenerateContent` chunk.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(transparent)]
pub struct GenerateContentChunk(pub Map<String, Value>);

/// Decode GenerateContent replies. A text or thought part continues the
/// block of the part before it while the kind stays the same, and a part
/// that carries only a thought signature joins that block; any other part is
/// a block of its own. The provider's end is held until EOF because
/// hosted-tool rounds can report intermediate finish reasons.
#[derive(Debug, Default)]
pub struct GenerateContentDecoder {
    /// The latest `finishReason` and `finishMessage`.
    finish: Option<String>,
    finish_message: Option<String>,
    /// The latest `usageMetadata`, as Gemini sent it.
    usage: Option<Value>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// The reply document a wire kept in place of the summary record.
    raw: Option<Value>,
    /// The text or thought run later parts continue, and whether it is
    /// thought.
    open: Option<(usize, bool)>,
    /// Whether the open run holds a signature.
    signed: bool,
    /// The index of the block written last.
    last: Option<usize>,
    /// A signature sent alone before any block, which the next block takes.
    signature: Option<String>,
}

impl GenerateContentDecoder {
    /// Report `raw` as the reply's raw document in place of the summary
    /// record, for a wire whose own reply is not REST JSON (an SDK's or
    /// a protobuf message).
    pub fn keep_raw(&mut self, raw: Value) {
        self.raw = Some(raw);
    }
}

impl<'id> Decoder<'id, Completion> for GenerateContentDecoder {
    type Event = GenerateContentChunk;

    fn classify(&self, frame: WireFrame) -> WireEvent<GenerateContentChunk> {
        // ID-only metadata must not create an unknown-content truncation tail.
        if GenerateContentDecoder::is_analysis_only(&frame) {
            return wire::classify_marker_keyed_frame(&frame.as_str(), &["responseId"]);
        }
        wire::classify_marker_keyed_frame(&frame.as_str(), RECOGNIZABLE_CHUNK_KEYS)
    }

    fn decode(
        &mut self,
        GenerateContentChunk(data): GenerateContentChunk,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let data = Value::Object(data);
        let span = tracing::Span::current();
        if let Some(id) = data.str("responseId").filter(|id| !id.is_empty()) {
            span.record("gen_ai.response.id", id);
            self.response_id = Some(id.to_owned());
        }
        if let Some(model) = data.str("modelVersion").filter(|model| !model.is_empty()) {
            span.record("gen_ai.response.model", model);
            self.model_version = Some(model.to_owned());
        }
        if let Some(usage) = data.get("usageMetadata") {
            self.usage = Some(usage.clone());
        }
        if let Some(error) = data.at("/error") {
            // An in-band failure is a provider error, not a truncation. Its
            // code is an HTTP status; only error statuses join the retry policy.
            let status = error
                .get("code")
                .and_then(Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .and_then(|code| http::StatusCode::from_u16(code).ok())
                .filter(|status| status.is_client_error() || status.is_server_error());
            let body = serde_json::json!({ "error": error }).to_string();
            return Err(match status {
                Some(status) => ProviderError::from_http_response(status, body),
                None => ProviderError::from_provider_body(body),
            });
        }
        if let Some(blocked) = data.get("promptFeedback").and_then(blocked_prompt_error) {
            return Err(blocked);
        }
        // The candidates, their content and its parts hold every block and
        // the finish, so a wrongly typed one fails the reply.
        let candidate = match data
            .get("candidates")
            .map(|candidates| (candidates, candidates.get(0)))
        {
            None | Some((Value::Null, _) | (Value::Array(_), None)) => return Ok(Flow::More),
            Some((Value::Array(_), Some(candidate @ Value::Object(_)))) => candidate,
            Some((Value::Array(_), Some(_))) => {
                return Err(malformed("a candidate that is not an object"));
            }
            Some(_) => return Err(malformed("candidates that are not a list")),
        };
        // Last one wins: an intermediate `finishReason` is superseded by the
        // reason the turn actually ended on. Proto3 JSON spells an enum
        // value its schema does not know as its number.
        match candidate.get("finishReason") {
            Some(Value::String(name)) => self.finish = Some(name.clone()),
            Some(Value::Number(number)) => self.finish = Some(format!("FINISH_REASON_{number}")),
            _ => {}
        }
        if let Some(message) = candidate.str("finishMessage") {
            self.finish_message = Some(message.to_owned());
        }
        let parts = match candidate.get("content") {
            None | Some(Value::Null) => None,
            Some(content @ Value::Object(_)) => content.get("parts"),
            Some(_) => return Err(malformed("candidate content that is not an object")),
        };
        match parts {
            None | Some(Value::Null) => {}
            Some(Value::Array(parts)) => {
                for part in parts {
                    self.part(part.clone(), &mut out)?;
                }
            }
            Some(_) => return Err(malformed("candidate parts that are not a list")),
        }
        // A failure is final: nothing after it is read.
        use crate::completion::FinishReason::{Length, Stop};
        let reason = self.finish.as_deref().map(map_google_finish_reason);
        match (reason, candidate.get("finishReason")) {
            (None | Some(Stop | Length), _) | (_, None) => Ok(Flow::More),
            _ => self.end(out),
        }
    }

    /// Gemini ends a reply at EOF, not at its first finish reason: a
    /// hosted-tool round can report one before more content arrives.
    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.end(out)
    }
}

impl GenerateContentDecoder {
    /// End the reply on the finish reason the candidate holds. Without one
    /// the reply did not end.
    fn end(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        let Some(reason) = self.finish.take() else {
            return Err(ProviderError::Truncated);
        };
        self.close(&mut out)?;
        let usage = self.usage.as_ref().map(usage_of).unwrap_or_default();
        let model = self.model_version.take();
        let response_id = self.response_id.take();
        let summary = serde_json::json!({
            "usage_metadata": self.usage.take().unwrap_or_else(|| Value::Object(Map::new())),
            "finish_reason": reason,
            "finish_message": self.finish_message.take(),
            "model_version": model,
            "response_id": response_id,
        });
        let Value::Object(mut summary) = summary else {
            return Err(malformed("a reply"));
        };
        summary.retain(|_, value| !value.is_null());
        out.raw(self.raw.take().unwrap_or(Value::Object(summary)));
        Ok(out.end(Finish {
            usage,
            reason: Some(map_google_finish_reason(&reason)),
            response_id,
            model,
            ..Finish::default()
        }))
    }

    /// Finish the open text or thought run: Gemini states each part whole.
    fn close(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        self.open
            .take()
            .map_or(Ok(()), |(index, _)| out.finish(index))
    }

    /// Write a block at a fresh index, closing the run before it. A text or
    /// thought block stays open as the run later parts continue; any other
    /// is whole.
    fn open(
        &mut self,
        block: Block,
        item: Value,
        text: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.close(out)?;
        let index = out.fresh_index();
        let mut item = item;
        if let (Some(signature), Some(fields)) = (self.signature.take(), item.as_object_mut()) {
            fields
                .entry("thoughtSignature")
                .or_insert(Value::String(signature));
        }
        self.last = Some(index);
        self.signed = item
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .is_some_and(|signature| !signature.is_empty());
        let run = match &block {
            Block::Text => Some(false),
            Block::Reasoning { .. } => Some(true),
            _ => None,
        };
        out.open(index, block, item)?;
        out.push(index, text)?;
        self.open = run.map(|thought| (index, thought));
        self.open.map_or_else(|| out.finish(index), |_| Ok(()))
    }

    /// Write one part of the candidate's content. A part this decoder does
    /// not model is an opaque item that replays; one with no data, or that
    /// is not an object, is kept as one that does not. A signature alone
    /// joins the block before it, as Gemini attaches it to the part before,
    /// or the next block when it comes first.
    fn part(&mut self, part: Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Value::Object(fields) = &part else {
            return self.open(Block::Opaque { replay: false }, part, "", out);
        };
        let thought = part.bool("thought") == Some(true);
        let signature = part
            .str("thoughtSignature")
            .filter(|signature| !signature.is_empty());
        if let Some(text) = part.str("text") {
            // An empty part without a signature carries nothing.
            if text.is_empty() && signature.is_none() {
                return Ok(());
            }
            // A second signed part starts a block of its own, so each
            // signature stays with the part that carried it.
            return match self.open {
                Some((index, kind)) if kind == thought && !(self.signed && signature.is_some()) => {
                    self.signed |= signature.is_some();
                    out.push(index, text)?;
                    out.edit(index, |item| merge_part(item, &part))
                }
                _ if thought => self.open(
                    Block::Reasoning { redacted: false },
                    part.clone(),
                    text,
                    out,
                ),
                _ => self.open(Block::Text, part.clone(), text, out),
            };
        }
        if let Some(call) = part.obj("functionCall") {
            let Ok(name) =
                ToolName::new(call.get("name").and_then(Value::as_str).unwrap_or_default())
            else {
                tracing::warn!("Gemini sent a function call without a name; nothing can answer it");
                return Ok(());
            };
            // Proto3 JSON leaves out an empty `args` Struct.
            let args = call
                .get("args")
                .map_or_else(|| "{}".to_owned(), Value::to_string);
            let id = CallId::from_wire(call.get("id").and_then(Value::as_str).unwrap_or_default());
            return self.open(Block::Call { id, name }, part.clone(), &args, out);
        }
        if !thought
            && let (Some(mime_type), Some(data)) =
                (part.at("/inlineData/mimeType"), part.at("/inlineData/data"))
            && let (Some(mime_type), Some(data)) = (mime_type.as_str(), data.as_str())
            && let Some(MediaType::Image(media_type)) = MediaType::from_mime_type(mime_type)
        {
            let image = Image {
                data: DocumentSourceKind::Base64(data.to_owned()),
                media_type: Some(media_type),
                detail: None,
                native: None,
            };
            return self.open(Block::Image(image), part.clone(), "", out);
        }
        let bare = ["thought", "thoughtSignature", "partMetadata"];
        let data = fields.keys().any(|key| !bare.contains(&key.as_str()));
        if !data && let Some(signature) = signature {
            let Some(index) = self.open.map(|(index, _)| index).or(self.last) else {
                self.signature = Some(signature.to_owned());
                return Ok(());
            };
            return out.edit(index, |item| {
                if let Some(item) = item.as_object_mut() {
                    item.insert("thoughtSignature".to_owned(), Value::from(signature));
                }
            });
        }
        self.open(Block::Opaque { replay: data }, part.clone(), "", out)
    }

    /// Whether `frame` carries a response id and nothing else, a repeated
    /// key included.
    pub(crate) fn is_analysis_only(frame: &WireFrame) -> bool {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ResponseIdOnly {
            #[serde(rename = "responseId")]
            _id: String,
        }
        matches!(
            wire::classify_marker_keyed_frame::<ResponseIdOnly>(&frame.as_str(), &["responseId"]),
            WireEvent::Known(_)
        )
    }

    /// Project provider verdicts, usage, response identity, and errors
    /// before normalization. A field of another type is left out, and a
    /// payload that is not JSON projects nothing.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(reply) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        if let Some(usage) = reply.get("usageMetadata").filter(|usage| usage.is_object()) {
            let count = |key: &str| usage.u64(key);
            let usage = AdapterUsage {
                input_tokens: count("promptTokenCount"),
                output_tokens: count("candidatesTokenCount"),
                total_tokens: count("totalTokenCount"),
                cached_input_tokens: count("cachedContentTokenCount"),
                reasoning_tokens: count("thoughtsTokenCount"),
                tool_input_tokens: count("toolUsePromptTokenCount"),
            };
            sink.emit(AdapterEvent::Usage { usage });
        }
        let candidate = reply.arr("candidates").first().unwrap_or(&Value::Null);
        let scrub = |value: Option<&str>| value.map(|value| sink.scrub(value));
        let block = reply
            .at("/promptFeedback/blockReason")
            .and_then(Value::as_str);
        let verdict = AdapterVerdict {
            finish_reason: scrub(candidate.str("finishReason")),
            block_reason: scrub(block),
            detail: scrub(candidate.str("finishMessage")),
            model: scrub(reply.str("modelVersion")),
        };
        let response_id = scrub(reply.str("responseId"));
        sink.provider(verdict, response_id);
        if let Some(error) = reply.get("error").filter(|error| error.is_object()) {
            let text = |key: &str| error.str(key).map(str::to_owned);
            let error = crate::observe::ObservedError {
                code: error.get("code").cloned(),
                kind: text("status").or_else(|| text("type")),
                message: text("message"),
            };
            error.emit(sink);
        }
    }
}

fn malformed(what: &str) -> ProviderError {
    ProviderError::Response(format!("Gemini sent {what}"))
}

/// Merge a streamed part into the part its block holds so far: the text
/// appends, a non-empty `thoughtSignature` replaces the one held, and any
/// other field replaces its namesake.
fn merge_part(item: &mut Value, part: &Value) {
    let (Value::Object(held), Value::Object(part)) = (&mut *item, part) else {
        *item = part.clone();
        return;
    };
    for (key, value) in part {
        if key == "text"
            && let (Some(Value::String(text)), Value::String(more)) = (held.get_mut("text"), value)
        {
            text.push_str(more);
        } else if key != "thoughtSignature" || value.as_str() != Some("") {
            held.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests;
