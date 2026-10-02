use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::completion::gemini_api_types::{
    Blob, FinishReason, FunctionCall, PromptFeedback, UsageMetadata, map_finish_reason,
};
use super::completion::{blocked_prompt_error, function_call_finish_reason_error};
use crate::error::ProviderError;
use crate::message::{AssistantContent, DocumentSourceKind, Image, MediaType, MimeType};
use crate::observe::ObservedError;
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};

/// Part-kind interpretation shared by the Gemini wires whose payloads
/// coincide: REST `streamGenerateContent` and the Interactions API both
/// deliver whole function calls.
pub(crate) mod shared_parts {
    use serde_json::Value;

    use crate::error::ProviderError;
    use crate::message::{CallId, ToolName};
    use crate::operation::{Block, Completion};
    use crate::wire::Out;

    /// Write a whole function call, `item` its provider item. An id-less
    /// call gets an id rig issues, never a fabricated provider id, even
    /// when it shares a tool name; a nameless one is malformed.
    pub(crate) fn function_call(
        out: &mut Out<'_, Completion>,
        name: String,
        args: Value,
        wire_id: Option<String>,
        item: Value,
    ) -> Result<(), ProviderError> {
        let Ok(name) = ToolName::new(name) else {
            return Err(ProviderError::Response(format!(
                "Gemini function call without a name: {item}"
            )));
        };
        let id = CallId::from_wire(wire_id.unwrap_or_default());
        let index = out.fresh_index();
        out.whole(index, Block::Call { id, name }, item, &args.to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage_metadata: UsageMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: &StreamingCompletionResponse) -> crate::completion::Usage {
        (&value.usage_metadata).into()
    }
}

impl From<StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: StreamingCompletionResponse) -> crate::completion::Usage {
        (&value).into()
    }
}

/// The recognizability markers of a `streamGenerateContent` chunk: every
/// genuine frame carries `candidates`, `usageMetadata` and/or
/// `promptFeedback` (a blocked prompt's only chunk may carry nothing but
/// the feedback), and the service's in-band abort carries only `error`. A
/// frame with any of them must fully decode (else `Corrupt`). A valid ID-only
/// frame is recognized separately as metadata; other JSON is `Unknown`.
const RECOGNIZABLE_CHUNK_KEYS: &[&str] =
    &["candidates", "usageMetadata", "promptFeedback", "error"];

/// A GenerateContent reply document as the completion decoder reads it:
/// the whole `generateContent` body or one `streamGenerateContent` chunk.
/// Candidates stay JSON, so each part reaches history as Gemini sent it.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentChunk {
    /// The response id; empty when Gemini did not report one.
    #[serde(default)]
    pub response_id: String,
    /// The candidates. The decoder reads the first.
    #[serde(default)]
    pub candidates: Vec<Map<String, Value>>,
    /// The prompt's content-filter verdict.
    pub prompt_feedback: Option<PromptFeedback>,
    /// Token usage so far.
    pub usage_metadata: Option<UsageMetadata>,
    /// The model that answered.
    pub model_version: Option<String>,
    /// Gemini's error envelope, sent as a frame of its own when the
    /// service aborts a stream in-band.
    pub error: Option<Value>,
}

/// Decode GenerateContent replies, a whole body or a stream of chunks.
/// The gRPC and Vertex AI wires restate their replies as chunks and share
/// it. A text or thought part continues the block of the part before it
/// while the kind stays the same; any other part ends that block. The
/// provider's end is held until EOF because hosted-tool rounds can report
/// intermediate finish reasons. A reply without assistant content fails
/// unless a truncating finish reason permits empty output.
#[derive(Debug, Default)]
pub struct GenerateContentDecoder {
    /// The candidate's fields beside its content, each as the last chunk
    /// sent it: finish reason, safety ratings, citations and grounding.
    /// It becomes the turn's message-level native.
    candidate: Map<String, Value>,
    usage: Option<UsageMetadata>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// At least one part mapped to assistant content.
    delivered: bool,
    /// The reply document a wire kept in place of the summary record.
    raw: Option<Value>,
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
        data: GenerateContentChunk,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let span = tracing::Span::current();
        if !data.response_id.is_empty() {
            span.record("gen_ai.response.id", data.response_id.as_str());
            self.response_id = Some(data.response_id);
        }
        if let Some(model_version) = data.model_version {
            span.record("gen_ai.response.model", model_version.as_str());
            self.model_version = Some(model_version);
        }
        if data.usage_metadata.is_some() {
            self.usage = data.usage_metadata;
        }

        if let Some(error) = data.error {
            // Preserve in-band failures as provider errors, not unknown-frame truncation.
            // GenerateContent codes are HTTP statuses; only error statuses
            // participate in the unary retry policy.
            let status = error
                .get("code")
                .and_then(serde_json::Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .and_then(|code| http::StatusCode::from_u16(code).ok())
                .filter(|status| status.is_client_error() || status.is_server_error());
            let body = serde_json::json!({ "error": error }).to_string();
            return Err(match status {
                Some(status) => ProviderError::from_http_response(status, body),
                None => crate::error::ProviderError::from_provider_body(body),
            });
        }

        if let Some(blocked) = data.prompt_feedback.as_ref().and_then(blocked_prompt_error) {
            // Preserve the refusal reason rather than reporting an unexplained truncation.
            return Err(blocked);
        }

        let Some(mut candidate) = data.candidates.into_iter().next() else {
            return Ok(Flow::More);
        };
        let content = candidate.shift_remove("content");
        if let Some(reason) = candidate.get("finishReason") {
            let reason: FinishReason = serde_json::from_value(reason.clone())?;
            let message = candidate.get("finishMessage").and_then(Value::as_str);
            if let Some(error) = function_call_finish_reason_error(&reason, message) {
                return Err(error);
            }
        }
        // Last one wins: an intermediate `finishReason` is superseded by
        // the reason the turn actually ended on.
        self.candidate.extend(candidate);
        let parts = match content {
            Some(Value::Object(mut content)) => content.shift_remove("parts"),
            _ => None,
        };
        match parts {
            Some(Value::Array(parts)) => {
                for part in parts {
                    self.part(part, &mut out)?;
                }
            }
            None => {}
            Some(_) => return Err(malformed("candidate parts that are not a list")),
        }
        Ok(Flow::More)
    }

    /// Gemini ends a reply at EOF, not at its first finish reason: a
    /// hosted-tool round can report one before more content arrives.
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        // Without a provider finish reason, the reply did not end.
        let Some(reason) = self.candidate.get("finishReason") else {
            return Err(ProviderError::Truncated);
        };
        let reason: FinishReason = serde_json::from_value(reason.clone())?;
        let finish_reason = map_finish_reason(&reason);
        // An empty reply needs a truncating finish reason to explain it.
        let cut_short = finish_reason
            .as_ref()
            .is_some_and(crate::completion::FinishReason::truncated_output);
        if !self.delivered && !cut_short {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        // Defaulting the raw usage shape does not imply reported usage.
        let usage = self
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        let raw = StreamingCompletionResponse {
            usage_metadata: self.usage.take().unwrap_or_default(),
            finish_reason: Some(reason),
            finish_message: self
                .candidate
                .get("finishMessage")
                .and_then(Value::as_str)
                .map(str::to_owned),
            model_version: self.model_version.take(),
            response_id: self.response_id.take(),
        };
        out.raw(match self.raw.take() {
            Some(kept) => kept,
            None => serde_json::to_value(&raw)?,
        });
        out.message_native(Value::Object(std::mem::take(&mut self.candidate)));
        Ok(out.end(Finish {
            usage,
            reason: finish_reason,
            response_id: raw.response_id,
            model: raw.model_version,
            ..Finish::default()
        }))
    }
}

impl GenerateContentDecoder {
    /// Write one part of the candidate's content as its block.
    fn part(&mut self, part: Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Value::Object(fields) = &part else {
            return Err(malformed("a part that is not an object"));
        };
        let thought = fields.get("thought").and_then(Value::as_bool) == Some(true);
        if let Some(text) = fields.get("text") {
            let text = text
                .as_str()
                .ok_or_else(|| malformed("a text part whose text is not a string"))?;
            let signed = fields
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .is_some_and(|signature| !signature.is_empty());
            // An empty part without a signature carries nothing.
            if text.is_empty() && !signed {
                return Ok(());
            }
            self.delivered = true;
            let block = if thought {
                Block::Reasoning { redacted: false }
            } else {
                Block::Text
            };
            let index = out.run(block, text)?;
            return out.edit(index, |item| merge_part(item, part));
        }
        out.end_run()?;
        if let Some(call) = fields.get("functionCall") {
            let call: FunctionCall = serde_json::from_value(call.clone())?;
            self.delivered = true;
            return shared_parts::function_call(out, call.name, call.args, call.id, part);
        }
        if let Some(blob) = fields.get("inlineData").filter(|_| !thought) {
            let blob: Blob = serde_json::from_value(blob.clone())?;
            if let Some(MediaType::Image(media_type)) = MediaType::from_mime_type(&blob.mime_type) {
                self.delivered = true;
                let image = Image {
                    data: DocumentSourceKind::Base64(blob.data),
                    media_type: Some(media_type),
                    detail: None,
                    native: None,
                };
                return out.content(AssistantContent::Image(image).with_native(part));
            }
        }
        // Hosted code execution alone is not an answer.
        self.delivered |=
            !fields.contains_key("executableCode") && !fields.contains_key("codeExecutionResult");
        let index = out.fresh_index();
        out.whole(index, Block::Opaque { replay: true }, part, "")
    }

    pub(crate) fn is_analysis_only(frame: &WireFrame) -> bool {
        #[derive(Deserialize)]
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

    /// Project provider verdicts, usage, response identity, and errors before normalization.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        // The observation projection must not inherit native response
        // defaults: omitted prompt/total counts in UsageMetadata otherwise
        // become zero. Parsing failure has no effect on the provider's
        // authoritative decoder.
        if let Ok(ObservedUsageOnly { usage: Some(usage) }) =
            serde_json::from_slice::<ObservedUsageOnly>(payload)
        {
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: usage.prompt_token_count,
                    output_tokens: usage.candidates_token_count,
                    total_tokens: usage.total_token_count,
                    cached_input_tokens: usage.cached_content_token_count,
                    reasoning_tokens: usage.thoughts_token_count,
                    tool_input_tokens: usage.tool_use_prompt_token_count,
                },
            });
        }
        // Project metadata independently so malformed candidate fields cannot
        // erase an otherwise valid usage report from a rejected response.
        let Ok(metadata) = serde_json::from_slice::<ObservedMetadata>(payload) else {
            return;
        };
        let candidate = metadata.candidates.into_iter().next().unwrap_or_default();
        let scrub = |value: String| sink.scrub(&value);
        let verdict = AdapterVerdict {
            finish_reason: candidate.finish_reason.map(scrub),
            block_reason: metadata
                .prompt_feedback
                .and_then(|f| f.block_reason)
                .map(scrub),
            detail: candidate.finish_message.map(scrub),
            model: metadata.model_version.map(scrub),
        };
        let response_id = metadata.response_id.map(scrub);
        sink.provider(verdict, response_id);
        if let Some(error) = metadata.error {
            error.emit(sink);
        }
    }
}

/// Merge a streamed part into the part its block holds so far: the text
/// appends, a non-empty `thoughtSignature` replaces the one held, and any
/// other field replaces its namesake.
fn merge_part(item: &mut Value, part: Value) {
    let (Value::Object(held), Value::Object(part)) = (&mut *item, &part) else {
        *item = part;
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

fn malformed(what: &str) -> ProviderError {
    ProviderError::Response(format!("Gemini sent {what}"))
}

/// The usage report alone, read off the payload before the verdict so a
/// malformed candidate cannot erase it.
#[derive(Deserialize)]
struct ObservedUsageOnly {
    #[serde(rename = "usageMetadata")]
    usage: Option<ObservedUsage>,
}

// Ignore all unrelated response fields rather than allocating another tree
// containing the completion text, tools, signatures and media.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedUsage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    prompt_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    candidates_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_content_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    thoughts_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    tool_use_prompt_token_count: Option<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedMetadata {
    #[serde(default)]
    candidates: Vec<ObservedCandidate>,
    prompt_feedback: Option<ObservedFeedback>,
    model_version: Option<String>,
    response_id: Option<String>,
    error: Option<ObservedError>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedCandidate {
    finish_reason: Option<String>,
    finish_message: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedFeedback {
    block_reason: Option<String>,
}

#[cfg(test)]
mod tests;
