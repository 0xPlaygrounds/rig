use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::{
    ContentCandidate, FinishReason, GenerateContentResponse, Part, PartKind, UsageMetadata,
    map_finish_reason,
};
use super::completion::{blocked_prompt_error, function_call_finish_reason_error, part_kind_name};
use crate::error::ProviderError;
use crate::observe::ObservedError;
use crate::operation::{Completion, Finish, TextPart};
use crate::providers::internal::thoughts::Thoughts;
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};

/// Part-kind interpretation shared by the Gemini wires whose payloads
/// coincide: REST `streamGenerateContent` and the Interactions API both
/// deliver whole function calls and identity-less thought fragments.
pub(crate) mod shared_parts {
    use serde_json::Value;

    use crate::error::ProviderError;
    use crate::message::{CallId, ToolCall, ToolFunction, ToolName};
    use crate::operation::Completion;
    use crate::wire::Out;

    /// Write a whole function call. An id-less call gets an id rig issues,
    /// never a fabricated provider id, even when it shares a tool name; a
    /// nameless one is dropped.
    pub(crate) fn function_call(
        out: &mut Out<'_, Completion>,
        name: String,
        args: Value,
        wire_id: Option<String>,
        signature: Option<String>,
    ) -> Result<(), ProviderError> {
        let Ok(name) = ToolName::new(name) else {
            return Ok(());
        };
        out.tool_call(ToolCall {
            id: CallId::from_wire(wire_id.unwrap_or_default()),
            function: ToolFunction {
                name,
                arguments: args,
            },
            signature,
            additional_params: None,
        })
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

fn tool_protocol_finish_reason_error(choice: &ContentCandidate) -> Option<ProviderError> {
    let reason = choice.finish_reason.as_ref()?;
    function_call_finish_reason_error(reason, choice.finish_message.as_deref())
}

/// The recognizability markers of a `streamGenerateContent` chunk: every
/// genuine frame carries `candidates`, `usageMetadata` and/or
/// `promptFeedback` (a blocked prompt's only chunk may carry nothing but
/// the feedback), and the service's in-band abort carries only `error`. A
/// frame with any of them must fully decode (else `Corrupt`). A valid ID-only
/// frame is recognized separately as metadata; other JSON is `Unknown`.
const RECOGNIZABLE_CHUNK_KEYS: &[&str] =
    &["candidates", "usageMetadata", "promptFeedback", "error"];

/// Decode GenerateContent replies, a whole body or a stream of chunks.
/// The provider's end is held until EOF because hosted-tool rounds can
/// report intermediate finish reasons. A reply without assistant content
/// fails unless a truncating finish reason permits empty output.
pub struct GenerateContentDecoder<'id> {
    /// Thought boundaries inferred from content transitions and signatures.
    thoughts: Thoughts<'id>,
    /// The answer text part text chunks extend.
    text: Option<TextPart<'id>>,
    final_usage: Option<UsageMetadata>,
    final_finish_reason: Option<FinishReason>,
    final_finish_message: Option<String>,
    final_model_version: Option<String>,
    final_response_id: Option<String>,
    /// A provider finish reason was received, possibly before a hosted-tool round ended.
    saw_finish_reason: bool,
    /// At least one part mapped to assistant content.
    delivered: bool,
}

impl GenerateContentDecoder<'_> {
    /// A decoder for one reply.
    pub(super) fn new() -> Self {
        Self {
            thoughts: Thoughts::new(),
            text: None,
            final_usage: None,
            final_finish_reason: None,
            final_finish_message: None,
            final_model_version: None,
            final_response_id: None,
            saw_finish_reason: false,
            delivered: false,
        }
    }
}

impl<'id> Decoder<'id, Completion> for GenerateContentDecoder<'id> {
    type Event = GenerateContentResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<GenerateContentResponse> {
        // ID-only metadata must not create an unknown-content truncation tail.
        if GenerateContentDecoder::is_analysis_only(&frame) {
            return wire::classify_marker_keyed_frame(&frame.as_str(), &["responseId"]);
        }
        wire::classify_marker_keyed_frame(&frame.as_str(), RECOGNIZABLE_CHUNK_KEYS)
    }

    fn decode(
        &mut self,
        data: GenerateContentResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let span = tracing::Span::current();
        // The document defaults an absent id to the empty string, which is
        // the same "not reported" the normalized record's setter filters.
        if !data.response_id.is_empty() {
            span.record("gen_ai.response.id", data.response_id.as_str());
            self.final_response_id = Some(data.response_id.clone());
        }
        if let Some(model_version) = &data.model_version {
            span.record("gen_ai.response.model", model_version.as_str());
            self.final_model_version = Some(model_version.clone());
        }
        if let Some(usage) = data.usage_metadata.as_ref() {
            self.final_usage = Some(usage.clone());
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

        let Some(choice) = data.candidates.into_iter().next() else {
            tracing::debug!("There is no content candidate");
            return Ok(Flow::More);
        };

        if let Some(finish_reason) = &choice.finish_reason {
            // Last one wins: an intermediate `finishReason` is superseded by
            // the reason the turn actually ended on.
            self.saw_finish_reason = true;
            self.final_finish_reason = Some(finish_reason.clone());
        }
        if let Some(message) = &choice.finish_message {
            self.final_finish_message = Some(message.clone());
        }

        if let Some(err) = tool_protocol_finish_reason_error(&choice) {
            return Err(err);
        }

        match choice.content {
            Some(content) => {
                if content.parts.is_empty() {
                    tracing::trace!(reason = ?self.final_finish_reason, "There is no part in the streaming content");
                }
                // Parts within one document are distinct parts; text chunks
                // across stream events continue one part.
                let mut previous_text = false;
                for part in content.parts {
                    let text = matches!(part.part, PartKind::Text(_)) && part.thought != Some(true);
                    if text && previous_text {
                        self.close_text(&mut out);
                    }
                    previous_text = text;
                    self.interpret_part(part, &mut out)?;
                }
            }
            None => {
                // Gemini's final chunk may carry finishReason with no content.
                tracing::debug!(finish_reason = ?self.final_finish_reason, "Streaming candidate missing content");
            }
        }
        Ok(Flow::More)
    }

    /// Gemini ends a reply at EOF, not at its first finish reason: a
    /// hosted-tool round can report one before more content arrives.
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        // Without a provider finish reason, the reply did not end.
        if !self.saw_finish_reason {
            return Err(ProviderError::Truncated);
        }
        // An empty reply needs a truncating finish reason to explain it.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .and_then(map_finish_reason)
            .is_some_and(|reason| reason.truncated_output());
        if !self.delivered && !cut_short {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        self.close_text(&mut out);
        self.thoughts.close(&mut out, None);
        // Defaulting the raw usage shape does not imply reported usage.
        let usage = self
            .final_usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        let native = StreamingCompletionResponse {
            usage_metadata: self.final_usage.take().unwrap_or_default(),
            finish_reason: self.final_finish_reason.take(),
            finish_message: self.final_finish_message.take(),
            model_version: self.final_model_version.take(),
            response_id: self.final_response_id.take(),
        };
        out.raw(serde_json::to_value(&native)?);
        let finish_reason = native.finish_reason.as_ref().and_then(map_finish_reason);
        Ok(out.end(Finish {
            usage,
            reason: finish_reason,
            response_id: native.response_id,
            model: native.model_version,
            ..Finish::default()
        }))
    }
}

impl GenerateContentDecoder<'_> {
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

impl<'id> GenerateContentDecoder<'id> {
    fn close_text(&mut self, out: &mut Out<'id, Completion>) {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
    }

    fn interpret_part(
        &mut self,
        part: Part,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        // Hosted code-execution parts alone do not constitute an assistant answer.
        if !matches!(
            part.part,
            PartKind::ExecutableCode(_) | PartKind::CodeExecutionResult(_)
        ) {
            self.delivered = true;
        }
        match part {
            Part {
                part: PartKind::Text(text),
                thought: Some(true),
                thought_signature,
                ..
            } => {
                // A signature closes accumulated reasoning or forms a signature-only part.
                if !text.is_empty() {
                    self.close_text(out);
                }
                self.thoughts.fragment(out, &text);
                if let Some(signature) = thought_signature {
                    self.thoughts.signature(out, signature);
                }
            }
            Part {
                part: PartKind::Text(text),
                thought_signature,
                ..
            } => {
                // A signature on answer text belongs to that text part, and
                // must return on it: it rides on the part's extras. A
                // signature on its own empty part keeps its own part, and a
                // signed part ends there.
                let signed = thought_signature.is_some();
                if signed && text.is_empty() {
                    self.close_text(out);
                }
                let params = thought_signature.and_then(|signature| {
                    super::text_signature_extras(super::GEMINI_TEXT_EXTRAS_KEY, signature)
                });
                if !text.is_empty() || params.is_some() {
                    self.thoughts.boundary();
                    let part = self.text.get_or_insert_with(|| out.text());
                    out.push_text(part, &text);
                    if let Some(params) = params {
                        out.text_params(part, params);
                    }
                }
                if signed {
                    self.close_text(out);
                }
            }
            Part {
                part: PartKind::FunctionCall(function_call),
                thought_signature,
                ..
            } => {
                // Tool content interleaving an open thought part stops it.
                self.thoughts.boundary();
                self.close_text(out);
                shared_parts::function_call(
                    out,
                    function_call.name,
                    function_call.args,
                    function_call.id,
                    thought_signature,
                )?;
            }
            Part {
                part: part @ PartKind::InlineData(_),
                thought_signature,
                ..
            } => {
                // Inline media rides on a text part's metadata: the choice
                // has no part for it.
                self.thoughts.boundary();
                self.close_text(out);
                let raw = Part {
                    thought: None,
                    thought_signature,
                    part,
                    additional_params: None,
                };
                if let Some(params) = crate::message::AdditionalParams::from_entries([(
                    super::GEMINI_RAW_CONTENT_KEY,
                    serde_json::json!(raw),
                )]) {
                    let part = out.text();
                    out.text_params(&part, params);
                    out.close_text(part);
                }
            }
            Part {
                part: part @ (PartKind::ExecutableCode(_) | PartKind::CodeExecutionResult(_)),
                ..
            } => {
                // Log only the unmodeled part's kind; hosted-tool payloads may be sensitive.
                crate::driver::warn_unmodeled("gemini_part", &part_kind_name(&part));
            }
            Part { part, .. } => {
                // Unexpected response parts must fail rather than silently discard content.
                return Err(ProviderError::Response(format!(
                    "Gemini response part kind {} carries no assistant content rig can account for",
                    part_kind_name(&part)
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
