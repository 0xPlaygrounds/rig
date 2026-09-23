use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::{
    ContentCandidate, FinishReason, GenerateContentResponse, Part, PartKind, UsageMetadata,
    map_finish_reason,
};
use super::completion::{
    PROVIDER_NAME, blocked_prompt_error, function_call_finish_reason_error, part_kind_name,
};
use crate::error::ProviderError;
use crate::observe::ObservedError;
use crate::operation::{AdapterOutput, Completion};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Mode, ObservationSink, Output, WireFrame,
};

/// Part-kind interpretation shared by the Gemini wires whose payloads
/// coincide: REST `streamGenerateContent` and the Interactions API both
/// deliver whole function calls and identity-less thought fragments.
pub(crate) mod shared_parts {
    use serde_json::Value;

    use crate::streaming::{BlockClose, BlockId, BlockKind, StreamEvent, ToolCallEnd};

    /// Convert a whole function call into canonical start and end events.
    pub(crate) fn function_call(
        name: String,
        args: Value,
        wire_id: Option<String>,
        signature: Option<String>,
        tool_ids: &mut crate::streaming::SyntheticIds,
    ) -> Vec<StreamEvent> {
        // Id-less calls need distinct local block keys, but must replay without
        // fabricated provider identifiers, even when they share a tool name.
        let tool_id = wire_id.and_then(crate::streaming::non_empty_id);
        let id = tool_id
            .as_ref()
            .map_or_else(|| tool_ids.mint(), |id| BlockId::wire(id.as_str()));
        let mut end = ToolCallEnd::whole(name, args).with_signature(signature);
        // Setting call_id as well would invent a second provider identity.
        end.tool_id = tool_id;
        vec![
            StreamEvent::BlockStart {
                id: id.clone(),
                kind: BlockKind::ToolCall,
            },
            StreamEvent::BlockEnd {
                id,
                end: BlockClose::ToolCall(end),
                block: None,
            },
        ]
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

/// Decode unary and streamed GenerateContent replies into canonical events.
/// Hold terminal metadata until EOF because hosted-tool rounds can report
/// intermediate finish reasons. Unary replies without assistant content fail
/// unless a truncating finish reason permits empty output.
pub struct GenerateContentDecoder {
    /// Thought boundaries inferred from content transitions and signatures.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// Distinct local block keys for calls without provider identifiers.
    tool_ids: crate::streaming::SyntheticIds,
    /// Per-reply minter for the raw-content blocks a part the stream
    /// vocabulary cannot express rides on (see `GEMINI_RAW_CONTENT_KEY`).
    raw_ids: crate::streaming::SyntheticIds,
    final_usage: Option<UsageMetadata>,
    final_finish_reason: Option<FinishReason>,
    final_finish_message: Option<String>,
    final_model_version: Option<String>,
    final_response_id: Option<String>,
    /// A provider finish reason was received, possibly before a hosted-tool round ended.
    saw_finish_reason: bool,
    /// At least one part mapped to assistant content.
    delivered: bool,
    /// Unary mode, where EOF completes the whole response document.
    whole: bool,
    /// A terminal error was emitted; subsequent frames must not produce output.
    failed: bool,
}

impl GenerateContentDecoder {
    /// A decoder for one reply read in `mode`.
    pub(super) fn new(mode: Mode) -> Self {
        Self {
            reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                crate::streaming::MintKind::Reasoning,
            ),
            tool_ids: crate::streaming::SyntheticIds::tool(),
            raw_ids: crate::streaming::SyntheticIds::new(crate::streaming::MintKind::Block),
            final_usage: None,
            final_finish_reason: None,
            final_finish_message: None,
            final_model_version: None,
            final_response_id: None,
            saw_finish_reason: false,
            delivered: false,
            whole: mode == Mode::Unary,
            failed: false,
        }
    }
}

impl Decoder<Completion> for GenerateContentDecoder {
    type Event = GenerateContentResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<GenerateContentResponse> {
        // ID-only metadata must not create an unknown-content truncation tail.
        if <Self as Decoder<Completion>>::is_analysis_only(self, &frame) {
            return wire::classify_marker_keyed_frame(&frame.as_str(), &["responseId"]);
        }
        wire::classify_marker_keyed_frame(&frame.as_str(), RECOGNIZABLE_CHUNK_KEYS)
    }

    fn is_analysis_only(&self, frame: &WireFrame) -> bool {
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

    fn interpret(&mut self, data: GenerateContentResponse, out: &mut Output<Completion>) {
        if self.failed {
            return;
        }

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
            // Carried for the terminal record only: the driver records usage
            // off the folded response, so the decoder states it once.
            self.final_usage = Some(usage.clone());
        }

        if let Some(error) = data.error {
            // Preserve in-band failures as provider errors, not unknown-frame truncation.
            self.failed = true;
            // GenerateContent codes are HTTP statuses; only error statuses
            // participate in the unary retry policy.
            let status = error
                .get("code")
                .and_then(serde_json::Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .and_then(|code| http::StatusCode::from_u16(code).ok())
                .filter(|status| status.is_client_error() || status.is_server_error());
            let body = serde_json::json!({ "error": error }).to_string();
            let error = match status {
                Some(status) => ProviderError::from_http_response(status, body),
                None => crate::error::ProviderError::from_provider_body(body),
            };
            out.push(Err(error));
            return;
        }

        if let Some(blocked) = data.prompt_feedback.as_ref().and_then(blocked_prompt_error) {
            // Preserve the refusal reason rather than reporting an unexplained truncation.
            self.failed = true;
            out.push(Err(blocked));
            return;
        }

        let Some(choice) = data.candidates.into_iter().next() else {
            tracing::debug!("There is no content candidate");
            return;
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
            self.failed = true;
            out.push(Err(err));
            return;
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
                        out.end_active_text();
                    }
                    previous_text = text;
                    self.interpret_part(part, out);
                }
            }
            None => {
                // Gemini's final chunk may carry finishReason with no content.
                tracing::debug!(finish_reason = ?self.final_finish_reason, "Streaming candidate missing content");
            }
        }
    }

    fn finish(&mut self, out: &mut Output<Completion>) {
        // Empty unary replies require a truncating finish reason; stream truncation
        // is represented by an absent terminal record instead.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .and_then(map_finish_reason)
            .is_some_and(|reason| reason.truncated_output());
        if self.whole && !self.delivered && !cut_short {
            out.error(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
            return;
        }

        // Without a provider finish reason, a terminal would falsely report completion.
        if !self.saw_finish_reason {
            return;
        }

        // Deferred completion retains the final hosted-tool round and its metadata.
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
        let raw = match serde_json::to_value(&native) {
            Ok(raw) => raw,
            Err(err) => {
                out.error(err.into());
                return;
            }
        };
        let finish_reason = native.finish_reason.as_ref().and_then(map_finish_reason);
        out.final_record(
            streaming::StreamFinal::new(PROVIDER_NAME, usage, raw)
                .with_optional_finish_reason(finish_reason)
                .with_optional_response_id(native.response_id)
                .with_optional_model(native.model_version),
        );
    }

    fn is_finished(&self) -> bool {
        // Stop after terminal errors so later unknown frames cannot escape the failure gate.
        self.failed
    }

    /// Project provider verdicts, usage, response identity, and errors before normalization.
    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
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

impl GenerateContentDecoder {
    fn interpret_part(&mut self, part: Part, out: &mut AdapterOutput) {
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
                // A signature closes accumulated reasoning or forms a signature-only block.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: Some(text),
                        reasoning_signature: thought_signature,
                        text: None,
                        text_meta: None,
                        tool_events: Vec::new(),
                    },
                    out,
                );
            }
            Part {
                part: PartKind::Text(text),
                thought_signature,
                ..
            } => {
                // A signature on answer text belongs to that text part, and
                // must return on it: it rides on the text block's extras. A
                // signature on its own empty part keeps its own part, and a
                // signed part ends there.
                let signed = thought_signature.is_some();
                if signed && text.is_empty() {
                    out.end_active_text();
                }
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: Some(text),
                        text_meta: thought_signature.and_then(|signature| {
                            super::text_signature_extras(super::GEMINI_TEXT_EXTRAS_KEY, signature)
                        }),
                        tool_events: Vec::new(),
                    },
                    out,
                );
                if signed {
                    out.end_active_text();
                }
            }
            Part {
                part: PartKind::FunctionCall(function_call),
                thought_signature,
                ..
            } => {
                // Tool content interleaving an open thought block: the
                // shared lifecycle synthesizes the boundary end.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: None,
                        text_meta: None,
                        tool_events: shared_parts::function_call(
                            function_call.name,
                            function_call.args,
                            function_call.id,
                            thought_signature,
                            &mut self.tool_ids,
                        ),
                    },
                    out,
                );
            }
            Part {
                part: part @ PartKind::InlineData(_),
                thought_signature,
                ..
            } => {
                // Preserve inline media in metadata because canonical stream blocks
                // cannot represent image content.
                let raw = Part {
                    thought: None,
                    thought_signature,
                    part,
                    additional_params: None,
                };
                let events = match crate::message::AdditionalParams::from_entries([(
                    super::GEMINI_RAW_CONTENT_KEY,
                    serde_json::json!(raw),
                )]) {
                    Some(params) => {
                        let id = self.raw_ids.mint();
                        vec![
                            streaming::StreamEvent::BlockStart {
                                id: id.clone(),
                                kind: streaming::BlockKind::Text {
                                    additional_params: Some(params),
                                },
                            },
                            streaming::StreamEvent::BlockEnd {
                                id,
                                end: streaming::BlockClose::Text,
                                block: None,
                            },
                        ]
                    }
                    None => Vec::new(),
                };
                // Raw media closes any open thought block before its own events.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: None,
                        text_meta: None,
                        tool_events: events,
                    },
                    out,
                );
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
                out.error(ProviderError::Response(format!(
                    "Gemini response part kind {} carries no assistant content rig can account for",
                    part_kind_name(&part)
                )));
                self.failed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests;
