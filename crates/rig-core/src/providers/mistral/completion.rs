use serde::{Deserialize, Deserializer, Serialize};

use super::client::{MistralExt, Usage};
use crate::providers::internal::openai_chat_completions_compatible::map_openai_finish_reason;
use crate::providers::openai;
use crate::{
    completion::{self, CompletionError},
    json_utils,
};

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `pixtral-large` Mistral multimodal model
pub const PIXTRAL_LARGE: &str = "pixtral-large-latest";
/// The latest version of the `mistral` Mistral multimodal model, trained on datasets from the Middle East & South Asia
pub const MISTRAL_SABA: &str = "mistral-saba-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";
/// The `24-09` version of the `pixtral-small` Mistral multimodal model
pub const PIXTRAL_SMALL: &str = "pixtral-12b-2409";
/// The `open-mistral-nemo` model
pub const MISTRAL_NEMO: &str = "open-mistral-nemo";
/// The `open-mistral-mamba` model
pub const CODESTRAL_MAMBA: &str = "open-codestral-mamba";

/// Mistral completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<MistralExt, H>;

/// Mistral's provider-native terminal streaming record: the value carried by
/// the final item of the stream returned by `CompletionModel::raw_stream`.
/// Shared with the OpenAI Chat Completions path but carrying Mistral's own
/// usage payload (cached-token fallbacks).
pub type MistralStreamingCompletionResponse =
    openai::StreamingCompletionResponse<super::client::Usage>;

// =================================================================
// Rig Implementation Types
// =================================================================

fn mistral_content_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => openai::completion::joined_text_parts(&parts),
        _ => String::new(),
    }
}

fn deserialize_mistral_content_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<serde_json::Value>::deserialize(deserializer)?
        .map(mistral_content_value_to_text)
        .unwrap_or_default())
}

/// Content chunk type Mistral carries for images, used by the Pixtral models.
const IMAGE_CHUNK_TYPE: &str = "image_url";
/// Content chunk type Mistral carries for prompt audio, used by the
/// audio-input models that report `prompt_audio_seconds` usage.
const AUDIO_CHUNK_TYPE: &str = "input_audio";

/// How one serialized content part of a request message relates to Mistral's
/// message content schema.
enum RequestContentPart {
    /// Text-bearing part that folds into Mistral's plain-string `content`.
    Text,
    /// Image part whose OpenAI-compatible shape is also valid for Mistral.
    Image,
    /// Audio part that needs conversion to Mistral's string payload.
    Audio,
}

/// Classify a serialized OpenAI-compatible content part for Mistral's request
/// schema, rejecting parts Mistral has no chunk for.
fn classify_request_content_part(
    part: &serde_json::Value,
) -> Result<RequestContentPart, CompletionError> {
    // Textuality is decided per key, matching the shared flattening helper
    // this guards, so the two cannot disagree about what counts as text.
    if part
        .get("text")
        .and_then(serde_json::Value::as_str)
        .is_some()
        || part
            .get("refusal")
            .and_then(serde_json::Value::as_str)
            .is_some()
    {
        return Ok(RequestContentPart::Text);
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(IMAGE_CHUNK_TYPE) => Ok(RequestContentPart::Image),
        Some(AUDIO_CHUNK_TYPE) => Ok(RequestContentPart::Audio),
        kind => Err(crate::message::MessageError::ConversionError(format!(
            "Mistral does not support `{}` message content; Mistral messages carry text, \
             `{IMAGE_CHUNK_TYPE}`, and `{AUDIO_CHUNK_TYPE}` chunks only. Convert the content to \
             text before sending it.",
            kind.unwrap_or("untyped"),
        ))
        .into()),
    }
}

/// Normalize one serialized request message `content` value for Mistral.
///
/// Mistral takes text-only content as a plain string, so text parts are joined
/// into one. The two chunk types it shares with the OpenAI-compatible
/// conversion stay in the array, each rendered the way Mistral's own schema
/// names it: `image_url` forwards untouched, because `ImageURLChunk` accepts an
/// `ImageURL` of `url` plus optional `detail`; `input_audio` collapses to its
/// base64 payload, because `AudioChunk` carries that string directly and has no
/// slot for a sibling `format` — the same normalization `mistral-common`
/// applies to OpenAI-shaped `{data, format}` input. Restoring the object form
/// there would build a body Mistral cannot parse.
///
/// Content Mistral cannot represent — documents converted to OpenAI `file`
/// parts, video — fails here instead of being flattened away, which used to
/// return an ordinary completion built from a request the caller never made.
pub(super) fn normalize_request_content(
    content: &mut serde_json::Value,
) -> Result<(), CompletionError> {
    let Some(parts) = content.as_array_mut() else {
        return Ok(());
    };

    let mut has_chunk = false;
    for part in parts.iter_mut() {
        match classify_request_content_part(part)? {
            RequestContentPart::Text => {}
            RequestContentPart::Image => has_chunk = true,
            RequestContentPart::Audio => {
                // Rig first serializes through the OpenAI-compatible shape:
                // `input_audio: { data, format }`. Mistral's native AudioChunk
                // instead requires `input_audio` to be the base64/URL string
                // itself, so translate rather than forwarding an invalid body.
                let input_audio = part.get_mut(AUDIO_CHUNK_TYPE).ok_or_else(|| {
                    crate::message::MessageError::ConversionError(
                        "Mistral `input_audio` content is missing its payload".to_string(),
                    )
                })?;
                if let Some(data) = input_audio
                    .get("data")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                {
                    *input_audio = serde_json::Value::String(data);
                } else if !input_audio.is_string() {
                    return Err(crate::message::MessageError::ConversionError(
                        "Mistral `input_audio` content must contain base64 data or a URL"
                            .to_string(),
                    )
                    .into());
                }
                has_chunk = true;
            }
        }
    }

    if !has_chunk {
        openai::completion::flatten_text_content_parts(content, "", false);
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// Mistral's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        #[serde(default, deserialize_with = "deserialize_mistral_content_string")]
        content: String,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        prefix: bool,
    },
    System {
        content: String,
    },
    Tool {
        /// The name of the tool that was called
        #[serde(skip_serializing_if = "String::is_empty")]
        name: String,
        /// The content of the tool call
        content: String,
        /// The id of the tool call
        tool_call_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type OutputMessage = Choice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.clone())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        let res = self
            .choices
            .iter()
            .filter_map(|choice| match choice.message {
                Message::Assistant { ref content, .. } => {
                    if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    }
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

/// Normalize a Mistral chat completion response.
///
/// The provider descriptor name is an *input* rather than a constant so the
/// shared OpenAI-compatible completion path labels the response with the
/// descriptor that actually produced it.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        let response = self;
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let finish_reason = Some(choice.finish_reason.as_str())
            .filter(|reason| !reason.is_empty())
            .map(map_openai_finish_reason);

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = if content.is_empty() {
                    vec![]
                } else {
                    vec![completion::AssistantContent::text(content.clone())]
                };

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = crate::message::require_non_empty_response(content)?;

        let usage = response
            .usage
            .as_ref()
            .map(completion::Usage::from)
            .unwrap_or_default();

        Ok(completion::CompletionResponse::new(choice, usage, provider)
            .with_response_id(response.id.as_str())
            .with_model(response.model.as_str())
            .with_optional_finish_reason(finish_reason))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::{CompletionModel as _, CompletionRequestBuilder};
    use crate::message;
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };
    use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

    /// Convert a Rig request the way Mistral's completion path does and
    /// finalize the serialized body, so the assertions see exactly what the
    /// provider would put on the wire — or the error raised in its place.
    fn finalized_body(
        request: crate::completion::CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: MISTRAL_SMALL.to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: MistralExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: MistralExt::SUPPORTS_TOOLS,
        })?;
        let mut body = serde_json::to_value(request)?;
        MistralExt.finalize_request_body(&mut body)?;
        Ok(body)
    }

    /// A one-turn request whose user message carries `content`.
    fn user_request(content: Vec<message::UserContent>) -> crate::completion::CompletionRequest {
        CompletionRequestBuilder::new(
            MockCompletionModel::default(),
            message::Message::User { content },
        )
        .build()
    }

    fn pdf_document() -> message::UserContent {
        message::UserContent::Document(message::Document {
            data: message::DocumentSourceKind::Base64("JVBERi0xLjQK".to_string()),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        })
    }

    fn mistral_client(
        http_client: RecordingHttpClient,
    ) -> crate::providers::mistral::Client<RecordingHttpClient> {
        crate::providers::mistral::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("client should build")
    }

    #[test]
    fn deserializes_response_with_array_and_null_content() {
        let data = r#"{
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-small-latest",
            "system_fingerprint": null,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{\"x\":1,\"y\":2}"}
                        }]
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(data).expect("response should deserialize");
        match &response.choices[0].message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello world"),
            _ => panic!("expected assistant message"),
        }
        match &response.choices[1].message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content, "");
                assert_eq!(tool_calls[0].function.name, "add");
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn usage_prefers_structured_cached_tokens_and_falls_back() {
        let structured: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "num_cached_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 7}
        }))
        .expect("usage should deserialize");
        assert_eq!(structured.cached_tokens(), 7);

        let fallback: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "num_cached_tokens": 2
        }))
        .expect("usage should deserialize");
        assert_eq!(fallback.cached_tokens(), 2);

        // The singular alias form used by some Mistral responses.
        let aliased: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_token_details": {"cached_tokens": 4}
        }))
        .expect("usage should deserialize");
        assert_eq!(aliased.cached_tokens(), 4);
    }

    #[test]
    fn finalize_rewrites_required_tool_choice_to_any() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required"
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(body["tool_choice"], "any");
    }

    #[test]
    fn finalize_preserves_specific_function_tool_choice() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "beta"}}
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "beta"}})
        );
    }

    #[test]
    fn finalize_flattens_assistant_history_and_adds_prefix() {
        let mut body = serde_json::json!({
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello."}],
                    "reasoning_content": "hidden thoughts"
                },
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{}"}
                    }]
                }
            ]
        });

        MistralExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(body["messages"][0]["content"], "Be brief.");
        assert_eq!(body["messages"][2]["content"], "Hello.");
        assert_eq!(body["messages"][2]["prefix"], false);
        assert!(
            body["messages"][2].get("reasoning_content").is_none(),
            "Mistral rejects unknown assistant fields; reasoning must be stripped"
        );
        assert_eq!(body["messages"][3]["content"], "");
        assert_eq!(body["messages"][3]["prefix"], false);
    }

    /// Text-only requests keep reaching Mistral as plain-string content, which
    /// is what the flattening this change guards exists for.
    #[test]
    fn finalize_flattens_text_only_content_including_text_documents() {
        let mut request = user_request(vec![
            message::UserContent::text("First."),
            message::UserContent::text("Second."),
            message::UserContent::Document(message::Document {
                data: message::DocumentSourceKind::String("# Notes".to_string()),
                media_type: None,
                additional_params: None,
            }),
        ]);
        request.preamble = Some("Be brief.".to_string());

        let body = finalized_body(request).expect("text-only content should finalize");

        assert_eq!(body["messages"][0]["content"], "Be brief.");
        assert_eq!(body["messages"][1]["content"], "First.Second.# Notes");
    }

    /// Mistral's Pixtral models take `image_url` chunks, so an image must reach
    /// the wire instead of being flattened out of the message.
    #[test]
    fn finalize_keeps_image_chunks_alongside_text() {
        let body = finalized_body(user_request(vec![
            message::UserContent::text("What is in this picture?"),
            message::UserContent::image_url("https://example.com/cat.png", None, None),
            message::UserContent::image_base64(
                "iVBORw0KGgo=",
                Some(message::ImageMediaType::PNG),
                None,
            ),
        ]))
        .expect("image content should finalize");

        let content = body["messages"][0]["content"]
            .as_array()
            .expect("image content must stay an array of chunks");
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["text"], "What is in this picture?");
        assert_eq!(
            content[1],
            serde_json::json!({
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png", "detail": "auto"}
            }),
            "Mistral's `ImageURLChunk` takes an `ImageURL` of `url` plus optional \
             `detail`, so the nested pair must survive finalization unreshaped"
        );
        assert_eq!(content[2]["type"], "image_url");
        assert_eq!(
            content[2]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
    }

    /// Mistral's audio-input models take `input_audio` chunks — the reason its
    /// usage payload reports `prompt_audio_seconds` at all.
    #[test]
    fn finalize_keeps_audio_chunks_alongside_text() {
        let body = finalized_body(user_request(vec![
            message::UserContent::text("Transcribe this."),
            message::UserContent::audio("SUQzBAA=", Some(message::AudioMediaType::MP3)),
        ]))
        .expect("audio content should finalize");

        let content = body["messages"][0]["content"]
            .as_array()
            .expect("audio content must stay an array of chunks");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Transcribe this.");
        assert_eq!(content[1]["type"], "input_audio");
        assert_eq!(content[1]["input_audio"], "SUQzBAA=");
    }

    /// Documents convert to OpenAI's `file` content part, which has no Mistral
    /// equivalent: the request must fail rather than complete without the
    /// document. Not a cassette test — the point of the fix is that no request
    /// is built, so there is no traffic to record.
    #[test]
    fn finalize_rejects_document_content_instead_of_dropping_it() {
        let error = finalized_body(user_request(vec![pdf_document()]))
            .expect_err("a PDF document must not be dropped from the request");

        assert!(matches!(error, CompletionError::RequestError(_)));
        let rendered = error.to_string();
        assert!(rendered.contains("`file`"), "{rendered}");
        assert!(rendered.contains("Mistral"), "{rendered}");

        let error = finalized_body(user_request(vec![message::UserContent::Document(
            message::Document {
                data: message::DocumentSourceKind::FileId("file_abc".to_string()),
                media_type: None,
                additional_params: None,
            },
        )]))
        .expect_err("a file-id document must not be dropped from the request");
        assert!(matches!(error, CompletionError::RequestError(_)));
    }

    /// The text of a mixed message is no consolation prize: dropping only the
    /// unrepresentable part would still answer a prompt the caller never sent.
    #[test]
    fn finalize_rejects_mixed_text_and_document_content() {
        let error = finalized_body(user_request(vec![
            message::UserContent::text("Summarize the attached report."),
            pdf_document(),
        ]))
        .expect_err("mixed text and document content must not silently lose the document");

        assert!(matches!(error, CompletionError::RequestError(_)));
    }

    /// Video has no Mistral chunk either, and an unknown part type must fail
    /// closed so a future content type cannot start disappearing silently.
    #[test]
    fn finalize_rejects_video_and_unrecognized_chunks() {
        let error = finalized_body(user_request(vec![message::UserContent::video(
            "AAAAIGZ0eXA=",
            Some(message::VideoMediaType::MP4),
        )]))
        .expect_err("video content must not be dropped from the request");
        assert!(matches!(error, CompletionError::RequestError(_)));

        let mut body = serde_json::json!({
            "model": MISTRAL_SMALL,
            "messages": [{
                "role": "user",
                "content": [{"type": "document_url", "document_url": "https://example.com/x.pdf"}]
            }]
        });
        let error = MistralExt
            .finalize_request_body(&mut body)
            .expect_err("unrecognized chunks must not be dropped from the request");
        assert!(matches!(error, CompletionError::RequestError(_)));
    }

    /// End-to-end guard for the reported symptom: the caller used to get an
    /// ordinary completion built from a request its content never reached.
    #[tokio::test]
    async fn completion_rejects_document_content_before_sending() {
        let http_client = RecordingHttpClient::new("{}");
        let model = mistral_client(http_client.clone()).completion_model(MISTRAL_SMALL);

        let error = model
            .completion(user_request(vec![
                message::UserContent::text("Summarize the attached report."),
                pdf_document(),
            ]))
            .await
            .expect_err("completion must fail instead of dropping the document");

        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(
            http_client.requests().is_empty(),
            "no request may be sent once content cannot be represented"
        );
    }

    /// Streaming shares the request finalization, so it must fail the same way,
    /// and a rejected turn never becomes a stream that could be sent.
    ///
    /// The captured-request list cannot witness that: `RecordingHttpClient`
    /// answers `send_streaming` without recording, so `requests()` stays empty
    /// either way. The error's position is the discriminator instead — content
    /// Mistral can carry returns a stream handle here and only fails once the
    /// absent transport is polled — which is what the control below pins, so
    /// the document's `Err` can only have come from finalization.
    #[tokio::test]
    async fn streaming_rejects_document_content_before_sending() {
        let http_client = RecordingHttpClient::new("{}");
        let model = mistral_client(http_client.clone()).completion_model(MISTRAL_SMALL);

        let error = model
            .stream(user_request(vec![pdf_document()]))
            .await
            .err()
            .expect("streaming must fail instead of dropping the document");

        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(
            model
                .stream(user_request(vec![message::UserContent::text("Hello!")]))
                .await
                .is_ok(),
            "representable content must still open a stream, or the rejection above proves nothing"
        );
    }

    /// The rejection is narrow: an ordinary text prompt still completes, and
    /// still leaves Rig as Mistral's plain-string `content`.
    #[tokio::test]
    async fn completion_sends_text_only_content_as_a_plain_string() {
        let response = r#"{
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-small-latest",
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there."},
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;
        let http_client = RecordingHttpClient::new(response);
        let model = mistral_client(http_client.clone()).completion_model(MISTRAL_SMALL);

        let completion = model
            .completion(user_request(vec![message::UserContent::text("Hello!")]))
            .await
            .expect("text-only completion should succeed");

        assert_eq!(completion.model.as_deref(), Some(MISTRAL_SMALL));
        let sent: serde_json::Value = serde_json::from_slice(&http_client.requests()[0].body)
            .expect("request body should be JSON");
        assert_eq!(sent["messages"][0]["content"], "Hello!");
    }
}
