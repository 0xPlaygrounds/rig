use serde::{Deserialize, Deserializer, Serialize};

use super::client::{MistralExt, Usage};
use crate::completion::GetTokenUsage;
use crate::providers::openai;
use crate::{
    OneOrMany,
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

/// Final streaming response, shared with the OpenAI Chat Completions path.
pub type MistralStreamingCompletionResponse = openai::StreamingCompletionResponse;

// =================================================================
// Rig Implementation Types
// =================================================================

fn mistral_content_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => parts
            .into_iter()
            .filter_map(|part| {
                (part.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                    .then(|| part.get("text").and_then(serde_json::Value::as_str))
                    .flatten()
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>()
            .join(""),
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

impl Message {
    pub fn user(content: String) -> Self {
        Message::User { content }
    }

    pub fn assistant(content: String, tool_calls: Vec<ToolCall>, prefix: bool) -> Self {
        Message::Assistant {
            content,
            tool_calls,
            prefix,
        }
    }

    pub fn system(content: String) -> Self {
        Message::System { content }
    }
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

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = self.completion_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage.cached_input_tokens = self.cached_tokens();
        usage
    }
}

impl GetTokenUsage for CompletionResponse {
    fn token_usage(&self) -> crate::completion::Usage {
        self.usage
            .as_ref()
            .map(GetTokenUsage::token_usage)
            .unwrap_or_default()
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
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

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: usage.completion_tokens as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage.cached_tokens(),
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();

        let message_id = response.id.clone();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: Some(message_id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::completion::OpenAICompatibleProvider;

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
}
