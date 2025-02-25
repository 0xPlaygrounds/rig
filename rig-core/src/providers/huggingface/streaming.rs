use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge_inplace;
use crate::streaming::{StreamingCompletionModel, StreamingResult};
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::str::FromStr;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase", tag = "type")]
/// Represents the content sent back in the StreamDelta for an Assistant
enum AssistantContent {
    Text { text: String },
}

// Ensure that string contents can be serialized correctly
impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase", tag = "role")]
enum StreamDelta {
    Assistant {
        #[serde(deserialize_with = "json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
    },
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct StreamingChoice {
    index: usize,
    delta: StreamDelta,
    logprobs: Value,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct CompletionChunk {
    id: String,
    created: i32,
    model: String,
    #[serde(default)]
    system_fingerprint: String,
    choices: Vec<StreamingChoice>,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_request_body(&completion_request)?;

        // Enable streaming
        merge_inplace(&mut request, json!({"stream": true}));

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut request, params.clone());
        }

        // HF Inference API uses the model in the path even though its specified in the request body
        let path = self.client.sub_provider.completion_endpoint(&self.model);

        let response = self.client.post(&path).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        Ok(Box::pin(stream! {
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(CompletionError::from(e));
                        break;
                    }
                };

                let text = match String::from_utf8(chunk.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        yield Err(CompletionError::ResponseError(e.to_string()));
                        break;
                    }
                };


                for line in text.lines() {
                    let Some(line) = line.strip_prefix("data: ") else { continue; };

                    if line == "[DONE]" {
                        break;
                    }

                    let Ok(data) = serde_json::from_str::<CompletionChunk>(line) else {
                        continue;
                    };

                    let choice = data.choices.first().expect("Should have at least one choice");

                    match &choice.delta {
                        StreamDelta::Assistant { content, .. } => match &content[0] {
                            AssistantContent::Text { text } => yield Ok(streaming::StreamingChoice::Message(text.clone())),
                        }
                    }
                }
            }
        }))
    }
}
