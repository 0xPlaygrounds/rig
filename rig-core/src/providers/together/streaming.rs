use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::{CompletionError, CompletionRequest}, json_utils::{self, merge}, providers::openai::{ToolCall, ToolType}, streaming::{self, StreamingCompletionModel, StreamingResult}
};

use super::completion::CompletionModel;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingToolCall {
    #[serde(default)]
    pub r#type: ToolType,
    pub function: StreamingFunction,
}

#[derive(Deserialize)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(deserialize_with="json_utils::null_or_vec")]
    tool_calls: Vec<ToolCall>
}

#[derive(Deserialize)]
struct StreamingChoice {
    #[serde(default)]
    text: Option<String>,
    delta: StreamingDelta
}

#[derive(Deserialize)]
struct StreamingCompletionResponse {
    choices: Vec<StreamingChoice>
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_request_body(completion_request)?;

        request = merge(request, json!({"stream_tokens": true}));

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(&request)
            .send()
            .await?;

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
                    // println!("{:?}", line);
                    let Some(line) = line.strip_prefix("data: ") else { continue; };

                    let data = serde_json::from_str::<StreamingCompletionResponse>(line);
                    
                    let Ok(data) = data else {
                        println!("{}", line);
                        println!("{}", data.err().unwrap());
                        continue;   
                    };

                    let choice = data.choices.first().expect("Should have at least one choice");

                    if let Some(text) = &choice.text {
                        yield Ok(streaming::StreamingChoice::Message(text.clone()))
                    }

                    let delta = &choice.delta;

                    if delta.tool_calls.len() > 0 {
                        for tool_call in &delta.tool_calls {
                            let function = tool_call.function.clone();

                            yield Ok(streaming::StreamingChoice::ToolCall(function.name, "".to_string(), function.arguments))
                        }
                    }
                    
                    if let Some(content) = &choice.delta.content {
                        yield Ok(streaming::StreamingChoice::Message(content.clone()))
                    }
                }
            }
        }))
    }
}
