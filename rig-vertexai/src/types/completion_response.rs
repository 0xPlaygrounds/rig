use google_cloud_aiplatform_v1 as vertexai;
use rig::OneOrMany;
use rig::completion::{CompletionError, CompletionResponse, Usage};
use rig::message::{AssistantContent, Text, ToolCall, ToolFunction};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct VertexGenerateContentOutput(pub vertexai::model::GenerateContentResponse);

impl TryFrom<VertexGenerateContentOutput> for CompletionResponse<VertexGenerateContentOutput> {
    type Error = CompletionError;

    fn try_from(value: VertexGenerateContentOutput) -> Result<Self, Self::Error> {
        let response = &value.0;

        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ProviderError("No candidates in response".to_string())
        })?;

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| CompletionError::ProviderError("No content in candidate".to_string()))?;

        let mut assistant_contents = Vec::new();

        // vertexai internally uses a wkt::Struct (serde_json::Map<String, serde_json::Value>) in
        // function calling args. We need to convert that to serde_json::Value for rig::completion type matching
        for part in content.parts.iter() {
            if let Some(function_call) = part.function_call() {
                let args_json = function_call
                    .args
                    .as_ref()
                    .map(|s| serde_json::Value::Object(s.clone()))
                    .unwrap_or_else(|| serde_json::json!({}));

                assistant_contents.push(AssistantContent::ToolCall(ToolCall {
                    id: function_call.name.clone(),
                    call_id: None,
                    function: ToolFunction {
                        name: function_call.name.clone(),
                        arguments: args_json,
                    },
                }));
            } else if let Some(text) = part.text() {
                assistant_contents.push(AssistantContent::Text(Text { text: text.clone() }));
            }
        }

        if assistant_contents.is_empty() {
            return Err(CompletionError::ProviderError(
                "No text or tool call content found in response".to_string(),
            ));
        }

        let choice = OneOrMany::many(assistant_contents).map_err(|e| {
            CompletionError::ProviderError(format!("Failed to create OneOrMany: {e}"))
        })?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            choice,
            usage,
            raw_response: value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use google_cloud_aiplatform_v1 as vertexai;
    use rig::OneOrMany;
    use rig::message::{AssistantContent, Text, ToolCall};

    fn create_text_response(text: &str) -> VertexGenerateContentOutput {
        let part = vertexai::model::Part::new().set_text(text.to_string());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new()
            .set_content(content)
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    fn create_tool_call_response(
        function_name: &str,
        args: serde_json::Value,
    ) -> VertexGenerateContentOutput {
        let struct_args = match args {
            serde_json::Value::Object(map) => map,
            _ => panic!("Expected JSON object for Struct conversion"),
        };
        let function_call = vertexai::model::FunctionCall::new()
            .set_name(function_name.to_string())
            .set_args(struct_args);
        let part = vertexai::model::Part::new().set_function_call(function_call);
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new()
            .set_content(content)
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    #[test]
    fn test_text_response_conversion() {
        let vertex_output = create_text_response("Hello, world!");
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();
        assert_eq!(
            response.choice,
            OneOrMany::one(AssistantContent::Text(Text {
                text: "Hello, world!".to_string()
            }))
        );
    }

    #[test]
    fn test_tool_call_response_conversion() {
        let args = serde_json::json!({
            "x": 5,
            "y": 3
        });
        let vertex_output = create_tool_call_response("add", args.clone());
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();

        match response.choice.first() {
            AssistantContent::ToolCall(ToolCall { id, function, .. }) => {
                assert_eq!(id, "add");
                assert_eq!(function.name, "add");
                assert_eq!(function.arguments, args);
            }
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_usage_metadata_conversion() {
        let mut response = create_text_response("test").0;
        let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
            .set_prompt_token_count(10)
            .set_candidates_token_count(20)
            .set_total_token_count(30);
        response = response.set_usage_metadata(usage_metadata);

        let vertex_output = VertexGenerateContentOutput(response);
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.total_tokens, 30);
    }

    #[test]
    fn test_empty_response_error() {
        // Create a response with no candidates
        let response = vertexai::model::GenerateContentResponse::new();
        let vertex_output = VertexGenerateContentOutput(response);
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_err());
    }
}
