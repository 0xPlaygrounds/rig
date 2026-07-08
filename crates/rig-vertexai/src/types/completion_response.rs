use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::OneOrMany;
use rig_core::completion::{CompletionError, CompletionResponse, Usage};
use rig_core::message::{AssistantContent, Reasoning, Text, ToolCall, ToolFunction};
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
        // function calling args. We need to convert that to serde_json::Value for rig_core::completion type matching
        for part in content.parts.iter() {
            // Gemini "thinking" models attach an opaque `thoughtSignature` to (usually) the
            // functionCall part. It must be echoed back verbatim on subsequent turns or Vertex
            // rejects the request with INVALID_ARGUMENT ("missing a thought_signature"). We carry
            // it through rig-core's `ToolCall.signature` (base64, since it is raw bytes).
            let signature = (!part.thought_signature.is_empty())
                .then(|| BASE64.encode(&part.thought_signature));

            if let Some(function_call) = part.function_call() {
                let args_json = function_call
                    .args
                    .as_ref()
                    .map(|s| serde_json::Value::Object(s.clone()))
                    .unwrap_or_else(|| serde_json::json!({}));

                assistant_contents.push(AssistantContent::ToolCall(
                    ToolCall::new(
                        function_call.name.clone(),
                        ToolFunction::new(function_call.name.clone(), args_json),
                    )
                    .with_signature(signature),
                ));
            } else if let Some(text) = part.text() {
                if part.thought {
                    assistant_contents.push(AssistantContent::Reasoning(
                        Reasoning::new_with_signature(text, signature),
                    ));
                } else {
                    assistant_contents.push(AssistantContent::Text(Text::new(text.clone())));
                }
            } else if signature.is_some() {
                // A signature-bearing part that is neither a function call nor text (e.g. a
                // standalone "thinking" part). rig-core has no carrier for it, so it is dropped —
                // log it so a later INVALID_ARGUMENT can be traced back here rather than being silent.
                tracing::warn!(
                    "Vertex response part carries a thought_signature but is neither a function \
                     call nor text; signature dropped (no rig-core carrier)."
                );
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
                cached_input_tokens: 0, // unreported at time of writing
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            choice,
            usage,
            raw_response: value,
            message_id: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use google_cloud_aiplatform_v1 as vertexai;
    use rig_core::OneOrMany;
    use rig_core::message::{AssistantContent, Text, ToolCall};

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

    fn create_signed_tool_call_response(
        function_name: &str,
        signature: &[u8],
    ) -> VertexGenerateContentOutput {
        let function_call = vertexai::model::FunctionCall::new()
            .set_name(function_name.to_string())
            .set_args(serde_json::Map::new());
        let part = vertexai::model::Part::new()
            .set_function_call(function_call)
            .set_thought_signature(signature.to_vec());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    #[test]
    fn test_tool_call_response_captures_thought_signature() {
        let raw = b"\x00\x01\x02thinking-sig\xff";
        let response: CompletionResponse<VertexGenerateContentOutput> =
            create_signed_tool_call_response("add", raw)
                .try_into()
                .unwrap();
        match response.choice.first() {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.signature, Some(BASE64.encode(raw))),
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_tool_call_response_without_signature_is_none() {
        let response: CompletionResponse<VertexGenerateContentOutput> =
            create_tool_call_response("add", serde_json::json!({"x": 1}))
                .try_into()
                .unwrap();
        match response.choice.first() {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.signature, None),
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_thought_text_response_captures_thought_signature() {
        let raw = b"\x00\x01\x02thinking-text-sig\xff";
        let part = vertexai::model::Part::new()
            .set_text("thinking text".to_string())
            .set_thought(true)
            .set_thought_signature(raw.to_vec());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);

        let response: CompletionResponse<VertexGenerateContentOutput> =
            VertexGenerateContentOutput(response).try_into().unwrap();

        match response.choice.first() {
            AssistantContent::Reasoning(reasoning) => {
                assert_eq!(reasoning.display_text(), "thinking text");
                assert_eq!(
                    reasoning.first_signature(),
                    Some(BASE64.encode(raw).as_str())
                );
            }
            _ => panic!("Expected Reasoning"),
        }
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
            OneOrMany::one(AssistantContent::Text(Text::new(
                "Hello, world!".to_string()
            )))
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
