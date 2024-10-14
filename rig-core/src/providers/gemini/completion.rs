// ================================================================
// Google Gemini Completion API
// ================================================================
//
// https://ai.google.dev/api/generate-conten

/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    providers::gemini::client::ApiResponse,
};

use super::Client;

//
// Gemini API Response Types
//

// Define the struct for the GenerateContentResponse
#[derive(Debug, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Vec<ContentCandidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
}

// Define the struct for a Candidate
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentCandidate {
    pub content: Content,
    pub finish_reason: Option<FinishReason>,
    pub safety_ratings: Option<Vec<SafetyRating>>,
    pub citation_metadata: Option<CitationMetadata>,
    pub token_count: Option<i32>,
    pub avg_logprobs: Option<f64>,
    pub logprobs_result: Option<LogprobsResult>,
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Content {
    pub parts: Vec<Part>,
    pub role: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Part {
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SafetyRating {
    pub category: HarmCategory,
    pub probability: HarmProbability,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    HarmProbabilityUnspecified,
    Negligible,
    Low,
    Medium,
    High,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexually,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
    HarmCategoryCivicIntegrity,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: i32,
    pub cached_content_token_count: i32,
    pub candidates_token_count: i32,
    pub total_token_count: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    pub block_reason: Option<BlockReason>,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
pub enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Other,
    Blocklist,
    ProhibitedContent,
}

#[derive(Debug, Deserialize)]
pub struct CitationMetadata {
    pub citation_sources: Vec<CitationSource>,
}

#[derive(Debug, Deserialize)]
pub struct CitationSource {
    pub uri: Option<String>,
    pub start_index: Option<i32>,
    pub end_index: Option<i32>,
    pub license: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LogprobsResult {
    pub top_candidate: Vec<TopCandidate>,
    pub chosen_candidate: Vec<LogProbCandidate>,
}

#[derive(Debug, Deserialize)]
pub struct TopCandidate {
    pub candidates: Vec<LogProbCandidate>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogProbCandidate {
    pub token: String,
    pub token_id: String,
    pub log_probability: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GenerationConfig {
    pub stop_sequences: Option<Vec<String>>,
    pub response_mime_type: Option<String>,
    pub response_schema: Option<Schema>,
    pub candidate_count: Option<i32>,
    pub max_output_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub response_logprobs: Option<bool>,
    pub logprobs: Option<i32>,
}

/// The Schema object allows the definition of input and output data types. These types can be objects, but also primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
/// https://ai.google.dev/api/caching#Schema
#[derive(Debug, Deserialize, Serialize)]
pub struct Schema {
    pub r#type: String,
    pub format: Option<String>,
    pub description: Option<String>,
    pub nullable: Option<bool>,
    pub r#enum: Option<Vec<String>>,
    pub max_items: Option<i32>,
    pub min_items: Option<i32>,
    pub properties: Option<HashMap<String, Schema>>,
    pub required: Option<Vec<String>>,
    pub items: Option<Box<Schema>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    pub tools: Option<Vec<Tool>>,
    pub tool_config: Option<ToolConfig>,
    pub generation_config: Option<GenerationConfig>,
    pub safety_settings: Option<Vec<SafetySetting>>,
    pub system_instruction: Option<String>,
    // cachedContent: Optional<String>
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declaration: FunctionDeclaration,
    pub code_execution: Option<CodeExecution>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Option<Vec<Schema>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub schema: Option<Schema>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecution {}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
    Off,
}

// =================================================================
// Rig Implementation Types
// =================================================================

impl From<completion::ToolDefinition> for Tool {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            function_declaration: FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters: None, // tool.parameters, TODO: Map Gemini
            },
            code_execution: None,
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        match response.candidates.as_slice() {
            [ContentCandidate { content, .. }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(
                    content.parts.first().unwrap().text.clone(),
                ),
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "No candidates found in response".into(),
            )),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        // QUESTION: Why do Anthropic/openAi implementation differ here? OpenAI adds the preamble but Anthropic does not.

        let mut full_history = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        full_history.append(&mut completion_request.chat_history);

        let prompt_with_context = completion_request.prompt_with_context();

        full_history.push(completion::Message {
            role: "user".into(),
            content: prompt_with_context,
        });

        let request = GenerateContentRequest {
            contents: full_history
                .into_iter()
                .map(|msg| Content {
                    parts: vec![Part { text: msg.content }],
                    role: match msg.role.as_str() {
                        "system" => Some("model".to_string()),
                        "user" => Some("user".to_string()),
                        "assistant" => Some("model".to_string()),
                        _ => None,
                    },
                })
                .collect(),
            // QUESTION: How to handle API config specifics?
            generation_config: Some(GenerationConfig {
                temperature: completion_request.temperature,
                max_output_tokens: completion_request.max_tokens,
                top_p: None,
                top_k: None,
                candidate_count: None,
                stop_sequences: None,
                response_mime_type: None,
                response_schema: None,
                presence_penalty: None,
                frequency_penalty: None,
                response_logprobs: None,
                logprobs: None,
            }),
            safety_settings: None,
            tools: Some(
                completion_request
                    .tools
                    .into_iter()
                    .map(Tool::from)
                    .collect(),
            ),
            tool_config: None,
            system_instruction: None,
        };

        let response = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<ApiResponse<GenerateContentResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => Ok(response.try_into()?),
            ApiResponse::Err(err) => Err(CompletionError::ResponseError(err.message)),
        }
    }
}
