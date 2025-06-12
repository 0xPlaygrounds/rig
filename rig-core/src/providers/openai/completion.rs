// ================================================================
// OpenAI Completion API
// ================================================================

use super::{ApiErrorResponse, ApiResponse, Client, StreamingCompletionResponse};
use crate::completion::{CompletionError, CompletionRequest};
use crate::message::{AudioMediaType, ImageDetail};
use crate::one_or_many::string_or_one_or_many;
use crate::providers::anthropic::completion::Role;
use crate::{completion, json_utils, message, OneOrMany};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::convert::Infallible;

use std::str::FromStr;

/// `o4-mini-2025-04-16` completion model
pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";
/// `o4-mini` completion model
pub const O4_MINI: &str = "o4-mini";
/// `o3` completion model
pub const O3: &str = "o3";
/// `o3-mini` completion model
pub const O3_MINI: &str = "o3-mini";
/// `o3-mini-2025-01-31` completion model
pub const O3_MINI_2025_01_31: &str = "o3-mini-2025-01-31";
/// `o1-pro` completion model
pub const O1_PRO: &str = "o1-pro";
/// `o1`` completion model
pub const O1: &str = "o1";
/// `o1-2024-12-17` completion model
pub const O1_2024_12_17: &str = "o1-2024-12-17";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";

/// `gpt-4.1-mini` completion model
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano` completion model
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-4.1-2025-04-14` completion model
pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
/// `gpt-4.1` completion model
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.5-preview` completion model
pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
/// `gpt-4.5-preview-2025-02-27` completion model
pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
/// `gpt-4o-2024-11-20` completion model (this is newer than 4o)
pub const GPT_4O_2024_11_20: &str = "gpt-4o-2024-11-20";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

/// The completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Debug, Deserialize, Serialize)]
pub struct NewCompletionRequest {
    input: Vec<Message>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    // TODO: Fix this before opening a PR!
    // tool_choice: Option<T>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    /// Additional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    addtl_params: Option<AddtlParams>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum OpenAITool {
    Function(InputFunction),
    FileSearch(InputFileSearch),
    WebSearchPreview(InputWebSearchPreview),
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311(InputWebSearchPreview),
    ComputerUsePreview(InputComputerUse),
    Mcp(InputMcpTool),
    CodeInterpreter(InputCodeInterpreter),
    ImageGeneration(InputImageGen),
    LocalShell,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct InputImageGen {
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<ImageGenBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_image_mask: Option<ImageMask>,
    /// The model to use. Defaults to gpt-image-1.
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Moderation level. Defaults to "auto".
    #[serde(skip_serializing_if = "Option::is_none")]
    moderation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_compression: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<ImageOutputFormat>,
    /// Number of partial images to generate in streaming mode, from 0 (default value) to 3.
    #[serde(skip_serializing_if = "Option::is_none")]
    partial_images: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<ImageQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<ImageSize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ImageSize {
    /// Default
    Auto,
    /// Square (1024x1024)
    Square,
    /// Wide (1024x1536)
    Wide,
    /// Tall (1536x1024)
    Tall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ImageQuality {
    Auto,
    Low,
    Medium,
    High,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ImageOutputFormat {
    Jpeg,
    Png,
    Webp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ImageMask {
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    /// base64 encoded image
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ImageGenBackground {
    Auto,
    Opaque,
    Transparent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputCodeInterpreter {
    container_id: InterpreterContainer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum InterpreterContainer {
    IdOnly(Id),
    Object(ContainerObjectConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Id(String);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ContainerObjectConfig {
    Auto { file_ids: Option<Vec<String>> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputMcpTool {
    server_label: String,
    server_url: String,
    allowed_tools: McpAllowedTools,
    headers: HashMap<String, String>,
    require_approval: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum McpAllowedTools {
    Array(Vec<String>),
    Object { tool_names: Vec<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputComputerUse {
    display_height: String,
    display_width: String,
    environment: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputWebSearchPreview {
    #[serde(skip_serializing_if = "Option::is_none")]
    search_size_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<UserLocArgs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum UserLocArgs {
    Approximate {
        city: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        country: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timezone: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputFileSearch {
    vector_store_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<FileSearchFilter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_num_results: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranking_options: Option<RankingOptions>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_threshold: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputFunction {
    name: String,
    parameters: serde_json::Value,
    strict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

/// Additional parameters for the completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Clone, Debug, Deserialize, Serialize)]
struct AddtlParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<Include>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<TruncationStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<OpenAIServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
}

/// Attempt to try and create a `NewCompletionRequest` from a model name and [`crate::completion::CompletionRequest`]
impl TryFrom<(String, CompletionRequest)> for NewCompletionRequest {
    type Error = CompletionError;
    fn try_from((model, req): (String, CompletionRequest)) -> Result<Self, Self::Error> {
        let input = {
            let mut partial_history = vec![];
            if let Some(docs) = req.normalized_documents() {
                partial_history.push(docs);
            }
            partial_history.extend(req.chat_history);

            // Initialize full history with preamble (or empty if non-existent)
            let mut full_history: Vec<Message> = Vec::new();

            // Convert and extend the rest of the history
            full_history.extend(
                partial_history
                    .into_iter()
                    .map(message::Message::try_into)
                    .collect::<Result<Vec<Vec<Message>>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            );
            full_history
        };

        let stream = req
            .additional_params
            .clone()
            .unwrap_or(Value::Null)
            .as_bool();

        let addtl_params = if let Some(map) = req.additional_params {
            serde_json::from_value::<AddtlParams>(map).ok()
        } else {
            None
        };

        Ok(Self {
            input,
            model,
            instructions: req.preamble,
            max_output_tokens: req.max_tokens,
            stream,
            tools: req.tools.into_iter().map(ToolDefinition::from).collect(),
            temperature: req.temperature,
            addtl_params,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum TruncationStrategy {
    Auto,
    #[default]
    Disabled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TextConfig {
    format: TextFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum TextFormat {
    JsonSchema(StructuredOutputsInput),
    Text,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StructuredOutputsInput {
    /// The name of your schema.
    name: String,
    /// Your required output schema. It is recommended that you use the JsonSchema macro, which you can check out at <https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html>.
    schema: serde_json::Value,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    strict: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Reasoning {
    effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ReasoningSummaryLevel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum OpenAIServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ReasoningSummaryLevel {
    #[default]
    Auto,
    Concise,
    Detailed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum Include {
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    #[serde(rename = "computer_call.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NewCompletionResponse {
    /// The ID of a completion response.
    id: String,
    /// The type of the object.
    object: ResponseObject,
    /// The time at which a given response has been created, in seconds from the UNIX epoch (01/01/1970 00:00:00).
    created_at: u64,
    status: ResponseStatus,
    error: Option<ResponseError>,
    incomplete_details: Option<IncompleteDetailsReason>,
    instructions: Option<String>,
    max_token_output: Option<u64>,
    metadata: Map<String, serde_json::Value>,
    model: String,
    usage: Usage,
    output: Vec<Output>,
    #[serde(flatten)]
    addtl_params: Option<AddtlParams>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum Output {
    Message(OutputMessage),
    FileSearchCall(OutputFileSearch),
    FunctionCall(OutputFunctionCall),
    WebSearchCall(OutputWebSearchCall),
    ComputerCall(OutputComputerCall),
    Reasoning(OutputReasoning),
    ImageGenerationCall(OutputImageGeneration),
    CodeInterpreterToolCall(OutputCodeInterpreterToolCall),
    LocalShellCall(OutputLocalShellCall),
    McpCall(OutputMcpToolCall),
    McpListTools(OutputMcpListTools),
    McpApprovalRequest(McpApprovalRequest),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct McpApprovalRequest {
    id: String,
    arguments: serde_json::Value,
    name: String,
    server_label: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ApprovalStatus {
    Pending,
    Approved,
    Denied,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputMcpListTools {
    id: String,
    server_label: String,
    tools: Vec<McpTool>,
    status: ToolStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct McpTool {
    name: String,
    input_schema: serde_json::Value,
    annotations: Option<Vec<serde_json::Value>>,
    description: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputMcpToolCall {
    id: String,
    arguments: serde_json::Value,
    name: String,
    server_label: String,
    error: Option<String>,
    output: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputLocalShellCall {
    id: String,
    call_id: String,
    action: ShellAction,
    status: ToolStatus,
    output: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ShellAction {
    Exec(ShellActionArgs),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShellActionArgs {
    /// The command to run (split up as a Vec<String>)
    command: Vec<String>,
    env: HashMap<String, String>,
    timeout_ms: Option<u64>,
    user: Option<String>,
    working_directory: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputCodeInterpreterToolCall {
    id: String,
    container_id: String,
    /// The code to run
    code: String,
    results: Vec<InterpreterOutput>,
    status: ToolStatus,
    outputs: Vec<CodeInterpreterOutput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum InterpreterOutput {
    Logs { logs: String },
    Files { files: Vec<FileId> },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FileId {
    file_id: String,
    mime_type: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum CodeInterpreterOutput {
    Text { text: String },
    Image { image_url: String },
    File { file_id: String },
    DataFrame { csv_url: String },
    // etc...
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputImageGeneration {
    id: String,
    status: ToolStatus,
    /// Base64
    result: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputReasoning {
    id: String,
    status: ToolStatus,
    summary: Vec<Summary>,
    encrypted_content: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum Summary {
    SummaryText { text: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputComputerCall {
    id: String,
    call_id: String,
    pending_safety_checks: Vec<ComputerCallSafetyCheck>,
    status: ToolStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ComputerCallSafetyCheck {
    code: String,
    id: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputWebSearchCall {
    id: String,
    status: ToolStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputFunctionCall {
    id: String,
    arguments: serde_json::Value,
    call_id: String,
    name: String,
    status: ToolStatus,
    content: Vec<AssistantContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputMessage {
    id: String,
    role: OutputRole,
    status: ResponseStatus,
    content: Vec<AssistantContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum OutputRole {
    Assistant,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputFileSearch {
    id: String,
    queries: Vec<String>,
    status: SearchStatus,
    results: Option<Vec<SearchResult>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SearchResult {
    attributes: Map<String, serde_json::Value>,
    file_id: String,
    filename: String,
    score: f64,
    text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum SearchStatus {
    InProgress,
    Searching,
    Incomplete,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Usage {
    input_tokens: u64,
    input_tokens_details: InputTokensDetails,
    output_tokens: u64,
    output_tokens_details: OutputTokensDetails,
    total_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputTokensDetails {
    cached_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OutputTokensDetails {
    reasoning_tokens: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct IncompleteDetailsReason {
    reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ResponseError {
    code: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ResponseObject {
    Response,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
enum ResponseStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Incomplete,
}

impl TryFrom<NewCompletionResponse> for completion::CompletionResponse<NewCompletionResponse> {
    type Error = CompletionError;
    fn try_from(response: NewCompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text } => text,
                            AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
                        }
                    })
                    .collect::<Vec<_>>();

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

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: OneOrMany<ToolResultContent>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioAssistant {
    id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    r#type: SystemContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Refusal { refusal: String },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
    },
    Audio {
        input_audio: InputAudio,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    #[serde(default)]
    r#type: ToolResultContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultContentType {
    #[default]
    Text,
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent {
            r#type: ToolResultContentType::default(),
            text: s,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                // If there are messages with both tool results and user content, openai will only
                //  handle tool results. It's unlikely that there will be both.
                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::ToolResult(message::ToolResult {
                                id,
                                content,
                            }) => Ok::<_, message::MessageError>(Message::ToolResult {
                                tool_call_id: id,
                                content: content.try_map(|content| match content {
                                    message::ToolResultContent::Text(message::Text { text }) => {
                                        Ok(text.into())
                                    }
                                    _ => Err(message::MessageError::ConversionError(
                                        "Tool result content does not support non-text".into(),
                                    )),
                                })?,
                            }),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    let other_content = OneOrMany::many(other_content).expect(
                        "There must be other content here if there were no tool result content",
                    );

                    Ok(vec![Message::User {
                        content: other_content.map(|content| match content {
                            message::UserContent::Text(message::Text { text }) => {
                                UserContent::Text { text }
                            }
                            message::UserContent::Image(message::Image {
                                data, detail, ..
                            }) => UserContent::Image {
                                image_url: ImageUrl {
                                    url: data,
                                    detail: detail.unwrap_or_default(),
                                },
                            },
                            message::UserContent::Document(message::Document { data, .. }) => {
                                UserContent::Text { text: data }
                            }
                            message::UserContent::Audio(message::Audio {
                                data,
                                media_type,
                                ..
                            }) => UserContent::Audio {
                                input_audio: InputAudio {
                                    data,
                                    format: match media_type {
                                        Some(media_type) => media_type,
                                        None => AudioMediaType::MP3,
                                    },
                                },
                            },
                            _ => unreachable!(),
                        }),
                        name: None,
                    }])
                }
            }
            message::Message::Assistant { content } => {
                let (text_content, tool_calls) = content.into_iter().fold(
                    (Vec::new(), Vec::new()),
                    |(mut texts, mut tools), content| {
                        match content {
                            message::AssistantContent::Text(text) => texts.push(text),
                            message::AssistantContent::ToolCall(tool_call) => tools.push(tool_call),
                        }
                        (texts, tools)
                    },
                );

                // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
                //  so either `content` or `tool_calls` will have some content.
                Ok(vec![Message::Assistant {
                    content: text_content
                        .into_iter()
                        .map(|content| content.text.into())
                        .collect::<Vec<_>>(),
                    refusal: None,
                    audio: None,
                    name: None,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            r#type: ToolType::default(),
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl From<ToolCall> for message::ToolCall {
    fn from(tool_call: ToolCall) -> Self {
        Self {
            id: tool_call.id,
            function: message::ToolFunction {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message {
            Message::User { content, .. } => message::Message::User {
                content: content.map(|content| content.into()),
            },
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),

                        // TODO: Currently, refusals are converted into text, but should be
                        //  investigated for generalization.
                        AssistantContent::Refusal { refusal } => {
                            message::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    content: OneOrMany::many(content).map_err(|_| {
                        message::MessageError::ConversionError(
                            "Neither `content` nor `tool_calls` was provided to the Message"
                                .to_owned(),
                        )
                    })?,
                }
            }

            Message::ToolResult {
                tool_call_id,
                content,
            } => message::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    tool_call_id,
                    content.map(|content| message::ToolResultContent::text(content.text)),
                )),
            },

            // System messages should get stripped out when converting message's, this is just a
            // stop gap to avoid obnoxious error handling or panic occurring.
            Message::System { content, .. } => message::Message::User {
                content: content.map(|content| message::UserContent::text(content.text)),
            },
        })
    }
}

impl From<UserContent> for message::UserContent {
    fn from(content: UserContent) -> Self {
        match content {
            UserContent::Text { text } => message::UserContent::text(text),
            UserContent::Image { image_url } => message::UserContent::image(
                image_url.url,
                Some(message::ContentFormat::default()),
                None,
                Some(image_url.detail),
            ),
            UserContent::Audio { input_audio } => message::UserContent::audio(
                input_audio.data,
                Some(message::ContentFormat::default()),
                Some(input_audio.format),
            ),
        }
    }
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text { text: s }
    }
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text {
            text: s.to_string(),
        })
    }
}

impl From<String> for AssistantContent {
    fn from(s: String) -> Self {
        AssistantContent::Text { text: s }
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string(),
        })
    }
}
impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        let req = NewCompletionRequest::try_from((self.model.clone(), completion_request))?;
        let json_req = serde_json::to_value(req)?;

        Ok(json_req)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = NewCompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self.client.post("/responses").json(&request).send().await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "OpenAI completion error: {}", t);

            match serde_json::from_str::<ApiResponse<Self::Response>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenAI completion token usage: {:?}",
                        response.usage.total_tokens.to_string()
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        Self::stream(self, request).await
    }
}
