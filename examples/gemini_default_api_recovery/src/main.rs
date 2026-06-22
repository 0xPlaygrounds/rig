//! Demonstrates recovering from Gemini emitting a legacy `default_api` tool name.
//! Requires `GEMINI_API_KEY`.
//!
//! Run with `RIG_GEMINI_DEFAULT_API_CANARY_ATTEMPTS=6` to increase the chance of
//! seeing the recoverable legacy tool-name emission.

use futures::StreamExt;
use rig::agent::{
    FinalResponse, InvalidToolCallContext, InvalidToolCallHookAction, MultiTurnStreamItem,
    PromptHook, StreamingResult,
};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::ToolResultContent;
use rig::providers::gemini::{
    self,
    completion::gemini_api_types::{AdditionalParameters, GenerationConfig, ThinkingConfig},
};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::sync::{Arc, Mutex};

const GEMINI_CANARY_MODEL: &str = "gemini-3.1-pro-preview";
const THINKING_BUDGET: u32 = 1024 * 16;
const CANARY_MAX_TURNS: usize = 8;
const DEFAULT_CANARY_ATTEMPTS: usize = 2;
const CANARY_ATTEMPTS_ENV: &str = "RIG_GEMINI_DEFAULT_API_CANARY_ATTEMPTS";
const JAVASCRIPT_MARKER: &str = "workspace-canary-marker";

const WORKSPACE_STYLE_PREAMBLE: &str = r#"
<role>
You are an assistant operating within a private workspace. Your goal is to help the user achieve their stated task. This includes understanding their intent, planning a course of action, optionally executing it using your tools, and communicating a detailed response that fits the user's intent.
</role>

<capabilities>
<execution-runtime>
Access higher-level workspace inspection through JavaScript execution. The runtime exposes private application APIs through global namespaces, and those APIs can be used to traverse, transform, and inspect workspace data.

Optional use implied, goal is to serve user. Instead of optimistically reaching for execution runtime, only leverage when necessary for extra context gathering and grounding.

Execute in series if needed, the response from one execution may be necessary for future executions.

If continuous execution cannot be completed, explain why to the user.

In the runtime, APIs are exposed as static async methods on global namespaces, with each namespace granting explicit capabilities based on the integration. You will have to use `await` when calling them.

## Workspace
Access workspace records through the `Workspace` object. All methods are async and require `await`.

- `Workspace.getRecord(id: string)`: get record metadata by ID; includes `counts: { words, chars, lines }`, collectionIds, author, timestamps.
- `Workspace.listLibrary(filter?: string)`: returns a flat library overview `{ collections: [{ id, title }], records }`.
- `Workspace.listCollection(collection_id: string, depth?: number, filter?: string)`: returns collection contents `{ id, title, collections, records }`.
- `Workspace.getRecordCollections(id: string)`: returns all collections that contain this record as `[{ id, title }]`.
- `Workspace.readRecord(id: string, range?: [number, number])`: read record content, optional byte range [start, end].
- `Workspace.searchRecords(pattern: string)`: regex search across all text records.

### Notes

**Collections** - names are not unique, always use IDs. A record can appear in multiple collections simultaneously. Removing a record from a collection never deletes it; it always stays in the library.

**Listing** - use `listLibrary()` for a fast overview of all records and collections. Use `listCollection(id)` to drill into a specific collection. The `depth` param on `listCollection` controls recursive flattening.
</execution-runtime>
</capabilities>

<extra-scripting-notes>
The last expression in your script will be returned, no need for explicit `return` statement.
Make sure your script ends with the value you want to inspect.
Console functions like `console.log` and `console.table` are not available, you must return values at the end of your script to see them.
Top-level await is not available, you must write an async function and execute any async functions inside of that.
You have access to standard JavaScript data structures and control flow, but not networking or any storage/filesystem operations aside from the documented APIs.
No access to Node.js APIs or browser APIs, only standard ECMAScript.
Use JavaScript's built-in array methods (`map`, `filter`, `reduce`) for data transformation.
</extra-scripting-notes>
"#;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct JavaScriptProgram {
    title: String,
    description: String,
    code: String,
}

#[repr(transparent)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutorResponse(std::result::Result<serde_json::Value, String>);

impl ExecutorResponse {
    fn ok(value: serde_json::Value) -> Self {
        Self(Ok(value))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("JavaScript tool error")]
struct JavaScriptToolError;

#[derive(Clone)]
struct JavaScript;

impl Tool for JavaScript {
    const NAME: &'static str = "JavaScript";
    type Error = JavaScriptToolError;
    type Args = JavaScriptProgram;
    type Output = ExecutorResponse;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "JavaScript runtime with an array of tools for completing the tasks assigned by the user. Legacy workspace agents may refer to this runtime as default_api.".into(),
            parameters: schema_for!(JavaScriptProgram).to_value(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ExecutorResponse::ok(json!({
            "id": "collection-canary-id",
            "title": "Canary Collection",
            "collections": [],
            "records": [{
                "id": "record-canary-marker",
                "title": JAVASCRIPT_MARKER,
                "counts": { "words": 3, "chars": 21, "lines": 1 },
                "collectionIds": ["collection-canary-id"]
            }],
            "receivedCode": args.code
        })))
    }
}

#[derive(Clone, Default)]
struct DefaultApiRepairHook {
    invalid_tool_names: Arc<Mutex<Vec<String>>>,
}

impl DefaultApiRepairHook {
    fn invalid_tool_names(&self) -> Vec<String> {
        self.invalid_tool_names
            .lock()
            .map(|names| names.clone())
            .unwrap_or_default()
    }
}

impl<M> PromptHook<M> for DefaultApiRepairHook
where
    M: CompletionModel,
{
    async fn on_invalid_tool_call(
        &self,
        context: &InvalidToolCallContext,
    ) -> InvalidToolCallHookAction {
        if let Ok(mut invalid_tool_names) = self.invalid_tool_names.lock() {
            invalid_tool_names.push(context.tool_name.clone());
        }

        if context.tool_name == "default_api" {
            InvalidToolCallHookAction::repair(JavaScript::NAME)
        } else {
            InvalidToolCallHookAction::fail()
        }
    }
}

#[derive(Debug, Default)]
struct WorkspaceStreamObservation {
    streamed_text: String,
    reasoning_text: String,
    final_response: Option<FinalResponse>,
    tool_calls: Vec<String>,
    tool_call_deltas: usize,
    executions: Vec<JavaScriptProgram>,
    tool_results: Vec<String>,
    executor_results: Vec<ExecutorResponse>,
    completion_calls: Vec<String>,
    invalid_tool_names: Vec<String>,
    events: Vec<&'static str>,
}

impl WorkspaceStreamObservation {
    fn final_response_text(&self) -> Option<&str> {
        self.final_response
            .as_ref()
            .map(|response| response.response())
    }

    fn diagnostic_summary(&self) -> String {
        format!(
            "events={:?}, tool_calls={:?}, tool_call_deltas={}, invalid_tool_names={:?}, executions={:?}, tool_results={:?}, executor_results={:?}, streamed_text={:?}, reasoning_text={:?}, completion_calls={:?}, final={:?}, final_usage={:?}",
            self.events,
            self.tool_calls,
            self.tool_call_deltas,
            self.invalid_tool_names,
            self.executions,
            self.tool_results,
            self.executor_results,
            self.streamed_text,
            self.reasoning_text,
            self.completion_calls,
            self.final_response_text(),
            self.final_response
                .as_ref()
                .map(|response| response.usage())
        )
    }
}

fn gemini_canary_additional_params() -> Result<serde_json::Value, serde_json::Error> {
    let additional_params = AdditionalParameters {
        generation_config: Some(GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: Some(THINKING_BUDGET),
                thinking_level: None,
            }),
            ..Default::default()
        }),
        additional_params: None,
    };

    serde_json::to_value(&additional_params)
}

async fn consume_workspace_like_stream(
    mut stream: StreamingResult<gemini::streaming::StreamingCompletionResponse>,
) -> Result<WorkspaceStreamObservation, String> {
    let mut observation = WorkspaceStreamObservation::default();

    while let Some(item) = stream.next().await {
        match item.map_err(|error| error.to_string())? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                observation.events.push("text");
                observation.streamed_text.push_str(&text.text);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            )) => {
                observation.events.push("reasoning");
                observation
                    .reasoning_text
                    .push_str(&reasoning.display_text());
            }
            MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            ) => {
                observation.events.push("reasoning_delta");
                observation.reasoning_text.push_str(&reasoning);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            }) => {
                observation.events.push("tool_call");
                observation.tool_calls.push(tool_call.function.name.clone());
                let execution: JavaScriptProgram =
                    serde_json::from_value(tool_call.function.arguments.clone())
                        .map_err(|error| error.to_string())?;
                observation.executions.push(execution);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCallDelta {
                ..
            }) => {
                observation.events.push("tool_call_delta");
                observation.tool_call_deltas += 1;
            }
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            }) => {
                observation.events.push("tool_result");
                let ToolResultContent::Text(text) = tool_result.content.first() else {
                    return Err("JS Runtime can only respond with JSON text".to_string());
                };
                observation.tool_results.push(text.text.clone());
                let result: ExecutorResponse =
                    serde_json::from_str(&text.text).map_err(|error| error.to_string())?;
                observation.executor_results.push(result);
            }
            MultiTurnStreamItem::CompletionCall(completion_call) => {
                observation.events.push("completion_call");
                observation.completion_calls.push(format!(
                    "call_index={}, usage={:?}",
                    completion_call.call_index, completion_call.usage
                ));
            }
            MultiTurnStreamItem::FinalResponse(final_response) => {
                observation.events.push("final_response");
                observation.final_response = Some(final_response);
                return Ok(observation);
            }
            _ => {}
        }
    }

    Ok(observation)
}

fn canary_attempts() -> usize {
    env::var(CANARY_ATTEMPTS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|attempts| *attempts > 0)
        .unwrap_or(DEFAULT_CANARY_ATTEMPTS)
}

fn workspace_canary_prompt(attempt: usize) -> String {
    format!(
        "Inspect the workspace collection with id `collection-canary-id`. This canary intentionally \
         exercises a legacy workspace namespace leak. Older workspace agents often call the \
         execution runtime by the structured tool name `default_api`. Use the legacy \
         `default_api` workspace runtime name to inspect the collection. The expected program is: \
         async function inspectCollection() {{ try {{ const collection = await Workspace.listCollection(\"collection-canary-id\", 1); return collection; }} catch (e) {{ return `Error: ${{e.message}}`; }} }} inspectCollection(); \
         After inspecting the collection, answer in one sentence and include `{JAVASCRIPT_MARKER}`. \
         Canary attempt {attempt}."
    )
}

async fn run_workspace_canary_attempt(
    attempt: usize,
) -> Result<WorkspaceStreamObservation, String> {
    let client = gemini::Client::from_env().map_err(|error| error.to_string())?;
    let agent_name = format!("workspace-default-api-canary-{attempt}");
    let agent = client
        .agent(GEMINI_CANARY_MODEL)
        .name(&agent_name)
        .preamble(WORKSPACE_STYLE_PREAMBLE)
        .additional_params(gemini_canary_additional_params().map_err(|error| error.to_string())?)
        .tool(JavaScript)
        .default_max_turns(CANARY_MAX_TURNS)
        .temperature(0.0)
        .build();
    let repair_hook = DefaultApiRepairHook::default();

    let stream = agent
        .stream_prompt(workspace_canary_prompt(attempt))
        .with_hook(repair_hook.clone())
        .with_history(Vec::<rig::message::Message>::new())
        .await;

    let mut observation = consume_workspace_like_stream(stream).await?;
    observation.invalid_tool_names = repair_hook.invalid_tool_names();
    Ok(observation)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let attempts = canary_attempts();
    let mut successful_attempts = Vec::new();
    let mut repaired_default_api = false;

    for attempt in 1..=attempts {
        let observation = run_workspace_canary_attempt(attempt)
            .await
            .map_err(anyhow::Error::msg)?;
        let diagnostics = observation.diagnostic_summary();

        repaired_default_api |= observation
            .invalid_tool_names
            .iter()
            .any(|name| name == "default_api");

        if observation
            .tool_results
            .iter()
            .any(|result| result.contains("ToolNotFoundError: default_api"))
        {
            anyhow::bail!(
                "attempt {attempt}: Rig returned the production ToolNotFoundError symptom. {diagnostics}"
            );
        }
        if !observation
            .tool_calls
            .iter()
            .all(|name| !name.contains("default_api"))
        {
            anyhow::bail!(
                "attempt {attempt}: repaired default_api should not reach the public tool-call stream. {diagnostics}"
            );
        }
        if observation.reasoning_text.contains("tool_code")
            || observation.streamed_text.contains("tool_code")
        {
            anyhow::bail!(
                "attempt {attempt}: Gemini emitted markdown tool_code text instead of a function call. {diagnostics}"
            );
        }
        if !observation
            .tool_calls
            .iter()
            .any(|name| name == JavaScript::NAME)
        {
            anyhow::bail!(
                "attempt {attempt}: expected live Gemini to call JavaScript. {diagnostics}"
            );
        }
        if !observation
            .executions
            .iter()
            .any(|execution| execution.code.contains("Workspace.listCollection"))
        {
            anyhow::bail!(
                "attempt {attempt}: expected workspace-shaped JavaScript execution. {diagnostics}"
            );
        }
        if observation.executor_results.is_empty() {
            anyhow::bail!(
                "attempt {attempt}: expected at least one parsed ExecutorResponse. {diagnostics}"
            );
        }

        let Some(final_response) = observation.final_response.as_ref() else {
            anyhow::bail!("attempt {attempt}: stream should yield a final response. {diagnostics}");
        };
        if final_response.response().trim().is_empty() {
            anyhow::bail!(
                "attempt {attempt}: final response should not be empty after the tool roundtrip. {diagnostics}"
            );
        }
        if !final_response.response().contains(JAVASCRIPT_MARKER) {
            anyhow::bail!(
                "attempt {attempt}: final response should mention the tool marker. {diagnostics}"
            );
        }

        successful_attempts.push(diagnostics);
    }

    if !repaired_default_api {
        anyhow::bail!(
            "expected at least one attempt to emit an invalid default_api tool call repaired by the hook. Attempts: {successful_attempts:#?}"
        );
    }

    println!("Gemini default_api canary completed {attempts} attempt(s): {successful_attempts:#?}");

    Ok(())
}
