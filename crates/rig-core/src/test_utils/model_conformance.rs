//! Provider-neutral behavioral scenarios for completion-model conformance.
//!
//! These helpers test the model/agent contract only. Provider wire formats,
//! authentication, HTTP streaming, and cassette matching remain provider-suite
//! responsibilities.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    agent::{AgentBuilder, MultiTurnStreamItem, NoToolConfig, OutputMode},
    completion::{AssistantContent, CompletionModel, Message, Prompt, ToolDefinition},
    message::{ToolChoice, UserContent},
    streaming::StreamingPrompt,
    tool::{Tool, ToolContext},
};

/// Summary emitted by a portable model-conformance scenario.
#[derive(Debug, Clone)]
pub struct ScenarioReport {
    /// Stable scenario name.
    pub name: &'static str,
    /// Number of model-invoked tool calls observed by the scenario.
    pub tool_calls: usize,
    /// Aggregated prompt tokens reported by the model across all turns.
    pub prompt_tokens: u64,
    /// Aggregated generated tokens reported by the model across all turns.
    pub generated_tokens: u64,
    /// Number of messages retained in the completed run history.
    pub history_messages: usize,
    /// End-to-end scenario duration.
    pub duration: Duration,
    /// The model's final user-visible response.
    pub response: String,
}

/// Error used by the deterministic conformance tools.
#[derive(Debug, thiserror::Error)]
#[error("model-conformance tool failed")]
pub struct ConformanceToolError;

fn has_tool_roundtrip(messages: Option<&[Message]>) -> bool {
    let saw_call = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            )
        })
    });
    let saw_result = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
            )
        })
    });
    saw_call && saw_result
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RepeatArgs {
    /// The text to repeat.
    text: String,
    /// Number of repetitions; defaults to 2 when omitted.
    times: Option<u32>,
}

#[derive(Clone)]
struct RepeatTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for RepeatTool {
    const NAME: &'static str = "repeat_text";
    type Error = ConformanceToolError;
    type Args = RepeatArgs;
    type Output = String;

    fn description(&self) -> String {
        "Repeat `text`. `times` is optional and defaults to 2.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(RepeatArgs)).unwrap_or_default()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![args.text.as_str(); args.times.unwrap_or(2) as usize].join(" "))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BinOpArgs {
    a: i64,
    b: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ArithmeticResult {
    answer: i64,
    explanation: Option<String>,
}

#[derive(Clone)]
struct AddTool(Arc<AtomicUsize>);

impl Tool for AddTool {
    const NAME: &'static str = "add";
    type Error = ConformanceToolError;
    type Args = BinOpArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add two integers a and b.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.a + args.b)
    }
}

#[derive(Clone)]
struct MultiplyTool(Arc<AtomicUsize>);

impl Tool for MultiplyTool {
    const NAME: &'static str = "multiply";
    type Error = ConformanceToolError;
    type Args = BinOpArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Multiply two integers a and b.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.a * args.b)
    }
}

/// Runs the portable optional-argument tool scenario.
///
/// `configure` is deliberately outside the scenario so a provider suite can
/// attach transport-only settings without putting them into the shared model
/// contract.
pub async fn optional_argument<M, F>(
    model: M,
    configure: F,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
    F: FnOnce(AgentBuilder<M, NoToolConfig>) -> AgentBuilder<M, NoToolConfig>,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(model))
        .preamble("Use the repeat_text tool whenever asked to repeat text.")
        .tool(RepeatTool {
            calls: calls.clone(),
        })
        .default_max_turns(4)
        .build();
    let result = agent
        .prompt(
            "Use the repeat_text tool to repeat the word \"banana\" 3 times, then show me the exact result.",
        )
        .extended_details()
        .await?;
    let response = result.output.clone();
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0
        || response.matches("banana").count() < 1
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(
            format!("optional_argument failed: calls={tool_calls}, response={response:?}").into(),
        );
    }
    Ok(ScenarioReport {
        name: "optional_argument",
        tool_calls,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages: result.messages.as_ref().map_or(0, Vec::len),
        duration: started.elapsed(),
        response,
    })
}

/// Runs a portable two-tool sequential arithmetic scenario.
pub async fn sequential_tools<M, F>(
    model: M,
    configure: F,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
    F: FnOnce(AgentBuilder<M, NoToolConfig>) -> AgentBuilder<M, NoToolConfig>,
{
    let add_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(model))
        .preamble(
            "You are a calculator. Use the add and multiply tools for arithmetic; never compute by hand.",
        )
        .tool(AddTool(add_calls.clone()))
        .tool(MultiplyTool(multiply_calls.clone()))
        .default_max_turns(6)
        .build();
    let result = agent
        .prompt(
            "Compute (4 + 6) * 2. First call the add tool, then call the multiply tool on the result. Tell me the final number.",
        )
        .extended_details()
        .await?;
    let response = result.output.clone();
    let add = add_calls.load(Ordering::SeqCst);
    let multiply = multiply_calls.load(Ordering::SeqCst);
    if add == 0
        || multiply == 0
        || !response.contains("20")
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(format!(
            "sequential_tools failed: add={add}, multiply={multiply}, response={response:?}"
        )
        .into());
    }
    Ok(ScenarioReport {
        name: "sequential_tools",
        tool_calls: add + multiply,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages: result.messages.as_ref().map_or(0, Vec::len),
        duration: started.elapsed(),
        response,
    })
}

/// Runs a tool through Rig's multi-turn streaming agent driver.
pub async fn streaming_tool<M, F>(
    model: M,
    configure: F,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
    F: FnOnce(AgentBuilder<M, NoToolConfig>) -> AgentBuilder<M, NoToolConfig>,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(model))
        .preamble("Use the add tool for arithmetic; do not calculate by hand.")
        .tool(AddTool(calls.clone()))
        .default_max_turns(4)
        .build();
    let mut stream = agent
        .stream_prompt("Use add to calculate 17 + 25, then state the final number.")
        .max_turns(4)
        .await;
    let mut response = None;
    let mut usage = crate::completion::Usage::new();
    let mut history_messages = 0;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(final_response) = item? {
            usage = final_response.usage;
            history_messages = final_response.messages.as_ref().map_or(0, Vec::len);
            response = Some(final_response.output().to_owned());
        }
    }
    let response = response.ok_or("streaming_tool produced no final response")?;
    let tool_calls = calls.load(Ordering::SeqCst);
    // The streaming final response must retain the same call/result variants
    // that the buffered driver exposes to caller-owned history.
    if tool_calls == 0 || !response.contains("42") || history_messages < 4 {
        return Err(
            format!("streaming_tool failed: calls={tool_calls}, response={response:?}").into(),
        );
    }
    Ok(ScenarioReport {
        name: "streaming_tool",
        tool_calls,
        prompt_tokens: usage.input_tokens,
        generated_tokens: usage.output_tokens,
        history_messages,
        duration: started.elapsed(),
        response,
    })
}

/// Runs a normal tool followed by Rig's synthetic structured-output tool.
pub async fn structured_after_tool<M, F>(
    model: M,
    configure: F,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
    F: FnOnce(AgentBuilder<M, NoToolConfig>) -> AgentBuilder<M, NoToolConfig>,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(model))
        .preamble(
            "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        )
        .output_schema::<ArithmeticResult>()
        .output_mode(OutputMode::Tool)
        .tool(AddTool(calls.clone()))
        .default_max_turns(5)
        .build();
    let result = agent
        .prompt("Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.")
        .extended_details()
        .await?;
    let response = result.output.clone();
    let parsed: ArithmeticResult = serde_json::from_str(&response)?;
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0 || parsed.answer != 42 || !has_tool_roundtrip(result.messages.as_deref()) {
        return Err(format!(
            "structured_after_tool failed: calls={tool_calls}, response={response:?}"
        )
        .into());
    }
    let _ = parsed.explanation;
    Ok(ScenarioReport {
        name: "structured_after_tool",
        tool_calls: tool_calls + 1,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages: result.messages.as_ref().map_or(0, Vec::len),
        duration: started.elapsed(),
        response,
    })
}

/// Runs all portable tool-choice modes directly against a completion model.
pub async fn tool_choice_modes<M>(
    model: M,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
{
    let definition = |name: &str| ToolDefinition {
        name: name.to_string(),
        description: format!("Return the supplied integer using {name}."),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"]
        }),
    };
    let tools = vec![definition("alpha"), definition("beta")];
    let started = Instant::now();
    let none = model
        .completion(
            model
                .completion_request("Answer with only the number 4. Do not call a function.")
                .tools(tools.clone())
                .tool_choice(ToolChoice::None)
                .temperature(0.0)
                .max_tokens(64)
                .build(),
        )
        .await?;
    if none
        .choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)))
    {
        return Err("tool_choice none emitted a tool call".into());
    }

    let required = model
        .completion(
            model
                .completion_request("Call alpha with value 7.")
                .tools(tools.clone())
                .tool_choice(ToolChoice::Required)
                .temperature(0.0)
                .max_tokens(96)
                .build(),
        )
        .await?;
    let required_calls = required
        .choice
        .iter()
        .filter(|item| matches!(item, AssistantContent::ToolCall(_)))
        .count();
    if required_calls == 0 {
        return Err("tool_choice required emitted no tool call".into());
    }

    let specific = model
        .completion(
            model
                .completion_request("Call beta with value 9.")
                .tools(tools)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["beta".to_string()],
                })
                .temperature(0.0)
                .max_tokens(96)
                .build(),
        )
        .await?;
    let specific_calls = specific
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    if specific_calls.is_empty()
        || specific_calls
            .iter()
            .any(|call| call.function.name != "beta")
    {
        return Err("specific tool choice did not select only beta".into());
    }

    Ok(ScenarioReport {
        name: "tool_choice_modes",
        tool_calls: required_calls + specific_calls.len(),
        prompt_tokens: none.usage.input_tokens
            + required.usage.input_tokens
            + specific.usage.input_tokens,
        generated_tokens: none.usage.output_tokens
            + required.usage.output_tokens
            + specific.usage.output_tokens,
        history_messages: 0,
        duration: started.elapsed(),
        response: "none, required, and specific modes passed".to_string(),
    })
}

/// Runs a streamed real-tool turn followed by Rig's synthetic output tool.
pub async fn streaming_structured_after_tool<M, F>(
    model: M,
    configure: F,
) -> Result<ScenarioReport, Box<dyn std::error::Error + Send + Sync>>
where
    M: CompletionModel + 'static,
    F: FnOnce(AgentBuilder<M, NoToolConfig>) -> AgentBuilder<M, NoToolConfig>,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(model))
        .preamble(
            "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        )
        .output_schema::<ArithmeticResult>()
        .output_mode(OutputMode::Tool)
        .tool(AddTool(calls.clone()))
        .default_max_turns(5)
        .build();
    let mut stream = agent
        .stream_prompt(
            "Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.",
        )
        .max_turns(5)
        .await;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response);
        }
    }
    let result = final_response.ok_or("streaming structured run produced no final response")?;
    let parsed: ArithmeticResult = serde_json::from_str(&result.output)?;
    let calls = calls.load(Ordering::SeqCst);
    if calls == 0 || parsed.answer != 42 || !has_tool_roundtrip(result.messages.as_deref()) {
        return Err(format!(
            "streaming_structured_after_tool failed: calls={calls}, response={:?}",
            result.output
        )
        .into());
    }
    Ok(ScenarioReport {
        name: "streaming_structured_after_tool",
        tool_calls: calls + 1,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages: result.messages.as_ref().map_or(0, Vec::len),
        duration: started.elapsed(),
        response: result.output,
    })
}
