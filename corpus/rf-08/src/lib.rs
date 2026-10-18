//! `tool-exec`: run one model-produced tool call against a registry of rig
//! tools, outside any agent loop.
//!
//! The registry is rig-agent's [`ToolSet`]. [`run_call`] reads one call
//! (`{"id": …, "name": …, "arguments": …}`, the arguments an object or a
//! JSON string holding one), looks the tool up, checks the arguments against
//! its schema, runs it under a timeout, and returns a [`CallResult`] whose
//! [`Outcome`] the caller matches on. [`run_lines`] is the CLI loop: one call
//! per input line, one JSON result per output line.

pub mod tools;

use std::io::{self, BufRead, Write};
use std::time::Duration;

use rig_agent::tool::ToolSet;
use rig_core::effect::FamilyDescriptor;
use rig_core::message::{self, CallId, ToolCall, ToolFunction, ToolName};
use rig_core::tool::feedback::arguments_refusal;
use rig_core::tool::validate::{Problem, validate};
use rig_core::tool::{ToolContext, ToolErrorKind, ToolExecutionError, ToolOutput};
use serde_json::{Value, json};

/// The result of one call: the call id it answers and what happened.
#[derive(Debug)]
pub struct CallResult {
    /// The id passed in, or one rig issued when the call had none.
    pub id: CallId,
    /// What happened.
    pub outcome: Outcome,
}

/// The distinct outcomes of one call.
#[derive(Debug)]
pub enum Outcome {
    /// The input was not a call: not JSON, not an object, or no `name`.
    Malformed {
        /// Why it is not a call.
        reason: String,
    },
    /// No tool of that name is registered.
    UnknownTool {
        /// The name the call asked for.
        name: String,
    },
    /// The arguments do not fit the tool's schema; the tool did not run.
    InvalidArguments {
        /// Where and how they do not fit (empty when the arguments were not
        /// an object at all, or when the tool's own parsing refused them).
        problems: Vec<Problem>,
        /// What the model reads.
        feedback: String,
    },
    /// The tool ran and failed.
    Failed {
        /// The full error, for logs: its message, kind and source.
        error: ToolExecutionError,
    },
    /// The tool did not finish within the timeout and was cancelled.
    TimedOut,
    /// The tool succeeded; the tool result an agent sends the model for it.
    Success(message::ToolResult),
}

impl Outcome {
    /// A stable name for the outcome, as the CLI prints it.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Malformed { .. } => "malformed",
            Self::UnknownTool { .. } => "unknown_tool",
            Self::InvalidArguments { .. } => "invalid_arguments",
            Self::Failed { .. } => "failed",
            Self::TimedOut => "timed_out",
            Self::Success(_) => "success",
        }
    }

    /// The text safe to show the model for a call that did not succeed;
    /// `None` for a success.
    pub fn model_feedback(&self) -> Option<String> {
        match self {
            Self::Malformed { reason } => Some(reason.clone()),
            Self::UnknownTool { name } => Some(format!("tool `{name}` not found")),
            Self::InvalidArguments { feedback, .. } => Some(feedback.clone()),
            Self::Failed { error } => Some(error.model_output().render()),
            Self::TimedOut => Some("tool execution timed out".to_owned()),
            Self::Success(_) => None,
        }
    }
}

/// Runs the call `call_json` against `registry`: the tool sees `context`'s
/// values (never echoed in the result) and is cancelled after `timeout`.
pub async fn run_call(
    registry: &ToolSet,
    call_json: &str,
    context: &ToolContext,
    timeout: Duration,
) -> CallResult {
    let call = match parse_call(call_json) {
        Ok(call) => call,
        Err((id, reason)) => {
            return CallResult {
                id,
                outcome: Outcome::Malformed { reason },
            };
        }
    };
    let outcome = run_parsed(registry, &call, context, timeout).await;
    CallResult {
        id: call.id,
        outcome,
    }
}

async fn run_parsed(
    registry: &ToolSet,
    call: &ToolCall,
    context: &ToolContext,
    timeout: Duration,
) -> Outcome {
    let name = call.function.name.as_str();
    let Some(tool) = registry.get(name) else {
        return Outcome::UnknownTool {
            name: name.to_owned(),
        };
    };
    let parameters = match tool.handler().descriptor().family {
        FamilyDescriptor::Tool { parameters, .. } => parameters,
        _ => Value::Object(Default::default()),
    };
    if let Some(feedback) = arguments_refusal(&parameters, call) {
        let problems = validate(&parameters, &call.function.arguments_value())
            .err()
            .unwrap_or_default();
        return Outcome::InvalidArguments { problems, feedback };
    }
    let mut dispatch = context.for_dispatch();
    let running = registry.execute(name, call.function.arguments_json(), &mut dispatch);
    let Ok(result) = tokio::time::timeout(timeout, running).await else {
        return Outcome::TimedOut;
    };
    match result.error().or(result.refusal()) {
        None => Outcome::Success(call.answer(&result)),
        Some(error) if error.kind() == ToolErrorKind::InvalidArgs => Outcome::InvalidArguments {
            problems: Vec::new(),
            feedback: error.model_output().render(),
        },
        Some(error) if error.kind() == ToolErrorKind::Timeout => Outcome::TimedOut,
        Some(error) => Outcome::Failed {
            error: error.clone(),
        },
    }
}

/// Reads one call. On failure, the id it carried (if any) and why.
fn parse_call(call_json: &str) -> Result<ToolCall, (CallId, String)> {
    let value: Value = serde_json::from_str(call_json)
        .map_err(|error| (CallId::from_wire(""), format!("not JSON: {error}")))?;
    let Value::Object(mut fields) = value else {
        return Err((CallId::from_wire(""), "a call is a JSON object".to_owned()));
    };
    let id = CallId::from_wire(fields.get("id").and_then(Value::as_str).unwrap_or_default());
    let name = match fields.get("name") {
        Some(Value::String(name)) => {
            ToolName::new(name.as_str()).map_err(|_| (id.clone(), "`name` is empty".to_owned()))?
        }
        Some(_) => return Err((id, "`name` must be a string".to_owned())),
        None => return Err((id, "missing `name`".to_owned())),
    };
    let arguments = fields.shift_remove("arguments").unwrap_or(Value::Null);
    Ok(ToolCall::new(id, ToolFunction::new(name, arguments)))
}

/// One result as the CLI prints it: `{"id", "ok"}` with the output (text as
/// a string, one JSON block as JSON, anything else as its content blocks),
/// or `{"id", "error": {"kind", "message"}}` with the model-safe text.
pub fn result_json(result: &CallResult) -> Value {
    let id = result
        .id
        .provider()
        .map_or(Value::Null, |id| Value::from(id.as_str()));
    match &result.outcome {
        Outcome::Success(answer) => {
            let ok = match ToolOutput::content(answer.content.clone()) {
                Ok(output) => match (output.as_text(), output.as_json()) {
                    (Some(text), _) => Value::from(text),
                    (None, Some(value)) => value.clone(),
                    (None, None) => json!(output),
                },
                Err(_) => Value::from(""),
            };
            json!({"id": id, "ok": ok})
        }
        outcome => json!({
            "id": id,
            "error": {"kind": outcome.kind(), "message": outcome.model_feedback()},
        }),
    }
}

/// The CLI loop: reads one call per line of `input` (any bytes; invalid
/// UTF-8 is replaced, blank lines are skipped), writes one result line per
/// call to `output`, and the full error of each failure to `log`.
pub async fn run_lines(
    registry: &ToolSet,
    context: &ToolContext,
    timeout: Duration,
    mut input: impl BufRead,
    mut output: impl Write,
    mut log: impl Write,
) -> io::Result<()> {
    let mut line = Vec::new();
    loop {
        line.clear();
        if input.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        let text = String::from_utf8_lossy(&line);
        if text.trim().is_empty() {
            continue;
        }
        let result = run_call(registry, &text, context, timeout).await;
        if let Outcome::Failed { error } = &result.outcome {
            writeln!(log, "{}", full_error(error))?;
        }
        writeln!(output, "{}", result_json(&result))?;
    }
}

/// A failure as an operator reads it: kind, message and source chain.
pub fn full_error(error: &ToolExecutionError) -> String {
    let mut text = format!("{}: {}", error.kind(), error.message());
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if cause_text != error.message() {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    text
}

#[cfg(test)]
mod tests;
