//! What the model reads about its tool calls: a tool's result shaped as a
//! tool result, the error result of a call that never ran, what a call
//! that did not finish says ([`not_finished`]), and the feedback for
//! unavailable tool names and arguments that do not fit a tool.
//!
//! ```
//! use rig_core::tool::feedback::invalid_arguments_feedback;
//!
//! assert!(invalid_arguments_feedback("add", "[1]").contains("not a JSON object"));
//! ```

use crate::message::{
    self, AssistantContent, CallId, ToolCall, ToolName, ToolResultContent, UserContent,
};
use crate::tool::{ToolResult, Touches};

/// Shape a tool's result as the tool result the model reads, without
/// reparsing text. Anything but a success is an error result.
pub fn tool_result_output(call: CallId, name: ToolName, result: &ToolResult) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: result.output().clone().into_content(),
        is_error: !result.is_success(),
    })
}

/// Constructs the error result of a call that never ran, its text verbatim,
/// such as recovery feedback or a skip reason. JSON-shaped text is not
/// reinterpreted as structured or multimodal output.
pub fn tool_result_message(call: CallId, name: ToolName, message: String) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: vec![ToolResultContent::text(message)],
        is_error: true,
    })
}

/// What the model reads when its call to `tool` sent `raw`, arguments that
/// are not a JSON object: the tool never ran, and the model calls it again.
/// `raw` is echoed up to [`ECHOED_CHARS`].
pub fn invalid_arguments_feedback(tool: &str, raw: &str) -> String {
    format!(
        "The arguments for tool `{tool}` are not a JSON object: {}. \
         Call the tool again with a JSON object as its arguments.",
        echoed(raw)
    )
}

/// Corrects a call to an unavailable `name`, listing `tools` in sorted order.
/// The caller must supply only names the agent is allowed to use.
pub fn unknown_tool_feedback<'a>(name: &str, tools: impl IntoIterator<Item = &'a str>) -> String {
    let mut tools: Vec<_> = tools.into_iter().collect();
    tools.sort_unstable();
    let available = if tools.is_empty() {
        "You have no tools available. Answer without calling tools.".to_owned()
    } else {
        format!(
            "Your tools are: `{}`. Call one by its exact name.",
            tools.join("`, `")
        )
    };
    format!("There is no tool named `{name}`. {available}")
}

/// What the model reads for its call to `tool` that its reply ended in the
/// middle of, having reached the output limit after sending `raw` of the
/// arguments (a response's
/// [`unfinished_calls`](crate::completion::CompletionResponse::unfinished_calls)):
/// the tool never ran, and sending as much again would be cut off again.
/// `raw` is echoed up to [`ECHOED_CHARS`].
pub fn output_limit_feedback(tool: &str, raw: &str) -> String {
    format!(
        "Not run: your reply reached the output limit before `{tool}`'s arguments ended. Send \
         less in one call: split the work into smaller calls, such as writing part of a file \
         and adding the rest with edits. The arguments began: {}",
        echoed(raw)
    )
}

/// How much of a call's raw arguments its feedback echoes, in characters.
pub const ECHOED_CHARS: usize = 200;

/// `raw` up to [`ECHOED_CHARS`], saying how much more there was.
fn echoed(raw: &str) -> String {
    match raw.char_indices().nth(ECHOED_CHARS) {
        Some((end, _)) => format!(
            "{}… ({} more characters)",
            raw.get(..end).unwrap_or_default(),
            raw.len().saturating_sub(end)
        ),
        None => raw.to_owned(),
    }
}

/// The result every other call of a turn gets when one call was retried or
/// skipped: none of the turn's calls ran.
pub const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

/// The tool results answering a turn with an invalid call, in call order:
/// `feedback` for the call `invalid`, [`TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER`]
/// for every other call. Empty when `content` has no tool calls.
pub fn invalid_call_feedback(
    content: &[AssistantContent],
    invalid: &CallId,
    feedback: &str,
) -> Vec<UserContent> {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) if &call.id == invalid => Some(tool_result_message(
                call.id.clone(),
                call.function.name.clone(),
                feedback.to_owned(),
            )),
            AssistantContent::ToolCall(call) => Some(not_executed(call)),
            _ => None,
        })
        .collect()
}

/// The result of a call the run did not execute because a call beside it
/// was invalid.
pub fn not_executed(call: &ToolCall) -> UserContent {
    tool_result_message(
        call.id.clone(),
        call.function.name.clone(),
        TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_owned(),
    )
}

/// Why `call`'s arguments do not fit a tool whose JSON schema is
/// `parameters`, for the model to read as the call's error: they are not a
/// JSON object ([`invalid_arguments_feedback`]), name an argument the
/// schema does not declare, or do not [`validate`](crate::tool::validate)
/// against it (up to five of the problems are named). `None` when they fit
/// as far as the validator checks; the tool checks the rest.
///
/// A schema that declares `properties` is read as naming every argument,
/// stricter than JSON Schema, so a model that guesses an argument name hears
/// the real ones: unless it allows more (`additionalProperties` other than
/// `false`, or `patternProperties`), an undeclared argument is refused. A
/// schema that declares no `properties`, such as `{}` or
/// `{"type": "object"}`, accepts any argument.
pub fn arguments_refusal(parameters: &serde_json::Value, call: &ToolCall) -> Option<String> {
    let name = call.function.name.as_str();
    if let Some(raw) = &call.function.invalid_arguments {
        return Some(invalid_arguments_feedback(name, raw));
    }
    let allows_more = parameters
        .get("additionalProperties")
        .is_some_and(|more| more != &serde_json::Value::Bool(false))
        || parameters.get("patternProperties").is_some();
    let declared = parameters
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .filter(|_| !allows_more);
    fn quoted<'a>(names: impl Iterator<Item = &'a String>) -> String {
        names
            .map(|arg| format!("`{arg}`"))
            .collect::<Vec<_>>()
            .join(", ")
    }
    let unknown = quoted(
        call.function
            .arguments
            .keys()
            .filter(|arg| declared.is_some_and(|declared| !declared.contains_key(arg.as_str()))),
    );
    if let Some(declared) = declared.filter(|_| !unknown.is_empty()) {
        let mut known = quoted(declared.keys());
        if known.is_empty() {
            known = "none".to_owned();
        }
        return Some(format!(
            "`{name}` has no argument {unknown}. Its arguments are: {known}. Call it again with \
             only those."
        ));
    }
    const SHOWN: usize = 5;
    let arguments = serde_json::Value::Object(call.function.arguments.clone());
    let problems = crate::tool::validate::validate(parameters, &arguments).err()?;
    let mut named: Vec<String> = problems
        .iter()
        .take(SHOWN)
        .map(ToString::to_string)
        .collect();
    if let Some(more) = problems.len().checked_sub(SHOWN).filter(|more| *more > 0) {
        named.push(format!("and {more} more"));
    }
    Some(format!(
        "`{name}`'s arguments do not fit its schema: {}. Call it again with arguments that fit.",
        named.join("; ")
    ))
}

/// What the model reads of a call that did not finish, for any early end
/// ("stopped" means any of them: the user stopped the turn, a plugin or a
/// failure ended it, or the session restarted), from whether it `started`
/// and its `footprint`: one never started was not run; one that started and
/// only reads ([`Touches::reads_only`]) is safe to run again; any other,
/// [`Touches::Independent`] too, may have partly run. `why` is said as it
/// is, without a final period.
///
/// ```
/// use rig_core::tool::feedback::not_finished;
/// use rig_core::tool::{Scope, Touches};
///
/// let reads = Touches::Reads(Scope::Anything);
/// let why = "the user stopped the turn";
/// assert_eq!(not_finished(why, false, &reads), "Not run: the user stopped the turn.");
/// assert!(not_finished(why, true, &reads).ends_with("run it again if you still need it."));
/// assert!(not_finished(why, true, &Touches::Independent).contains("may have partly run"));
/// ```
pub fn not_finished(why: &str, started: bool, footprint: &Touches) -> String {
    let why = why.trim().trim_end_matches('.');
    match (started, footprint.reads_only()) {
        (false, _) => format!("Not run: {why}."),
        (true, true) => format!("Stopped: {why}. It only read; run it again if you still need it."),
        (true, false) => {
            format!("Stopped: {why}. It may have partly run; check before running it again.")
        }
    }
}

#[cfg(test)]
mod tests;
