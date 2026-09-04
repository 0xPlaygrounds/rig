//! The pure policy: the verbatim strings the goldens pin, the one fold
//! from the run graph to the wire request, and the small derivations the
//! systems share. Every function here is over plain data and has its own
//! tests; nothing here touches the world.

use rig_core::{
    completion::{
        CompletionRequest, Document, ToolDefinition,
        message::{AssistantContent, Message, ProviderCallId, ToolCallId, ToolChoice, UserContent},
    },
    effect::{HandlerDescriptor, Outcome},
    error::{ErrorKind, ErrorReport},
    json_utils::to_canonical_string,
    tool::{ToolExecutionError, ToolResult},
    transcript::{tool_result_message, tool_result_output},
};

use crate::agent::{Failure, MessageParts, OutputKind};

/// The strings the goldens pin, written once. Each has a test in
/// `policy::tests` that compares it to the golden that pins it, cited by
/// fixture and JSON pointer in `CONTRACT.md`.
pub mod text {
    /// The output tool's default name.
    pub const OUTPUT_TOOL_NAME: &str = "final_result";

    /// The output tool's description.
    pub const OUTPUT_TOOL_DESCRIPTION: &str = "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.";

    /// Appended to the preamble (after a blank line) when the answer is
    /// asked for through the output tool; `{name}` is the tool's name.
    pub fn output_tool_augmentation(name: &str) -> String {
        format!(
            "When you have gathered enough information to answer, call the `{name}` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text."
        )
    }

    /// Appended to the preamble (after a blank line) when the answer is
    /// asked for as prompted JSON; `{schema}` is the schema's canonical
    /// rendering.
    pub fn prompted_augmentation(schema: &str) -> String {
        format!(
            "Respond with ONLY a single JSON object that conforms to this JSON Schema. Do not include any prose, explanation, or markdown code fences.\n{schema}"
        )
    }

    /// The reprompt when the model answered as text instead of calling
    /// the output tool.
    pub fn reprompt_text_answer(name: &str) -> String {
        format!(
            "Provide your final answer by calling the `{name}` tool with the structured result as its arguments, not as plain text."
        )
    }

    /// The separator between the preamble and an augmentation.
    pub const AUGMENTATION_SEPARATOR: &str = "\n\n";

    /// The result every other call of a turn gets when one call was
    /// retried or skipped: none of the turn's calls ran.
    pub const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
        "Tool not executed because another tool call in the same assistant turn was invalid.";
}

/// Whether the tool choice permits the output tool's call.
pub fn output_tool_callable(choice: Option<&ToolChoice>, name: &str) -> bool {
    match choice {
        None | Some(ToolChoice::Auto) | Some(ToolChoice::Required) => true,
        Some(ToolChoice::None) => false,
        Some(ToolChoice::Specific { function_names }) => {
            function_names.iter().any(|named| named == name)
        }
    }
}

/// The output mode a turn runs under once `Auto` is resolved, never `Auto`:
/// no schema is `Native`; an explicit `Tool` the choice forbids degrades to
/// `Native` (the constraint is still enforced, natively); `Auto` is `Tool`
/// only with a real tool of the program's own, a permitting choice, and a
/// provider that does not compose native output with tools — else
/// `Native`.
pub fn resolve_output(
    mode: OutputKind,
    has_schema: bool,
    granted_tools: usize,
    callable: bool,
    provider_composes_native: bool,
) -> OutputKind {
    if !has_schema {
        return OutputKind::Native;
    }
    match mode {
        OutputKind::Native => OutputKind::Native,
        OutputKind::Prompted => OutputKind::Prompted,
        OutputKind::Tool => {
            if callable {
                OutputKind::Tool
            } else {
                OutputKind::Native
            }
        }
        OutputKind::Auto => {
            if granted_tools > 0 && callable && !provider_composes_native {
                OutputKind::Tool
            } else {
                OutputKind::Native
            }
        }
    }
}

/// The output tool's name for a run: the default, numbered from 1 on a
/// collision with a granted tool's name (`final_result`, `final_result_1`,
/// `final_result_2`, ...).
pub fn output_tool_name(granted: &[&str]) -> String {
    let base = text::OUTPUT_TOOL_NAME;
    if !granted.contains(&base) {
        return base.to_owned();
    }
    let mut suffix = 1;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !granted.contains(&candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

/// The required fields of an object schema the arguments lack, in the
/// schema's order; a non-object argument lacks every one.
pub fn missing_required_fields(
    schema: &serde_json::Value,
    arguments: &serde_json::Value,
) -> Vec<String> {
    let required: Vec<&str> = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .map(|names| names.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    match arguments.as_object() {
        Some(object) => required
            .into_iter()
            .filter(|name| !object.contains_key(*name))
            .map(str::to_owned)
            .collect(),
        None => required.into_iter().map(str::to_owned).collect(),
    }
}

/// The reprompt when the output tool was called without every required
/// field: a tool result on the call, naming the fields.
pub fn reprompt_missing_fields(name: &str, missing: &[String]) -> String {
    format!(
        "The `{name}` arguments were missing required field(s): {}. Call `{name}` again with every required field.",
        missing.join(", ")
    )
}

/// Whether an assistant turn belongs in history: not when it has no parts,
/// or exactly one unannotated empty text part.
pub fn turn_is_empty(content: &[AssistantContent]) -> bool {
    match content {
        [] => true,
        [AssistantContent::Text(text)] => text.text.is_empty() && text.additional_params.is_none(),
        _ => false,
    }
}

/// What the fold reads: the graph, gathered by `Assemble` in order. Borrows
/// the world's data; owns nothing.
pub struct RequestGraph<'a> {
    /// The preamble, if the program has one.
    pub preamble: Option<&'a str>,
    /// The utterances in order.
    pub utterances: Vec<&'a MessageParts>,
    /// The documents attached to the turn, in order.
    pub documents: Vec<Document>,
    /// The tools granted, in advertisement order, by their descriptors.
    pub tools: Vec<&'a HandlerDescriptor>,
    /// Sampling.
    pub temperature: Option<f64>,
    /// The token budget.
    pub max_tokens: Option<u64>,
    /// Provider parameters.
    pub additional_params: Option<&'a serde_json::Value>,
    /// The program's tool choice.
    pub tool_choice: Option<&'a ToolChoice>,
    /// The output mode, resolved, and its schema.
    pub output: OutputKind,
    /// The schema, if any.
    pub schema: Option<&'a serde_json::Value>,
    /// The output tool's name, when the mode is `Tool`.
    pub output_tool: Option<&'a str>,
}

/// The one fold: the wire request from the graph. The only constructor of
/// a `CompletionRequest` in the crate (a root guard refuses another).
pub fn fold_request(graph: &RequestGraph<'_>) -> CompletionRequest {
    let mut chat_history: Vec<Message> = Vec::with_capacity(graph.utterances.len() + 2);
    let system = system_message(graph);
    if let Some(content) = system {
        chat_history.push(Message::System { content });
    }
    chat_history.extend(graph.utterances.iter().map(|parts| parts.to_message()));

    let mut tools: Vec<ToolDefinition> = graph
        .tools
        .iter()
        .filter_map(|descriptor| tool_definition(descriptor))
        .collect();
    if graph.output == OutputKind::Tool
        && let (Some(name), Some(schema)) = (graph.output_tool, graph.schema)
    {
        tools.push(ToolDefinition {
            name: name.to_owned(),
            description: text::OUTPUT_TOOL_DESCRIPTION.to_owned(),
            parameters: schema.clone(),
        });
    }

    let output_schema = match (graph.output, graph.schema) {
        (OutputKind::Native, Some(schema)) => schemars::Schema::try_from(schema.clone()).ok(),
        (OutputKind::Native, None)
        | (OutputKind::Auto | OutputKind::Tool | OutputKind::Prompted, _) => None,
    };

    CompletionRequest {
        model: None,
        chat_history,
        documents: graph.documents.clone(),
        tools,
        temperature: graph.temperature,
        max_tokens: graph.max_tokens,
        tool_choice: graph.tool_choice.cloned(),
        additional_params: graph.additional_params.cloned(),
        output_schema,
        record_telemetry_content: false,
    }
}

/// The system message: the preamble with the output mode's augmentation,
/// or none when the program has no preamble and nothing to add.
fn system_message(graph: &RequestGraph<'_>) -> Option<String> {
    let augmentation = match (graph.output, graph.output_tool, graph.schema) {
        (OutputKind::Tool, Some(name), _) => Some(text::output_tool_augmentation(name)),
        (OutputKind::Prompted, _, Some(schema)) => {
            Some(text::prompted_augmentation(&to_canonical_string(schema)))
        }
        (OutputKind::Tool, None, _)
        | (OutputKind::Prompted, _, None)
        | (OutputKind::Auto | OutputKind::Native, _, _) => None,
    };
    match (graph.preamble, augmentation) {
        (None, None) => None,
        (Some(preamble), None) => Some(preamble.to_owned()),
        (None, Some(augmentation)) => Some(augmentation),
        (Some(preamble), Some(augmentation)) => Some(format!(
            "{preamble}{}{augmentation}",
            text::AUGMENTATION_SEPARATOR
        )),
    }
}

/// A tool handler's descriptor as the definition the model sees.
pub fn tool_definition(descriptor: &HandlerDescriptor) -> Option<ToolDefinition> {
    match &descriptor.family {
        rig_core::effect::FamilyDescriptor::Tool {
            name,
            description,
            parameters,
            ..
        } => Some(ToolDefinition {
            name: name.clone(),
            description: description.clone(),
            parameters: parameters.clone(),
        }),
        rig_core::effect::FamilyDescriptor::Completion { .. }
        | rig_core::effect::FamilyDescriptor::Embed { .. }
        | rig_core::effect::FamilyDescriptor::Rerank { .. }
        | rig_core::effect::FamilyDescriptor::Memory { .. }
        | rig_core::effect::FamilyDescriptor::Retrieve { .. }
        | rig_core::effect::FamilyDescriptor::Custom { .. } => None,
    }
}

/// A tool call's result as the model sees it (CONTRACT §8.1): the
/// outcome's model-visible output for a result, a skipped result for a
/// denial, a failed result carrying the report's message for any other
/// report — or the failure the run ends in (a cancel; a report the bus
/// could not serve).
pub fn tool_result_part(
    id: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    outcome: &Result<Outcome, ErrorReport>,
) -> Result<UserContent, Failure> {
    if let Some(failure) = tool_failure(outcome) {
        return Err(failure);
    }
    let result = match outcome {
        Ok(Outcome::ToolResult { result }) => result.clone(),
        Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
            "the tool handler answered with a {} outcome",
            other.family()
        ))),
        Err(report) if report.kind == ErrorKind::Denied => {
            ToolResult::skipped(report.message.clone())
        }
        Err(report) => ToolResult::failed(
            ToolExecutionError::other(report.message.clone())
                .with_model_feedback(report.message.clone()),
        ),
    };
    Ok(tool_result_output(
        id,
        provider,
        name,
        result.output().clone(),
    ))
}

/// The failure a tool call's outcome ends the run in, if any: a cancel,
/// or a report the bus could not serve the call with (closed, no handler,
/// a replay divergence). Every other outcome is a result the model sees.
pub fn tool_failure(outcome: &Result<Outcome, ErrorReport>) -> Option<Failure> {
    match outcome {
        Ok(_) => None,
        Err(report) if report.kind == ErrorKind::Cancelled => {
            Some(Failure::Cancelled(report.clone()))
        }
        Err(report)
            if matches!(
                report.kind,
                ErrorKind::BusClosed | ErrorKind::HandlerUnavailable | ErrorKind::Divergence
            ) =>
        {
            Some(Failure::Tool(report.clone()))
        }
        Err(_) => None,
    }
}

/// The user utterance a retried or skipped turn answers with (CONTRACT
/// §8.2): `text` as the tool result of the call `id`, the invalid-peer
/// notice for every other call of `content`, in call order.
pub fn invalid_peer_results(content: &[AssistantContent], id: &str, text: &str) -> MessageParts {
    let parts = content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(tool_result_message(
                call.id.clone(),
                call.provider.clone(),
                call.function.name.clone(),
                if call.id.as_str() == id {
                    text.to_owned()
                } else {
                    text::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_owned()
                },
            )),
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => None,
        })
        .collect();
    MessageParts::User { content: parts }
}

/// The turn as it stood when an invalid call surfaced mid-stream (CONTRACT
/// §8.2, the delta wire): a streamed turn whose invalid call is retried or
/// skipped is abandoned at that call — history keeps the parts before it
/// and the call as it stood: a call built from deltas has no provider id
/// yet and its arguments only when they streamed before its name (`events`
/// says which: the call's block's first delta); a call whose block was
/// delivered whole, or a unary turn, is kept whole.
pub fn partial_turn_at(
    content: &[AssistantContent],
    events: Option<&[rig_core::streaming::StreamEvent]>,
    invalid_id: &str,
) -> Vec<AssistantContent> {
    use rig_core::streaming::{BlockKind, Delta, StreamEvent};
    let Some(events) = events else {
        return content.to_vec();
    };
    // The tool-call blocks in the order they started, each with whether its
    // first delta was the name.
    let mut blocks: Vec<(rig_core::streaming::BlockId, Option<bool>)> = Vec::new();
    for event in events {
        match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::ToolCall,
            } => blocks.push((id.clone(), None)),
            StreamEvent::BlockDelta { id, delta } => {
                if let Some((_, first)) = blocks.iter_mut().find(|(block, _)| block == id)
                    && first.is_none()
                {
                    *first = Some(match delta {
                        Delta::ToolName { .. } => true,
                        Delta::ToolArguments { .. }
                        | Delta::Text { .. }
                        | Delta::TextMeta { .. }
                        | Delta::Reasoning { .. } => false,
                    });
                }
            }
            StreamEvent::BlockStart { .. }
            | StreamEvent::BlockEnd { .. }
            | StreamEvent::Final(_)
            | StreamEvent::Unknown(_) => {}
        }
    }
    if blocks.is_empty() {
        return content.to_vec();
    }
    let mut kept = Vec::new();
    let mut call_index = 0;
    for part in content {
        match part {
            AssistantContent::ToolCall(call) => {
                let name_first = blocks.get(call_index).and_then(|(_, first)| *first);
                call_index += 1;
                if call.id.as_str() == invalid_id {
                    // A call built from deltas surfaced before its block
                    // ended: no provider id yet, and no arguments unless
                    // they streamed before the name. A block delivered
                    // whole is kept whole.
                    let mut partial = call.clone();
                    match name_first {
                        Some(true) => {
                            partial.provider = None;
                            partial.function.arguments = serde_json::Value::Null;
                        }
                        Some(false) => partial.provider = None,
                        None => {}
                    }
                    kept.push(AssistantContent::ToolCall(partial));
                    return kept;
                }
                kept.push(part.clone());
            }
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => kept.push(part.clone()),
        }
    }
    kept
}

/// The text of an assistant answer: its text parts concatenated.
pub fn answer_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            AssistantContent::Reasoning(_)
            | AssistantContent::Image(_)
            | AssistantContent::ToolCall(_) => None,
        })
        .collect()
}

/// A user message of one text part.
pub fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserContent::text(text)],
    }
}

#[cfg(test)]
mod tests;
