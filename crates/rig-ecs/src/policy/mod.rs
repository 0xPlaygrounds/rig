//! Pure request assembly, output selection, and tool-result shaping over plain data.
//!
//! ```
//! use rig_ecs::{agent::OutputKind, policy::resolve_output};
//! let mode = resolve_output(OutputKind::Auto, true, 0, true, false);
//! assert_eq!(mode, OutputKind::Native);
//! ```

use rig_core::{
    completion::{
        CompletionRequest, Document, ToolDefinition,
        message::{CallId, Message, ToolChoice, ToolName, UserContent},
    },
    effect::{HandlerDescriptor, Outcome},
    error::{ErrorKind, ErrorReport},
    json_utils::to_canonical_string,
    structured_output::{
        AUGMENTATION_SEPARATOR, OUTPUT_TOOL_DESCRIPTION, output_tool_augmentation,
        prompted_augmentation,
    },
    tool::{ToolExecutionError, ToolResult},
    transcript::tool_result_output,
};

use crate::agent::{
    Failure, MessageParts, OutputKind, OutputToolConfig,
    content::parts::{ToolResultLimit, ToolResultStatus},
};

/// The output mode a turn runs under once `Auto` is resolved, never `Auto`:
/// no schema is `Native`; an explicit `Tool` the choice forbids degrades to
/// `Native` (the constraint is still enforced, natively); `Auto` is `Tool`
/// only with a real tool of the program's own, a permitting choice, and a
/// provider that does not compose native output with tools; otherwise `Native`.
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

/// Ordered request inputs gathered by assembly, borrowing message parts and
/// handler descriptors while owning the attached document list.
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
    /// Optional custom description and preamble behavior of the output tool.
    pub output_tool_config: Option<&'a OutputToolConfig>,
}

/// Construct a completion request from ordered graph inputs and resolved output
/// policy. Non-tool descriptors are omitted; an invalid native schema is omitted.
/// A graph with neither a preamble nor an utterance holds no conversation and
/// is [`ContentError::Missing`](crate::agent::content::parts::ContentError::Missing),
/// as is an empty output tool name.
pub fn fold_request(
    graph: &RequestGraph<'_>,
) -> Result<CompletionRequest, crate::agent::content::parts::ContentError> {
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
            name: ToolName::new(name)
                .map_err(|_| crate::agent::content::parts::ContentError::Missing)?,
            description: graph
                .output_tool_config
                .and_then(|config| config.description.as_deref())
                .unwrap_or(OUTPUT_TOOL_DESCRIPTION)
                .to_owned(),
            parameters: schema.clone(),
        });
    }

    let output_schema = match (graph.output, graph.schema) {
        (OutputKind::Native, Some(schema)) => {
            rig_core::schemars::Schema::try_from(schema.clone()).ok()
        }
        (OutputKind::Native, None)
        | (OutputKind::Auto | OutputKind::Tool | OutputKind::Prompted, _) => None,
    };

    if chat_history.is_empty() {
        return Err(crate::agent::content::parts::ContentError::Missing);
    }
    Ok(CompletionRequest {
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
    })
}

/// The system message: the preamble with the output mode's augmentation,
/// or none when the program has no preamble and nothing to add.
fn system_message(graph: &RequestGraph<'_>) -> Option<String> {
    let augmentation = match (graph.output, graph.output_tool, graph.schema) {
        (OutputKind::Tool, Some(name), _) => graph
            .output_tool_config
            .is_none_or(|config| config.augment_preamble)
            .then(|| output_tool_augmentation(name)),
        (OutputKind::Prompted, _, Some(schema)) => {
            Some(prompted_augmentation(&to_canonical_string(schema)))
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
            AUGMENTATION_SEPARATOR
        )),
    }
}

/// The name a tool descriptor is called by; `None` for other effect families.
pub(crate) fn tool_name(descriptor: &HandlerDescriptor) -> Option<&str> {
    match &descriptor.family {
        rig_core::effect::FamilyDescriptor::Tool { name, .. } => Some(name),
        rig_core::effect::FamilyDescriptor::Completion { .. }
        | rig_core::effect::FamilyDescriptor::Embed { .. }
        | rig_core::effect::FamilyDescriptor::Rerank { .. }
        | rig_core::effect::FamilyDescriptor::Memory { .. }
        | rig_core::effect::FamilyDescriptor::Retrieve { .. }
        | rig_core::effect::FamilyDescriptor::Custom { .. } => None,
    }
}

/// Convert a tool descriptor to its model-facing definition; return `None` for
/// other effect families and for a tool with an empty name.
pub fn tool_definition(descriptor: &HandlerDescriptor) -> Option<ToolDefinition> {
    match &descriptor.family {
        rig_core::effect::FamilyDescriptor::Tool {
            name,
            description,
            parameters,
            ..
        } => Some(ToolDefinition {
            name: ToolName::new(name.as_str()).ok()?,
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

/// Convert a tool outcome to model-visible content and a separate graph status.
/// Denials become skipped results; other nonterminal reports become error results.
/// Returns a run failure for cancellation, unavailable handlers, a closed bus,
/// or replay divergence.
pub fn tool_result_part(
    id: CallId,
    name: ToolName,
    outcome: &Result<Outcome, ErrorReport>,
) -> Result<(UserContent, ToolResultStatus), Failure> {
    if let Some(failure) = tool_failure(outcome) {
        return Err(failure);
    }
    let (result, status) = match outcome {
        Ok(Outcome::ToolResult { result }) => (
            result.clone(),
            if result.is_success() {
                ToolResultStatus::Ok
            } else if result.is_refused() {
                ToolResultStatus::Refused
            } else if result.is_skipped() {
                ToolResultStatus::Skipped
            } else {
                ToolResultStatus::Error
            },
        ),
        Ok(other) => (
            ToolResult::failed(ToolExecutionError::other(format!(
                "the tool handler answered with a {} outcome",
                other.family()
            ))),
            ToolResultStatus::WrongFamily,
        ),
        Err(report) if report.kind == ErrorKind::Denied => (
            ToolResult::skipped(report.message.clone()),
            ToolResultStatus::Denied,
        ),
        Err(report) => {
            let result = ToolResult::failed(report.clone().into());
            let status = if result.is_refused() {
                ToolResultStatus::Refused
            } else {
                ToolResultStatus::Error
            };
            (result, status)
        }
    };
    Ok((tool_result_output(id, name, &result), status))
}

/// Return `None` if text fits the limit; otherwise retain head and tail totaling
/// at most `limit.max_bytes`, cut at UTF-8 boundaries around the omission marker.
pub fn limit_tool_result_text(text: &str, limit: &ToolResultLimit) -> Option<String> {
    if text.len() <= limit.max_bytes {
        return None;
    }
    let mut head = limit.max_bytes / 2;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - (limit.max_bytes - limit.max_bytes / 2);
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    let omitted = tail - head;
    let marker = limit.marker.replace("{omitted}", &omitted.to_string());
    let head = text.get(..head).unwrap_or_default();
    let tail = text.get(tail..).unwrap_or_default();
    let mut cut = String::with_capacity(head.len() + marker.len() + tail.len());
    cut.push_str(head);
    cut.push_str(&marker);
    cut.push_str(tail);
    Some(cut)
}

/// Whether `limit_tool_results` would cut anything of `parts`: a text item
/// of a tool-result part longer than the limit.
#[must_use]
pub fn tool_results_exceed(parts: &MessageParts, limit: &ToolResultLimit) -> bool {
    let MessageParts::User { content } = parts else {
        return false;
    };
    content.iter().any(|part| match part {
        UserContent::ToolResult(result) => result.content.iter().any(|item| {
            matches!(item, rig_core::message::ToolResultContent::Text(text) if text.text.len() > limit.max_bytes)
        }),
        _ => false,
    })
}

/// `limit_tool_result_text` over every text item of every tool-result part
/// of a message; JSON and image items and every other part are untouched.
pub fn limit_tool_results(parts: &mut MessageParts, limit: &ToolResultLimit) {
    let MessageParts::User { content } = parts else {
        return;
    };
    for part in content {
        let UserContent::ToolResult(result) = part else {
            continue;
        };
        for item in &mut result.content {
            if let rig_core::message::ToolResultContent::Text(text) = item
                && let Some(cut) = limit_tool_result_text(&text.text, limit)
            {
                text.text = cut;
            }
        }
    }
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

/// Return retrieval text from the last utterance with usable text, or an empty
/// string when none exists. Tool-result-only utterances do not replace the query.
pub fn retrieval_query(utterances: &[MessageParts]) -> String {
    utterances
        .iter()
        .rev()
        .find_map(|parts| parts.to_message().rag_text())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
