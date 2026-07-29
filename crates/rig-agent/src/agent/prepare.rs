//! Pure, sans-IO completion-request preparation.
//!
//! [`prepare_request`] is the de-genericized heart of request building: given
//! an [`AgentConfig`], a [`ToolCatalog`] (plain tool-definition data), the
//! current step's prompt/history from
//! [`AgentRun::next_step`](super::run::AgentRun::next_step), and an optional
//! per-turn [`RequestPatch`](super::hook::RequestPatch), it produces the
//! complete [`CompletionRequest`] plus the tool-name sets that feed
//! [`ModelTurn`](super::run::ModelTurn). It performs no IO and holds no
//! behavior — any driver (the classic runner, a hand-rolled loop, an ECS
//! system) can call it between `next_step` and its provider fulfilment.
//!
//! The one provider-dependent input is
//! `composes_native_output_with_tools`, a plain capability flag (see
//! issue #1928); drivers obtain it from their provider descriptor or model
//! handle.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use rig_core::OneOrMany;
use rig_core::completion::{CompletionError, CompletionRequest, ToolDefinition};

use super::completion::{
    allowed_tool_names_for_choice, output_tool_callable, pick_output_tool_name,
    resolve_output_mode, tool_choice_permits_output_tool,
};
use super::config::AgentConfig;
use super::hook::RequestPatch;
use super::run::OutputMode;
use crate::completion::Message;
use crate::json_utils;

/// The tool definitions advertised for a turn, as plain data.
///
/// This is the data half of the classic tool-server snapshot: what the
/// provider is told about (`definitions`, in advertisement order) and which
/// of those names the driver will actually execute (`executable`). Retrieval,
/// refresh cadence, and execution all belong to the driver; the catalog is
/// just the contract for one turn.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCatalog {
    /// Provider-facing tool definitions, in advertisement order.
    pub definitions: Vec<ToolDefinition>,
    /// Names the driver will actually execute. Normally the definitions'
    /// names; a host may advertise more than it executes (or vice versa)
    /// at its own risk.
    pub executable: BTreeSet<String>,
}

impl ToolCatalog {
    /// Build a catalog whose executable set is exactly the definitions'
    /// names.
    pub fn new(definitions: Vec<ToolDefinition>) -> Self {
        let executable = definitions.iter().map(|def| def.name.clone()).collect();
        Self {
            definitions,
            executable,
        }
    }

    /// Narrow the catalog to the named tools (per-turn `active_tools`
    /// allow-lists).
    pub fn retain_names(&mut self, allow: &BTreeSet<String>) {
        self.definitions.retain(|def| allow.contains(&def.name));
        self.executable.retain(|name| allow.contains(name));
    }

    /// Whether the catalog advertises no tools.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

/// The complete, ready-to-send request for one turn plus the tool-name sets
/// the run machine validates model output against.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PreparedRequest {
    /// The finished provider-agnostic request.
    pub request: CompletionRequest,
    /// Executable Rig tools advertised for this turn (feeds
    /// [`ModelTurn::executable_tool_names`](super::run::ModelTurn)).
    pub executable_tool_names: BTreeSet<String>,
    /// Tools the active tool choice permits (feeds
    /// [`ModelTurn::allowed_tool_names`](super::run::ModelTurn)); includes
    /// the synthetic output tool when Tool output mode is active.
    pub allowed_tool_names: BTreeSet<String>,
    /// The synthetic output-tool name when Tool output mode is active
    /// (feed back via
    /// [`AgentRun::set_output_tool_name`](super::run::AgentRun) and pass as
    /// `committed_output_tool` on later turns).
    pub output_tool_name: Option<String>,
}

/// Build the complete completion request for one turn. Pure: no IO, no
/// locks, no behavior — see the [module docs](self).
///
/// `committed_output_tool` is the output-tool name the run pinned on an
/// earlier turn ([`AgentRun::output_tool_name`](super::run::AgentRun)), and
/// `patch` is the merged per-turn hook patch
/// ([`fold_completion_actions`](super::hook::fold_completion_actions)).
///
/// # Errors
/// [`CompletionError::RequestError`] when the effective configuration cannot
/// produce a valid request: an `active_tools` allow-list naming an
/// unavailable tool, a reserved output-tool name colliding with a real tool,
/// or a tool choice that no advertised tool can satisfy.
#[allow(clippy::too_many_arguments)]
pub fn prepare_request(
    config: &AgentConfig,
    catalog: &ToolCatalog,
    composes_native_output_with_tools: bool,
    prompt: Message,
    history: &[Message],
    committed_output_tool: Option<&str>,
    patch: Option<&RequestPatch>,
) -> Result<PreparedRequest, CompletionError> {
    // Apply the per-turn request patch: each set field replaces the
    // configured value for this turn, unset fields inherit it,
    // `additional_params` is shallow-merged, and `extra_context`/`history`
    // are applied below. Per-turn only — the baseline is never mutated.
    let preamble = patch
        .and_then(|o| o.preamble.as_deref())
        .or(config.preamble.as_deref());
    let temperature = patch.and_then(|o| o.temperature).or(config.temperature);
    let max_tokens = patch.and_then(|o| o.max_tokens).or(config.max_tokens);
    let tool_choice = patch
        .and_then(|o| o.tool_choice.as_ref())
        .or(config.tool_choice.as_ref());
    let additional_params: Option<serde_json::Value> = match (
        config.additional_params.as_ref(),
        patch.and_then(|o| o.additional_params.as_ref()),
    ) {
        (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
            Some(json_utils::merge(base.clone(), patch.clone()))
        }
        (base, patch) => patch.or(base).cloned(),
    };
    let active_tools = patch.and_then(|o| o.active_tools.as_deref());

    let mut catalog = catalog.clone();

    // Capture the full tool set BEFORE an `active_tools` filter: the
    // synthetic output-tool name must avoid colliding with ANY advertised
    // tool, not just this turn's narrowed set (the pin outlives the filter).
    let pre_filter_tool_names: Option<BTreeSet<String>> = active_tools.map(|_| {
        catalog
            .definitions
            .iter()
            .map(|tool| tool.name.clone())
            .collect()
    });

    if let Some(allow) = active_tools {
        if let Some(missing) = allow
            .iter()
            .find(|name| !catalog.definitions.iter().any(|tool| &tool.name == *name))
        {
            return Err(CompletionError::RequestError(
                format!(
                    "active_tools requested tool `{missing}`, which is not available this turn"
                )
                .into(),
            ));
        }
        let allowed: BTreeSet<String> = allow.iter().cloned().collect();
        catalog.retain_names(&allowed);
    }

    let mut tooldefs = catalog.definitions.clone();

    // Executable tools are the real driver-owned tools, computed BEFORE any
    // synthetic output tool is appended.
    let executable_tool_names: BTreeSet<String> = catalog.executable.clone();

    // Resolve the effective output mode (#1928); once a run commits to a
    // Tool-mode output tool, stay pinned (see the classic path's comments).
    let resolved_mode = if committed_output_tool.is_some() && config.output_schema.is_some() {
        OutputMode::Tool
    } else {
        resolve_output_mode(
            config.output_schema.is_some(),
            !executable_tool_names.is_empty(),
            tool_choice_permits_output_tool(tool_choice),
            composes_native_output_with_tools,
            &config.output_mode,
        )
    };

    let output_tool_name = matches!(resolved_mode, OutputMode::Tool).then(|| {
        committed_output_tool.map(str::to_owned).unwrap_or_else(|| {
            pick_output_tool_name(
                pre_filter_tool_names
                    .as_ref()
                    .unwrap_or(&executable_tool_names),
            )
        })
    });

    // A name pinned on turn 1 can collide if a real tool with that name
    // becomes effective later; fail before provider IO.
    if let Some(name) = &output_tool_name
        && executable_tool_names.contains(name)
    {
        return Err(CompletionError::RequestError(
            format!(
                "real tool `{name}` conflicts with the structured-output tool reserved for this \
                 run; rename or remove the real tool, exclude it with `active_tools`, or make it \
                 visible before starting a new run so Rig can reserve a different output-tool name"
            )
            .into(),
        ));
    }

    if let Some(name) = &output_tool_name
        && !output_tool_callable(tool_choice, name)
    {
        tracing::warn!(
            "the active tool_choice forbids calling the structured-output tool while the \
             run is pinned to Tool output mode; this turn cannot emit the structured \
             result (check for a `RequestPatch` setting `tool_choice` to None or a \
             Specific set that excludes the output tool)"
        );
    }

    // Augment the preamble for Tool/Prompted modes.
    let effective_preamble: Option<String> = {
        let base = preamble.map(str::to_owned);
        let instruction = match &resolved_mode {
            OutputMode::Tool if config.augment_output_preamble => {
                output_tool_name.as_deref().map(|name| {
                    format!(
                        "When you have gathered enough information to answer, call the `{name}` \
                     tool exactly once with your final answer. Its arguments are the structured \
                     result and must satisfy the required schema. Do not return the final answer \
                     as plain text."
                    )
                })
            }
            OutputMode::Tool => None,
            OutputMode::Prompted => config.output_schema.as_ref().map(|schema| {
                let schema_json = serde_json::to_string(schema.as_value()).unwrap_or_default();
                format!(
                    "Respond with ONLY a single JSON object that conforms to this JSON Schema. \
                     Do not include any prose, explanation, or markdown code fences.\n{schema_json}"
                )
            }),
            OutputMode::Native | OutputMode::Auto => None,
        };
        match (base, instruction) {
            (Some(b), Some(i)) => Some(format!("{b}\n\n{i}")),
            (Some(b), None) => Some(b),
            (None, Some(i)) => Some(i),
            (None, None) => None,
        }
    };

    // A per-turn `history` patch replaces the messages sent this turn only.
    let messages_history: &[Message] = patch.and_then(|o| o.history.as_deref()).unwrap_or(history);
    let mut chat_history: Vec<Message> = if let Some(preamble) = &effective_preamble {
        std::iter::once(Message::system(preamble.clone()))
            .chain(messages_history.iter().cloned())
            .collect()
    } else {
        messages_history.to_vec()
    };
    chat_history.push(prompt.clone());
    let chat_history =
        OneOrMany::from_iter_optional(chat_history).unwrap_or_else(|| OneOrMany::one(prompt));

    // In Tool mode, advertise the synthetic output tool (allowed below, never
    // executable).
    if let (Some(name), Some(schema)) = (&output_tool_name, config.output_schema.as_ref()) {
        tooldefs.push(ToolDefinition {
            name: name.clone(),
            description: config
                .output_tool_description
                .as_deref()
                .unwrap_or(
                    "Call this tool exactly once with your final answer when you are done. \
                     Its arguments are the structured result and must satisfy the output schema.",
                )
                .to_string(),
            parameters: schema.clone().to_value(),
        });
    }

    // Hook-supplied extra context documents follow static context, in hook
    // registration order (they were merged in that order).
    let mut documents = config.static_context.clone();
    if let Some(patch) = patch {
        documents.extend(patch.extra_context.iter().cloned());
    }

    // Only Native mode sets the provider's native structured-output
    // constraint.
    let output_schema = matches!(resolved_mode, OutputMode::Native)
        .then(|| config.output_schema.clone())
        .flatten();

    let request = CompletionRequest {
        model: None,
        preamble: None,
        chat_history,
        documents,
        tools: tooldefs,
        temperature,
        max_tokens,
        tool_choice: tool_choice.cloned(),
        additional_params,
        output_schema,
        record_telemetry_content: config.record_telemetry_content,
    };

    // Validate the effective request locally before any provider IO.
    let mut allowed_tool_names = allowed_tool_names_for_choice(
        &executable_tool_names,
        request.tool_choice.as_ref(),
        output_tool_name.as_deref(),
        pre_filter_tool_names.as_ref(),
    )?;
    if let Some(name) = &output_tool_name {
        allowed_tool_names.insert(name.clone());
    }

    Ok(PreparedRequest {
        request,
        executable_tool_names,
        allowed_tool_names,
        output_tool_name,
    })
}
