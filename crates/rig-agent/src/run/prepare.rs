//! Pure request preparation from run settings, model capabilities, the prompt,
//! history, tools, and per-turn overrides. Retrieval and model execution remain
//! driver responsibilities.
//!
//! ```
//! use rig_agent::run::{prepare::prepare_request, spec::RunSpec};
//! let prepared =
//!     prepare_request(&RunSpec::new(), &Default::default(), "hi", &[], vec![], None, None)?;
//! assert!(prepared.request.tools.is_empty());
//! # Ok::<(), rig_agent::run::prepare::PrepareError>(())
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::completion::{CompletionRequest, Message, ProviderCapabilities, ToolDefinition};
use rig_core::error::ProviderError;
use rig_core::message::ToolName;
use rig_core::structured_output::{
    AUGMENTATION_SEPARATOR, OUTPUT_TOOL_DESCRIPTION, output_tool_augmentation,
    output_tool_callable, output_tool_name, prompted_augmentation,
};

use super::output::OutputMode;
use super::patch::{RequestPatch, overlay_params};
use super::policy::TurnPolicy;
use super::spec::RunSpec;

/// Why a request could not be prepared: a local, pre-IO error. The effective
/// request is invalid (an impossible tool choice, an `active_tools` name that
/// is not available, an output-tool collision).
#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    /// The request the spec, patch and tool set describe.
    #[error("{0}")]
    Request(String),
}

impl From<PrepareError> for ProviderError {
    fn from(error: PrepareError) -> Self {
        ProviderError::request(error.to_string())
    }
}

/// Everything a model call carries that the protocol decides, as owned data.
///
/// The driver adds only what is its own (telemetry flags) to
/// [`request`](Self::request), then sends it. The turn's
/// [`policy`](Self::policy) is what the driver feeds back into the run when
/// the response arrives.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PreparedRequest {
    /// The request to send. Its history is the effective (possibly augmented)
    /// preamble as a leading system message, then the patch's `history` when
    /// set (else the caller's), then the prompt. Its documents are the static
    /// context followed by the patch's `extra_context`. Its tools are the
    /// executable tools (after any `active_tools` allow-list) plus, in Tool
    /// output mode, the synthetic output tool last. Temperature, output-token
    /// cap and tool choice take the patch over the spec; `additional_params`
    /// shallow-merge the patch over the spec when both are objects and carry
    /// the spec's provider-hosted tools in `tools`. The output schema is set
    /// only when the resolved mode is [`OutputMode::Native`].
    pub request: CompletionRequest,
    /// The mode this turn actually runs in (never [`OutputMode::Auto`]).
    pub output_mode: OutputMode,
    /// The turn's tool policy: executable tools, effective tool choice
    /// (patch over spec), output tool and the names the choice allows.
    pub policy: TurnPolicy,
}

/// Prepare one model call from settings, capability snapshot, the prompt,
/// ordered history, retrieved tools in advertisement order, and an optional
/// merged patch.
/// A committed output-tool name pins tool mode while a schema is present.
/// Returns errors for incompatible tool choices, unavailable allow-list names,
/// output-tool collisions, invalid native schemas, or provider tools beside
/// `additional_params` that is not an object. Performs no I/O.
pub fn prepare_request(
    spec: &RunSpec,
    capabilities: &ProviderCapabilities,
    prompt: impl Into<Message>,
    history: &[Arc<Message>],
    tools: Vec<Arc<ToolDefinition>>,
    committed_output_tool: Option<&str>,
    patch: Option<&RequestPatch>,
) -> Result<PreparedRequest, PrepareError> {
    let output_schema = spec.output_schema.as_ref();
    let preamble = patch
        .and_then(|o| o.preamble.as_deref())
        .or(spec.preamble.as_deref());
    let temperature = patch.and_then(|o| o.temperature).or(spec.temperature);
    let max_tokens = patch.and_then(|o| o.max_tokens).or(spec.max_tokens);
    let tool_choice = patch
        .and_then(|o| o.tool_choice.as_ref())
        .or(spec.tool_choice.as_ref());
    let additional_params = overlay_params(
        spec.additional_params.clone(),
        patch.and_then(|o| o.additional_params.clone()),
    );
    // Provider tools are sent in `additional_params.tools`, which a string,
    // array or number cannot carry.
    if !spec.provider_tools.is_empty()
        && let Some(
            params @ (serde_json::Value::String(_)
            | serde_json::Value::Array(_)
            | serde_json::Value::Number(_)),
        ) = &additional_params
    {
        return Err(PrepareError::Request(format!(
            "additional_params must be a JSON object to carry provider tools, got {params}"
        )));
    }
    let active_tools = patch.and_then(|o| o.active_tools.as_deref());
    // Hosted tools also arrive as raw `additional_params.tools`; providers
    // advertise both after the function tools.
    let hosted = !spec.provider_tools.is_empty()
        || additional_params
            .as_ref()
            .and_then(|params| params.get("tools"))
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tools| !tools.is_empty());

    // Reserve output names against the full tool set because filtered tools may
    // return on later turns while the output name remains pinned.
    let pre_filter_tool_names: Option<BTreeSet<String>> =
        active_tools.map(|_| tools.iter().map(|tool| tool.name.to_string()).collect());

    // Filter before computing executable names so request advertisements and
    // runtime validation agree. Synthetic output remains independent of this filter.
    let mut tooldefs = tools;
    if let Some(allow) = active_tools {
        if let Some(missing) = allow
            .iter()
            .find(|name| !tooldefs.iter().any(|tool| tool.name == **name))
        {
            return Err(PrepareError::Request(format!(
                "active_tools requested tool `{missing}`, which is not available this turn"
            )));
        }
        let allowed: BTreeSet<String> = allow.iter().cloned().collect();
        tooldefs.retain(|tool| allowed.contains(tool.name.as_str()));
    }

    // The synthetic output tool must never enter the executable set.
    let executable_tool_names: BTreeSet<String> =
        tooldefs.iter().map(|tool| tool.name.to_string()).collect();

    // Resolve against the actual candidate name so Specific choices can permit
    // the output tool. A committed name prevents changing output mode mid-run.
    let candidate_output_tool = committed_output_tool.map_or_else(
        || {
            let taken = pre_filter_tool_names
                .as_ref()
                .unwrap_or(&executable_tool_names);
            output_tool_name(|name| taken.contains(name))
        },
        str::to_owned,
    );
    let resolved_mode = if committed_output_tool.is_some() && output_schema.is_some() {
        OutputMode::Tool
    } else {
        resolve_output_mode(
            output_schema.is_some(),
            hosted || !executable_tool_names.is_empty(),
            output_tool_callable(tool_choice, &candidate_output_tool),
            capabilities.composes_native_output_with_tools,
            &spec.output_mode,
        )
    };

    let output_tool_name =
        matches!(resolved_mode, OutputMode::Tool).then_some(candidate_output_tool);

    // A later real tool may collide with a pinned output name; reject it before
    // a real tool call can be mistaken for run finalization.
    if let Some(name) = &output_tool_name
        && executable_tool_names.contains(name)
    {
        return Err(PrepareError::Request(format!(
            "real tool `{name}` conflicts with the structured-output tool reserved for this \
             run; rename or remove the real tool, exclude it with `active_tools`, or make it \
             visible before starting a new run so Rig can reserve a different output-tool name"
        )));
    }

    // Pinned tool mode cannot fall back to native output when a later choice
    // forbids its output call, so report the incompatible policy.
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

    let effective_preamble: Option<String> = {
        let base = preamble.map(str::to_owned);
        let instruction = match &resolved_mode {
            OutputMode::Tool if spec.augment_output_preamble => {
                output_tool_name.as_deref().map(output_tool_augmentation)
            }
            OutputMode::Tool => None,
            OutputMode::Prompted => output_schema.map(|schema| {
                prompted_augmentation(&rig_core::json_utils::to_canonical_string(
                    schema.as_value(),
                ))
            }),
            OutputMode::Native | OutputMode::Auto => None,
        };
        match (base, instruction) {
            (Some(b), Some(i)) => Some(format!("{b}{AUGMENTATION_SEPARATOR}{i}")),
            (Some(b), None) => Some(b),
            (None, Some(i)) => Some(i),
            (None, None) => None,
        }
    };

    // Request history replacement must not mutate persisted conversation history.
    let messages_history: &[Arc<Message>] =
        patch.and_then(|o| o.history.as_deref()).unwrap_or(history);
    let preamble = effective_preamble.map(|preamble| Arc::new(Message::system(preamble)));
    let chat_history: Vec<Arc<Message>> = preamble
        .into_iter()
        .chain(messages_history.iter().cloned())
        .collect();

    if let (Some(name), Some(schema)) = (&output_tool_name, output_schema) {
        tooldefs.push(Arc::new(ToolDefinition {
            name: ToolName::new(name.clone()).map_err(|_| {
                PrepareError::Request("the output tool name cannot be empty".to_string())
            })?,
            description: spec
                .output_tool_description
                .as_deref()
                .unwrap_or(OUTPUT_TOOL_DESCRIPTION)
                .to_string(),
            parameters: schema.as_value().clone(),
            annotations: Default::default(),
        }));
    }

    let native_schema = match (&resolved_mode, output_schema) {
        (OutputMode::Native, Some(schema)) => Some(schema.clone()),
        _ => None,
    };

    let mut documents = spec.static_context.clone();
    if let Some(patch) = patch {
        documents.extend(patch.extra_context.iter().cloned());
    }

    // Reject impossible choices before spending a provider request.
    let policy = TurnPolicy::resolve(
        executable_tool_names,
        tool_choice.cloned(),
        output_tool_name,
        pre_filter_tool_names.as_ref(),
        hosted,
    )?;

    let mut request = CompletionRequest::new(prompt);
    request.accept_unknown_finish_reasons = spec.accept_unknown_finish_reasons;
    let request = request
        .messages(chat_history)
        .temperature(temperature)
        .max_tokens(max_tokens)
        .additional_params(additional_params)
        .options(spec.options.clone())
        .provider_options(spec.provider_options.clone())
        .documents(documents)
        .tools(tooldefs)
        .provider_tools(spec.provider_tools.clone())
        .output_schema(native_schema);
    let request = match policy.tool_choice() {
        Some(tool_choice) => request.tool_choice(tool_choice.clone()),
        None => request,
    };

    Ok(PreparedRequest {
        request,
        output_mode: resolved_mode,
        policy,
    })
}

/// Resolve to a non-auto mode. Without a schema, use native output. Auto selects
/// tool output only with advertised tools (executable or hosted), a callable
/// output tool, and no native composition support. Explicit tool mode falls
/// back to native if uncallable; native and prompted modes remain unchanged
/// when a schema is present.
fn resolve_output_mode(
    has_schema: bool,
    has_tools: bool,
    output_tool_callable: bool,
    provider_composes_native: bool,
    requested: &OutputMode,
) -> OutputMode {
    if !has_schema {
        return OutputMode::Native;
    }
    match requested {
        OutputMode::Native => OutputMode::Native,
        OutputMode::Prompted => OutputMode::Prompted,
        OutputMode::Tool if output_tool_callable => OutputMode::Tool,
        OutputMode::Tool => OutputMode::Native,
        OutputMode::Auto if has_tools && output_tool_callable && !provider_composes_native => {
            OutputMode::Tool
        }
        OutputMode::Auto => OutputMode::Native,
    }
}

#[cfg(test)]
mod tests;
