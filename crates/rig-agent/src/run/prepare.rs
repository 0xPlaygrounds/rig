//! Pure request preparation from run settings, model capabilities, history, tools,
//! and per-turn overrides. Retrieval and model execution remain driver responsibilities.
//!
//! ```
//! use rig_agent::run::{prepare::prepare_request, spec::RunSpec};
//! let request = prepare_request(&RunSpec::new(), &Default::default(), &[], vec![], None, None)?;
//! assert!(request.tools.is_empty());
//! # Ok::<(), rig_agent::run::prepare::PrepareError>(())
//! ```

use std::collections::BTreeSet;

use rig_core::completion::{
    CompletionRequestBuilder, Document, Message, ProviderCapabilities, ToolDefinition,
};
use rig_core::error::ProviderError;
use rig_core::message::ToolChoice;

use super::output::OutputMode;
use super::patch::RequestPatch;
use super::spec::RunSpec;

/// Why a request could not be prepared. Every variant is a local, pre-IO
/// error: the spec, patch and tool set cannot produce a request the model
/// could honor.
#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    /// The effective request is invalid (an impossible tool choice, an
    /// `active_tools` name that is not available, an output-tool collision).
    #[error("{0}")]
    Request(String),
    /// `output_schema` is not a valid JSON schema.
    #[error("invalid output schema: {0}")]
    InvalidOutputSchema(#[source] serde_json::Error),
}

impl From<PrepareError> for ProviderError {
    fn from(error: PrepareError) -> Self {
        ProviderError::Request(error.to_string().into())
    }
}

/// Everything a model call carries that the protocol decides, as owned data.
///
/// Apply it to a provider's request builder with [`apply`](Self::apply); the
/// driver adds only what is its own (telemetry flags), then sends. The
/// bookkeeping fields (`executable_tool_names`, `allowed_tool_names`,
/// `output_tool_name`, `output_mode`) are what the driver feeds back into the
/// run when the response arrives.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PreparedRequest {
    /// Prior messages to send: the effective (possibly augmented) preamble as a
    /// leading system message, then the history (the patch's `history` when
    /// set, else the caller's).
    pub chat_history: Vec<Message>,
    /// Static context documents followed by the patch's `extra_context`.
    pub documents: Vec<Document>,
    /// The tools advertised to the model this turn, in order: the executable
    /// tools (after any `active_tools` allow-list) plus, in Tool output mode,
    /// the synthetic output tool last.
    pub tools: Vec<ToolDefinition>,
    /// Effective sampling temperature (patch over spec).
    pub temperature: Option<f64>,
    /// Effective output-token cap, with patch values overriding the spec.
    pub max_tokens: Option<u64>,
    /// Effective provider passthrough parameters (patch shallow-merged over
    /// spec when both are objects).
    pub additional_params: Option<serde_json::Value>,
    /// Effective tool choice (patch over spec).
    pub tool_choice: Option<ToolChoice>,
    /// The provider-native structured-output constraint, set only when the
    /// resolved mode is [`OutputMode::Native`].
    pub output_schema: Option<rig_core::schemars::Schema>,
    /// The mode this turn actually runs in (never [`OutputMode::Auto`]).
    pub output_mode: OutputMode,
    /// Names of the real, dispatchable tools advertised this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Names the model may call without it being an invalid tool call: the
    /// executable tools narrowed by the tool choice, plus the output tool.
    pub allowed_tool_names: BTreeSet<String>,
    /// In Tool output mode, the synthetic output tool's name (allowed but never
    /// executable); reuse it as `committed_output_tool` on later turns.
    pub output_tool_name: Option<String>,
}

impl PreparedRequest {
    /// Apply every prepared field to a provider request builder, in the
    /// protocol's canonical order. The builder keeps its prompt; the prepared
    /// sampling fields (`temperature`, `max_tokens`, `output_schema`)
    /// overwrite whatever the driver set on it, while messages, documents,
    /// tools and additional parameters accumulate.
    pub fn apply<M>(self, builder: CompletionRequestBuilder<M>) -> CompletionRequestBuilder<M> {
        let builder = builder
            .messages(self.chat_history)
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .additional_params(self.additional_params)
            .documents(self.documents)
            .tools(self.tools)
            .output_schema(self.output_schema);
        match self.tool_choice {
            Some(tool_choice) => builder.tool_choice(tool_choice),
            None => builder,
        }
    }
}

/// Prepare one model call from settings, capability snapshot, ordered history,
/// retrieved tools in advertisement order, and an optional merged patch.
/// A committed output-tool name pins tool mode while a schema is present.
/// Returns errors for incompatible tool choices, unavailable allow-list names,
/// output-tool collisions, or invalid native schemas. Performs no I/O.
pub fn prepare_request(
    spec: &RunSpec,
    capabilities: &ProviderCapabilities,
    history: &[Message],
    tools: Vec<ToolDefinition>,
    committed_output_tool: Option<&str>,
    patch: Option<&RequestPatch>,
) -> Result<PreparedRequest, PrepareError> {
    let request_patch = patch;
    let chat_history = history;
    let preamble = spec.preamble.as_deref();
    let static_context = &spec.static_context;
    let temperature = spec.temperature;
    let max_tokens = spec.max_tokens;
    let additional_params = spec.additional_params.as_ref();
    let tool_choice = spec.tool_choice.as_ref();
    let output_schema = spec.output_schema.as_ref();
    let output_mode = &spec.output_mode;
    let output_tool_description = spec.output_tool_description.as_deref();
    let augment_output_preamble = spec.augment_output_preamble;
    let preamble = request_patch
        .and_then(|o| o.preamble.as_deref())
        .or(preamble);
    let temperature = request_patch.and_then(|o| o.temperature).or(temperature);
    let max_tokens = request_patch.and_then(|o| o.max_tokens).or(max_tokens);
    let tool_choice = request_patch
        .and_then(|o| o.tool_choice.as_ref())
        .or(tool_choice);
    // Non-object patches must replace the baseline; the object merge helper
    // otherwise retains its first argument for these inputs.
    let additional_params: Option<serde_json::Value> = match (
        additional_params,
        request_patch.and_then(|o| o.additional_params.as_ref()),
    ) {
        (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
            Some(rig_core::json_utils::merge(base.clone(), patch.clone()))
        }
        (base, patch) => patch.or(base).cloned(),
    };
    let active_tools = request_patch.and_then(|o| o.active_tools.as_deref());

    // Reserve output names against the full tool set because filtered tools may
    // return on later turns while the output name remains pinned.
    let pre_filter_tool_names: Option<BTreeSet<String>> =
        active_tools.map(|_| tools.iter().map(|tool| tool.name.clone()).collect());

    // Filter before computing executable names so request advertisements and
    // runtime validation agree. Synthetic output remains independent of this filter.
    let mut tooldefs = tools;
    if let Some(allow) = active_tools {
        if let Some(missing) = allow
            .iter()
            .find(|name| !tooldefs.iter().any(|tool| &tool.name == *name))
        {
            return Err(PrepareError::Request(format!(
                "active_tools requested tool `{missing}`, which is not available this turn"
            )));
        }
        let allowed: BTreeSet<String> = allow.iter().cloned().collect();
        tooldefs.retain(|tool| allowed.contains(&tool.name));
    }

    // The synthetic output tool must never enter the executable set.
    let executable_tool_names: BTreeSet<String> =
        tooldefs.iter().map(|tool| tool.name.clone()).collect();

    // Resolve against the actual candidate name so Specific choices can permit
    // the output tool. A committed name prevents changing output mode mid-run.
    let candidate_output_tool = committed_output_tool.map_or_else(
        || {
            pick_output_tool_name(
                pre_filter_tool_names
                    .as_ref()
                    .unwrap_or(&executable_tool_names),
            )
        },
        str::to_owned,
    );
    let resolved_mode = if committed_output_tool.is_some() && output_schema.is_some() {
        OutputMode::Tool
    } else {
        resolve_output_mode(
            output_schema.is_some(),
            !executable_tool_names.is_empty(),
            output_tool_callable(tool_choice, &candidate_output_tool),
            capabilities.composes_native_output_with_tools,
            output_mode,
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
            OutputMode::Tool if augment_output_preamble => {
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
            OutputMode::Prompted => output_schema.map(|schema| {
                let schema_json = rig_core::json_utils::to_canonical_string(schema);
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

    // Request history replacement must not mutate persisted conversation history.
    let messages_history: &[Message] = request_patch
        .and_then(|o| o.history.as_deref())
        .unwrap_or(chat_history);
    let chat_history: Vec<Message> = if let Some(preamble) = &effective_preamble {
        std::iter::once(Message::system(preamble.clone()))
            .chain(messages_history.iter().cloned())
            .collect()
    } else {
        messages_history.to_vec()
    };

    if let (Some(name), Some(schema)) = (&output_tool_name, output_schema) {
        tooldefs.push(ToolDefinition {
            name: name.clone(),
            description: output_tool_description
                .unwrap_or(
                    "Call this tool exactly once with your final answer when you are done. \
                     Its arguments are the structured result and must satisfy the output schema.",
                )
                .to_string(),
            parameters: schema.clone(),
        });
    }

    let native_schema = match (&resolved_mode, output_schema) {
        (OutputMode::Native, Some(schema)) => Some(
            rig_core::schemars::Schema::try_from(schema.clone())
                .map_err(PrepareError::InvalidOutputSchema)?,
        ),
        _ => None,
    };

    let mut documents = static_context.clone();
    if let Some(patch) = request_patch {
        documents.extend(patch.extra_context.iter().cloned());
    }

    // Reject impossible choices before spending a provider request.
    let mut allowed_tool_names = allowed_tool_names_for_choice(
        &executable_tool_names,
        tool_choice,
        output_tool_name.as_deref(),
        pre_filter_tool_names.as_ref(),
    )?;
    // The output tool must be allowed (so it isn't flagged as an invalid tool
    // call) even though it is not executable.
    if let Some(name) = &output_tool_name {
        allowed_tool_names.insert(name.clone());
    }

    Ok(PreparedRequest {
        chat_history,
        documents,
        tools: tooldefs,
        temperature,
        max_tokens,
        additional_params,
        tool_choice: tool_choice.cloned(),
        output_schema: native_schema,
        output_mode: resolved_mode,
        executable_tool_names,
        allowed_tool_names,
        output_tool_name,
    })
}

/// Base name of the synthetic output tool used by [`OutputMode::Tool`].
const DEFAULT_OUTPUT_TOOL_NAME: &str = "final_result";

/// Return whether tool choice permits the named output tool. Unspecified, auto,
/// required, and specific choices naming it permit the call.
fn output_tool_callable(tool_choice: Option<&ToolChoice>, output_tool_name: &str) -> bool {
    match tool_choice {
        None | Some(ToolChoice::Auto | ToolChoice::Required) => true,
        Some(ToolChoice::None) => false,
        Some(ToolChoice::Specific { function_names }) => function_names
            .iter()
            .any(|name| name.as_str() == output_tool_name),
    }
}

/// Resolve to a non-auto mode. Without a schema, use native output. Auto selects
/// tool output only with executable tools, a callable output tool, and no native
/// composition support. Explicit tool mode falls back to native if uncallable;
/// native and prompted modes remain unchanged when a schema is present.
fn resolve_output_mode(
    has_schema: bool,
    has_executable_tools: bool,
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
        OutputMode::Auto
            if has_executable_tools && output_tool_callable && !provider_composes_native =>
        {
            OutputMode::Tool
        }
        OutputMode::Auto => OutputMode::Native,
    }
}

/// Pick a collision-safe name for the synthetic output tool, never shadowing a
/// real executable tool (which would make the model's output call dispatchable).
fn pick_output_tool_name(executable_tool_names: &BTreeSet<String>) -> String {
    let mut name = DEFAULT_OUTPUT_TOOL_NAME.to_string();
    let mut suffix = 1u32;
    while executable_tool_names.contains(&name) {
        name = format!("{DEFAULT_OUTPUT_TOOL_NAME}_{suffix}");
        suffix += 1;
    }
    name
}

/// Validate tool choice and return permitted names. Auto and required choices
/// return executable names; specific choices return their requested names,
/// including the output tool if named. None returns an empty set.
/// Errors on required choice without any advertised tool, empty specific choices,
/// or names absent from executable tools and the output tool. Supply pre-filter
/// names only when an allow-list was applied, for filter-specific diagnostics.
pub fn allowed_tool_names_for_choice(
    executable_tool_names: &BTreeSet<String>,
    tool_choice: Option<&ToolChoice>,
    output_tool_name: Option<&str>,
    pre_filter_tool_names: Option<&BTreeSet<String>>,
) -> Result<BTreeSet<String>, PrepareError> {
    let has_advertised_tool = !executable_tool_names.is_empty() || output_tool_name.is_some();
    let hint = |active_tools_caused: bool| {
        if active_tools_caused {
            " A per-turn `active_tools` allow-list narrowed the advertised tools this turn; \
             set a compatible `tool_choice` in the same `RequestPatch`, or widen `active_tools`."
        } else {
            ""
        }
    };
    let advertised = || {
        executable_tool_names
            .iter()
            .map(String::as_str)
            .chain(output_tool_name)
            .collect::<Vec<_>>()
    };

    let allowed = match tool_choice {
        None | Some(ToolChoice::Auto) => executable_tool_names.clone(),
        Some(ToolChoice::Required) => {
            if !has_advertised_tool {
                let active_tools_caused = pre_filter_tool_names.is_some_and(|pf| !pf.is_empty());
                return Err(PrepareError::Request(format!(
                    "ToolChoice::Required forces the model to call a tool, but no tools are \
                     advertised this turn.{}",
                    hint(active_tools_caused)
                )));
            }
            executable_tool_names.clone()
        }
        Some(ToolChoice::None) => BTreeSet::new(),
        Some(ToolChoice::Specific { function_names }) => {
            if function_names.is_empty() {
                return Err(PrepareError::Request(
                    "ToolChoice::Specific requires at least one function name".to_string(),
                ));
            }

            let requested = function_names.iter().cloned().collect::<BTreeSet<String>>();
            let missing = function_names
                .iter()
                .map(String::as_str)
                .filter(|name| {
                    !executable_tool_names.contains(*name) && Some(*name) != output_tool_name
                })
                .collect::<Vec<_>>();

            if !missing.is_empty() {
                // Attribute missing names to filtering only if they existed before it.
                let active_tools_caused = pre_filter_tool_names
                    .is_some_and(|pf| missing.iter().any(|name| pf.contains(*name)));
                return Err(PrepareError::Request(format!(
                    "ToolChoice::Specific requested tool names not advertised this turn: \
                     {missing:?}. Advertised: {:?}.{}",
                    advertised(),
                    hint(active_tools_caused)
                )));
            }

            requested
        }
    };

    Ok(allowed)
}

#[cfg(test)]
mod tests;
