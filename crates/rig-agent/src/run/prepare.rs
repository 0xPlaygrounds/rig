//! Pure request preparation: `(spec, tools, patch) → the request`.
//!
//! [`prepare_request`] is the protocol's answer to "given this run
//! specification, the tools available this turn, the history to send, and a
//! per-turn [`RequestPatch`], what exactly goes to the model?" It performs no
//! IO — tool *retrieval* (which tools are available) is a driver concern that
//! happens before, model *execution* happens after — so every driver builds
//! byte-identical requests from the same inputs, and the output-mode
//! resolution, synthetic output-tool synthesis, preamble augmentation and
//! tool-choice validation live in exactly one place.

use std::collections::BTreeSet;

use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequestBuilder, Document, Message,
    ProviderCapabilities, ToolDefinition,
};
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

impl From<PrepareError> for CompletionError {
    fn from(error: PrepareError) -> Self {
        CompletionError::RequestError(error.to_string().into())
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
    /// Effective output-token cap (patch over spec) — the value the request
    /// carries, so a driver can report it without reading the builder back.
    pub max_tokens: Option<u64>,
    /// Effective provider passthrough parameters (patch shallow-merged over
    /// spec when both are objects).
    pub additional_params: Option<serde_json::Value>,
    /// Effective tool choice (patch over spec).
    pub tool_choice: Option<ToolChoice>,
    /// The provider-native structured-output constraint — set only when the
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
    /// protocol's canonical order. The builder keeps its prompt and anything
    /// the driver set on it.
    pub fn apply<M: CompletionModel>(
        self,
        builder: CompletionRequestBuilder<M>,
    ) -> CompletionRequestBuilder<M> {
        let builder = builder
            .messages(self.chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params)
            .documents(self.documents)
            .tools(self.tools)
            .output_schema_opt(self.output_schema);
        match self.tool_choice {
            Some(tool_choice) => builder.tool_choice(tool_choice),
            None => builder,
        }
    }
}

/// Prepare one model call.
///
/// * `spec` — the run's configuration (preamble, static context, sampling,
///   tool choice, structured-output policy).
/// * `capabilities` — the selected model's capability snapshot; only
///   `composes_native_output_with_tools` is read, for output-mode resolution.
/// * `history` — the prior messages (the run's history for this call).
/// * `tools` — the real tools available this turn, already retrieved, in
///   advertisement order.
/// * `committed_output_tool` — the output-tool name the run committed to on
///   an earlier turn, if any; pins Tool mode and its name for the whole run.
/// * `patch` — the merged per-turn [`RequestPatch`], if any.
///
/// The result is pure data; nothing here touches a model, a registry, or an
/// executor.
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
    // Apply a per-turn request patch (the merged patch from every `CompletionCall`
    // hook): each set field replaces the agent's configured value for this turn,
    // unset fields inherit it, `additional_params` is shallow-merged, and
    // `extra_context`/`history` are applied below. This is per-turn only — it
    // never mutates the agent's baseline.
    let preamble = request_patch
        .and_then(|o| o.preamble.as_deref())
        .or(preamble);
    let temperature = request_patch.and_then(|o| o.temperature).or(temperature);
    let max_tokens = request_patch.and_then(|o| o.max_tokens).or(max_tokens);
    let tool_choice = request_patch
        .and_then(|o| o.tool_choice.as_ref())
        .or(tool_choice);
    // Provider passthrough params: when both the baseline and the override are
    // JSON objects, shallow-merge them (top-level keys, the override winning);
    // otherwise the override value wins wholesale when set, else the baseline.
    // This keeps the override winning consistently instead of silently dropping a
    // non-object patch — `json_utils::merge` returns its first argument unchanged
    // when either side isn't an object.
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

    // When a per-turn `active_tools` allow-list is present, capture the full tool
    // set BEFORE filtering: the synthetic output-tool name must avoid colliding
    // with ANY advertised tool, not just this turn's narrowed set — a tool
    // filtered out this turn can be advertised again on a later turn, while the
    // output-tool name is pinned for the whole run, so picking against only the
    // narrowed set could commit a name that collides once the filter lifts.
    // Without a filter the full set equals `executable_tool_names` below, so we
    // skip the extra allocation and reuse that.
    let pre_filter_tool_names: Option<BTreeSet<String>> =
        active_tools.map(|_| tools.iter().map(|tool| tool.name.clone()).collect());

    // Apply a per-turn `active_tools` allow-list (from a `CompletionCall` hook):
    // narrow the advertised tool set to the named tools BEFORE computing the
    // executable set, so tool-choice resolution and invalid-tool-call validation
    // all operate on the narrowed set. The synthetic output tool is appended
    // later and is unaffected, so structured output still works under an empty
    // allow-list. A name that isn't available this turn is a hook bug, surfaced
    // as a request error (mirroring `ToolChoice::Specific`'s contract).
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

    // Executable tools are the real tool-server tools, computed BEFORE any
    // synthetic output tool is appended.
    let executable_tool_names: BTreeSet<String> =
        tooldefs.iter().map(|tool| tool.name.clone()).collect();

    // Resolve the effective output mode (#1928). Once the run has committed to a
    // Tool-mode output tool on an earlier turn (signaled by `committed_output_
    // tool`, which is persisted on the run via `output_tool_name`), stay in Tool
    // mode and reuse that name — so a later turn whose tool set differs (e.g. RAG
    // retrieved no tools) can't flip Tool -> Native and re-apply the native
    // constraint that suppressed tools in the first place. Only Tool mode is
    // pinned; Native/Prompted re-resolve, so a tool-less first turn can still
    // become Tool once tools appear. Otherwise resolve from the request, the
    // schema, the tool set, whether the tool choice permits the output-tool call,
    // and whether the provider composes native structured output with tools.
    let resolved_mode = if committed_output_tool.is_some() && output_schema.is_some() {
        OutputMode::Tool
    } else {
        resolve_output_mode(
            output_schema.is_some(),
            !executable_tool_names.is_empty(),
            tool_choice_permits_output_tool(tool_choice),
            capabilities.composes_native_output_with_tools,
            output_mode,
        )
    };

    // In Tool mode, reuse the run's committed name or pick a collision-safe one
    // against the full pre-filter set (or the executable set when unfiltered).
    let output_tool_name = matches!(resolved_mode, OutputMode::Tool).then(|| {
        committed_output_tool.map_or_else(
            || {
                pick_output_tool_name(
                    pre_filter_tool_names
                        .as_ref()
                        .unwrap_or(&executable_tool_names),
                )
            },
            str::to_owned,
        )
    });

    // A freshly picked name never collides, but a name pinned on turn 1 can if a
    // real tool with that name becomes effective later (for example through a
    // shared tool server, retrieval, or an MCP refresh). The output-tool
    // intercept matches by name, so fail before provider I/O: advertising both
    // definitions would make a call to the real tool finalize the run instead
    // of reaching normal dispatch.
    if let Some(name) = &output_tool_name
        && executable_tool_names.contains(name)
    {
        return Err(PrepareError::Request(format!(
            "real tool `{name}` conflicts with the structured-output tool reserved for this \
             run; rename or remove the real tool, exclude it with `active_tools`, or make it \
             visible before starting a new run so Rig can reserve a different output-tool name"
        )));
    }

    // In committed Tool mode the run can only finalize by calling the synthetic
    // output tool, and the mode is pinned (it cannot degrade to Native mid-run,
    // see #1928). A `tool_choice` that forbids the output-tool call — `None`, or
    // a `Specific` set that excludes it, e.g. from a per-turn `RequestPatch` —
    // therefore produces a turn that cannot emit the structured result. The
    // non-committed path degrades to Native via `resolve_output_mode`, so this
    // only fires once a turn has committed Tool mode; warn rather than silently
    // stall the run. Use the name-aware check so a `Specific` set that *names*
    // the output tool (which `allowed_tool_names_for_choice` accepts) is not
    // falsely flagged as unable to finalize.
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

    // Augment the preamble for Tool/Prompted modes, then prepend it as a system
    // message (deferred from the original position so it can reference the tool).
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
                let schema_json = serde_json::to_string(schema).unwrap_or_default();
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

    // A per-turn `history` patch replaces the prior messages sent to the provider
    // *this turn only* (context-window compaction / summarization). The RAG query
    // text above deliberately still derives from the original `chat_history`, so
    // this changes only what is sent, never what is retrieved or persisted.
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

    // In Tool mode, advertise the synthetic output tool to the provider (its name
    // is added to `allowed_tool_names` below but never to `executable_tool_names`,
    // so it is never dispatched to the tool server).
    // `output_tool_name` is only `Some` when `output_schema` is `Some` (Tool mode
    // requires a schema), so this match always fires in Tool mode.
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

    // Only Native mode sets the provider's native structured-output constraint.
    let native_schema = match (&resolved_mode, output_schema) {
        (OutputMode::Native, Some(schema)) => Some(
            rig_core::schemars::Schema::try_from(schema.clone())
                .map_err(PrepareError::InvalidOutputSchema)?,
        ),
        _ => None,
    };

    // Hook-supplied extra context documents (passive RAG) follow static context,
    // with extras in hook registration order (they were merged in that order).
    // Per-turn and non-sticky: the next turn re-resolves from the baseline.
    let mut documents = static_context.clone();
    if let Some(patch) = request_patch {
        documents.extend(patch.extra_context.iter().cloned());
    }

    // Validate the effective request locally (Required/Specific vs the effective
    // advertised tool set, incl. the output tool) *before* building the send —
    // so an impossible tool_choice/tool-set combination fails here with no
    // provider round-trip, and names the `active_tools` filter when it caused it.
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

/// Whether the active [`ToolChoice`] lets the model call the synthetic output
/// tool. Tool output mode finalizes via that call, so when the choice forbids it
/// (`None`, or a `Specific` allow-list that lists only the caller's real tools)
/// Tool mode cannot work and must fall back to native structured output.
fn tool_choice_permits_output_tool(tool_choice: Option<&ToolChoice>) -> bool {
    matches!(
        tool_choice,
        None | Some(ToolChoice::Auto | ToolChoice::Required)
    )
}

/// Whether the active [`ToolChoice`] can call the *named* synthetic output tool.
///
/// Unlike [`tool_choice_permits_output_tool`] — which runs during output-mode
/// resolution, before the output-tool name is known, and so conservatively
/// treats every `Specific` set as forbidding the call — this knows the committed
/// output-tool name, so a `Specific` set that names it counts as callable. That
/// matches [`allowed_tool_names_for_choice`], which advertises the output tool
/// for exactly that choice. Only a `None` choice or a `Specific` set that omits
/// the output tool genuinely cannot finalize a pinned Tool-mode turn.
fn output_tool_callable(tool_choice: Option<&ToolChoice>, output_tool_name: &str) -> bool {
    match tool_choice {
        Some(ToolChoice::Specific { function_names }) => function_names
            .iter()
            .any(|name| name.as_str() == output_tool_name),
        other => tool_choice_permits_output_tool(other),
    }
}

/// Resolve the caller-facing [`OutputMode`] to a concrete mode for one request.
///
/// With no schema there is nothing to enforce, so the result is always `Native`
/// (the synthetic tool and prompt injection only make sense with a schema).
/// `Auto` becomes `Tool` only when a real executable tool is present, the tool
/// choice permits the output-tool call, AND the provider does *not* compose
/// native structured output with tools — i.e. only where the native constraint
/// would actually suppress tool calls (#1928). On providers that compose them
/// (OpenAI, Anthropic), `Auto` keeps guaranteed native structured output.
/// `Tool` (explicit or via `Auto`) requires that the active [`ToolChoice`]
/// permit the output-tool call; when it does not, it degrades to `Native` so
/// structured output is still enforced rather than silently dropped. Explicit
/// `Prompted`/`Native` are honored when a schema is present. The returned mode is
/// never `Auto`.
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

/// Compute the allowed tool names for a `tool_choice` **and** validate the
/// effective request locally (no provider round-trip).
///
/// The effective advertised tool set for a turn is the executable tools (after
/// any per-turn `active_tools` filtering) plus the synthetic output tool
/// (`output_tool_name`) when structured output runs in Tool mode. Validation:
///
/// - [`ToolChoice::Required`] with **no** advertised tool (no executable tool and
///   no output tool) is a local error — the model is forced to call a tool but
///   none is advertised.
/// - [`ToolChoice::Specific`] must name only advertised tools (executable tools
///   or the output tool); an empty specific set is also an error.
///
/// `pre_filter_tool_names` is the full executable tool set *before* any per-turn
/// `active_tools` filtering — `Some` only when an `active_tools` allow-list was
/// applied. When the incompatibility was actually **caused** by that filter (a
/// tool that would otherwise satisfy the choice was dropped), the error says so
/// and suggests setting a compatible `tool_choice` in the same `RequestPatch`.
/// A plain typo naming a tool that never existed is *not* blamed on the filter.
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
    // The advertised tools the model may call: executable tools + the output tool.
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
                // The filter caused this only if there *were* tools before it ran.
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
                // The filter caused this only if a missing name existed pre-filter
                // (i.e. `active_tools` dropped it) — not for a plain typo.
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
