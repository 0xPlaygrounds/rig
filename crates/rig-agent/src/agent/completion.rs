use super::hook::{HookStack, RequestPatch};
use super::model::ModelHandle;
use super::prompt_request::{self, PromptRequest};
use super::run::{AgentRun, ModelTurn, OutputMode, PendingToolCall};
use super::runner::{
    AgentRunner, DEFAULT_INVALID_TOOL_CALL_RETRIES, DEFAULT_MAX_TURNS, build_agent_run,
};
use crate::{
    agent::prompt_request::streaming::StreamingPromptRequest,
    completion::{
        Chat, CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse,
        Document, Message, Prompt, PromptError, ToolDefinition, TypedPrompt,
    },
    json_utils,
    streaming::{StreamingChat, StreamingPrompt},
    tool::{
        ToolContext,
        server::{ToolRegistrySnapshot, ToolServerError, ToolServerHandle},
    },
};
use rig_core::{
    message::{ToolChoice, UserContent},
    wasm_compat::WasmCompatSend,
};
use std::{borrow::Cow, collections::BTreeSet, sync::Arc};

use super::UNKNOWN_AGENT_NAME;

/// A prepared completion request plus the executable Rig tool names advertised
/// to the provider for this turn.
pub(crate) struct PreparedCompletionRequest {
    /// Builder carrying the selected model handle: request preparation ran
    /// against this handle's captured capabilities, and the same handle
    /// executes the prepared request.
    pub(crate) builder: CompletionRequestBuilder<ModelHandle>,
    /// Exact implementations behind this turn's provider definitions.
    pub(crate) tool_snapshot: Arc<ToolRegistrySnapshot>,
    pub(crate) executable_tool_names: BTreeSet<String>,
    pub(crate) allowed_tool_names: BTreeSet<String>,
    /// When Tool output mode is active, the name of the synthetic output tool
    /// advertised to the model (allowed but not executable). See #1928.
    pub(crate) output_tool_name: Option<String>,
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
pub(crate) fn allowed_tool_names_for_choice(
    executable_tool_names: &BTreeSet<String>,
    tool_choice: Option<&ToolChoice>,
    output_tool_name: Option<&str>,
    pre_filter_tool_names: Option<&BTreeSet<String>>,
) -> Result<BTreeSet<String>, CompletionError> {
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
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Required forces the model to call a tool, but no tools are \
                         advertised this turn.{}",
                        hint(active_tools_caused)
                    )
                    .into(),
                ));
            }
            executable_tool_names.clone()
        }
        Some(ToolChoice::None) => BTreeSet::new(),
        Some(ToolChoice::Specific { function_names }) => {
            if function_names.is_empty() {
                return Err(CompletionError::RequestError(
                    "ToolChoice::Specific requires at least one function name".into(),
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
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Specific requested tool names not advertised this turn: \
                         {missing:?}. Advertised: {:?}.{}",
                        advertised(),
                        hint(active_tools_caused)
                    )
                    .into(),
                ));
            }

            requested
        }
    };

    Ok(allowed)
}

/// Inputs to [`build_prepared_completion_request`], as named fields.
///
/// The parameter list carries several adjacent same-typed values (three
/// `Option<&str>`, two `bool`s); positional passing would let a call site
/// silently transpose them (e.g. `committed_output_tool` with
/// `output_tool_description`, breaking Tool-mode pinning) with no compiler
/// signal. Named fields make every call site self-checking.
pub(crate) struct PreparedRequestInputs<'a> {
    pub(crate) model: &'a ModelHandle,
    pub(crate) prompt: Message,
    /// Borrowed for per-turn drivers that keep their history; owned for the
    /// manual surface, whose `CallModel` step hands the driver a fresh `Vec`
    /// that would otherwise be cloned and dropped.
    pub(crate) chat_history: Cow<'a, [Message]>,
    pub(crate) preamble: Option<&'a str>,
    pub(crate) static_context: &'a [Document],
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) additional_params: Option<&'a serde_json::Value>,
    pub(crate) record_telemetry_content: bool,
    pub(crate) tool_choice: Option<&'a ToolChoice>,
    pub(crate) tool_server_handle: &'a ToolServerHandle,
    pub(crate) output_schema: Option<&'a schemars::Schema>,
    pub(crate) output_mode: &'a OutputMode,
    pub(crate) output_tool_description: Option<&'a str>,
    pub(crate) augment_output_preamble: bool,
    pub(crate) request_patch: Option<&'a RequestPatch>,
}

/// Build a prepared request under `run`'s committed output-tool policy.
///
/// The single home for the #1928 read/commit pairing shared by the classic
/// driver and [`Agent::prepare_completion_request`] — the only two callers of
/// [`build_prepared_completion_request`]: the run's committed Tool-mode
/// output-tool name feeds preparation (so a committed run cannot flip back to
/// native on a later turn), and the name the prepared request actually
/// advertises is committed back onto the run before it is returned.
pub(crate) async fn build_prepared_completion_request_for_run(
    run: &mut AgentRun,
    inputs: PreparedRequestInputs<'_>,
) -> Result<PreparedCompletionRequest, CompletionError> {
    let committed_output_tool = run.output_tool_name();
    let prepared = build_prepared_completion_request(inputs, committed_output_tool).await?;
    run.set_output_tool_name(prepared.output_tool_name.clone());
    Ok(prepared)
}

/// Helper function to build a completion request from agent components while
/// preserving the executable Rig tool names sent to the provider.
/// `committed_output_tool` is the run's already-committed Tool-mode
/// output-tool name, if any (#1928); call through
/// [`build_prepared_completion_request_for_run`] so the read/commit pairing
/// stays in one place.
pub(crate) async fn build_prepared_completion_request(
    inputs: PreparedRequestInputs<'_>,
    committed_output_tool: Option<&str>,
) -> Result<PreparedCompletionRequest, CompletionError> {
    let PreparedRequestInputs {
        model,
        prompt,
        chat_history,
        preamble,
        static_context,
        temperature,
        max_tokens,
        additional_params,
        record_telemetry_content,
        tool_choice,
        tool_server_handle,
        output_schema,
        output_mode,
        output_tool_description,
        augment_output_preamble,
        request_patch,
    } = inputs;
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
            Some(json_utils::merge(base.clone(), patch.clone()))
        }
        (base, patch) => patch.or(base).cloned(),
    };
    let active_tools = request_patch.and_then(|o| o.active_tools.as_deref());

    // Retrieved tools keep their existing query-selection behavior: prefer the
    // current prompt's RAG text, then the latest matching history message.
    let retrieval_query = prompt.rag_text().or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    let mut tool_snapshot = tool_server_handle
        .snapshot_tool_defs(retrieval_query)
        .await
        .map_err(|_| CompletionError::RequestError("Failed to get tool definitions".into()))?;

    // When a per-turn `active_tools` allow-list is present, capture the full tool
    // set BEFORE filtering: the synthetic output-tool name must avoid colliding
    // with ANY advertised tool, not just this turn's narrowed set — a tool
    // filtered out this turn can be advertised again on a later turn, while the
    // output-tool name is pinned for the whole run, so picking against only the
    // narrowed set could commit a name that collides once the filter lifts.
    // Without a filter the full set equals `executable_tool_names` below, so we
    // skip the extra allocation and reuse that.
    let pre_filter_tool_names: Option<BTreeSet<String>> = active_tools.map(|_| {
        tool_snapshot
            .definitions()
            .iter()
            .map(|tool| tool.name.clone())
            .collect()
    });

    // Apply a per-turn `active_tools` allow-list (from a `CompletionCall` hook):
    // narrow the advertised tool set to the named tools BEFORE computing the
    // executable set, so tool-choice resolution and invalid-tool-call validation
    // all operate on the narrowed set. The synthetic output tool is appended
    // later and is unaffected, so structured output still works under an empty
    // allow-list. A name that isn't available this turn is a hook bug, surfaced
    // as a request error (mirroring `ToolChoice::Specific`'s contract).
    if let Some(allow) = active_tools {
        if let Some(missing) = allow.iter().find(|name| {
            !tool_snapshot
                .definitions()
                .iter()
                .any(|tool| &tool.name == *name)
        }) {
            return Err(CompletionError::RequestError(
                format!(
                    "active_tools requested tool `{missing}`, which is not available this turn"
                )
                .into(),
            ));
        }
        let allowed: BTreeSet<String> = allow.iter().cloned().collect();
        tool_snapshot.retain_names(&allowed);
    }

    let mut tooldefs = tool_snapshot.definitions().to_vec();

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
            model.capabilities().composes_native_output_with_tools,
            output_mode,
        )
    };

    // In Tool mode, reuse the run's committed name or pick a collision-safe one
    // against the full pre-filter set (or the executable set when unfiltered).
    let output_tool_name = matches!(resolved_mode, OutputMode::Tool).then(|| {
        committed_output_tool.map(str::to_owned).unwrap_or_else(|| {
            pick_output_tool_name(
                pre_filter_tool_names
                    .as_ref()
                    .unwrap_or(&executable_tool_names),
            )
        })
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
        return Err(CompletionError::RequestError(
            format!(
                "real tool `{name}` conflicts with the structured-output tool reserved for this \
                 run; rename or remove the real tool, exclude it with `active_tools`, or make it \
                 visible before starting a new run so Rig can reserve a different output-tool name"
            )
            .into(),
        ));
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

    // A per-turn `history` patch replaces the prior messages sent to the provider
    // *this turn only* (context-window compaction / summarization). The RAG query
    // text above deliberately still derives from the original `chat_history`, so
    // this changes only what is sent, never what is retrieved or persisted.
    // `into_owned` moves an already-owned history (the manual surface) and
    // clones a borrowed one (the classic driver, or a patched history).
    let messages_history: Cow<'_, [Message]> =
        match request_patch.and_then(|o| o.history.as_deref()) {
            Some(patched) => Cow::Borrowed(patched),
            None => chat_history,
        };
    let mut chat_history: Vec<Message> = messages_history.into_owned();
    if let Some(preamble) = &effective_preamble {
        chat_history.insert(0, Message::system(preamble.clone()));
    }

    // In Tool mode, advertise the synthetic output tool to the provider (its name
    // is added to `allowed_tool_names` below but never to `executable_tool_names`,
    // so it is never dispatched to the tool server).
    // `output_tool_name` is only `Some` when `output_schema` is `Some` (Tool mode
    // requires a schema), so this match always fires in Tool mode.
    if let (Some(name), Some(schema)) = (&output_tool_name, output_schema) {
        tooldefs.push(crate::completion::ToolDefinition {
            name: name.clone(),
            description: output_tool_description
                .unwrap_or(
                    "Call this tool exactly once with your final answer when you are done. \
                     Its arguments are the structured result and must satisfy the output schema.",
                )
                .to_string(),
            parameters: schema.clone().to_value(),
        });
    }

    let mut completion_request = model
        .completion_request(prompt)
        .messages(chat_history)
        .temperature_opt(temperature)
        .max_tokens_opt(max_tokens)
        .additional_params_opt(additional_params)
        .record_content_telemetry(record_telemetry_content)
        .documents(static_context.to_vec())
        .tools(tooldefs);

    // Hook-supplied extra context documents (passive RAG) follow static context,
    // with extras in hook registration order (they were merged in that order).
    // Per-turn and non-sticky: the next turn re-resolves from the baseline.
    if let Some(patch) = request_patch
        && !patch.extra_context.is_empty()
    {
        completion_request = completion_request.documents(patch.extra_context.clone());
    }

    // Only Native mode sets the provider's native structured-output constraint.
    if matches!(resolved_mode, OutputMode::Native) {
        completion_request = completion_request.output_schema_opt(output_schema.cloned());
    }

    let completion_request = if let Some(tool_choice) = tool_choice {
        completion_request.tool_choice(tool_choice.clone())
    } else {
        completion_request
    };

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

    Ok(PreparedCompletionRequest {
        builder: completion_request,
        tool_snapshot: Arc::new(tool_snapshot),
        executable_tool_names,
        allowed_tool_names,
        output_tool_name,
    })
}

/// One fully configured, hook-free provider request prepared from an
/// [`Agent`], paired with the exact turn metadata the request was built with.
///
/// Produced by [`Agent::prepare_completion_request`] for callers manually
/// driving the sans-IO [`AgentRun`] state machine. Split it with
/// [`into_parts`](Self::into_parts): the request half is sent (or built and
/// sent through a custom transport), while the [`PreparedAgentTurn`] half must
/// outlive the send so the response and the turn's tool calls can be paired
/// with the metadata and implementations this exact request advertised.
///
/// The pair is in-process state for one issued request. It is deliberately not
/// serializable and claims no durability: the durable state of a manual loop is
/// the [`AgentRun`] itself. After a cross-process resume, rebuild the agent and
/// dispatch pending calls through [`Agent::tool_server_handle`] instead.
pub struct PreparedAgentRequest {
    builder: CompletionRequestBuilder<ModelHandle>,
    turn: PreparedAgentTurn,
}

impl std::fmt::Debug for PreparedAgentRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedAgentRequest")
            .field("turn", &self.turn)
            .finish_non_exhaustive()
    }
}

impl PreparedAgentRequest {
    /// Split into the sendable request builder and the turn pairing.
    ///
    /// A consuming split, because sending consumes the builder while the turn
    /// must outlive the send. The returned builder is fully configured from the
    /// agent's baseline; send it as-is with
    /// [`send`](CompletionRequestBuilder::send) (or
    /// [`build`](CompletionRequestBuilder::build) it for a custom transport).
    /// Modifying it further is caller-owned risk: the paired
    /// [`PreparedAgentTurn`] keeps describing the request *as prepared*.
    pub fn into_parts(self) -> (CompletionRequestBuilder<ModelHandle>, PreparedAgentTurn) {
        (self.builder, self.turn)
    }
}

/// The turn-pairing half of a [`PreparedAgentRequest`]: the executable and
/// allowed tool-name sets captured during preparation, plus the exact registry
/// snapshot whose definitions were sent to the provider.
///
/// It owns exactly two operations — [`model_turn`](Self::model_turn) to convert
/// the provider's response into a [`ModelTurn`] carrying the captured name
/// sets, and [`execute_call`](Self::execute_call) to execute the resulting
/// [`AgentRunStep::CallTools`](super::run::AgentRunStep::CallTools) calls
/// through the pinned implementations. Tool registrations changed *after*
/// preparation are invisible to this turn: a replaced implementation still
/// dispatches to the one whose definition was sent, and a newly registered tool
/// is rejected even though the live registry has it.
///
/// In-process only, for the one request it was prepared with. It is not
/// serializable; after a cross-process resume use the rebuilt agent's live
/// [`Agent::tool_server_handle`], which follows current registry state instead.
pub struct PreparedAgentTurn {
    /// Exact implementations behind this turn's provider definitions.
    tool_snapshot: Arc<ToolRegistrySnapshot>,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
}

impl std::fmt::Debug for PreparedAgentTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedAgentTurn")
            .field("executable_tool_names", &self.executable_tool_names)
            .field("allowed_tool_names", &self.allowed_tool_names)
            .finish_non_exhaustive()
    }
}

impl PreparedAgentTurn {
    /// Convert the provider's response into the [`ModelTurn`] to feed
    /// [`AgentRun::model_response`], supplying the executable and allowed
    /// tool-name sets captured when the request was prepared.
    ///
    /// This removes the manual driver's name-set hazard: the caller never
    /// re-derives either set from the live registry or orders
    /// [`ModelTurn::new`]'s two same-typed set arguments by hand.
    pub fn model_turn(&self, response: CompletionResponse) -> ModelTurn {
        ModelTurn::new(
            response.message_id,
            response.choice,
            response.usage,
            self.executable_tool_names.clone(),
            self.allowed_tool_names.clone(),
        )
    }

    /// Execute one pending tool call from this turn's
    /// [`AgentRunStep::CallTools`](super::run::AgentRunStep::CallTools) step,
    /// returning the correlated tool-result content for
    /// [`AgentRun::tool_results`].
    ///
    /// A call carrying a
    /// [`preresolved_result`](PendingToolCall::preresolved_result) (from
    /// invalid tool-call recovery) is returned unchanged without executing or
    /// invoking anything. Every other call dispatches through the registry
    /// snapshot pinned at preparation, never the live registry: a name outside
    /// the prepared turn's executable set — including a tool registered after
    /// preparation — is rejected with a not-found tool result rather than
    /// executed.
    ///
    /// Like every dispatch surface, this clears result metadata left in
    /// `context` by a previous dispatch before resolving the call (including
    /// on the pre-resolved path) and publishes the new dispatch's result
    /// metadata back into it. It runs no hooks, applies no concurrency policy,
    /// and records no telemetry.
    pub async fn execute_call(
        &self,
        call: &PendingToolCall,
        context: &mut ToolContext,
    ) -> UserContent {
        if let Some(result) = &call.preresolved_result {
            // Same context hygiene as a real dispatch: stale result metadata
            // from a previous dispatch must not survive a pre-resolved call.
            context.clear_dispatch_result();
            return result.clone();
        }

        let tool_call = &call.tool_call;
        let args = json_utils::serialize_json_value(&tool_call.function.arguments);
        let result = self
            .tool_snapshot
            .execute(&tool_call.function.name, args, context)
            .await;
        call.result_content(result.output().clone())
    }
}

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// Default hooks attached with [`AgentBuilder::add_hook`](crate::agent::AgentBuilder::add_hook)
/// are used for every prompt request, plus any added on the request or runner.
///
/// # Example
/// ```no_run
/// use rig_agent::prelude::*;
/// use rig_core::{client::ProviderClient, providers::openai};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = openai::Client::from_env()?;
///
/// let comedian_agent = openai
///     .agent(openai::GPT_5_2)
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct Agent {
    /// Name of the agent used for logging and debugging
    pub(crate) name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    pub(crate) description: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub(crate) model: ModelHandle,
    /// System prompt
    pub(crate) preamble: Option<String>,
    /// Context documents always available to the agent
    pub(crate) static_context: Vec<Document>,
    /// Temperature of the model
    pub(crate) temperature: Option<f64>,
    /// Maximum number of tokens for the completion
    pub(crate) max_tokens: Option<u64>,
    /// Additional parameters to be passed to the model
    pub(crate) additional_params: Option<serde_json::Value>,
    /// Whether to record sensitive request, response, and tool content on GenAI spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs.
    pub(crate) record_telemetry_content: bool,
    pub(crate) tool_server_handle: ToolServerHandle,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    pub(crate) tool_choice: Option<ToolChoice>,
    /// Default total model-call budget, including the initial call and every
    /// retry or continuation. `None` uses the implicit budget of one.
    pub(crate) default_max_turns: Option<usize>,
    /// Default hook stack applied to every prompt request and runner created
    /// from this agent. Empty by default.
    pub(crate) hooks: HookStack,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub(crate) output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced — tool call, native structured output, or
    /// prompt injection (see [`OutputMode`] and issue #1928).
    pub(crate) output_mode: OutputMode,
    /// Optional conversation memory backend that loads/saves history per conversation id.
    pub(crate) memory: Option<Arc<dyn rig_core::memory::ConversationMemory>>,
    /// Optional default conversation id used when none is set per-request.
    pub(crate) default_conversation_id: Option<String>,
}

impl Agent {
    /// Returns the configured agent name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the configured agent description.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub(crate) fn name_or_default(&self) -> &str {
        self.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Build a hook-aware [`AgentRunner`] for this agent, seeded with the
    /// agent's default hook stack. Attach more hooks with
    /// [`AgentRunner::add_hook`], then call [`AgentRunner::run`].
    pub fn runner(&self, prompt: impl Into<Message>) -> AgentRunner {
        AgentRunner::from_agent(self, prompt)
    }

    /// Returns the agent's current default model handle.
    pub fn model_handle(&self) -> &ModelHandle {
        &self.model
    }

    /// Replace the default model used by runners created after this call.
    ///
    /// Existing runners retain their model snapshot, and replacing one cloned
    /// agent does not mutate another clone. Model-selection hooks may replace
    /// the captured default at each model-call boundary.
    pub fn set_model_handle(&mut self, model: ModelHandle) {
        self.model = model;
    }

    /// Erase and install a typed completion model as this agent's new default.
    pub fn set_model<M>(&mut self, model: M)
    where
        M: CompletionModel + 'static,
    {
        self.set_model_handle(ModelHandle::new(model));
    }

    /// Return this agent with a replacement default model handle.
    ///
    /// Model-selection hooks may replace this default for individual calls.
    pub fn with_model_handle(mut self, model: ModelHandle) -> Self {
        self.set_model_handle(model);
        self
    }

    /// Return this agent with an erased typed model as its new default.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.set_model(model);
        self
    }

    /// Resolve the provider-facing tool definitions available for a prompt.
    ///
    /// This read-only view does not expose tool dispatch. Agent execution and
    /// tool lifecycle hooks remain owned by [`Self::runner`].
    pub async fn tool_definitions(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        self.tool_server_handle.get_tool_defs(prompt).await
    }

    /// Returns an owned clone of the live handle to the tool registry this
    /// agent's tools (from `.tool()` / `.dynamic_tools()` / a shared
    /// [`ToolServer`](crate::tool::server::ToolServer)) are registered in.
    ///
    /// Most callers want [`Agent::runner`], which executes tools with hooks,
    /// memory, telemetry, and the classic concurrency and invalid-call
    /// policies. Dispatching directly through this handle runs none of those.
    ///
    /// The handle is **live**: it observes registry changes made after this
    /// call (registrations, replacements, removals). That makes it the
    /// capability available after a cross-process resume — rebuild the agent
    /// and dispatch an [`AgentRun`]'s pending tool calls through the rebuilt
    /// registry under live-registry semantics. In contrast, the prepared turn
    /// from [`Agent::prepare_completion_request`] **pins** the definitions and
    /// implementations advertised for one issued request and never sees later
    /// registry changes.
    pub fn tool_server_handle(&self) -> ToolServerHandle {
        self.tool_server_handle.clone()
    }

    /// Create a sans-IO [`AgentRun`] seeded with this agent's durable run
    /// policy: the default turn budget, tool choice, and structured-output
    /// validation schema, exactly as [`Agent::runner`] would seed its internal
    /// run.
    ///
    /// This is construction, not execution: the returned run holds no model,
    /// tools, hooks, memory, or telemetry, and the caller drives it by hand
    /// (see the [`run`](super::run) module docs). Because it is a plain
    /// [`AgentRun`], the run's builder methods keep working on it, e.g.
    /// `agent.new_run(prompt).max_turns(10)`.
    ///
    /// Pair it with [`Agent::prepare_completion_request`], which prepares each
    /// model call under the **run's** policy — including a later
    /// [`AgentRun::with_tool_choice`] override — and commits the run's
    /// structured-output tool expectation. [`AgentRun::new`] remains available
    /// for intentionally custom runs; such callers own keeping the run's
    /// policy (tool choice, output validation, output-tool name) consistent
    /// with the requests they prepare.
    pub fn new_run(&self, prompt: impl Into<Message>) -> AgentRun {
        // Same policy defaults as `AgentRunner::from_agent` (shared constants,
        // so the two cannot drift) and the same single construction site
        // (`build_agent_run`) — without materializing a throwaway runner,
        // which would deep-clone static context, preamble, and params only to
        // discard them.
        build_agent_run(
            prompt.into(),
            self.default_max_turns.unwrap_or(DEFAULT_MAX_TURNS),
            DEFAULT_INVALID_TOOL_CALL_RETRIES,
            self.output_schema.as_ref(),
            None,
            self.tool_choice.clone(),
        )
    }

    /// Prepare one fully configured, hook-free provider request from this
    /// agent's baseline configuration, for a caller manually driving an
    /// [`AgentRun`].
    ///
    /// Feed it directly from an
    /// [`AgentRunStep::CallModel`](super::run::AgentRunStep::CallModel) step:
    /// pass that step's `prompt` and `history` unchanged, plus the run being
    /// driven. The request carries the agent's preamble, static context,
    /// sampling parameters, additional provider parameters, resolved tool
    /// definitions, tool choice, and structured-output constraint. An
    /// impossible tool choice ([`ToolChoice::Required`] with no advertised
    /// tool, or [`ToolChoice::Specific`] naming an unknown tool) fails here,
    /// before any provider IO.
    ///
    /// `run` keeps request preparation and run policy paired. For structured
    /// output, preparation reads the run's committed output-tool name so a run
    /// that committed to Tool-mode output on an earlier turn cannot flip back
    /// to native later, and commits the name this request advertises back onto
    /// the run — exactly as the classic driver does — so the run's output-tool
    /// interception matches what the model was told. The advertised tool
    /// choice is likewise the **run's** (seeded from the agent by
    /// [`Agent::new_run`], and following an [`AgentRun::with_tool_choice`]
    /// override), so the run always classifies calls under the policy the
    /// provider actually saw. Nothing else about the run is read or advanced.
    ///
    /// **Hook-free means hook-free.** This runs no hooks, appends no memory,
    /// opens no classic lifecycle or telemetry spans, and applies no tool
    /// concurrency or invalid-call policy. Behavior that exists only as a hook
    /// — including passive dynamic context from
    /// [`AgentBuilder::dynamic_context`](crate::agent::AgentBuilder::dynamic_context)
    /// — simply does not happen on this surface. It also does not send the
    /// request, execute tools, or create or advance the run.
    pub async fn prepare_completion_request(
        &self,
        prompt: impl Into<Message>,
        history: Vec<Message>,
        run: &mut AgentRun,
    ) -> Result<PreparedAgentRequest, CompletionError> {
        // The run's tool choice — not the agent baseline — so a
        // `with_tool_choice` override on the run reaches the provider and the
        // run never classifies calls under a policy the request didn't carry.
        // Cloned up front because the inputs below cannot borrow `run` while
        // the wrapper holds it mutably.
        let tool_choice = run.tool_choice().cloned();
        let prepared = build_prepared_completion_request_for_run(
            run,
            PreparedRequestInputs {
                model: &self.model,
                prompt: prompt.into(),
                chat_history: Cow::Owned(history),
                preamble: self.preamble.as_deref(),
                static_context: &self.static_context,
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                additional_params: self.additional_params.as_ref(),
                record_telemetry_content: self.record_telemetry_content,
                tool_choice: tool_choice.as_ref(),
                tool_server_handle: &self.tool_server_handle,
                output_schema: self.output_schema.as_ref(),
                output_mode: &self.output_mode,
                output_tool_description: None,
                augment_output_preamble: true,
                request_patch: None,
            },
        )
        .await?;

        let PreparedCompletionRequest {
            builder,
            tool_snapshot,
            executable_tool_names,
            allowed_tool_names,
            output_tool_name: _,
        } = prepared;
        Ok(PreparedAgentRequest {
            builder,
            turn: PreparedAgentTurn {
                tool_snapshot,
                executable_tool_names,
                allowed_tool_names,
            },
        })
    }
}

// Here, we need to ensure that usage of `.prompt` on agent uses these redefinitions on the opaque
//  `Prompt` trait so that when `.prompt` is used at the call-site, it'll use the more specific
//  `PromptRequest` implementation for `Agent`, making the builder's usage fluent.
//
// References:
//  - https://github.com/rust-lang/rust/issues/121718 (refining_impl_trait)

#[allow(refining_impl_trait)]
impl Prompt for Agent {
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard> {
        PromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl Prompt for &Agent {
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name_or_default()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard> {
        PromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl Chat for Agent {
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name_or_default()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let response = PromptRequest::from_agent(self, prompt)
            .history(chat_history.clone())
            .extended_details()
            .await?;

        if let Some(messages) = response.messages {
            chat_history.extend(messages);
        }

        Ok(response.output)
    }
}

impl StreamingPrompt for Agent {
    fn stream_prompt(&self, prompt: impl Into<Message> + WasmCompatSend) -> StreamingPromptRequest {
        StreamingPromptRequest::from_agent(self, prompt)
    }
}

impl StreamingChat for Agent {
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        StreamingPromptRequest::from_agent(self, prompt).history(chat_history)
    }
}

use crate::agent::prompt_request::TypedPromptRequest;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[allow(refining_impl_trait)]
impl TypedPrompt for Agent {
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    /// Send a prompt and receive a typed structured response.
    ///
    /// The JSON schema for `T` is automatically generated and sent to the provider.
    /// Providers that support native structured outputs will constrain the model's
    /// response to match this schema.
    ///
    /// # Example
    /// ```rust,ignore
    /// use rig_core::prelude::*;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct WeatherForecast {
    ///     city: String,
    ///     temperature_f: f64,
    ///     conditions: String,
    /// }
    ///
    /// let agent = client.agent("gpt-4o").build();
    ///
    /// // Type inferred from variable
    /// let forecast: WeatherForecast = agent
    ///     .prompt_typed("What's the weather in NYC?")
    ///     .await?;
    ///
    /// // Or explicit turbofish syntax
    /// let forecast = agent
    ///     .prompt_typed::<WeatherForecast>("What's the weather in NYC?")
    ///     .max_turns(3)
    ///     .await?;
    /// ```
    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<T, prompt_request::Standard>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl TypedPrompt for &Agent {
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<T, prompt_request::Standard>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(self, prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn allowed_tool_names_defaults_to_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, None, None, None).unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_auto_and_required_allow_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Auto), None, None)
                .unwrap(),
            executable
        );
        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Required), None, None)
                .unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_none_allows_no_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::None), None, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn allowed_tool_names_specific_allows_requested_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["add".to_string()],
        };

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&choice), None, None).unwrap(),
            tool_names(&["add"])
        );
    }

    #[test]
    fn allowed_tool_names_specific_rejects_missing_tools() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["missing".to_string()],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
            .expect_err("missing specific tool should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("missing")
                    && err.to_string().contains("add")
        ));
    }

    #[test]
    fn allowed_tool_names_specific_rejects_empty_names() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec![],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
            .expect_err("empty specific tool choice should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("requires at least one function name")
        ));
    }

    #[test]
    fn output_tool_callable_honors_specific_naming_the_output_tool() {
        // Auto / Required / no explicit choice all permit the output-tool call.
        assert!(output_tool_callable(None, "final_result"));
        assert!(output_tool_callable(
            Some(&ToolChoice::Auto),
            "final_result"
        ));
        assert!(output_tool_callable(
            Some(&ToolChoice::Required),
            "final_result"
        ));
        // A `Specific` set that NAMES the output tool can call it — the case the
        // pinned Tool-mode stall warning must not flag (it is accepted by
        // `allowed_tool_names_for_choice`, which advertises the output tool).
        assert!(output_tool_callable(
            Some(&ToolChoice::Specific {
                function_names: vec!["final_result".to_string()],
            }),
            "final_result",
        ));
        // A `Specific` set that omits it — or `ToolChoice::None` — genuinely cannot
        // finalize a pinned Tool-mode turn, so the warning should still fire there.
        assert!(!output_tool_callable(
            Some(&ToolChoice::Specific {
                function_names: vec!["search".to_string()],
            }),
            "final_result",
        ));
        assert!(!output_tool_callable(
            Some(&ToolChoice::None),
            "final_result"
        ));
    }

    #[test]
    fn required_with_no_advertised_tool_is_local_error() {
        let empty = tool_names(&[]);
        let err = allowed_tool_names_for_choice(&empty, Some(&ToolChoice::Required), None, None)
            .expect_err("Required with no advertised tool must fail locally");
        assert!(matches!(
            err,
            CompletionError::RequestError(err) if err.to_string().contains("Required")
        ));
    }

    #[test]
    fn required_with_only_the_output_tool_is_allowed() {
        // Structured-output Tool mode with no real tools: the model can still be
        // forced to call the synthetic output tool, so Required is valid.
        let empty = tool_names(&[]);
        let allowed = allowed_tool_names_for_choice(
            &empty,
            Some(&ToolChoice::Required),
            Some("final_result"),
            None,
        )
        .expect("Required is satisfiable by the output tool");
        // The output tool is added to the allowed set by the caller, so the
        // executable-derived allowed set is empty here.
        assert!(allowed.is_empty());
    }

    #[test]
    fn required_with_active_tools_filter_names_the_filter_in_the_error() {
        let empty = tool_names(&[]);
        let err = allowed_tool_names_for_choice(
            &empty,
            Some(&ToolChoice::Required),
            None,
            Some(&tool_names(&["add"])),
        )
        .expect_err("Required after active_tools filtered everything must fail locally");
        let msg = err.to_string();
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
        assert!(
            msg.contains("RequestPatch"),
            "error should suggest RequestPatch: {msg}"
        );
    }

    #[test]
    fn specific_naming_a_filtered_out_tool_is_a_local_error_with_hint() {
        // active_tools narrowed the advertised set to {add}; Specific still names
        // the now-filtered-out `subtract`.
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["subtract".to_string()],
        };
        let err = allowed_tool_names_for_choice(
            &executable,
            Some(&choice),
            None,
            Some(&tool_names(&["add", "subtract"])),
        )
        .expect_err("Specific naming a filtered-out tool must fail locally");
        let msg = err.to_string();
        assert!(
            msg.contains("subtract"),
            "error should name the missing tool: {msg}"
        );
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
    }

    #[test]
    fn specific_may_name_the_output_tool() {
        // The effective advertised set includes the synthetic output tool.
        let empty = tool_names(&[]);
        let choice = ToolChoice::Specific {
            function_names: vec!["final_result".to_string()],
        };
        let allowed =
            allowed_tool_names_for_choice(&empty, Some(&choice), Some("final_result"), None)
                .expect("Specific naming the output tool is valid");
        assert_eq!(allowed, tool_names(&["final_result"]));
    }

    #[test]
    fn specific_typo_is_not_blamed_on_active_tools() {
        // Specific names a tool that never existed (a typo), even though an
        // active_tools filter was applied. The error must NOT blame active_tools,
        // because the filter never had that tool to drop.
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["nonexistent".to_string()],
        };
        let err = allowed_tool_names_for_choice(
            &executable,
            Some(&choice),
            None,
            Some(&tool_names(&["add"])),
        )
        .expect_err("Specific naming a non-existent tool must fail locally");
        let msg = err.to_string();
        assert!(msg.contains("nonexistent"), "error names the typo: {msg}");
        assert!(
            !msg.contains("active_tools"),
            "a plain typo must not be blamed on active_tools: {msg}"
        );
    }

    #[test]
    fn resolve_output_mode_without_schema_is_always_native() {
        // No schema => nothing to enforce, regardless of the requested mode or tools.
        for requested in [
            OutputMode::Auto,
            OutputMode::Tool,
            OutputMode::Native,
            OutputMode::Prompted,
        ] {
            assert_eq!(
                resolve_output_mode(false, true, true, false, &requested),
                OutputMode::Native,
                "no schema should force Native for {requested:?}"
            );
            assert_eq!(
                resolve_output_mode(false, false, true, false, &requested),
                OutputMode::Native,
            );
        }
    }

    #[test]
    fn resolve_output_mode_auto_picks_tool_only_when_tools_present() {
        // This is the #1928 fix: with tools on a provider that does NOT compose
        // native output with tools, the schema must not be a native `format`
        // constraint on every turn, so Auto routes to Tool.
        assert_eq!(
            resolve_output_mode(true, true, true, false, &OutputMode::Auto),
            OutputMode::Tool,
        );
        // No tools => native structured output is safe and preferred.
        assert_eq!(
            resolve_output_mode(true, false, true, false, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_auto_keeps_native_when_provider_composes() {
        // On providers that compose native structured output with tools (OpenAI,
        // Anthropic), Auto keeps guaranteed native output even with tools present.
        assert_eq!(
            resolve_output_mode(true, true, true, true, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_honors_explicit_choice_with_schema() {
        for (requested, expected) in [
            (OutputMode::Tool, OutputMode::Tool),
            (OutputMode::Native, OutputMode::Native),
            (OutputMode::Prompted, OutputMode::Prompted),
        ] {
            // Explicit modes are honored regardless of tools or provider support.
            assert_eq!(
                resolve_output_mode(true, true, true, false, &requested),
                expected
            );
            assert_eq!(
                resolve_output_mode(true, false, true, true, &requested),
                expected
            );
        }
    }

    #[test]
    fn resolve_output_mode_degrades_to_native_when_output_tool_not_callable() {
        // Tool mode finalizes via the output-tool call; when the tool choice
        // forbids it (None / Specific), structured output must still be enforced
        // via Native rather than silently dropped (#1928 regression guard).
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Auto),
            OutputMode::Native,
        );
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Tool),
            OutputMode::Native,
        );
        // Prompted does not rely on tools, so it is unaffected.
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Prompted),
            OutputMode::Prompted,
        );
    }

    #[test]
    fn tool_choice_permits_output_tool_only_for_auto_required_or_unset() {
        assert!(tool_choice_permits_output_tool(None));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Auto)));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Required)));
        assert!(!tool_choice_permits_output_tool(Some(&ToolChoice::None)));
        assert!(!tool_choice_permits_output_tool(Some(
            &ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            }
        )));
    }

    #[test]
    fn pick_output_tool_name_defaults_when_unused() {
        let executable = tool_names(&["add", "subtract"]);
        assert_eq!(pick_output_tool_name(&executable), DEFAULT_OUTPUT_TOOL_NAME);
    }

    #[test]
    fn pick_output_tool_name_avoids_collision_with_real_tools() {
        // A user tool literally named `final_result` must not be shadowed, or
        // the model's output call would be dispatched to the tool server.
        let executable = tool_names(&["final_result"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_1");

        let executable = tool_names(&["final_result", "final_result_1"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_2");
    }
}

/// Tests for the minimal configured-`AgentRun` integration surface:
/// [`Agent::tool_server_handle`], [`Agent::new_run`],
/// [`Agent::prepare_completion_request`], [`PreparedAgentRequest`], and
/// [`PreparedAgentTurn`].
#[cfg(test)]
mod agent_run_surface_tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use serde_json::json;

    use super::*;
    use crate::{
        agent::{
            AgentBuilder,
            hook::InvalidToolCallAction,
            hook::{AgentHook, CompletionCall, CompletionCallAction, HookContext},
            run::{AgentRunStep, ModelTurnOutcome},
        },
        test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockTurn},
        tool::{DynamicTool, Tool, ToolExecutionError, ToolOutput},
    };
    use rig_core::message::{ToolCall, ToolFunction, ToolResultContent};

    /// Concatenated literal text of a tool-result `UserContent`.
    fn result_text(content: &UserContent) -> String {
        let UserContent::ToolResult(result) = content else {
            panic!("expected a tool result, got {content:?}");
        };
        result
            .content
            .iter()
            .filter_map(ToolResultContent::as_text)
            .collect()
    }

    /// Advance a fresh run to its first `CallModel` step.
    fn first_call_model(run: &mut AgentRun) -> (Message, Vec<Message>) {
        match run.next_step().expect("first step") {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => (prompt, history),
            other => panic!("expected CallModel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_server_handle_sees_and_executes_agent_built_tools() {
        let agent = AgentBuilder::new(MockCompletionModel::text("done"))
            .tool(MockAddTool)
            .build();

        let handle = agent.tool_server_handle();
        let defs = handle.get_tool_defs(None).await.expect("tool definitions");
        assert!(defs.iter().any(|def| def.name == "add"));

        let mut context = ToolContext::new();
        let result = handle
            .execute("add", r#"{"x":2,"y":3}"#, &mut context)
            .await;
        assert_eq!(result.output().render(), "5");
    }

    #[tokio::test]
    async fn tool_server_handle_sees_and_executes_dynamic_tools() {
        let echo = DynamicTool::new(
            "echo",
            "Echoes its arguments",
            json!({"type": "object", "properties": {}}),
            |_context, args| Box::pin(async move { Ok(ToolOutput::json(args)) }),
        );
        let agent = AgentBuilder::new(MockCompletionModel::text("done"))
            .dynamic_tools(vec![echo])
            .build();

        let handle = agent.tool_server_handle();
        let defs = handle.get_tool_defs(None).await.expect("tool definitions");
        assert!(defs.iter().any(|def| def.name == "echo"));

        let mut context = ToolContext::new();
        let result = handle
            .execute("echo", r#"{"hello":"world"}"#, &mut context)
            .await;
        assert_eq!(result.output().render(), r#"{"hello":"world"}"#);
    }

    #[tokio::test]
    async fn prepared_request_carries_the_agent_baseline() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .preamble("You are the baseline preamble")
            .context("baseline context document")
            .temperature(0.3)
            .max_tokens(77)
            .additional_params(json!({"top_p": 0.5}))
            .tool_choice(ToolChoice::Auto)
            .tool(MockAddTool)
            .build();

        let mut run = agent.new_run("go");
        let (prompt, history) = first_call_model(&mut run);
        let (builder, _turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let request = builder.build();

        assert!(request.chat_history.iter().any(|message| matches!(
            message,
            Message::System { content } if content == "You are the baseline preamble"
        )));
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.text == "baseline context document")
        );
        assert_eq!(request.temperature, Some(0.3));
        assert_eq!(request.max_tokens, Some(77));
        assert_eq!(request.additional_params, Some(json!({"top_p": 0.5})));
        assert_eq!(request.tool_choice, Some(ToolChoice::Auto));
        assert!(request.tools.iter().any(|tool| tool.name == "add"));
    }

    #[tokio::test]
    async fn model_turn_hands_off_the_exact_name_sets() {
        // Two executable tools plus a restrictive Specific tool choice. The
        // test never reconstructs either name set by hand: the sets reach the
        // run exclusively through `PreparedAgentTurn::model_turn`.
        let build_agent = |model: MockCompletionModel| {
            AgentBuilder::new(model)
                .tool(MockAddTool)
                .tool(MockSubtractTool)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["add".to_string()],
                })
                .build()
        };

        // An allowed call ("add") is accepted and reaches CallTools.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call-1",
            "add",
            json!({"x": 1, "y": 2}),
        )]);
        let agent = build_agent(model);
        let mut run = agent.new_run("go").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        let outcome = run
            .model_response(turn.model_turn(response))
            .expect("model response accepted");
        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
        match run.next_step().expect("next step") {
            AgentRunStep::CallTools { calls } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].tool_call.function.name, "add");
            }
            other => panic!("expected CallTools, got {other:?}"),
        }

        // A disallowed call ("subtract" — executable but outside the Specific
        // allow-list) is flagged for resolution.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call-2",
            "subtract",
            json!({"x": 5, "y": 2}),
        )]);
        let agent = build_agent(model);
        let mut run = agent.new_run("go").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        let outcome = run
            .model_response(turn.model_turn(response))
            .expect("model response accepted");
        match outcome {
            ModelTurnOutcome::NeedsResolution(context) => {
                assert_eq!(context.tool_name, "subtract");
                assert!(context.available_tools.contains(&"subtract".to_string()));
                assert!(!context.allowed_tools.contains(&"subtract".to_string()));
            }
            other => panic!("expected NeedsResolution, got {other:?}"),
        }
    }

    #[derive(Clone)]
    struct ProbeA;

    impl Tool for ProbeA {
        const NAME: &'static str = "probe";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Implementation A".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok("A".to_string())
        }
    }

    #[derive(Clone)]
    struct ProbeB;

    impl Tool for ProbeB {
        const NAME: &'static str = "probe";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Implementation B".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok("B".to_string())
        }
    }

    #[derive(Clone)]
    struct LateTool;

    impl Tool for LateTool {
        const NAME: &'static str = "late";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Registered after preparation".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok("late".to_string())
        }
    }

    #[tokio::test]
    async fn prepared_turn_executes_through_the_pinned_snapshot() {
        let model =
            MockCompletionModel::from_turns([MockTurn::tool_call("call-1", "probe", json!({}))]);
        let agent = AgentBuilder::new(model).tool(ProbeA).build();
        let mut run = agent.new_run("go").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        run.model_response(turn.model_turn(response))
            .expect("model response accepted");
        let AgentRunStep::CallTools { calls } = run.next_step().expect("next step") else {
            panic!("expected CallTools");
        };

        // Mutate the live registry AFTER preparation: replace `probe` with
        // implementation B and register a brand-new tool.
        let handle = agent.tool_server_handle();
        handle.add_tool(ProbeB).await;
        handle.add_tool(LateTool).await;

        // The prepared turn still executes implementation A...
        let mut context = ToolContext::new();
        let result = turn.execute_call(&calls[0], &mut context).await;
        assert_eq!(result_text(&result), "A");

        // ...while the live handle now dispatches implementation B.
        let live = handle.execute("probe", "{}", &mut context).await;
        assert_eq!(live.output().render(), "B");

        // A tool registered after preparation is rejected by the prepared
        // turn even though the live registry has it.
        let late_call = PendingToolCall {
            tool_call: ToolCall::from_wire("call-2", ToolFunction::new("late".into(), json!({}))),
            preresolved_result: None,
            internal_call_id: None,
        };
        let rejected = turn.execute_call(&late_call, &mut context).await;
        assert!(
            result_text(&rejected).contains("not found"),
            "post-preparation registration must be rejected: {rejected:?}"
        );
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Summary {
        answer: String,
    }

    #[tokio::test]
    async fn seeded_run_and_prepared_request_agree_on_output_policy() {
        // Schema-configured agent with a real tool on a provider that does not
        // compose native structured output with tools: output mode resolves to
        // Tool. The run must intercept exactly the output tool the request
        // advertises — proving `new_run` + `prepare_completion_request` agree.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call-1",
            "final_result",
            json!({"answer": "42"}),
        )]);
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .output_schema::<Summary>()
            .build();

        let mut run = agent.new_run("go").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();

        // Preparation committed the output-tool expectation onto the run.
        assert_eq!(run.output_tool_name(), Some("final_result"));

        let response = builder.send().await.expect("mock send");
        run.model_response(turn.model_turn(response))
            .expect("model response accepted");

        // The request advertised the synthetic output tool (Tool mode, no
        // native constraint), and the run finalizes by intercepting its call.
        let requests = model.requests();
        let request = requests.first().expect("one request sent");
        assert!(request.tools.iter().any(|tool| tool.name == "final_result"));
        assert!(request.output_schema.is_none());
        match run.next_step().expect("final step") {
            AgentRunStep::Done(response) => {
                assert_eq!(response.output, r#"{"answer":"42"}"#);
            }
            other => panic!("expected Done via output-tool interception, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_tool_choice_override_reaches_the_prepared_request() {
        // The run — not the agent baseline — is the source of truth for tool
        // choice: an `AgentRun::with_tool_choice` override after `new_run`
        // must reach the provider request and the run's allowed set alike.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call-1",
            "subtract",
            json!({"x": 5, "y": 2}),
        )]);
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Auto)
            .build();

        let mut run = agent
            .new_run("go")
            .max_turns(2)
            .with_tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            });
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");

        // The provider saw the override, not the agent's Auto baseline...
        let requests = model.requests();
        let request = requests.first().expect("one request");
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
        );

        // ...and the run classifies under that same policy: the executable
        // `subtract` call is disallowed by the overridden choice.
        match run
            .model_response(turn.model_turn(response))
            .expect("model response accepted")
        {
            ModelTurnOutcome::NeedsResolution(context) => {
                assert_eq!(context.tool_name, "subtract");
            }
            other => panic!("expected NeedsResolution under the override, got {other:?}"),
        }

        // An override that is impossible against the agent's tools still
        // fails before provider IO.
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone()).tool(MockAddTool).build();
        let mut run = agent.new_run("go").with_tool_choice(ToolChoice::Specific {
            function_names: vec!["missing".to_string()],
        });
        let (prompt, history) = first_call_model(&mut run);
        let error = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect_err("an impossible override must fail locally");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert_eq!(model.request_count(), 0, "no provider IO may happen");
    }

    #[tokio::test]
    async fn seeded_run_validates_output_against_the_agent_schema() {
        // `new_run` seeds the run's output validation from the agent's schema:
        // an output-tool call missing a required field is re-prompted (within
        // the default output-retry budget) instead of finalized as-is.
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call-1", "final_result", json!({"wrong_field": true})),
            MockTurn::tool_call("call-2", "final_result", json!({"answer": "42"})),
        ]);
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .output_schema::<Summary>()
            .build();

        let mut run = agent.new_run("go").max_turns(3);
        loop {
            match run.next_step().expect("step") {
                AgentRunStep::CallModel {
                    prompt, history, ..
                } => {
                    let (builder, turn) = agent
                        .prepare_completion_request(prompt, history, &mut run)
                        .await
                        .expect("preparation succeeds")
                        .into_parts();
                    let response = builder.send().await.expect("mock send");
                    run.model_response(turn.model_turn(response))
                        .expect("model response accepted");
                }
                AgentRunStep::CallTools { calls } => {
                    panic!("output-tool calls must be intercepted, got {calls:?}")
                }
                AgentRunStep::Done(response) => {
                    assert_eq!(response.output, r#"{"answer":"42"}"#);
                    break;
                }
            }
        }
        assert_eq!(
            model.request_count(),
            2,
            "the incomplete output must be re-prompted exactly once"
        );
    }

    #[tokio::test]
    async fn seeded_run_uses_agent_turn_budget_and_retry_budget() {
        // `new_run` seeds the agent's `default_max_turns`: with a budget of 2,
        // a third model call fails with MaxTurnsError { max_turns: 2 }.
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call-1", "add", json!({"x": 1, "y": 1})),
            MockTurn::tool_call("call-2", "add", json!({"x": 2, "y": 2})),
        ]);
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .default_max_turns(2)
            .build();

        let mut run = agent.new_run("go");
        let error = loop {
            match run.next_step() {
                Ok(AgentRunStep::CallModel {
                    prompt, history, ..
                }) => {
                    let (builder, turn) = agent
                        .prepare_completion_request(prompt, history, &mut run)
                        .await
                        .expect("preparation succeeds")
                        .into_parts();
                    let response = builder.send().await.expect("mock send");
                    run.model_response(turn.model_turn(response))
                        .expect("model response accepted");
                }
                Ok(AgentRunStep::CallTools { calls }) => {
                    let mut results = Vec::with_capacity(calls.len());
                    for call in &calls {
                        // Execution source is irrelevant here; the budget is
                        // the subject under test.
                        results.push(UserContent::tool_result_for(
                            call.tool_call.id.clone(),
                            call.tool_call.provider.clone(),
                            call.tool_call.function.name.clone(),
                            vec![ToolResultContent::text("2")],
                        ));
                    }
                    run.tool_results(results).expect("results accepted");
                }
                Ok(AgentRunStep::Done(response)) => {
                    panic!("run must exhaust its turn budget, got {response:?}")
                }
                Err(error) => break error,
            }
        };
        assert!(
            matches!(error, PromptError::MaxTurnsError { max_turns: 2, .. }),
            "expected the agent-seeded budget of 2, got {error:?}"
        );

        // `new_run` also seeds the runner's default invalid-call retry budget
        // of zero: a Retry resolution is immediately rejected.
        let model =
            MockCompletionModel::from_turns([MockTurn::tool_call("call-1", "bogus", json!({}))]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let mut run = agent.new_run("go").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        let ModelTurnOutcome::NeedsResolution(context) = run
            .model_response(turn.model_turn(response))
            .expect("model response accepted")
        else {
            panic!("an unknown tool call must need resolution");
        };
        assert_eq!(context.tool_name, "bogus");
        let error = run
            .resolve_invalid_tool_call(InvalidToolCallAction::retry("use a real tool"))
            .expect_err("a zero retry budget must reject Retry");
        assert!(matches!(error, PromptError::UnknownToolCall { .. }));
    }

    #[tokio::test]
    async fn impossible_tool_choice_fails_before_provider_io() {
        // Required with no tools at all.
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone())
            .tool_choice(ToolChoice::Required)
            .build();
        let mut run = agent.new_run("go");
        let (prompt, history) = first_call_model(&mut run);
        let error = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect_err("Required with no tools must fail locally");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert_eq!(model.request_count(), 0, "no provider IO may happen");

        // Specific naming an unknown tool.
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .build();
        let mut run = agent.new_run("go");
        let (prompt, history) = first_call_model(&mut run);
        let error = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect_err("Specific naming an unknown tool must fail locally");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert_eq!(model.request_count(), 0, "no provider IO may happen");
    }

    /// A completion-call hook that counts invocations and visibly patches the
    /// request, to pin the hook-free boundary.
    #[derive(Clone, Default)]
    struct CountingPatchHook(Arc<AtomicUsize>);

    impl AgentHook for CountingPatchHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: CompletionCall<'_>,
        ) -> CompletionCallAction {
            self.0.fetch_add(1, Ordering::SeqCst);
            CompletionCallAction::Patch(RequestPatch {
                preamble: Some("patched by hook".to_string()),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn prepare_completion_request_runs_no_hooks() {
        let hook = CountingPatchHook::default();
        let model = MockCompletionModel::text("done");
        let agent = AgentBuilder::new(model.clone())
            .preamble("baseline preamble")
            .add_hook(hook.clone())
            .build();

        let mut run = agent.new_run("go");
        let (prompt, history) = first_call_model(&mut run);
        let (builder, _turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let request = builder.build();

        assert_eq!(hook.0.load(Ordering::SeqCst), 0, "no hook may fire");
        assert!(
            request.chat_history.iter().any(|message| matches!(
                message,
                Message::System { content } if content == "baseline preamble"
            )),
            "the baseline preamble must be untouched by the hook patch"
        );
    }

    /// Guard: the same hook still fires (and patches) under `AgentRunner`.
    #[tokio::test]
    async fn completion_call_hook_still_fires_under_the_runner() {
        let hook = CountingPatchHook::default();
        let model = MockCompletionModel::text("done");
        let agent = AgentBuilder::new(model.clone())
            .preamble("baseline preamble")
            .add_hook(hook.clone())
            .build();

        agent.runner("go").run().await.expect("runner succeeds");

        assert_eq!(hook.0.load(Ordering::SeqCst), 1, "the hook must fire once");
        let requests = model.requests();
        let request = requests.first().expect("one request");
        assert!(
            request.chat_history.iter().any(|message| matches!(
                message,
                Message::System { content } if content == "patched by hook"
            )),
            "the runner must apply the hook's patch"
        );
    }

    #[tokio::test]
    async fn full_manual_round_trip_through_the_new_surface() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call-1", "add", json!({"x": 2, "y": 5})),
            MockTurn::text("7"),
        ]);
        let agent = AgentBuilder::new(model)
            .preamble("You are a calculator")
            .tool(MockAddTool)
            .build();

        let mut run = agent.new_run("What is 2 + 5?").max_turns(2);
        let mut prepared_turn: Option<PreparedAgentTurn> = None;
        let mut context = ToolContext::new();
        loop {
            match run.next_step().expect("step") {
                AgentRunStep::CallModel {
                    prompt, history, ..
                } => {
                    let (builder, turn) = agent
                        .prepare_completion_request(prompt, history, &mut run)
                        .await
                        .expect("preparation succeeds")
                        .into_parts();
                    let response = builder.send().await.expect("mock send");
                    let outcome = run
                        .model_response(turn.model_turn(response))
                        .expect("model response accepted");
                    assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    prepared_turn = Some(turn);
                }
                AgentRunStep::CallTools { calls } => {
                    let turn = prepared_turn.as_ref().expect("turn retained across send");
                    let mut results = Vec::with_capacity(calls.len());
                    for call in &calls {
                        results.push(turn.execute_call(call, &mut context).await);
                    }
                    run.tool_results(results).expect("results accepted");
                }
                AgentRunStep::Done(response) => {
                    assert_eq!(response.output, "7");
                    assert_eq!(response.completion_calls.len(), 2);
                    break;
                }
            }
        }
    }

    /// A tool named `add` that fails the test if its body ever runs.
    #[derive(Clone, Default)]
    struct MustNotRunTool(Arc<AtomicUsize>);

    impl Tool for MustNotRunTool {
        const NAME: &'static str = "add";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Must not execute".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("executed".to_string())
        }
    }

    #[tokio::test]
    async fn preresolved_calls_execute_nothing_and_clear_stale_metadata() {
        let executions = Arc::new(AtomicUsize::new(0));
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MustNotRunTool(executions.clone()))
            .build();
        let mut run = agent.new_run("go");
        let (prompt, history) = first_call_model(&mut run);
        let (_builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();

        let tool_call = ToolCall::from_wire("call-1", ToolFunction::new("add".into(), json!({})));
        let preresolved = UserContent::tool_result_for(
            tool_call.id.clone(),
            tool_call.provider.clone(),
            "add".to_string(),
            vec![ToolResultContent::text("already handled")],
        );
        let call = PendingToolCall {
            tool_call,
            preresolved_result: Some(preresolved.clone()),
            internal_call_id: None,
        };

        // Seed the context with stale metadata from a previous dispatch.
        let mut context = ToolContext::new();
        context.insert_result("stale".to_string());

        let result = turn.execute_call(&call, &mut context).await;
        assert_eq!(result_text(&result), "already handled");
        assert_eq!(
            executions.load(Ordering::SeqCst),
            0,
            "a pre-resolved call must execute nothing"
        );
        assert!(
            context.result::<String>().is_none(),
            "stale dispatch metadata must be cleared on the pre-resolved path"
        );
    }

    /// Guard: cross-process resume finishes pending calls through a rebuilt
    /// agent's live `tool_server_handle()`, asserting only what the existing
    /// API promises (no snapshot survival, no drift detection).
    #[tokio::test]
    async fn cross_process_resume_finishes_via_the_live_handle() {
        // "Process one": drive to CallTools, then suspend.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call-1",
            "add",
            json!({"x": 2, "y": 5}),
        )]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let mut run = agent.new_run("What is 2 + 5?").max_turns(2);
        let (prompt, history) = first_call_model(&mut run);
        let (builder, turn) = agent
            .prepare_completion_request(prompt, history, &mut run)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        run.model_response(turn.model_turn(response))
            .expect("model response accepted");
        let AgentRunStep::CallTools { .. } = run.next_step().expect("next step") else {
            panic!("expected CallTools");
        };
        let suspended = serde_json::to_string(&run).expect("run serializes");

        // "Process two": rebuild an equivalent agent, deserialize the run, and
        // finish through the live handle. The prepared turn did not survive.
        let model = MockCompletionModel::from_turns([MockTurn::text("7")]);
        let rebuilt = AgentBuilder::new(model).tool(MockAddTool).build();
        let mut resumed: AgentRun = serde_json::from_str(&suspended).expect("run deserializes");

        let AgentRunStep::CallTools { calls } = resumed.next_step().expect("re-emitted step")
        else {
            panic!("resumed run must re-emit the pending tool calls");
        };
        let handle = rebuilt.tool_server_handle();
        let mut context = ToolContext::new();
        let mut results = Vec::with_capacity(calls.len());
        for call in &calls {
            let args = json_utils::serialize_json_value(&call.tool_call.function.arguments);
            let result = handle
                .execute(&call.tool_call.function.name, &args, &mut context)
                .await;
            // `result_content` applies the id/provider/name correlation so
            // the resume path never copies those fields by hand.
            results.push(call.result_content(result.output().clone()));
        }
        resumed.tool_results(results).expect("results accepted");

        let (prompt, history) = match resumed.next_step().expect("follow-up step") {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => (prompt, history),
            other => panic!("expected CallModel, got {other:?}"),
        };
        let (builder, turn) = rebuilt
            .prepare_completion_request(prompt, history, &mut resumed)
            .await
            .expect("preparation succeeds")
            .into_parts();
        let response = builder.send().await.expect("mock send");
        resumed
            .model_response(turn.model_turn(response))
            .expect("model response accepted");
        match resumed.next_step().expect("final step") {
            AgentRunStep::Done(response) => assert_eq!(response.output, "7"),
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
