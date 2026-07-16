use super::hook::{HookStack, RequestPatch};
use super::prompt_request::{self, PromptRequest};
use super::run::OutputMode;
use super::runner::AgentRunner;
use crate::{
    agent::prompt_request::streaming::StreamingPromptRequest,
    completion::{
        Chat, CompletionError, CompletionModel, CompletionRequestBuilder, Document, GetTokenUsage,
        Message, Prompt, PromptError, ToolDefinition, TypedPrompt,
    },
    json_utils,
    message::ToolChoice,
    streaming::{StreamingChat, StreamingPrompt},
    tool::server::{ToolRegistrySnapshot, ToolServerError, ToolServerHandle},
    vector_store::{VectorStoreError, request::VectorSearchRequest},
    wasm_compat::WasmCompatSend,
};
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use super::UNKNOWN_AGENT_NAME;

pub type DynamicContextStore = Arc<
    Vec<(
        usize,
        Arc<dyn crate::vector_store::VectorStoreIndexDyn + Send + Sync>,
    )>,
>;

/// A prepared completion request plus the executable Rig tool names advertised
/// to the provider for this turn.
pub(crate) struct PreparedCompletionRequest<M: CompletionModel> {
    pub(crate) builder: CompletionRequestBuilder<M>,
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

/// Helper function to build a completion request from agent components while
/// preserving the executable Rig tool names sent to the provider.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_prepared_completion_request<M: CompletionModel>(
    model: &Arc<M>,
    prompt: Message,
    chat_history: &[Message],
    preamble: Option<&str>,
    static_context: &[Document],
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<&serde_json::Value>,
    record_telemetry_content: bool,
    tool_choice: Option<&ToolChoice>,
    tool_server_handle: &ToolServerHandle,
    dynamic_context: &DynamicContextStore,
    output_schema: Option<&schemars::Schema>,
    output_mode: &OutputMode,
    committed_output_tool: Option<&str>,
    output_tool_description: Option<&str>,
    augment_output_preamble: bool,
    request_patch: Option<&RequestPatch>,
) -> Result<PreparedCompletionRequest<M>, CompletionError> {
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

    // Find the latest message in the chat history that contains RAG text
    let rag_text = prompt.rag_text();
    let rag_text = rag_text.or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    // Fetch dynamic (RAG) documents and the real executable tool set first, so we
    // can resolve the output mode (which depends on whether tools exist) before
    // building the preamble and request.
    let (mut tool_snapshot, fetched_context): (ToolRegistrySnapshot, Vec<Document>) =
        match &rag_text {
            Some(text) => {
                let search_futures = dynamic_context.iter().map(|(num_sample, index)| {
                    let text = text.clone();
                    let num_sample = *num_sample;
                    let index = index.clone();

                    async move {
                        let req = VectorSearchRequest::builder()
                            .query(text)
                            .samples(num_sample as u64)
                            .build();

                        let docs = index
                            .top_n(req)
                            .await?
                            .into_iter()
                            .map(|(_, id, doc)| {
                                let text = serde_json::to_string_pretty(&doc)
                                    .unwrap_or_else(|_| doc.to_string());

                                Document {
                                    id,
                                    text,
                                    additional_props: HashMap::new(),
                                }
                            })
                            .collect::<Vec<_>>();

                        Ok::<_, VectorStoreError>(docs)
                    }
                });

                let fetched_context: Vec<Document> = futures::future::try_join_all(search_futures)
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?
                    .into_iter()
                    .flatten()
                    .collect();

                let tool_snapshot = tool_server_handle
                    .snapshot_tool_defs(Some(text.to_string()))
                    .await
                    .map_err(|_| {
                        CompletionError::RequestError("Failed to get tool definitions".into())
                    })?;

                (tool_snapshot, fetched_context)
            }
            None => {
                let tool_snapshot =
                    tool_server_handle
                        .snapshot_tool_defs(None)
                        .await
                        .map_err(|_| {
                            CompletionError::RequestError("Failed to get tool definitions".into())
                        })?;

                (tool_snapshot, Vec::new())
            }
        };

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
            model.composes_native_output_with_tools(),
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

    if !fetched_context.is_empty() {
        completion_request = completion_request.documents(fetched_context);
    }

    // Hook-supplied extra context documents (passive RAG) are appended last, so
    // document order is static → dynamic (vector-store) → hook extras, with the
    // extras in the hooks' registration order (they were merged in that order).
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

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// Default hooks attached with [`AgentBuilder::add_hook`](crate::agent::AgentBuilder::add_hook)
/// are used for every prompt request, plus any added on the request or runner.
///
/// # Example
/// ```no_run
/// use rig_core::{
///     client::{CompletionClient, ProviderClient},
///     completion::Prompt,
///     providers::openai,
/// };
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
pub struct Agent<M>
where
    M: CompletionModel,
{
    /// Name of the agent used for logging and debugging
    pub(crate) name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    pub(crate) description: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub(crate) model: Arc<M>,
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
    /// List of vector store, with the sample number
    pub(crate) dynamic_context: DynamicContextStore,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    pub(crate) tool_choice: Option<ToolChoice>,
    /// Default total model-call budget, including the initial call and every
    /// retry or continuation. `None` uses the implicit budget of one.
    pub(crate) default_max_turns: Option<usize>,
    /// Default hook stack applied to every prompt request and runner created
    /// from this agent. Empty by default.
    pub(crate) hooks: HookStack<M>,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub(crate) output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced — tool call, native structured output, or
    /// prompt injection (see [`OutputMode`] and issue #1928).
    pub(crate) output_mode: OutputMode,
    /// Optional conversation memory backend that loads/saves history per conversation id.
    pub(crate) memory: Option<Arc<dyn crate::memory::ConversationMemory>>,
    /// Optional default conversation id used when none is set per-request.
    pub(crate) default_conversation_id: Option<String>,
}

impl<M> Agent<M>
where
    M: CompletionModel,
{
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
    pub fn runner(&self, prompt: impl Into<Message>) -> AgentRunner<M> {
        AgentRunner::from_agent(self, prompt)
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
}

// Here, we need to ensure that usage of `.prompt` on agent uses these redefinitions on the opaque
//  `Prompt` trait so that when `.prompt` is used at the call-site, it'll use the more specific
//  `PromptRequest` implementation for `Agent`, making the builder's usage fluent.
//
// References:
//  - https://github.com/rust-lang/rust/issues/121718 (refining_impl_trait)

#[allow(refining_impl_trait)]
impl<M> Prompt for Agent<M>
where
    M: CompletionModel + 'static,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard, M> {
        PromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Prompt for &Agent<M>
where
    M: CompletionModel + 'static,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name_or_default()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard, M> {
        PromptRequest::from_agent(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Chat for Agent<M>
where
    M: CompletionModel + 'static,
{
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

impl<M> StreamingPrompt<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M> {
        StreamingPromptRequest::<M>::from_agent(self, prompt)
    }
}

impl<M> StreamingChat<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M>
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        StreamingPromptRequest::<M>::from_agent(self, prompt).history(chat_history)
    }
}

use crate::agent::prompt_request::TypedPromptRequest;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[allow(refining_impl_trait)]
impl<M> TypedPrompt for Agent<M>
where
    M: CompletionModel + 'static,
{
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard, M>
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
    ) -> TypedPromptRequest<T, prompt_request::Standard, M>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> TypedPrompt for &Agent<M>
where
    M: CompletionModel + 'static,
{
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard, M>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<T, prompt_request::Standard, M>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(*self, prompt)
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
