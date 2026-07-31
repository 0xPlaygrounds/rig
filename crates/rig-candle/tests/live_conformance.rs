#![cfg(not(target_family = "wasm"))]

//! Live conformance checks for the pinned Qwen3 model.
//!
//! Candle has no `ProviderConfig` arm (model tensors are not expressible as
//! plain configuration), so the shared `rig_agent::test_utils` scenarios —
//! which now take a `ProviderConfig` — cannot run here. This file reproduces
//! each scenario's core assertion by driving the sans-IO agent protocol
//! directly against the loaded model: `prepare_request` builds each turn's
//! request, `rig_candle::functions::complete`/`open_stream` executes it, and
//! `AgentRun` sequences turns and tool batches.
//!
//! Deliberate simplifications versus the classic-runner scenarios:
//! - Hook rewrites (`hook_rewrites_and_request_patch`) are classic-runner
//!   machinery; only the per-turn `RequestPatch` half is reproduced here.
//! - The serial-tool-concurrency variant of `parallel_tools` is dropped: the
//!   sans-IO driver executes a parallel call batch serially by construction,
//!   so the single parallel scenario already covers it.
//! - Cancellation is reduced to the run's `cancel_error` surface plus the
//!   max-turns exhaustion failure; there is no background runner to abort.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use rig_agent::agent::{
    AgentConfig, AgentRun, AgentRunStep, ModelTurn, OutputMode, RequestPatch, ToolCatalog,
    prepare_request,
};
use rig_candle::{CandleModel, ModelArtifacts, ModelData};
use rig_core::completion::{AssistantContent, ToolDefinition, Usage};
use rig_core::message::{Message, ToolChoice, ToolResultContent, UserContent};
use rig_core::streaming::StreamedAssistantContent;
use rig_core::{OneOrMany, completion::CompletionRequest};
use serde::Deserialize;

type TestError = Box<dyn std::error::Error + Send + Sync>;

static MODEL: OnceLock<Result<CandleModel, String>> = OnceLock::new();

fn model() -> Result<CandleModel, TestError> {
    let result = MODEL.get_or_init(|| -> Result<CandleModel, String> {
        let directory = PathBuf::from(
            std::env::var_os("RIG_CANDLE_TEST_MODEL_DIR")
                .ok_or_else(|| "RIG_CANDLE_TEST_MODEL_DIR is not set".to_string())?,
        );
        let data = ModelData {
            config: std::fs::read(directory.join("config.json"))
                .map_err(|error| error.to_string())?,
            tokenizer: std::fs::read(directory.join("tokenizer.json"))
                .map_err(|error| error.to_string())?,
            weights: std::fs::read(directory.join("model.gguf"))
                .map_err(|error| error.to_string())?,
        };
        CandleModel::builder_from_artifacts(ModelArtifacts::Gguf(data))
            .temperature(0.0)
            .seed(42)
            .max_tokens(384)
            .max_concurrent_requests(1)
            .build()
            .map_err(|error| error.to_string())
    });
    result.clone().map_err(Into::into)
}

/// A conformance tool: its advertised definition plus a synchronous
/// deterministic implementation over raw JSON arguments.
struct ToolSpec {
    definition: ToolDefinition,
    run: Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
}

/// The observable outcome of one sans-IO agent run.
struct RunOutcome {
    output: String,
    usage: Usage,
    history: Vec<Message>,
    tool_calls: usize,
}

fn base_config() -> AgentConfig {
    AgentConfig::new()
        .with_preamble(
            "You are a calculator assistant. You MUST use the provided tools for every \
             requested operation instead of computing results yourself. Once you have all \
             the tool results you need, reply with the final answer in plain text.",
        )
        .with_temperature(0.0)
        .with_max_tokens(384)
}

/// Execute one model turn with buffered completion.
async fn buffered_turn(
    model: &CandleModel,
    prepared: rig_agent::agent::PreparedRequest,
) -> Result<ModelTurn, TestError> {
    let response = rig_candle::functions::complete(model, prepared.request).await?;
    Ok(ModelTurn::new(
        response.message_id.clone(),
        response.choice.clone(),
        response.usage,
        prepared.executable_tool_names,
        prepared.allowed_tool_names,
    ))
}

/// Execute one model turn by draining the streaming API into a buffered turn.
async fn streamed_turn(
    model: &CandleModel,
    prepared: rig_agent::agent::PreparedRequest,
) -> Result<ModelTurn, TestError> {
    let mut stream = rig_candle::functions::open_stream(model, prepared.request).await?;
    let mut text = String::new();
    let mut contents: Vec<AssistantContent> = Vec::new();
    let mut final_info = None;
    while let Some(item) = stream.next().await {
        match item? {
            StreamedAssistantContent::Text(fragment) => text.push_str(&fragment.text),
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                contents.push(AssistantContent::ToolCall(tool_call));
            }
            StreamedAssistantContent::Final(raw) => final_info = Some(raw),
            _ => {}
        }
    }
    let raw = final_info.ok_or("stream did not emit a final response")?;
    if !text.is_empty() {
        contents.insert(0, AssistantContent::text(text));
    }
    let choice = OneOrMany::many(contents)
        .map_err(|_| "stream produced neither text nor tool calls".to_string())?;
    Ok(ModelTurn::new(
        raw.message_id.clone(),
        choice,
        raw.usage,
        prepared.executable_tool_names,
        prepared.allowed_tool_names,
    ))
}

/// Drive a full sans-IO agent run: prepare each turn's request, call the
/// model (buffered or via the streaming API), execute surfaced tool calls,
/// and return the finished run's observable outcome.
async fn drive(
    model: &CandleModel,
    config: &AgentConfig,
    tools: &[ToolSpec],
    prompt: &str,
    max_turns: usize,
    streaming: bool,
) -> Result<RunOutcome, TestError> {
    let catalog = ToolCatalog::new(tools.iter().map(|tool| tool.definition.clone()).collect());
    let mut run = AgentRun::new(prompt).max_turns(max_turns);
    let mut tool_calls = 0usize;
    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => {
                let prepared = prepare_request(
                    config,
                    &catalog,
                    false,
                    prompt,
                    &history,
                    run.output_tool_name(),
                    None,
                )?;
                if run.output_tool_name().is_none() {
                    run.set_output_tool_name(prepared.output_tool_name.clone());
                }
                let turn = if streaming {
                    streamed_turn(model, prepared).await?
                } else {
                    buffered_turn(model, prepared).await?
                };
                run.model_response(turn)?;
            }
            AgentRunStep::CallTools { calls } => {
                let mut results = Vec::new();
                for call in &calls {
                    tool_calls += 1;
                    let name = call.tool_call.function.name.as_str();
                    let spec = tools
                        .iter()
                        .find(|tool| tool.definition.name == name)
                        .ok_or_else(|| format!("model called unknown tool {name:?}"))?;
                    let rendered = match (spec.run)(call.tool_call.function.arguments.clone()) {
                        Ok(serde_json::Value::String(text)) => text,
                        Ok(other) => other.to_string(),
                        Err(message) => format!("error: {message}"),
                    };
                    results.push(UserContent::tool_result(
                        call.tool_call.id.clone(),
                        OneOrMany::one(ToolResultContent::text(rendered)),
                    ));
                }
                run.tool_results(results)?;
            }
            AgentRunStep::Done(response) => {
                return Ok(RunOutcome {
                    output: response.output,
                    usage: response.usage,
                    history: response.messages.unwrap_or_default(),
                    tool_calls,
                });
            }
        }
    }
}

fn print_pass(name: &str, outcome: &RunOutcome) {
    println!(
        "PASS {} prompt_tokens={} generated_tokens={} tool_calls={} history_messages={} output={:?}",
        name,
        outcome.usage.input_tokens,
        outcome.usage.output_tokens,
        outcome.tool_calls,
        outcome.history.len(),
        outcome.output,
    );
}

/// Protocol hygiene: chat-template markers must never leak into the visible
/// output or history text.
fn check_hygiene(name: &str, outcome: &RunOutcome) -> Result<(), TestError> {
    const MARKERS: &[&str] = &[
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
    ];
    for marker in MARKERS {
        if outcome.output.contains(marker) {
            return Err(format!("{name}: output leaked protocol marker {marker:?}").into());
        }
    }
    Ok(())
}

/// Canonical call/result correlation: every assistant tool call in the history
/// has exactly one matching tool result, and no results dangle.
fn check_tool_correlation(name: &str, outcome: &RunOutcome) -> Result<(), TestError> {
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for message in &outcome.history {
        match message {
            Message::Assistant { content, .. } => {
                calls.extend(content.iter().filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.id.clone()),
                    _ => None,
                }));
            }
            Message::User { content } => {
                results.extend(content.iter().filter_map(|item| match item {
                    UserContent::ToolResult(result) => Some(result.id.clone()),
                    _ => None,
                }));
            }
            Message::System { .. } => {}
        }
    }
    if calls.is_empty() {
        return Err(format!("{name}: history has no assistant tool calls").into());
    }
    for id in &calls {
        let matches = results.iter().filter(|result| *result == id).count();
        if matches != 1 {
            return Err(format!("{name}: tool call id={id:?} has {matches} results").into());
        }
    }
    if results.len() != calls.len() {
        return Err(format!("{name}: dangling tool calls or results").into());
    }
    Ok(())
}

fn integer_args(
    arguments: &serde_json::Value,
    first: &str,
    second: &str,
) -> Result<(i64, i64), String> {
    let read = |key: &str| -> Result<i64, String> {
        let value = arguments
            .get(key)
            .ok_or_else(|| format!("missing argument {key:?}"))?;
        value
            .as_i64()
            .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
            .ok_or_else(|| format!("argument {key:?} is not an integer"))
    };
    Ok((read(first)?, read(second)?))
}

fn binop_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "x": { "type": "number", "description": "The first operand" },
            "y": { "type": "number", "description": "The second operand" }
        },
        "required": ["x", "y"]
    })
}

fn add_tool(counter: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        definition: ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: binop_parameters(),
        },
        run: Box::new(move |arguments| {
            counter.fetch_add(1, Ordering::SeqCst);
            let (x, y) = integer_args(&arguments, "x", "y")?;
            Ok(serde_json::json!(x + y))
        }),
    }
}

fn subtract_tool(counter: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        definition: ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e. x - y)".to_string(),
            parameters: binop_parameters(),
        },
        run: Box::new(move |arguments| {
            counter.fetch_add(1, Ordering::SeqCst);
            let (x, y) = integer_args(&arguments, "x", "y")?;
            Ok(serde_json::json!(x - y))
        }),
    }
}

fn multiply_tool(counter: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        definition: ToolDefinition {
            name: "multiply".to_string(),
            description: "Multiply x and y together".to_string(),
            parameters: binop_parameters(),
        },
        run: Box::new(move |arguments| {
            counter.fetch_add(1, Ordering::SeqCst);
            let (x, y) = integer_args(&arguments, "x", "y")?;
            Ok(serde_json::json!(x * y))
        }),
    }
}

#[derive(Debug, Deserialize)]
struct ExtractedPerson {
    first_name: Option<String>,
    last_name: Option<String>,
    job: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArithmeticResult {
    answer: i64,
}

/// Build a `schemars::Schema` from a raw JSON Schema value (rig-candle does
/// not depend on schemars directly; the type is re-exported by rig-core).
fn schema(value: serde_json::Value) -> Result<rig_core::schemars::Schema, TestError> {
    Ok(serde_json::from_value(value)?)
}

fn person_schema() -> Result<rig_core::schemars::Schema, TestError> {
    schema(serde_json::json!({
        "type": "object",
        "properties": {
            "first_name": { "type": ["string", "null"] },
            "last_name": { "type": ["string", "null"] },
            "job": { "type": ["string", "null"] }
        },
        "required": ["first_name", "last_name", "job"]
    }))
}

fn arithmetic_schema() -> Result<rig_core::schemars::Schema, TestError> {
    schema(serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "integer" }
        },
        "required": ["answer"]
    }))
}

fn plain_request(prompt: &str, max_tokens: u64) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: OneOrMany::one(Message::user(prompt)),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn choice_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

const SECS: fn(u64) -> Duration = Duration::from_secs;

#[tokio::test(flavor = "current_thread")]
#[ignore = "downloads are opt-in; run tests/download_qwen3.sh and set RIG_CANDLE_TEST_MODEL_DIR"]
async fn pinned_qwen3_model_contract() -> Result<(), TestError> {
    let model = model()?;

    // Simple buffered completion: model quality baseline.
    let simple = tokio::time::timeout(SECS(300), async {
        rig_candle::functions::complete(
            &model,
            plain_request("Answer with only the capital of France.", 32),
        )
        .await
    })
    .await??;
    let simple_text = choice_text(&simple.choice);
    if !simple_text.contains("Paris") {
        return Err(format!("model-quality failure in simple completion: {simple_text:?}").into());
    }
    println!(
        "PASS simple_buffered prompt_tokens={} generated_tokens={} output={:?}",
        simple.usage.input_tokens, simple.usage.output_tokens, simple_text,
    );

    // Buffered/streaming text parity (was `buffered_streaming_text_parity`).
    tokio::time::timeout(SECS(600), async {
        let request = plain_request("Answer with only the capital of France.", 32);
        let buffered = rig_candle::functions::complete(&model, request.clone()).await?;
        let mut stream = rig_candle::functions::open_stream(&model, request).await?;
        let mut streamed_text = String::new();
        let mut final_info = None;
        while let Some(item) = stream.next().await {
            match item? {
                StreamedAssistantContent::Text(fragment) => streamed_text.push_str(&fragment.text),
                StreamedAssistantContent::Final(raw) => final_info = Some(raw),
                _ => {}
            }
        }
        let raw = final_info.ok_or("stream did not emit a final response")?;
        if streamed_text != choice_text(&buffered.choice) || raw.usage != buffered.usage {
            return Err(format!(
                "streaming parity failure: streamed={streamed_text:?} buffered={:?}",
                choice_text(&buffered.choice)
            )
            .into());
        }
        println!("PASS buffered_streaming_text_parity output={streamed_text:?}");
        Ok::<(), TestError>(())
    })
    .await??;

    // Parallel tool batch with canonical correlation (was `parallel_tools`;
    // the serial-concurrency variant is subsumed by this driver's serial
    // execution).
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let parallel = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config(),
            &[
                add_tool(add_calls.clone()),
                subtract_tool(subtract_calls.clone()),
            ],
            "Compute 3 + 4 and 10 - 2. You MUST call the add tool and the subtract tool \
             together in your first response, as two parallel function calls, then report \
             both results.",
            4,
            false,
        ),
    )
    .await??;
    if add_calls.load(Ordering::SeqCst) == 0 || subtract_calls.load(Ordering::SeqCst) == 0 {
        return Err("parallel_tools: both tools must run".into());
    }
    if !parallel.output.contains('7') || !parallel.output.contains('8') {
        return Err(format!("parallel_tools: wrong answer {:?}", parallel.output).into());
    }
    check_hygiene("parallel_tools", &parallel)?;
    check_tool_correlation("parallel_tools", &parallel)?;
    print_pass("parallel_tools", &parallel);

    // Zero-argument tool (was `zero_argument_tool`).
    const PING_OUTPUT: &str = "pong-crimson-7423";
    let ping_calls = Arc::new(AtomicUsize::new(0));
    let ping_counter = ping_calls.clone();
    let ping = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config().with_preamble(
                "Use the ping tool to fetch the current ping marker, then repeat it verbatim.",
            ),
            &[ToolSpec {
                definition: ToolDefinition {
                    name: "ping".to_string(),
                    description: "Return the current ping marker. Takes no arguments.".to_string(),
                    parameters: serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
                },
                run: Box::new(move |_| {
                    ping_counter.fetch_add(1, Ordering::SeqCst);
                    Ok(serde_json::json!(PING_OUTPUT))
                }),
            }],
            "What is the current ping marker?",
            3,
            false,
        ),
    )
    .await??;
    if ping_calls.load(Ordering::SeqCst) == 0 || !ping.output.contains(PING_OUTPUT) {
        return Err(format!("zero_argument_tool: {:?}", ping.output).into());
    }
    check_hygiene("zero_argument_tool", &ping)?;
    print_pass("zero_argument_tool", &ping);

    // Structured (JSON object) tool output round trip (was
    // `tool_output_serialization`).
    let config_calls = Arc::new(AtomicUsize::new(0));
    let config_counter = config_calls.clone();
    let serialized = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config().with_preamble(
                "Use the fetch_config tool, then report the service name and its max_retries.",
            ),
            &[ToolSpec {
                definition: ToolDefinition {
                    name: "fetch_config".to_string(),
                    description: "Fetch the service configuration object.".to_string(),
                    parameters: serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
                },
                run: Box::new(move |_| {
                    config_counter.fetch_add(1, Ordering::SeqCst);
                    Ok(serde_json::json!({ "service": "cassette-lab", "max_retries": 3 }))
                }),
            }],
            "What service is configured and how many retries does it allow?",
            3,
            false,
        ),
    )
    .await??;
    if config_calls.load(Ordering::SeqCst) == 0
        || !serialized.output.contains("cassette-lab")
        || !serialized.output.contains('3')
    {
        return Err(format!("tool_output_serialization: {:?}", serialized.output).into());
    }
    check_hygiene("tool_output_serialization", &serialized)?;
    print_pass("tool_output_serialization", &serialized);

    // Nested/optional/enum/quoted arguments arrive intact (was
    // `complex_tool_arguments`).
    const COMPLEX_QUOTE: &str = "steady hands, calm waters";
    let complex_captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let complex_slot = complex_captured.clone();
    let complex = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config().with_preamble(
                "Store the requested profile with the store_profile tool, then confirm.",
            ),
            &[ToolSpec {
                definition: ToolDefinition {
                    name: "store_profile".to_string(),
                    description:
                        "Store one profile with its nested tags, mode, and exact quoted text."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "profile": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "tags": { "type": "array", "items": { "type": "string" } }
                                },
                                "required": ["name", "tags"]
                            },
                            "mode": { "type": "string", "enum": ["careful", "fast"] },
                            "quote": { "type": "string" }
                        },
                        "required": ["profile", "mode", "quote"]
                    }),
                },
                run: Box::new(move |arguments| {
                    *complex_slot
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(arguments.clone());
                    Ok(arguments)
                }),
            }],
            &format!(
                "Store a profile named \"harbor\" with tags [\"tide\", \"anchor\"], mode careful, \
                 and this exact quote: \"{COMPLEX_QUOTE}\"."
            ),
            3,
            false,
        ),
    )
    .await??;
    let captured = complex_captured
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or("complex_tool_arguments: tool never ran")?;
    let quote_ok = captured
        .get("quote")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|quote| quote == COMPLEX_QUOTE);
    let tags_ok = captured
        .pointer("/profile/tags")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tags| tags.len() == 2);
    if !quote_ok || !tags_ok {
        return Err(format!("complex_tool_arguments: captured={captured:?}").into());
    }
    check_hygiene("complex_tool_arguments", &complex)?;
    print_pass("complex_tool_arguments", &complex);

    // Optional argument may be omitted and defaulted host-side (was
    // `optional_argument`).
    let repeat_calls = Arc::new(AtomicUsize::new(0));
    let repeat_counter = repeat_calls.clone();
    let optional = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config()
                .with_preamble("Use the repeat_text tool, then report its result verbatim."),
            &[ToolSpec {
                definition: ToolDefinition {
                    name: "repeat_text".to_string(),
                    description: "Repeat `text`. `times` is optional and defaults to 2.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "The text to repeat." },
                            "times": { "type": "integer", "description": "Number of repetitions; defaults to 2 when omitted." }
                        },
                        "required": ["text"]
                    }),
                },
                run: Box::new(move |arguments| {
                    repeat_counter.fetch_add(1, Ordering::SeqCst);
                    let text = arguments
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .ok_or("missing text argument")?;
                    let times = arguments
                        .get("times")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(2) as usize;
                    Ok(serde_json::json!(vec![text; times].join(" ")))
                }),
            }],
            "Repeat the word lighthouse.",
            3,
            false,
        ),
    )
    .await??;
    if repeat_calls.load(Ordering::SeqCst) == 0 || !optional.output.contains("lighthouse") {
        return Err(format!("optional_argument: {:?}", optional.output).into());
    }
    check_hygiene("optional_argument", &optional)?;
    print_pass("optional_argument", &optional);

    // The model recovers after a tool reports an error (was
    // `invalid_tool_recovery`, simplified: the error is injected by the tool
    // rather than provoked through the runner's invalid-call retry path).
    let recovery_calls = Arc::new(AtomicUsize::new(0));
    let recovery_counter = recovery_calls.clone();
    let recovery = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config(),
            &[ToolSpec {
                definition: ToolDefinition {
                    name: "add".to_string(),
                    description: "Add x and y together".to_string(),
                    parameters: binop_parameters(),
                },
                run: Box::new(move |arguments| {
                    let attempt = recovery_counter.fetch_add(1, Ordering::SeqCst);
                    if attempt == 0 {
                        return Err("transient failure, please call add again".to_string());
                    }
                    let (x, y) = integer_args(&arguments, "x", "y")?;
                    Ok(serde_json::json!(x + y))
                }),
            }],
            "Use the add tool to compute 3 + 4, then report the result. If the tool reports \
             a transient failure, call it again.",
            5,
            false,
        ),
    )
    .await??;
    if recovery_calls.load(Ordering::SeqCst) < 2 || !recovery.output.contains('7') {
        return Err(format!(
            "invalid_tool_recovery: calls={} output={:?}",
            recovery_calls.load(Ordering::SeqCst),
            recovery.output
        )
        .into());
    }
    check_hygiene("invalid_tool_recovery", &recovery)?;
    print_pass("invalid_tool_recovery", &recovery);

    // Per-turn request patching (was `hook_rewrites_and_request_patch`; hook
    // rewrites are classic-runner machinery and are not reproduced).
    tokio::time::timeout(SECS(900), async {
        let config = base_config().with_preamble("Answer using the provided context.");
        // `RequestPatch` is non_exhaustive, so it must be built from
        // `default()` and mutated.
        #[allow(clippy::field_reassign_with_default)]
        let mut patch = RequestPatch::default();
        patch.extra_context = vec![rig_core::completion::Document {
            id: "patched-doc".to_string(),
            text: "The code word is zephyr-9931.".to_string(),
            additional_props: std::collections::HashMap::new(),
        }];
        let mut run = AgentRun::new("What is the code word?").max_turns(2);
        loop {
            match run.next_step()? {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    let prepared = prepare_request(
                        &config,
                        &ToolCatalog::default(),
                        false,
                        prompt,
                        &history,
                        run.output_tool_name(),
                        (turn == 1).then_some(&patch),
                    )?;
                    let turn = buffered_turn(&model, prepared).await?;
                    run.model_response(turn)?;
                }
                AgentRunStep::CallTools { .. } => {
                    return Err("request_patch: unexpected tool calls".into());
                }
                AgentRunStep::Done(response) => {
                    if !response.output.contains("zephyr-9931") {
                        return Err(format!("request_patch: {:?}", response.output).into());
                    }
                    println!("PASS request_patch output={:?}", response.output);
                    return Ok::<(), TestError>(());
                }
            }
        }
    })
    .await??;

    // Run controls: max-turns exhaustion fails the run, and the run can mint
    // a cancellation error (was `cancellation_and_max_turns`).
    tokio::time::timeout(SECS(900), async {
        let counter = Arc::new(AtomicUsize::new(0));
        let error = drive(
            &model,
            &base_config(),
            &[add_tool(counter)],
            "Use the add tool to compute 1 + 1, then use it again to add 1 to the result, \
             and keep going until you reach 5.",
            1,
            false,
        )
        .await
        .err()
        .ok_or("max_turns: run with a 1-turn budget unexpectedly completed")?;
        println!("PASS max_turns error={error}");
        let run = AgentRun::new("unused");
        let cancel = run.cancel_error("stopped by test");
        if !cancel.to_string().contains("stopped by test") {
            return Err(format!("cancellation: unexpected error {cancel}").into());
        }
        println!("PASS cancellation error={cancel}");
        Ok::<(), TestError>(())
    })
    .await??;

    // Structured extraction through the synthetic output tool (was
    // `structured_extraction`).
    let extraction = tokio::time::timeout(SECS(900), async {
        let mut config = AgentConfig::new()
            .with_preamble(
                "Extract structured data from the provided text. Always call the submit \
                 function with every field filled in.",
            )
            .with_temperature(0.0)
            .with_max_tokens(384)
            .with_tool_choice(ToolChoice::Required);
        config.output_schema = Some(person_schema()?);
        config.output_mode = OutputMode::Tool;
        drive(
            &model,
            &config,
            &[],
            "Ada Lovelace was a mathematician who wrote the first computer program.",
            3,
            false,
        )
        .await
    })
    .await??;
    let person: ExtractedPerson = serde_json::from_str(&extraction.output)
        .map_err(|error| format!("structured_extraction: {error}: {:?}", extraction.output))?;
    let fields_ok = person
        .first_name
        .as_deref()
        .is_some_and(|name| name.eq_ignore_ascii_case("Ada"))
        && person
            .last_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case("Lovelace"))
        && person
            .job
            .as_deref()
            .is_some_and(|job| job.to_ascii_lowercase().contains("mathematician"));
    if !fields_ok {
        return Err(format!("structured_extraction: {person:?}").into());
    }
    print_pass("structured_extraction", &extraction);

    // Sequential dependent tool calls across turns (was `sequential_tools`).
    let seq_add = Arc::new(AtomicUsize::new(0));
    let seq_multiply = Arc::new(AtomicUsize::new(0));
    let sequential = tokio::time::timeout(
        SECS(1200),
        drive(
            &model,
            &base_config(),
            &[
                add_tool(seq_add.clone()),
                multiply_tool(seq_multiply.clone()),
            ],
            "Compute (2 + 3) * 4. First use the add tool, then multiply its result by 4 \
             with the multiply tool, then report the final result.",
            6,
            false,
        ),
    )
    .await??;
    if seq_add.load(Ordering::SeqCst) == 0
        || seq_multiply.load(Ordering::SeqCst) == 0
        || !sequential.output.contains("20")
    {
        return Err(format!("sequential_tools: {:?}", sequential.output).into());
    }
    check_hygiene("sequential_tools", &sequential)?;
    check_tool_correlation("sequential_tools", &sequential)?;
    print_pass("sequential_tools", &sequential);

    // Tool round trip over the streaming API (was `streaming_tool`).
    let stream_add = Arc::new(AtomicUsize::new(0));
    let streaming = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config(),
            &[add_tool(stream_add.clone())],
            "Use the add tool to compute 3 + 4, then report the result.",
            4,
            true,
        ),
    )
    .await??;
    if stream_add.load(Ordering::SeqCst) == 0 || !streaming.output.contains('7') {
        return Err(format!("streaming_tool: {:?}", streaming.output).into());
    }
    check_hygiene("streaming_tool", &streaming)?;
    print_pass("streaming_tool", &streaming);

    // Structured output after a tool round trip, buffered and streaming (were
    // `structured_after_tool` / `streaming_structured_after_tool`).
    for (name, use_streaming) in [
        ("structured_after_tool", false),
        ("streaming_structured_after_tool", true),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let outcome = tokio::time::timeout(SECS(900), async {
            let mut config = base_config();
            config.output_schema = Some(arithmetic_schema()?);
            config.output_mode = OutputMode::Tool;
            drive(
                &model,
                &config,
                &[add_tool(calls.clone())],
                "Use the add tool to compute 3 + 4, then submit the structured result.",
                5,
                use_streaming,
            )
            .await
        })
        .await??;
        let result: ArithmeticResult = serde_json::from_str(&outcome.output)
            .map_err(|error| format!("{name}: {error}: {:?}", outcome.output))?;
        if calls.load(Ordering::SeqCst) == 0 || result.answer != 7 {
            return Err(format!(
                "{name}: answer={} output={:?}",
                result.answer, outcome.output
            )
            .into());
        }
        print_pass(name, &outcome);
    }

    // Tool-choice policy: `none` suppresses tool execution even when tools
    // are advertised (was `tool_choice_modes`, reduced to the suppressive
    // half — the Required half is exercised by structured extraction above).
    let choice_calls = Arc::new(AtomicUsize::new(0));
    let choice = tokio::time::timeout(
        SECS(900),
        drive(
            &model,
            &base_config()
                .with_preamble("Answer arithmetic questions directly in plain text.")
                .with_tool_choice(ToolChoice::None),
            &[add_tool(choice_calls.clone())],
            "What is 3 + 4? Answer with just the number.",
            2,
            false,
        ),
    )
    .await??;
    if choice_calls.load(Ordering::SeqCst) != 0 || !choice.output.contains('7') {
        return Err(format!(
            "tool_choice_modes: calls={} output={:?}",
            choice_calls.load(Ordering::SeqCst),
            choice.output
        )
        .into());
    }
    print_pass("tool_choice_modes", &choice);

    Ok(())
}
