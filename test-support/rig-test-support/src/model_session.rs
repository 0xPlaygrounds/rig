//! One long recorded session per model: every capability the model documents
//! and rig supports without a model-specific setting, driven phase by phase
//! through one cassette.
//!
//! A session has a main conversation (one agent, a preamble long enough to
//! cache, a fixed tool set, a history that only grows, turns alternating
//! unary and streamed) and side conversations for what needs a differently
//! configured agent: the shared model-contract scenarios, native structured
//! output, strict tools, stop sequences, truncation and hosted tools. Every
//! phase prints one `MODEL_RUN <provider>/<model> <phase> <result> {json}`
//! line, with `result` either `recorded` or `refused` (a documented refusal
//! the phase asserts), so a coverage matrix is generated from a replay. A
//! phase that fails panics, failing the session.

use std::future::Future;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_agent::AgentBuilder;
use rig_agent::agent::Agent;
use rig_agent::test_utils::{
    COMPLEX_ARGUMENTS_RAW_PROMPT, ScenarioError, ScenarioReport, buffered_streaming_text_parity,
    cancellation_and_max_turns, complex_tool_arguments_with_prompt,
    hook_rewrites_and_request_patch, hook_rewrites_and_request_patch_with_choice,
    invalid_tool_recovery, invalid_tool_recovery_with_choice, optional_argument, parallel_tools,
    sequential_tools, streaming_structured_after_tool, streaming_tool, structured_after_tool,
    structured_extraction, tool_choice_modes, tool_output_serialization, without_temperature,
    zero_argument_tool,
};
use rig_cassette::http::CassetteClock;
use rig_core::completion::{
    CacheRetention, CompletionRequest, Effort, FinishReason, GenerationOptions, Message,
    ProviderOptions, ProviderToolDefinition, Reasoning,
};
use rig_core::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType,
    ToolChoice, UserContent,
};
use rig_core::providers::anthropic;
use rig_core::providers::anthropic::extension::AnthropicExt;
use rig_core::providers::openai::extension::{Include, OpenAiOptions};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cache_longrun::{
    self, CacheWire, Limits, LongRun, LookupOrder, RunLog, SUPPORT_PREAMBLE, chat, chat_streamed,
};
use crate::cassette_models::{AnthropicModels, MapWire, OpenAiModels};
use rig_core::completion::CacheRates;

/// Text the main conversation's first user message carries, so the cache
/// figures read only that conversation from a fixture that holds many.
pub const MAIN_MARKER: &str = "MAIN SESSION";

/// The committed solid-red PNG, sent as a base64 image.
const RED_SQUARE: &[u8] = include_bytes!("../../../tests/data/red_square.png");
/// A public PDF the providers fetch themselves.
const PDF_URL: &str = "https://bitcoin.org/bitcoin.pdf";
/// A PDF whose pages carry exact verifier tokens, for the file-id phase.
pub const VERIFIER_PDF: &[u8] = include_bytes!("../../../tests/data/file-id-verifiers.pdf");
/// The first page's verifier token in [`VERIFIER_PDF`].
const PAGE_ONE_VERIFIER: &str = "rig-file-id-page-one-verifier-3a91";

// ---------------------------------------------------------------------------
// Recording phases.

/// One phase's line.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseLine {
    /// The phase.
    pub phase: String,
    /// The wire it ran on (`messages`, `responses`, `chat`).
    pub wire: &'static str,
    /// `recorded` or `refused`.
    pub result: &'static str,
    /// What it checked.
    pub detail: Value,
}

/// The phases a session ran, and the main conversation's usage.
pub struct Session {
    run: String,
    /// Every phase, in order.
    pub phases: Vec<PhaseLine>,
    /// Usage of the main conversation's calls, for the cache checks.
    pub main: RunLog,
}

impl Session {
    fn new(provider: &str, model: &str) -> Self {
        Self {
            run: format!("{provider}/{model}"),
            phases: Vec::new(),
            main: RunLog::default(),
        }
    }

    fn record(&mut self, phase: &str, wire: &'static str, result: &'static str, detail: Value) {
        let line = PhaseLine {
            phase: phase.to_owned(),
            wire,
            result,
            detail,
        };
        println!(
            "MODEL_RUN {} {} {} {}",
            self.run,
            line.phase,
            line.result,
            serde_json::to_string(&line).unwrap_or_default()
        );
        self.phases.push(line);
    }

    fn recorded(&mut self, phase: &str, wire: &'static str, detail: Value) {
        self.record(phase, wire, "recorded", detail);
    }

    fn refused(&mut self, phase: &str, wire: &'static str, detail: Value) {
        self.record(phase, wire, "refused", detail);
    }

    /// Run a shared model-contract scenario as a recorded phase.
    async fn scenario(
        &mut self,
        wire: &'static str,
        future: impl Future<Output = Result<ScenarioReport, ScenarioError>>,
    ) {
        let report = future
            .await
            .unwrap_or_else(|error| panic!("{}: scenario failed: {error}", self.run));
        self.recorded(
            report.name,
            wire,
            json!({
                "tool_calls": report.tool_calls,
                "prompt_tokens": report.prompt_tokens,
                "generated_tokens": report.generated_tokens,
            }),
        );
    }

    /// Run a scenario the model is documented to refuse, and assert the
    /// provider's refusal (`needle` in the error).
    async fn scenario_refused(
        &mut self,
        name: &str,
        wire: &'static str,
        needle: &str,
        future: impl Future<Output = Result<ScenarioReport, ScenarioError>>,
    ) {
        match future.await {
            Ok(_) => panic!(
                "{}: {name} succeeded, but the model is documented to refuse it",
                self.run
            ),
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.contains(needle),
                    "{}: {name} failed differently from the documented refusal: {message}",
                    self.run
                );
                self.refused(name, wire, json!({ "error": message }));
            }
        }
    }
}

/// Read the recorded session back and run the long-run cache checks on its
/// main conversation: rig's `Usage` matches the wire on every call and, when
/// `reads_every_call`, no call after the first cache read reads nothing. The
/// cached share and the input saving at `rates` are reported in the
/// `CACHE_LONGRUN` line. `reads_every_call` is false only for a model whose
/// provider is shown to miss byte-identical prefixes (the model's module
/// carries the evidence).
pub fn check_main(
    wire: CacheWire,
    scenario: &str,
    model: &str,
    rates: CacheRates,
    output_price: f64,
    session: &Session,
    reads_every_call: bool,
) -> cache_longrun::Figures {
    let run = LongRun {
        wire,
        scenario,
        model,
        rates,
        output_price,
        limits: Some(Limits {
            min_saving: None,
            min_call_share: reads_every_call.then_some(0.0),
            max_writes_share: None,
        }),
        drops_signatures: false,
        conversation: Some(MAIN_MARKER),
    };
    cache_longrun::check(&run, &session.main, None).1
}

// ---------------------------------------------------------------------------
// The main conversation's tools.

/// [`WarehouseTime`] takes no arguments.
#[derive(Deserialize)]
pub struct NoArgs {}

/// The warehouse clock: a tool with no arguments and a fixed answer.
pub struct WarehouseTime;

impl rig_core::tool::Tool for WarehouseTime {
    const NAME: &'static str = "warehouse_time";
    type Error = std::convert::Infallible;
    type Args = NoArgs;
    type Output = Value;

    fn description(&self) -> String {
        "The current local time at the warehouse. Takes no arguments.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(json!({ "time": "14:05", "timezone": "UTC" }))
    }
}

/// [`ListOrders`]' arguments: the status filter is optional.
#[derive(Deserialize)]
pub struct ListArgs {
    status: Option<String>,
}

/// Lists the customer's orders, optionally by status.
pub struct ListOrders;

impl rig_core::tool::Tool for ListOrders {
    const NAME: &'static str = "list_orders";
    type Error = std::convert::Infallible;
    type Args = ListArgs;
    type Output = Value;

    fn description(&self) -> String {
        "List the customer's orders. `status` is optional and filters by status.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "status": { "type": "string" } }
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let orders = [("A-1", "shipped"), ("A-2", "delivered"), ("B-1", "shipped")];
        Ok(json!(
            orders
                .iter()
                .filter(|(_, status)| args
                    .status
                    .as_deref()
                    .is_none_or(|wanted| wanted == *status))
                .map(|(id, status)| json!({ "order_id": id, "status": status }))
                .collect::<Vec<_>>()
        ))
    }
}

/// [`SchedulePickup`]'s arguments: a nested address and a list of orders.
#[derive(Deserialize)]
pub struct PickupArgs {
    address: Address,
    orders: Vec<String>,
}

/// A street address.
#[derive(Deserialize)]
pub struct Address {
    street: String,
    city: String,
}

/// Schedules a courier pickup: complex, nested arguments.
pub struct SchedulePickup;

impl rig_core::tool::Tool for SchedulePickup {
    const NAME: &'static str = "schedule_pickup";
    type Error = std::convert::Infallible;
    type Args = PickupArgs;
    type Output = Value;

    fn description(&self) -> String {
        "Schedule a courier pickup of some orders from an address.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": { "type": "string" },
                        "city": { "type": "string" }
                    },
                    "required": ["street", "city"]
                },
                "orders": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["address", "orders"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "pickup_id": "PU-7",
            "street": args.address.street,
            "city": args.address.city,
            "orders": args.orders.len(),
        }))
    }
}

/// The instructions the main conversation adds to the handbook.
const SESSION_RULES: &str = "\n\n## Tools in this session\n\n\
    Use `lookup_order` for an order's status, `list_orders` to list orders, \
    `warehouse_time` for the warehouse clock and `schedule_pickup` to book a courier. \
    Call a tool whenever a question needs one; never guess what a tool would return. \
    In this session the customer also sends images, documents and questions that are \
    not about orders: answer those directly and briefly, exactly as asked.";

/// The typed options every request of a conversation carries.
#[derive(Clone, Default)]
struct RequestOptions {
    generation: GenerationOptions,
    provider: ProviderOptions,
}

impl RequestOptions {
    fn generation(generation: GenerationOptions) -> Self {
        Self {
            generation,
            provider: ProviderOptions::new(),
        }
    }

    /// `generation` beside OpenAI's provider options.
    fn openai(generation: GenerationOptions, options: OpenAiOptions) -> Self {
        Self {
            generation,
            provider: ProviderOptions::new().set(options),
        }
    }

    /// `store: false`, which keeps a Responses call stateless.
    fn stateless() -> Self {
        Self::openai(
            GenerationOptions::default(),
            OpenAiOptions::new().store(false),
        )
    }

    fn agent<Tools>(&self, builder: AgentBuilder<Tools>) -> AgentBuilder<Tools> {
        builder
            .options(self.generation.clone())
            .provider_options(self.provider.clone())
    }

    fn request(&self, request: CompletionRequest) -> CompletionRequest {
        request
            .options(self.generation.clone())
            .provider_options(self.provider.clone())
    }
}

fn main_agent(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    options: &RequestOptions,
    max_tokens: u64,
) -> Agent {
    options
        .agent(AgentBuilder::new(model))
        .preamble(format!("{SUPPORT_PREAMBLE}{SESSION_RULES}"))
        .tool(LookupOrder)
        .tool(WarehouseTime)
        .tool(ListOrders)
        .tool(SchedulePickup)
        .max_tokens(max_tokens)
        .default_max_turns(8)
        .build()
}

/// Calls the history made to `tool` from index `from` on.
fn calls_to(history: &[Message], from: usize, tool: &str) -> usize {
    history[from..]
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(rig_core::message::AssistantMessage { content, .. }) => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter(|content| matches!(content, AssistantContent::ToolCall(call) if call.function.name == tool))
        .count()
}

/// The last assistant text in `history`.
fn last_text(history: &[Message]) -> String {
    history
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant(rig_core::message::AssistantMessage { content, .. }) => {
                let text: String = content
                    .iter()
                    .filter_map(|part| match part {
                        AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect();
                (!text.is_empty()).then_some(text)
            }
            _ => None,
        })
        .unwrap_or_default()
}

/// Whether any assistant turn from index `from` on carries reasoning.
fn has_reasoning(history: &[Message], from: usize) -> bool {
    history[from..].iter().any(|message| {
        matches!(message, Message::Assistant(rig_core::message::AssistantMessage { content, .. })
            if content.iter().any(|part| matches!(part, AssistantContent::Reasoning(_))))
    })
}

fn user(parts: Vec<UserContent>) -> Message {
    Message::User { content: parts }
}

fn red_square() -> UserContent {
    UserContent::Image(Image {
        data: DocumentSourceKind::base64(BASE64_STANDARD.encode(RED_SQUARE)),
        media_type: Some(ImageMediaType::PNG),
        ..Default::default()
    })
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_lowercase();
    needles.iter().any(|needle| lower.contains(needle))
}

/// One turn of the main conversation, unary or streamed.
async fn turn(
    session: &mut Session,
    agent: &Agent,
    clock: &CassetteClock,
    history: &mut Vec<Message>,
    prompt: impl Into<Message>,
    streamed: bool,
) -> usize {
    let from = history.len();
    if streamed {
        chat_streamed(agent, clock, prompt, history, &mut session.main).await;
    } else {
        chat(agent, clock, prompt, history, &mut session.main).await;
    }
    from
}

/// The main conversation's turns every provider shares. Returns the history.
async fn main_conversation(
    session: &mut Session,
    agent: &Agent,
    clock: &CassetteClock,
    wire: &'static str,
    pdf: UserContent,
    opening: Option<UserContent>,
) -> Vec<Message> {
    let mut history = Vec::new();

    let greeting = UserContent::text(format!(
        "{MAIN_MARKER}. Hello! In one sentence, what can you help me with?"
    ));
    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        user(opening.into_iter().chain([greeting]).collect()),
        false,
    )
    .await;
    assert!(
        !last_text(&history[from..]).is_empty(),
        "{}: the greeting has text",
        session.run
    );
    session.recorded("main_text", wire, json!({ "streamed": false }));

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "Look up order A-1. After you have its status, look up order A-2.",
        true,
    )
    .await;
    let lookups = calls_to(&history, from, "lookup_order");
    assert!(
        lookups >= 2,
        "{}: two sequential lookups, saw {lookups}",
        session.run
    );
    session.recorded(
        "main_sequential_tools",
        wire,
        json!({ "streamed": true, "lookups": lookups }),
    );

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "Check orders B-1 and B-2 at the same time.",
        false,
    )
    .await;
    let lookups = calls_to(&history, from, "lookup_order");
    assert!(lookups >= 2, "{}: two lookups, saw {lookups}", session.run);
    session.recorded(
        "main_multiple_tools",
        wire,
        json!({ "streamed": false, "lookups": lookups }),
    );

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "What time is it at the warehouse right now?",
        true,
    )
    .await;
    assert!(
        calls_to(&history, from, "warehouse_time") >= 1,
        "{}: the zero-argument tool",
        session.run
    );
    assert!(
        last_text(&history).contains("14:05"),
        "{}: the warehouse time",
        session.run
    );
    session.recorded("main_zero_argument_tool", wire, json!({ "streamed": true }));

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "List all my orders, whatever their status.",
        false,
    )
    .await;
    assert!(
        calls_to(&history, from, "list_orders") >= 1,
        "{}: the optional-argument tool",
        session.run
    );
    session.recorded("main_optional_argument", wire, json!({ "streamed": false }));

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "Schedule a courier pickup of orders A-1 and B-2 from 12 Main Street, Springfield.",
        true,
    )
    .await;
    assert!(
        calls_to(&history, from, "schedule_pickup") >= 1,
        "{}: the complex-argument tool",
        session.run
    );
    // The tool ran only if its nested arguments deserialized; its result
    // (the pickup id) is in the history, whether or not the reply repeats it.
    assert!(
        serde_json::to_string(&history[from..])
            .unwrap_or_default()
            .contains("PU-7"),
        "{}: the pickup's tool result",
        session.run
    );
    session.recorded("main_complex_arguments", wire, json!({ "streamed": true }));

    turn(
        session,
        agent,
        clock,
        &mut history,
        user(vec![
            red_square(),
            UserContent::text(
                "What single colour fills this image? Answer with one lowercase word.",
            ),
        ]),
        false,
    )
    .await;
    assert!(
        contains_any(&last_text(&history), &["red"]),
        "{}: the image is red",
        session.run
    );
    session.recorded("main_image_base64", wire, json!({ "streamed": false }));

    turn(
        session,
        agent,
        clock,
        &mut history,
        user(vec![
            pdf,
            UserContent::text("What is the title of this paper? Answer in one short sentence."),
        ]),
        true,
    )
    .await;
    assert!(
        contains_any(&last_text(&history), &["bitcoin"]),
        "{}: the PDF's title",
        session.run
    );
    session.recorded("main_pdf_url", wire, json!({ "streamed": true }));

    let from = turn(
        session,
        agent,
        clock,
        &mut history,
        "A customer ordered 17 items at $23.45 each. A $5 coupon applies before 8% tax. \
         Work out the total they pay, then reply with the amount only.",
        false,
    )
    .await;
    // (17 * 23.45 - 5) * 1.08 = 425.142
    assert!(
        contains_any(&last_text(&history), &["425.14"]),
        "{}: the total",
        session.run
    );
    session.recorded(
        "main_reasoning",
        wire,
        json!({ "streamed": false, "reasoning_in_history": has_reasoning(&history, from) }),
    );

    turn(
        session,
        agent,
        clock,
        &mut history,
        "How many different orders did you look up with lookup_order in this conversation? \
         Answer with the number only.",
        true,
    )
    .await;
    assert!(
        contains_any(&last_text(&history), &["4", "four"]),
        "{}: four orders",
        session.run
    );
    session.recorded("main_history_recall", wire, json!({ "streamed": true }));

    history
}

// ---------------------------------------------------------------------------
// Anthropic.

/// What an Anthropic model's session covers.
pub struct AnthropicProfile {
    /// The model.
    pub model: &'static str,
    /// The model answers a forced tool choice with a 400.
    pub rejects_forced_tool_choice: bool,
    /// The model takes `role: "system"` inside `messages`.
    pub mid_conversation_system: bool,
    /// Only the phases a fix changed (for a model `main` already covers).
    pub fixes_only: bool,
}

/// The refusal Anthropic returns for a forced tool choice.
const FORCED_CHOICE_REFUSAL: &str = "not supported for this model";

/// Run an Anthropic model's session. `files` is a Files-API client and an
/// uploaded [`VERIFIER_PDF`]'s id, for the file-id phase.
pub async fn anthropic(
    models: AnthropicModels,
    files: Option<(AnthropicModels, String)>,
    clock: CassetteClock,
    profile: &AnthropicProfile,
) -> Session {
    let mut session = Session::new("anthropic", profile.model);
    let wire = "messages";
    let plain = || models.completion(profile.model);
    let configure = |builder: AgentBuilder| without_temperature(builder);

    if !profile.fixes_only {
        let agent = main_agent(
            models.completion(profile.model),
            &RequestOptions::generation(GenerationOptions::default().cache(CacheRetention::Short)),
            4096,
        );
        // Enabling citations changes the rendered system prompt, so the
        // conversation carries a citations document from its first request
        // (a later first one would miss the whole cache), and Anthropic
        // requires citations on every document or none, so the PDF has them.
        let rust_goals = UserContent::Document(Document {
            data: DocumentSourceKind::String(
                "Rust is a systems programming language focused on three goals: \
                 safety, speed, and concurrency."
                    .to_owned(),
            ),
            media_type: Some(DocumentMediaType::TXT),
            additional_params: Some(json!({
                "title": "Rust Goals",
                "citations": { "enabled": true }
            })),
        });
        let pdf = UserContent::Document(Document {
            data: DocumentSourceKind::Url(PDF_URL.to_owned()),
            media_type: None,
            additional_params: Some(json!({
                "title": "Bitcoin Whitepaper",
                "citations": { "enabled": true }
            })),
        });
        let mut history =
            main_conversation(&mut session, &agent, &clock, wire, pdf, Some(rust_goals)).await;
        let from = history.len();
        // A cited answer from the plaintext document of the first turn, then
        // a mid-conversation system message: both appended, so the history
        // stays append-only.
        turn(
            &mut session,
            &agent,
            &clock,
            &mut history,
            "Using citations from the Rust Goals document I attached at the start, name its \
             three goals in one sentence.",
            false,
        )
        .await;
        let cited = history[from..].iter().any(|message| match message {
            Message::Assistant(rig_core::message::AssistantMessage { content, .. }) => {
                content.iter().any(|part| match part {
                    AssistantContent::Text(text) => !text.citations().is_empty(),
                    _ => false,
                })
            }
            _ => false,
        });
        assert!(cited, "{}: the answer carries citations", session.run);
        session.recorded(
            "main_document_citations",
            wire,
            json!({ "streamed": false }),
        );

        if profile.mid_conversation_system {
            // The system message follows an assistant turn, where Anthropic
            // does not take one: rig moves it after the user's question, so
            // the prompt prefix (and every earlier thinking block's binding)
            // is unchanged. Hoisting it into `system` answered 400.
            history.push(Message::system("From now on, answer in Spanish only."));
            turn(
                &mut session,
                &agent,
                &clock,
                &mut history,
                "What colour was the image I sent you? One word.",
                true,
            )
            .await;
            assert!(
                contains_any(&last_text(&history), &["rojo"]),
                "{}: Spanish",
                session.run
            );
            session.recorded(
                "main_mid_conversation_system",
                wire,
                json!({ "streamed": true }),
            );
        }
        let reasoning = has_reasoning(&history, 0);
        assert!(
            reasoning,
            "{}: the model reasoned somewhere in the conversation",
            session.run
        );
        let reasoning_tokens: u64 = session
            .main
            .usages
            .iter()
            .filter_map(|usage| usage.reasoning_tokens)
            .sum();
        session.recorded(
            "main_reasoning_round_trip",
            wire,
            json!({ "reasoning_tokens": reasoning_tokens }),
        );

        session
            .scenario(
                wire,
                buffered_streaming_text_parity(plain(), |mut request| {
                    request.temperature = None;
                    request
                }),
            )
            .await;
        session
            .scenario(wire, sequential_tools(plain(), configure))
            .await;
        session
            .scenario(wire, parallel_tools(plain(), configure, None))
            .await;
        session
            .scenario(wire, streaming_tool(plain(), configure))
            .await;
        session
            .scenario(wire, zero_argument_tool(plain(), configure))
            .await;
        session
            .scenario(wire, optional_argument(plain(), configure))
            .await;
        session
            .scenario(
                wire,
                complex_tool_arguments_with_prompt(
                    plain(),
                    configure,
                    COMPLEX_ARGUMENTS_RAW_PROMPT,
                ),
            )
            .await;
        session
            .scenario(wire, tool_output_serialization(plain(), configure))
            .await;
        session
            .scenario(wire, cancellation_and_max_turns(plain(), configure))
            .await;
    }

    session
        .scenario(wire, structured_extraction(plain(), None))
        .await;
    session
        .scenario(wire, structured_after_tool(plain(), configure))
        .await;
    session
        .scenario(wire, streaming_structured_after_tool(plain(), configure))
        .await;

    let no_temperature = |mut request: CompletionRequest| {
        request.temperature = None;
        request
    };
    if profile.rejects_forced_tool_choice {
        session
            .scenario_refused(
                "tool_choice_modes",
                wire,
                FORCED_CHOICE_REFUSAL,
                tool_choice_modes(plain(), no_temperature),
            )
            .await;
        // The forced choice is refused once, above. These two force a call
        // only as scaffolding, so they run with `auto` and the prompt asking.
        if !profile.fixes_only {
            session
                .scenario(
                    wire,
                    invalid_tool_recovery_with_choice(plain(), configure, ToolChoice::Auto),
                )
                .await;
            session
                .scenario(
                    wire,
                    hook_rewrites_and_request_patch_with_choice(
                        plain(),
                        configure,
                        ToolChoice::Auto,
                    ),
                )
                .await;
        }
    } else {
        session
            .scenario(wire, tool_choice_modes(plain(), no_temperature))
            .await;
        session
            .scenario(wire, invalid_tool_recovery(plain(), configure))
            .await;
        session
            .scenario(wire, hook_rewrites_and_request_patch(plain(), configure))
            .await;
    }

    if profile.fixes_only {
        return session;
    }

    native_structured_output(&mut session, plain(), wire, &RequestOptions::default()).await;
    strict_tools(
        &mut session,
        models
            .completion(profile.model)
            .map_wire(anthropic::Messages::with_strict_tools),
        wire,
        &RequestOptions::default(),
    )
    .await;

    // Stop sequence: generation stops at the sequence, which is not returned.
    let response = plain()
        .call(
            CompletionRequest::new(
                "Write these words separated by spaces: alpha bravo charlie delta echo.",
            )
            .max_tokens(256)
            .stop(["charlie"]),
        )
        .await
        .unwrap_or_else(|error| panic!("{}: stop sequence: {error}", session.run));
    let text = text_of(&response.choice);
    assert!(
        !text.contains("delta"),
        "{}: stopped before delta: {text:?}",
        session.run
    );
    let stop_reason = response.extras_lossy::<AnthropicExt>().stop_reason;
    assert!(
        stop_reason.as_deref() == Some("stop_sequence"),
        "{}: stop_reason stop_sequence, saw {stop_reason:?}",
        session.run
    );
    session.recorded(
        "stop_sequence",
        wire,
        json!({ "message_id": response.response_id().is_some(), "request_id": response.provider_request_id.is_some() }),
    );
    assert!(
        response.response_id().is_some(),
        "{}: a message id",
        session.run
    );
    session.recorded(
        "response_metadata",
        wire,
        json!({ "message_id": true, "usage": response.usage.is_reported() }),
    );

    truncation(&mut session, plain(), wire, &RequestOptions::default(), 40).await;

    // A hosted tool: web search, streamed.
    let mut stream = plain()
        .stream(
            CompletionRequest::new(
                "Use web search to check the latest stable Rust release. You must run a search \
                 before answering. Keep the final answer under ten words.",
            )
            .provider_tool(
                ProviderToolDefinition::new("web_search_20250305")
                    .with_config("name", json!("web_search")),
            )
            .max_tokens(2048),
        )
        .unwrap_or_else(|error| panic!("{}: web search stream: {error}", session.run));
    let mut raw_types = Vec::new();
    use futures::StreamExt;
    while let Some(item) = stream.next().await {
        if let rig_core::streaming::Item::Event(rig_core::streaming::StreamEvent::End {
            content: AssistantContent::Opaque(opaque),
            ..
        }) = item.unwrap_or_else(|error| panic!("{}: web search item: {error}", session.run))
            && let Some(kind) = opaque.kind()
        {
            raw_types.push(kind.to_owned());
        }
    }
    let terminal = stream
        .finish()
        .await
        .unwrap_or_else(|error| panic!("{}: web search terminal: {error}", session.run));
    assert!(
        raw_types
            .iter()
            .any(|kind| kind == "web_search_tool_result"),
        "{}: a web search result block, saw {raw_types:?}",
        session.run
    );
    session.recorded(
        "hosted_web_search",
        wire,
        json!({ "blocks": raw_types, "usage": terminal.usage.is_reported() }),
    );

    if let Some((files, file_id)) = files {
        let agent = AgentBuilder::new(files.completion(profile.model))
            .preamble("Answer using only the attached PDF. Return exact visible tokens.")
            .build();
        let answer = agent
            .prompt(user(vec![
                UserContent::Document(Document {
                    data: DocumentSourceKind::FileId(file_id),
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
                UserContent::text(
                    "What verifier token is printed on page one? Reply with the token only.",
                ),
            ]))
            .await
            .unwrap_or_else(|error| panic!("{}: file id: {error}", session.run))
            .output();
        assert!(
            answer.contains(PAGE_ONE_VERIFIER),
            "{}: page one's token: {answer:?}",
            session.run
        );
        session.recorded("pdf_file_id", wire, json!({}));
    }
    session
}

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Native structured output through the agent's typed prompt.
async fn native_structured_output(
    session: &mut Session,
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    wire: &'static str,
    options: &RequestOptions,
) {
    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct City {
        name: String,
        country: String,
    }
    let city = options
        .agent(AgentBuilder::new(model).max_tokens(2048))
        .build()
        .prompt_typed::<City>("Name the capital of France and its country.")
        .await
        .unwrap_or_else(|error| panic!("{}: native structured output: {error}", session.run))
        .output;
    assert!(
        city.name.contains("Paris") && city.country.contains("France"),
        "{}: {city:?}",
        session.run
    );
    session.recorded(
        "native_structured_output",
        wire,
        json!({ "name": city.name }),
    );
}

/// A tool call under the provider's strict tool schemas.
async fn strict_tools(
    session: &mut Session,
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    wire: &'static str,
    options: &RequestOptions,
) {
    let builder = AgentBuilder::new(model)
        .preamble("Use lookup_order to answer order questions.")
        .tool(LookupOrder)
        .max_tokens(2048)
        .default_max_turns(4);
    let mut history = Vec::new();
    let response = options
        .agent(builder)
        .build()
        .chat("What is the status of order A-9?", &mut history)
        .await
        .unwrap_or_else(|error| panic!("{}: strict tools: {error}", session.run));
    assert!(
        calls_to(&history, 0, "lookup_order") >= 1,
        "{}: a strict tool call",
        session.run
    );
    session.recorded(
        "strict_tools",
        wire,
        json!({ "calls": response.completion_calls.len() }),
    );
}

/// A call cut short by a small output cap reports `Length`.
async fn truncation<W, T>(
    session: &mut Session,
    model: rig_core::driver::Model<W, T>,
    wire: &'static str,
    options: &RequestOptions,
    max_tokens: u64,
) where
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
{
    let request = options.request(
        CompletionRequest::new("Count from one to two hundred in words, separated by commas.")
            .max_tokens(max_tokens),
    );
    let response = model
        .call(request)
        .await
        .unwrap_or_else(|error| panic!("{}: truncation: {error}", session.run));
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::Length),
        "{}: cut at the cap",
        session.run
    );
    session.recorded(
        "max_tokens_truncation",
        wire,
        json!({ "finish_reason": "length" }),
    );
}

// ---------------------------------------------------------------------------
// OpenAI.

/// How an OpenAI model takes Chat Completions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatSupport {
    /// Responses only (the pro models).
    None,
    /// Text only: function tools are refused at every effort.
    TextOnly,
    /// Function tools only at `reasoning_effort: "none"`.
    ToolsAtEffortNone,
    /// Function tools at the default effort.
    Tools,
}

/// What an OpenAI model's session covers: its model page's rows.
pub struct OpenAiProfile {
    /// The model.
    pub model: &'static str,
    /// The model takes `temperature`.
    pub takes_temperature: bool,
    /// How it takes Chat Completions.
    pub chat: ChatSupport,
    /// A pro model: slow and expensive, so every phase keeps a small output
    /// limit and the default effort.
    pub pro: bool,
    /// The page lists structured outputs.
    pub structured_outputs: bool,
    /// The page lists the Responses web search tool.
    pub web_search: bool,
}

/// The output cap of a pro model's phases, USD 0.37 at most per call.
const PRO_MAX_TOKENS: u64 = 2048;

/// The `prompt_cache_key` a session's main conversation sends.
pub fn cache_key(model: &str) -> String {
    format!("rig-model-session-{model}")
}

/// The shared model-contract scenarios on one OpenAI route. `options` ride
/// every request (`store: false` on Responses, the caller's effort on Chat
/// Completions). The structured extraction scenario takes raw parameters
/// only, so `extractor_params` is the same setting as JSON.
async fn openai_scenarios(
    session: &mut Session,
    wire: &'static str,
    models: &OpenAiModels,
    profile: &OpenAiProfile,
    options: RequestOptions,
    extractor_params: Option<Value>,
) {
    let plain = || models.completion(profile.model);
    let takes_temperature = profile.takes_temperature;
    // A pro model's every phase has a small output cap (a scenario that sets
    // its own keeps it).
    let pro_cap = profile.pro.then_some(PRO_MAX_TOKENS);
    let configure = {
        let options = options.clone();
        move |builder: AgentBuilder| {
            let builder = options.agent(builder);
            let builder = match pro_cap {
                Some(cap) => builder.max_tokens(cap),
                None => builder,
            };
            if takes_temperature {
                builder
            } else {
                without_temperature(builder)
            }
        }
    };
    let adjust = {
        move |mut request: CompletionRequest| {
            if !takes_temperature {
                request.temperature = None;
            }
            request.options = options.generation.clone();
            request.provider_options = options.provider.clone();
            if request.max_tokens.is_none() {
                request.max_tokens = pro_cap;
            }
            request
        }
    };

    session
        .scenario(
            wire,
            buffered_streaming_text_parity(plain(), adjust.clone()),
        )
        .await;
    session
        .scenario(wire, sequential_tools(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, parallel_tools(plain(), configure.clone(), None))
        .await;
    session
        .scenario(wire, streaming_tool(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, zero_argument_tool(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, optional_argument(plain(), configure.clone()))
        .await;
    session
        .scenario(
            wire,
            complex_tool_arguments_with_prompt(
                plain(),
                configure.clone(),
                COMPLEX_ARGUMENTS_RAW_PROMPT,
            ),
        )
        .await;
    session
        .scenario(wire, tool_output_serialization(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, invalid_tool_recovery(plain(), configure.clone()))
        .await;
    session
        .scenario(
            wire,
            hook_rewrites_and_request_patch(plain(), configure.clone()),
        )
        .await;
    session
        .scenario(wire, cancellation_and_max_turns(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, tool_choice_modes(plain(), adjust))
        .await;
    session
        .scenario(wire, structured_extraction(plain(), extractor_params))
        .await;
    session
        .scenario(wire, structured_after_tool(plain(), configure.clone()))
        .await;
    session
        .scenario(wire, streaming_structured_after_tool(plain(), configure))
        .await;
}

/// The hosted web search tool on Responses: the reply carries a
/// `web_search_call` output item.
async fn openai_web_search(session: &mut Session, models: &OpenAiModels, model: &str) {
    let response = models
        .completion(model)
        .call(
            CompletionRequest::new(
                "Use web search to check the latest stable Rust release. You must run a search \
                 before answering. Keep the final answer under ten words.",
            )
            .provider_options(RequestOptions::stateless().provider)
            .provider_tool(ProviderToolDefinition::new("web_search"))
            .max_tokens(4096),
        )
        .await
        .unwrap_or_else(|error| panic!("{}: web search: {error}", session.run));
    let items: Vec<String> = response.raw["output"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item["type"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        items.iter().any(|kind| kind == "web_search_call"),
        "{}: a web_search_call item, saw {items:?}",
        session.run
    );
    session.recorded(
        "hosted_web_search",
        "responses",
        json!({ "items": items, "usage": response.usage.is_reported() }),
    );
}

/// Run an OpenAI model's session: `responses` and `chat` are the same
/// configuration on the two routes.
pub async fn openai(
    responses: OpenAiModels,
    chat_models: OpenAiModels,
    clock: CassetteClock,
    profile: &OpenAiProfile,
) -> Session {
    let mut session = Session::new("openai", profile.model);
    let wire = "responses";
    let stateless = RequestOptions::stateless();
    let plain = || responses.completion(profile.model);

    // Low effort, so every model reasons and its encrypted reasoning rides
    // the history; the pro models keep their default effort and ask for the
    // encrypted reasoning themselves (rig adds it only beside `reasoning`).
    let shared = OpenAiOptions::new()
        .prompt_cache_key(cache_key(profile.model))
        .store(false);
    let options = if profile.pro {
        RequestOptions::openai(
            GenerationOptions::default(),
            shared.include([Include::ReasoningEncryptedContent]),
        )
    } else {
        RequestOptions::openai(GenerationOptions::default().reasoning(Effort::Low), shared)
    };
    let max_tokens = if profile.pro { PRO_MAX_TOKENS } else { 4096 };
    let agent = main_agent(responses.completion(profile.model), &options, max_tokens);
    let pdf = UserContent::document_url(PDF_URL, None);
    let history = main_conversation(&mut session, &agent, &clock, wire, pdf, None).await;
    let reasoning_tokens: u64 = session
        .main
        .usages
        .iter()
        .filter_map(|usage| usage.reasoning_tokens)
        .sum();
    assert!(
        has_reasoning(&history, 0),
        "{}: encrypted reasoning in the history",
        session.run
    );
    session.recorded(
        "main_reasoning_round_trip",
        wire,
        json!({ "reasoning_tokens": reasoning_tokens }),
    );

    openai_scenarios(
        &mut session,
        wire,
        &responses,
        profile,
        stateless.clone(),
        Some(json!({ "store": false })),
    )
    .await;
    if profile.structured_outputs {
        native_structured_output(&mut session, plain(), wire, &stateless).await;
    }
    strict_tools(
        &mut session,
        responses
            .completion(profile.model)
            .map_wire(|wire| wire.with_strict_tools()),
        wire,
        &stateless,
    )
    .await;
    truncation(&mut session, plain(), wire, &stateless, 40).await;
    if profile.web_search {
        openai_web_search(&mut session, &responses, profile.model).await;
    }
    let response = plain()
        .call(stateless.request(CompletionRequest::new("Say ok.")))
        .await
        .unwrap_or_else(|error| panic!("{}: metadata: {error}", session.run));
    assert!(
        response.response_id().is_some(),
        "{}: a response id",
        session.run
    );
    session.recorded(
        "response_metadata",
        wire,
        json!({ "response_id": true, "request_id": response.provider_request_id.is_some(), "usage": response.usage.is_reported() }),
    );

    chat_conversation(&mut session, &chat_models, &clock, profile).await;
    session
}

/// The Chat Completions half: every phase the model's page allows there,
/// with function tools only where the model takes them.
async fn chat_conversation(
    session: &mut Session,
    models: &OpenAiModels,
    clock: &CassetteClock,
    profile: &OpenAiProfile,
) {
    let wire = "chat";
    if profile.chat == ChatSupport::None {
        return;
    }
    let plain = || models.completion(profile.model);
    let text_agent = AgentBuilder::new(plain())
        .preamble("You are a concise assistant.")
        .max_tokens(2048)
        .build();
    let mut log = RunLog::default();
    let mut history = Vec::new();
    chat(
        &text_agent,
        clock,
        "Say hello in one short sentence.".to_owned(),
        &mut history,
        &mut log,
    )
    .await;
    assert!(
        !last_text(&history).is_empty(),
        "{}: chat text",
        session.run
    );
    session.recorded("chat_text", wire, json!({ "streamed": false }));
    chat_streamed(
        &text_agent,
        clock,
        user(vec![
            red_square(),
            UserContent::text(
                "What single colour fills this image? Answer with one lowercase word.",
            ),
        ]),
        &mut history,
        &mut log,
    )
    .await;
    assert!(
        contains_any(&last_text(&history), &["red"]),
        "{}: chat image",
        session.run
    );
    session.recorded("chat_image_base64", wire, json!({ "streamed": true }));
    chat(
        &text_agent,
        clock,
        "What colour did you just name? One word.".to_owned(),
        &mut history,
        &mut log,
    )
    .await;
    assert!(
        contains_any(&last_text(&history), &["red"]),
        "{}: chat history recall",
        session.run
    );
    session.recorded("chat_history_recall", wire, json!({ "streamed": false }));

    native_structured_output(session, plain(), wire, &RequestOptions::default()).await;
    // Chat Completions answers a cap reached during reasoning with a 400
    // ("Could not finish the message because max_tokens ... was reached"),
    // so the GPT-6 models truncate at their lowest effort: `none` where they
    // take it, else `low` under a cap reasoning leaves room in.
    let (reasoning, cap) = match profile.chat {
        ChatSupport::ToolsAtEffortNone => (Some(Reasoning::Off), 40),
        ChatSupport::TextOnly => (Some(Reasoning::Effort(Effort::Low)), 300),
        ChatSupport::Tools | ChatSupport::None => (None, 40),
    };
    let options = reasoning.map_or_else(RequestOptions::default, |reasoning| {
        RequestOptions::generation(GenerationOptions::default().reasoning(reasoning))
    });
    truncation(session, plain(), wire, &options, cap).await;
    let response = plain()
        .call(CompletionRequest::new("Say ok."))
        .await
        .unwrap_or_else(|error| panic!("{}: chat metadata: {error}", session.run));
    assert!(
        response.response_id().is_some(),
        "{}: a chat completion id",
        session.run
    );
    session.recorded(
        "response_metadata",
        wire,
        json!({ "response_id": true, "request_id": response.provider_request_id.is_some(), "usage": response.usage.is_reported() }),
    );

    match profile.chat {
        ChatSupport::Tools => {
            openai_scenarios(
                session,
                wire,
                models,
                profile,
                RequestOptions::default(),
                None,
            )
            .await;
        }
        ChatSupport::ToolsAtEffortNone => {
            openai_scenarios(
                session,
                wire,
                models,
                profile,
                RequestOptions::generation(GenerationOptions::default().reasoning(Reasoning::Off)),
                Some(json!({ "reasoning_effort": "none" })),
            )
            .await;
        }
        ChatSupport::TextOnly | ChatSupport::None => {}
    }
    if matches!(
        profile.chat,
        ChatSupport::TextOnly | ChatSupport::ToolsAtEffortNone
    ) {
        // Function tools at the default effort are refused by OpenAI, whose
        // 400 names the fix, and so is the extractor, whose `submit` is a
        // function tool.
        let tools_agent = AgentBuilder::new(plain())
            .preamble("Use lookup_order to answer order questions.")
            .tool(LookupOrder)
            .max_tokens(2048)
            .build();
        let error = tools_agent
            .prompt("What is the status of order A-3?")
            .await
            .expect_err("tools at the default effort are refused on Chat Completions");
        let message = error.to_string();
        assert!(
            message.contains("/v1/responses"),
            "{}: {message}",
            session.run
        );
        session.refused(
            "chat_tools_default_effort",
            wire,
            json!({ "error": message }),
        );
        if profile.chat == ChatSupport::TextOnly {
            session
                .scenario_refused(
                    "structured_extraction",
                    wire,
                    "/v1/responses",
                    structured_extraction(plain(), None),
                )
                .await;
        }
    }
}
