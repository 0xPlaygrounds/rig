//! Long-run workloads beyond the support chat: mixed unary and streamed
//! delivery, a tool-heavy agent loop, a document reused every turn, a
//! mid-conversation system message every few turns, active tools that change
//! mid-run, and conversations run side by side. Every generated input is
//! deterministic, so a re-recording sends the same bytes, and each workload
//! names its conversation with a marker in the first user message, so one
//! fixture can hold several conversations and [`LongRun::conversation`] can
//! pick one out.
//!
//! [`LongRun::conversation`]: super::LongRun::conversation

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_agent::agent::{
    Agent, AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch,
};
use rig_cassette::http::CassetteClock;
use rig_core::completion::Message;
use rig_core::message::{AssistantContent, Document, DocumentData, DocumentMediaType, UserContent};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{RunLog, chat, chat_streamed, question, try_chat};

/// Deterministic hash of `text`, for picking generated facts.
fn hash(text: &str) -> u32 {
    text.bytes().fold(7u32, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(u32::from(byte))
    })
}

/// `question(turn, prefix)`, with `marker` in front on the first turn.
fn marked(marker: &str, turn: usize, text: String) -> String {
    if turn == 1 {
        format!("{marker}. {text}")
    } else {
        text
    }
}

// ---------------------------------------------------------------------------
// Mixed delivery.

/// `turns` turns of the support chat about orders `<prefix>-<turn>`, odd
/// turns unary and even turns streamed, in one conversation.
///
/// # Panics
/// When a turn fails (see [`chat`]).
pub async fn mixed_delivery(
    agent: &Agent,
    clock: &CassetteClock,
    turns: usize,
    marker: &str,
    prefix: &str,
) -> RunLog {
    let mut history = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=turns {
        let prompt = marked(marker, turn, question(turn, prefix));
        if turn % 2 == 0 {
            chat_streamed(agent, clock, prompt, &mut history, &mut log).await;
        } else {
            chat(agent, clock, prompt, &mut history, &mut log).await;
        }
    }
    log
}

// ---------------------------------------------------------------------------
// The tool-heavy agent loop.

/// The agent-loop preamble: the support handbook plus how to work the two
/// account tools.
pub fn tool_loop_preamble() -> String {
    format!(
        "{}\n\n## Account investigations\n\nFor an account investigation, first call \
         `order_history` for every account the customer names, as parallel calls in one \
         response (never one account per response). When the histories are back, call \
         `shipping_log` with the tracking number of the newest order in the first account's \
         history. Then answer in two sentences.\n",
        super::SUPPORT_PREAMBLE
    )
}

/// [`OrderHistory`]'s arguments.
#[derive(Deserialize)]
pub struct AccountArgs {
    account_id: String,
}

/// An account's order history: twelve orders, newest first, about a
/// thousand tokens of JSON. The same account always gives the same history.
pub struct OrderHistory;

const PRODUCTS: [&str; 8] = [
    "linen shirt",
    "canvas tote",
    "wool scarf",
    "cotton cap",
    "leather wallet",
    "sock pack",
    "bamboo apron",
    "rain jacket",
];

const CARRIERS: [&str; 3] = ["DHL Parcel", "UPS Standard", "FedEx Economy"];
const CITIES: [&str; 4] = ["Leipzig DE", "Leeds GB", "Lyon FR", "Leiden NL"];

/// The history [`OrderHistory`] returns for `account_id`.
pub fn order_history(account_id: &str) -> Value {
    let seed = hash(account_id);
    let orders: Vec<Value> = (0..12u32)
        .map(|index| {
            // Kept small, so the arithmetic below cannot overflow.
            let key = seed.wrapping_add(index.wrapping_mul(7919)) % 1_000_003;
            let items: Vec<Value> = (0..(key % 3 + 1))
                .map(|item| {
                    let product = PRODUCTS[((key + item) % 8) as usize];
                    json!({
                        "sku": format!("SKU-{:04}", (key + item * 97) % 10_000),
                        "name": product,
                        "qty": (key + item) % 4 + 1,
                        "unit_price": format!("{}.{:02}", 9 + (key + item) % 90, (key * 7) % 100),
                    })
                })
                .collect();
            json!({
                "order_id": format!("{account_id}-O{:02}", 12 - index),
                "placed": format!("2026-{:02}-{:02}", 9 - index % 9, 28 - (index * 2) % 27),
                "status": super::STATUSES[(key % 6) as usize],
                "tracking": format!("TRK{:08}", (key.wrapping_mul(2_654_435_761)) % 100_000_000),
                "carrier": CARRIERS[(key % 3) as usize],
                "ship_to": CITIES[(key % 4) as usize],
                "items": items,
            })
        })
        .collect();
    json!({ "account_id": account_id, "orders": orders })
}

impl rig_core::tool::Tool for OrderHistory {
    const NAME: &'static str = "order_history";
    type Error = std::convert::Infallible;
    type Args = AccountArgs;
    type Output = Value;

    fn description(&self) -> String {
        "An account's order history, newest first, with each order's tracking number.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "account_id": { "type": "string" } },
            "required": ["account_id"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(order_history(&args.account_id))
    }
}

/// [`ShippingLog`]'s arguments.
#[derive(Deserialize)]
pub struct TrackingArgs {
    tracking: String,
}

/// A parcel's carrier log: thirty-six scan events, about a thousand tokens
/// of text. The same tracking number always gives the same log.
pub struct ShippingLog;

const EVENTS: [&str; 8] = [
    "ARRIVED AT SORT FACILITY",
    "DEPARTED SORT FACILITY",
    "IN TRANSIT TO NEXT FACILITY",
    "CUSTOMS CLEARANCE COMPLETE",
    "LOADED ONTO LINEHAUL VEHICLE",
    "HELD: ADDRESS VERIFICATION",
    "OUT FOR DELIVERY",
    "DELIVERY ATTEMPTED: NO ACCESS",
];

const FACILITIES: [&str; 6] = [
    "LEIPZIG HUB DE",
    "EAST MIDLANDS GB",
    "LOUISVILLE KY US",
    "MEMPHIS TN US",
    "LIEGE BE",
    "ROTTERDAM NL",
];

/// The log [`ShippingLog`] returns for `tracking`.
pub fn shipping_log(tracking: &str) -> String {
    let seed = hash(tracking);
    let mut log = format!("Carrier scan log for {tracking}\n");
    for index in 0..36u32 {
        let key = seed.wrapping_add(index.wrapping_mul(104_729));
        log.push_str(&format!(
            "2026-09-{:02}T{:02}:{:02}Z  {:<20}  {:<32}  ref {:06}\n",
            1 + index / 2,
            (key % 24),
            (key / 24) % 60,
            FACILITIES[(key % 6) as usize],
            EVENTS[((key / 7) % 8) as usize],
            key % 1_000_000,
        ));
    }
    log
}

impl rig_core::tool::Tool for ShippingLog {
    const NAME: &'static str = "shipping_log";
    type Error = std::convert::Infallible;
    type Args = TrackingArgs;
    type Output = String;

    fn description(&self) -> String {
        "A parcel's carrier scan log by tracking number.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "tracking": { "type": "string" } },
            "required": ["tracking"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(shipping_log(&args.tracking))
    }
}

/// The customer's investigation request for `turn`: two accounts at once,
/// then the newest parcel of the first.
pub fn investigation(turn: usize) -> String {
    format!(
        "Account investigation: accounts C{turn:03}-a and C{turn:03}-b (look both up at \
         once). Where is the newest parcel of C{turn:03}-a?"
    )
}

/// The calls one turn of the loop made to each tool, and whether it called
/// `order_history` for both accounts in one assistant message.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurnTools {
    /// Calls to `order_history`.
    pub histories: usize,
    /// Calls to `shipping_log`.
    pub logs: usize,
    /// Both histories were asked for in one assistant message.
    pub parallel: bool,
}

fn turn_tools(messages: &[Message]) -> TurnTools {
    let mut tools = TurnTools::default();
    for message in messages {
        let Message::Assistant(rig_core::message::AssistantMessage { content, .. }) = message
        else {
            continue;
        };
        let mut histories = 0;
        for part in content.iter() {
            if let AssistantContent::ToolCall(call) = part {
                let name = call.function.name.as_str();
                if name == <OrderHistory as rig_core::tool::Tool>::NAME {
                    histories += 1;
                } else if name == <ShippingLog as rig_core::tool::Tool>::NAME {
                    tools.logs += 1;
                }
            }
        }
        tools.histories += histories;
        tools.parallel |= histories >= 2;
    }
    tools
}

/// `turns` account investigations in one conversation, odd turns unary and
/// even turns streamed. Each turn asks for two histories at once, then a
/// carrier log that needs a tracking number from them, then the answer: three
/// model calls whose tool results are about a thousand tokens each.
///
/// # Panics
/// When a turn fails, or does not call `order_history` for both accounts in
/// one message and then `shipping_log`.
pub async fn tool_loop(
    agent: &Agent,
    clock: &CassetteClock,
    turns: usize,
    marker: &str,
) -> (RunLog, Vec<TurnTools>) {
    let mut history = Vec::new();
    let mut log = RunLog::default();
    let mut per_turn = Vec::new();
    for turn in 1..=turns {
        let from = history.len();
        let prompt = marked(marker, turn, investigation(turn));
        if turn % 2 == 0 {
            chat_streamed(agent, clock, prompt, &mut history, &mut log).await;
        } else {
            chat(agent, clock, prompt, &mut history, &mut log).await;
        }
        let tools = turn_tools(&history[from..]);
        assert!(
            tools.parallel && tools.logs >= 1,
            "turn {turn}: expected both histories in one message, then a log: {tools:?}"
        );
        per_turn.push(tools);
    }
    (log, per_turn)
}

// ---------------------------------------------------------------------------
// The document session.

/// The document session's preamble.
pub const DOCUMENT_PREAMBLE: &str = "You answer questions about the store policy the customer \
    attached at the start of the conversation. Answer in one sentence, from the policy only.";

const POLICY_TOPICS: [&str; 10] = [
    "returns",
    "exchanges",
    "refunds",
    "shipping",
    "customs",
    "warranties",
    "gift cards",
    "price matching",
    "loyalty points",
    "data privacy",
];

/// The store policy the document session reuses: forty-eight numbered sections,
/// about eight thousand tokens of plain text.
pub fn policy_text() -> String {
    let mut text = String::from("Northwind Outfitters store policy (edition 2026-09)\n\n");
    for section in 1..=48usize {
        let topic = POLICY_TOPICS[(section - 1) % POLICY_TOPICS.len()];
        let key = hash(&format!("section-{section}"));
        text.push_str(&format!(
            "Section {section}: {topic}, part {part}.\n\
             {section}.1 A customer may raise a {topic} request within {days} days of delivery, \
             through the account page or by writing to support with the order number.\n\
             {section}.2 Requests about orders above {limit} EUR need a supervisor's approval, \
             which support records on the order before replying to the customer.\n\
             {section}.3 The store answers every {topic} request within {hours} business hours \
             and confirms the outcome by email, quoting this section's number.\n\
             {section}.4 Items marked final sale are excluded from this section unless the item \
             arrived damaged, in which case section {damaged} applies instead.\n\
             {section}.5 Fees under this section are {fee} EUR per order, waived for loyalty \
             members of tier {tier} or above.\n\n",
            part = (section - 1) / POLICY_TOPICS.len() + 1,
            days = 14 + key % 47,
            limit = 150 + (key % 20) * 25,
            hours = 12 + key % 60,
            damaged = 1 + (key % 48),
            fee = key % 9,
            tier = 1 + key % 4,
        ));
    }
    text
}

/// [`policy_text`] as the first turn's document. With `citations`, the
/// document enables citations (Anthropic): it must do so from the first
/// request, because enabling citations later changes the rendered system
/// prompt and so every cached prefix.
pub fn policy_document(citations: bool) -> UserContent {
    let params = citations.then(|| {
        json!({
            "title": "Northwind Outfitters store policy",
            "citations": { "enabled": true }
        })
    });
    UserContent::Document(Document {
        data: DocumentData::Text(policy_text()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: params,
    })
}

/// The customer's question about the policy for `turn`.
pub fn policy_question(turn: usize) -> String {
    let section = (turn * 7) % 48 + 1;
    let asks = [
        "how many days after delivery can a customer raise a request under section {s}?",
        "above what order value does section {s} need a supervisor's approval?",
        "within how many business hours does the store answer under section {s}?",
        "which section applies instead of section {s} when a final-sale item arrived damaged?",
        "what fee does section {s} charge, and from which loyalty tier is it waived?",
    ];
    format!(
        "According to the policy, {}",
        asks[turn % asks.len()].replace("{s}", &section.to_string())
    )
}

/// `turns` questions about one document attached on the first turn, odd
/// turns unary and even turns streamed.
///
/// # Panics
/// When a turn fails (see [`chat`]).
pub async fn document_session(
    agent: &Agent,
    clock: &CassetteClock,
    turns: usize,
    marker: &str,
    document: UserContent,
) -> RunLog {
    let mut history = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=turns {
        let text = marked(marker, turn, policy_question(turn));
        let prompt = if turn == 1 {
            Message::User {
                content: vec![document.clone(), UserContent::text(text)],
            }
        } else {
            Message::user(text)
        };
        if turn % 2 == 0 {
            chat_streamed(agent, clock, prompt, &mut history, &mut log).await;
        } else {
            chat(agent, clock, prompt, &mut history, &mut log).await;
        }
    }
    log
}

// ---------------------------------------------------------------------------
// A mid-conversation system message every few turns.

/// The instruction sent before turn `turn`.
pub fn signature_instruction(turn: usize) -> String {
    format!("From now on, end every reply with the line \"Support desk {turn}\".")
}

/// `turns` turns of the support chat with a system message before every
/// `every`-th turn after the first. Alternate messages are placed where
/// Anthropic does not take one (after an assistant turn, before the user's
/// question), which rig moves after that question, and where it does (the
/// request's last entry, after the question).
///
/// # Panics
/// When a turn fails (see [`chat`]).
pub async fn mid_system_chat(
    agent: &Agent,
    clock: &CassetteClock,
    turns: usize,
    every: usize,
    marker: &str,
    prefix: &str,
) -> (RunLog, usize) {
    let mut history: Vec<Message> = Vec::new();
    let mut log = RunLog::default();
    let mut sent = 0;
    for turn in 1..=turns {
        let prompt = marked(marker, turn, question(turn, prefix));
        if turn > 1 && (turn - 1) % every == 0 {
            let instruction = Message::system(signature_instruction(turn));
            if sent % 2 == 0 {
                history.push(instruction);
                chat(agent, clock, prompt, &mut history, &mut log).await;
            } else {
                history.push(Message::user(prompt));
                chat(agent, clock, instruction, &mut history, &mut log).await;
            }
            sent += 1;
        } else {
            chat(agent, clock, prompt, &mut history, &mut log).await;
        }
    }
    (log, sent)
}

// ---------------------------------------------------------------------------
// Active tools that change mid-run.

/// A hook that narrows the advertised tools by conversation turn: turns
/// `1..=every` get the first set, the next `every` turns the second, and so
/// on, cycling. Set the turn with [`ToolSchedule::start_turn`] before each.
#[derive(Clone)]
pub struct ToolSchedule {
    turn: Arc<AtomicUsize>,
    every: usize,
    sets: Arc<Vec<Vec<String>>>,
}

impl ToolSchedule {
    /// A schedule switching between `sets` every `every` turns.
    pub fn new(every: usize, sets: Vec<Vec<&str>>) -> Self {
        Self {
            turn: Arc::new(AtomicUsize::new(1)),
            every: every.max(1),
            sets: Arc::new(
                sets.into_iter()
                    .map(|set| set.into_iter().map(str::to_owned).collect())
                    .collect(),
            ),
        }
    }

    /// Mark the start of conversation turn `turn`.
    pub fn start_turn(&self, turn: usize) {
        self.turn.store(turn, Ordering::SeqCst);
    }

    /// The tools advertised on turn `turn`.
    pub fn tools_for(&self, turn: usize) -> &[String] {
        let index = (turn.saturating_sub(1) / self.every) % self.sets.len().max(1);
        self.sets.get(index).map_or(&[], Vec::as_slice)
    }
}

impl AgentHook for ToolSchedule {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let tools = self.tools_for(self.turn.load(Ordering::SeqCst)).to_vec();
        CompletionCallAction::patch(RequestPatch::new().active_tools(tools))
    }
}

/// How a dynamic-tools run ended.
pub struct DynamicToolsRun {
    /// The successful calls' usage.
    pub log: RunLog,
    /// The turns that completed.
    pub completed: usize,
    /// The turn that failed and its error, when one did: the run stops there.
    pub refused: Option<(usize, String)>,
}

/// `turns` turns of the support chat on an agent carrying `schedule` as a
/// hook. A turn that fails ends the run and is returned, not raised: a model
/// that binds its reasoning to the tools it saw answers the first changed set
/// with a 400.
pub async fn dynamic_tools_chat(
    agent: &Agent,
    schedule: &ToolSchedule,
    clock: &CassetteClock,
    turns: usize,
    marker: &str,
    prefix: &str,
) -> DynamicToolsRun {
    let mut history = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=turns {
        schedule.start_turn(turn);
        let prompt = marked(marker, turn, question(turn, prefix));
        if let Err(error) = try_chat(agent, clock, prompt, &mut history, &mut log).await {
            return DynamicToolsRun {
                log,
                completed: turn - 1,
                refused: Some((turn, error)),
            };
        }
    }
    DynamicToolsRun {
        log,
        completed: turns,
        refused: None,
    }
}

// ---------------------------------------------------------------------------
// Conversations side by side.

/// One support-chat conversation per agent, `turns` turns each, about orders
/// `<prefix>-<turn>`. Turn 1 runs agent by agent, so which conversation first
/// meets the shared prefix is the same on replay; the later turns run all
/// conversations at once. Each conversation's requests differ from every
/// other's, so an unordered cassette replays them deterministically.
///
/// # Panics
/// When a turn fails (see [`chat`]).
pub async fn fan_out(
    agents: &[Agent],
    clock: &CassetteClock,
    turns: usize,
    markers: &[&str],
    prefixes: &[&str],
) -> RunLog {
    let mut histories: Vec<Vec<Message>> = vec![Vec::new(); agents.len()];
    let mut logs: Vec<RunLog> = agents.iter().map(|_| RunLog::default()).collect();
    for (index, agent) in agents.iter().enumerate() {
        let prompt = marked(markers[index], 1, question(1, prefixes[index]));
        chat(
            agent,
            clock,
            prompt,
            &mut histories[index],
            &mut logs[index],
        )
        .await;
    }
    for turn in 2..=turns {
        futures::future::join_all(
            agents
                .iter()
                .zip(histories.iter_mut())
                .zip(logs.iter_mut())
                .zip(prefixes)
                .map(|(((agent, history), log), prefix)| async move {
                    chat(agent, clock, question(turn, prefix), history, log).await;
                }),
        )
        .await;
    }
    let mut log = RunLog::default();
    for part in logs {
        log.usages.extend(part.usages);
        log.retries += part.retries;
    }
    log
}

#[cfg(test)]
mod tests;
