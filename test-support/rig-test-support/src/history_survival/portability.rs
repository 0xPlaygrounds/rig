//! Cross-wire continuation: a history one wire produced, decoded from its
//! committed recording through Rig's real decoder, continued on another wire.
//!
//! The source turn is the first reply of each provider's
//! `reasoning_tool_roundtrip/nonstreaming` recording: signed, encrypted or
//! plain reasoning beside a `get_weather` call. The cell answers that call
//! with the deterministic report and asks the target wire to continue. The
//! target either accepts the foreign history or rejects it, and the recorded
//! request shows which opaque fields were forwarded.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use rig_agent::agent::AgentBuilder;
use rig_agent::completion::CompletionModel;
use rig_core::completion::CompletionResponse;
use rig_core::driver::WireDriver;
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Message, ToolResult, ToolResultContent, UserContent};
use rig_core::operation::{Completion, CompletionFold};
use rig_core::providers::anthropic::wire::Anthropic;
use rig_core::providers::gemini::Gemini;
use rig_core::providers::gemini::completion::GenerateContent;
use rig_core::providers::openai::wire::{DEEPSEEK, OpenAI};
use rig_core::wire::{Fold, HasCompletion, Mode, Reply, Wire, WireFrame};

use super::{Dialect, response_tokens, string_values, unpaired_tool_calls};
use crate::reasoning::{TOOL_SYSTEM_PROMPT, TOOL_USER_PROMPT, WeatherTool};

/// The recording every source turn comes from.
pub const SOURCE_SCENARIO: &str = "reasoning_tool_roundtrip/nonstreaming";

/// The follow-up the target wire answers from the ported history.
pub const FOLLOW_UP: &str = "Using the weather you already retrieved, should I pack an umbrella \
    or sunscreen? Answer in one sentence and do not call any tool.";

/// Decode one whole recorded reply through the wire's own decoder and fold,
/// without I/O: the exact normalization a live call would perform.
pub fn decode_whole_reply<W>(wire: &W, body: &str) -> Result<CompletionResponse, ProviderError>
where
    W: Wire<Op = Completion>,
{
    let mut driver = WireDriver::new(wire.decoder(Mode::Unary));
    driver.push(WireFrame::Text(body.to_owned()));
    driver.finish();
    let mut fold = CompletionFold::default();
    for item in driver.drain() {
        fold.absorb(item?)?;
    }
    let raw = serde_json::from_str::<Value>(body)
        .map_err(|error| ProviderError::Response(error.to_string()))?;
    fold.finish(Reply {
        provider: wire.name().to_owned(),
        raw,
        provider_request_id: None,
    })
}

/// The wire whose recording supplies the ported turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Anthropic Messages: signed thinking beside a `tool_use`.
    Anthropic,
    /// OpenAI Responses: encrypted reasoning beside a `function_call`.
    OpenAiResponses,
    /// Gemini: a thought signature beside a `functionCall`.
    Gemini,
    /// DeepSeek: plain `reasoning_content` beside a `tool_calls` entry.
    DeepSeek,
}

impl Source {
    /// The cassette directory of the source recording.
    pub fn provider(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAiResponses => "openai",
            Self::Gemini => "gemini",
            Self::DeepSeek => "deepseek",
        }
    }

    /// The dialect of the source recording, for reading its tokens.
    pub fn dialect(self) -> Dialect {
        match self {
            Self::Anthropic => Dialect::AnthropicMessages,
            Self::OpenAiResponses => Dialect::OpenAiResponses,
            Self::Gemini => Dialect::GeminiGenerateContent,
            Self::DeepSeek => Dialect::ChatCompletions,
        }
    }

    fn body(self) -> String {
        crate::cassettes::recorded_interaction_bodies(self.provider(), SOURCE_SCENARIO)
            .into_iter()
            .next()
            .map(|(_, response)| response)
            .unwrap_or_else(|| panic!("{} {SOURCE_SCENARIO} records a reply", self.provider()))
    }

    /// The opaque fields the source reply delivered.
    pub fn tokens(self) -> Vec<super::Token> {
        response_tokens(self.dialect(), &self.body())
    }

    /// The source reply normalized by its own wire's decoder.
    pub fn reply(self) -> CompletionResponse {
        let body = self.body();
        // Decoding needs no credential; the placeholder never reaches a socket.
        let decoded = match self {
            Self::Anthropic => decode_whole_reply(
                &Anthropic::new("decode-only").completion("claude-sonnet-4-6"),
                &body,
            ),
            Self::OpenAiResponses => {
                decode_whole_reply(&OpenAI::new("decode-only").responses("gpt-5.2"), &body)
            }
            Self::Gemini => decode_whole_reply(
                &GenerateContent::new(Gemini::new("decode-only"), "gemini-2.5-flash"),
                &body,
            ),
            Self::DeepSeek => decode_whole_reply(
                &OpenAI::with_key(&DEEPSEEK, "decode-only").chat("deepseek-v4-flash"),
                &body,
            ),
        };
        decoded.unwrap_or_else(|error| panic!("{} source reply decodes: {error}", self.provider()))
    }

    /// The history the target continues: the prompt, the source turn as
    /// Rig normalized it, and the deterministic result of every call it made.
    pub fn history(self) -> Vec<Message> {
        let reply = self.reply();
        let results: Vec<UserContent> = reply
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .map(|call| {
                let city = call.function.arguments["city"]
                    .as_str()
                    .unwrap_or("Tokyo")
                    .to_owned();
                UserContent::ToolResult(ToolResult {
                    call: call.id.clone(),
                    provider: call.provider.clone(),
                    name: call.function.name.clone(),
                    content: vec![ToolResultContent::text(weather_report(&city))],
                })
            })
            .collect();
        assert!(
            !results.is_empty(),
            "{} source turn calls get_weather",
            self.provider()
        );
        vec![
            Message::user(TOOL_USER_PROMPT),
            Message::Assistant {
                id: reply.message_id,
                content: reply.choice,
            },
            Message::User { content: results },
        ]
    }
}

/// The exact text [`WeatherTool`] returns.
pub fn weather_report(city: &str) -> String {
    format!("Weather in {city}: 72F (22C), sunny with light clouds, humidity 45%, wind 8 mph NW")
}

/// One target cell.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// The target's cassette directory.
    pub provider: &'static str,
    /// The target model.
    pub model: &'static str,
    /// Provider-specific request parameters.
    pub params: fn() -> Option<Value>,
    /// Output budget per model call.
    pub max_tokens: u64,
    /// Which wire's recording supplies the history.
    pub source: Source,
}

/// Whether the source's reasoning issuer is the target model's own: direct
/// Anthropic reasoning continued on Claude through OpenRouter.
pub fn shares_issuer(cell: Cell) -> bool {
    cell.source == Source::Anthropic
        && cell.provider == "openrouter"
        && cell.model.starts_with("anthropic/")
}

/// What the continuation produced.
#[derive(Clone, Debug)]
pub struct Observation {
    /// The history after the continuation.
    pub history: Vec<Message>,
    /// The target's answer.
    pub final_text: String,
    /// How many times the target re-ran the weather tool.
    pub tool_calls: usize,
}

/// Shared slot the cell body fills and the post-cassette assertion reads.
pub type Observed = Arc<std::sync::Mutex<Option<Observation>>>;

/// Continue the ported history on the target wire.
pub async fn run<M>(model: M, cell: Cell) -> Observation
where
    M: CompletionModel + 'static,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = AgentBuilder::new(model)
        .preamble(TOOL_SYSTEM_PROMPT)
        .max_tokens(cell.max_tokens)
        .default_max_turns(4);
    if let Some(params) = (cell.params)() {
        builder = builder.additional_params(params);
    }
    let agent = builder.tool(WeatherTool::new(Arc::clone(&calls))).build();
    let mut history = cell.source.history();
    let response = agent
        .chat(FOLLOW_UP, &mut history)
        .await
        .unwrap_or_else(|error| {
            panic!(
                "[{} from {}] the target rejected the ported history: {error}",
                cell.provider,
                cell.source.provider()
            )
        });
    Observation {
        history,
        final_text: response.output,
        tool_calls: calls.load(Ordering::SeqCst),
    }
}

/// The target answered from the ported result.
pub fn assert_run(cell: Cell, observation: &Observation) {
    let text = observation.final_text.to_ascii_lowercase();
    assert!(
        text.contains("sunscreen") || text.contains("umbrella") || text.contains("sunny"),
        "[{} from {}] the answer should draw on the ported weather, got {:?}",
        cell.provider,
        cell.source.provider(),
        observation.final_text
    );
}

/// Which source fields the target request carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forwarded {
    /// Token kinds the source delivered.
    pub delivered: Vec<&'static str>,
    /// Token kinds whose exact values reached the target request.
    pub forwarded: Vec<&'static str>,
}

/// The target's first recorded request carries the ported content and
/// pairs the ported call with its result. Returns which source opaque
/// fields were forwarded verbatim, for the evidence table.
pub fn assert_recorded(cell: Cell, scenario: &str) -> Forwarded {
    let provider = cell.provider;
    let paths = crate::cassettes::recorded_request_paths(provider, scenario);
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let (path, (request, _)) = paths
        .iter()
        .zip(bodies)
        .next()
        .unwrap_or_else(|| panic!("[{provider} {scenario}] records a request"));
    let request = serde_json::from_str::<Value>(&request)
        .unwrap_or_else(|error| panic!("[{provider}] request JSON: {error}"));
    let dialect = Dialect::from_path(path);
    let strings = string_values(&request);
    let joined = strings.iter().cloned().collect::<Vec<_>>().join("\n");
    for needle in ["get_weather", "Tokyo", "72F (22C)", FOLLOW_UP] {
        assert!(
            joined.contains(needle),
            "[{provider} from {}] the request should carry {needle:?}",
            cell.source.provider()
        );
    }
    let unpaired = unpaired_tool_calls(dialect, &request);
    assert!(
        unpaired.is_empty(),
        "[{provider} from {}] the ported call is unpaired: {}",
        cell.source.provider(),
        unpaired
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    );

    let tokens = cell.source.tokens();
    let mut delivered: Vec<&'static str> = tokens.iter().map(|token| token.kind).collect();
    delivered.sort();
    delivered.dedup();
    let mut forwarded: Vec<&'static str> = tokens
        .iter()
        .filter(|token| strings.contains(&token.value))
        .map(|token| token.kind)
        .collect();
    forwarded.sort();
    forwarded.dedup();
    // Reasoning state is only meaningful to its issuer: none of it may reach
    // another issuer's model. Tool-call ids are correlation, not state. Claude
    // through OpenRouter shares the Anthropic issuer (its thinking signatures
    // verified valid between OpenRouter and the Claude API both ways), so
    // there the signature must arrive.
    let shared = shares_issuer(cell);
    let leaked: Vec<&str> = forwarded
        .iter()
        .copied()
        .filter(|kind| *kind != "tool_call_id" && !(shared && *kind == "signature"))
        .collect();
    assert!(
        leaked.is_empty(),
        "[{provider} from {}] foreign reasoning state reached the target request: {leaked:?}",
        cell.source.provider()
    );
    if shared {
        assert!(
            forwarded.contains(&"signature"),
            "[{provider} from {}] the shared issuer's signature reaches the target",
            cell.source.provider()
        );
    }
    let report = Forwarded {
        delivered,
        forwarded,
    };
    if let Some(path) = std::env::var_os("RIG_PORTABILITY_REPORT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("portability report opens");
        writeln!(
            file,
            "{provider}\t{}\t{scenario}\tdelivered={}\tforwarded={}",
            cell.source.provider(),
            report.delivered.join(","),
            report.forwarded.join(",")
        )
        .expect("portability report written");
    }
    report
}
