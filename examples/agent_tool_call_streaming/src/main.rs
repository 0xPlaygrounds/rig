//! Demonstrates tool calls streaming their arguments as the provider sends
//! them, first on a raw completion stream, then through an agent run.
//!
//! The prompt asks for two calls in one turn: `write_file`, whose long
//! `content` arrives over many fragments, and `add`. Every event prints
//! with its time and part: a call's start names its tool, each argument
//! fragment shows the object its text states so far
//! ([`parse_partial_arguments`](rig::streaming::parse_partial_arguments)),
//! and its end brings the call's id. A summary checks that each call's
//! joined fragments are the arguments its end states. Tools only run once
//! a call ends.
//!
//! Run it with a provider, which reads its key from the environment:
//!
//! ```sh
//! cargo run -p agent_tool_call_streaming -- anthropic
//! cargo run -p agent_tool_call_streaming -- openai --model gpt-5.4-mini
//! cargo run -p agent_tool_call_streaming -- openai-responses --mode agent
//! cargo run -p agent_tool_call_streaming -- gemini --mode core
//! cargo run -p agent_tool_call_streaming -- deepseek --stop-after 5
//! ```
//!
//! Providers: `anthropic` (`ANTHROPIC_API_KEY`), `openai` (Chat Completions)
//! and `openai-responses` (`OPENAI_API_KEY`), `gemini` and
//! `gemini-interactions` (`GEMINI_API_KEY`), `deepseek` (`DEEPSEEK_API_KEY`),
//! `openrouter` (`OPENROUTER_API_KEY`). Gemini's generateContent sends each
//! call whole, so its calls start, carry one fragment, and end together.
//! `--stop-after N` stops the agent run from a hook after N argument
//! fragments, before any call ends, and shows that no tool ran.

mod hook;
mod tools;
mod view;

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;
use rig::DynModel;
use rig::agent::MultiTurnStreamItem;
use rig::completion::CompletionRequest;
use rig::message::{Message, ToolChoice, ToolResultContent};
use rig::operation::Completion;
use rig::prelude::*;
use rig::providers::{anthropic, deepseek, gemini, openai, openrouter};
use rig::streaming::Item;
use rig::tool::ToolSet;

use hook::FragmentHook;
use tools::{Add, Executed, WriteFile};
use view::StreamView;

const PREAMBLE: &str = "You are a careful assistant. Use the tools you are given. \
                        When asked for several tool calls, make them all in one turn.";

const PROMPT: &str = "In one turn, call both tools: use write_file to save a note to \
                      `notes/streaming.md` of three short paragraphs explaining why streaming \
                      tool-call arguments helps a user interface, and use add to compute \
                      1234 + 5678. After the tools answer, reply with one sentence giving the sum.";

#[derive(Clone, Copy, Debug)]
enum Provider {
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
    Gemini,
    GeminiInteractions,
    DeepSeek,
    OpenRouter,
}

impl Provider {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "anthropic" => Self::Anthropic,
            "openai" => Self::OpenAiChat,
            "openai-responses" => Self::OpenAiResponses,
            "gemini" => Self::Gemini,
            "gemini-interactions" => Self::GeminiInteractions,
            "deepseek" => Self::DeepSeek,
            "openrouter" => Self::OpenRouter,
            other => bail!(
                "unknown provider `{other}`: use anthropic, openai, openai-responses, gemini, \
                 gemini-interactions, deepseek or openrouter"
            ),
        })
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => anthropic::completion::CLAUDE_HAIKU_4_5,
            Self::OpenAiChat | Self::OpenAiResponses => openai::GPT_5_4_MINI,
            Self::Gemini | Self::GeminiInteractions => gemini::completion::GEMINI_2_5_FLASH,
            Self::DeepSeek => deepseek::DEEPSEEK_V4_FLASH,
            Self::OpenRouter => "openai/gpt-4o-mini",
        }
    }

    /// The completion model, its client built from the provider's
    /// environment variables.
    fn model(self, model: &str) -> Result<DynModel<Completion>> {
        let context = || format!("building the {self:?} client from the environment");
        Ok(match self {
            Self::Anthropic => anthropic::Anthropic::from_env()
                .with_context(context)?
                .completion(model)
                .erase(),
            Self::OpenAiChat => openai::OpenAI::from_env()
                .with_context(context)?
                .chat(model)
                .erase(),
            Self::OpenAiResponses => openai::OpenAI::from_env()
                .with_context(context)?
                .responses(model)
                .erase(),
            Self::Gemini => gemini::Gemini::from_env()
                .with_context(context)?
                .completion(model)
                .erase(),
            Self::GeminiInteractions => gemini::Gemini::from_env()
                .with_context(context)?
                .interactions(model)
                .erase(),
            Self::DeepSeek => deepseek::from_env()
                .with_context(context)?
                .chat(model)
                .erase(),
            Self::OpenRouter => openrouter::from_env()
                .with_context(context)?
                .chat(model)
                .erase(),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Core,
    Agent,
    Both,
}

struct Options {
    provider: Provider,
    model: String,
    mode: Mode,
    stop_after: Option<usize>,
}

impl Options {
    fn from_args() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let provider = Provider::parse(
            &args
                .next()
                .ok_or_else(|| anyhow!("name a provider, for example `anthropic`"))?,
        )?;
        let mut options = Self {
            provider,
            model: provider.default_model().to_owned(),
            mode: Mode::Both,
            stop_after: None,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| anyhow!("`{flag}` needs a value"));
            match flag.as_str() {
                "--model" => options.model = value()?,
                "--mode" => {
                    options.mode = match value()?.as_str() {
                        "core" => Mode::Core,
                        "agent" => Mode::Agent,
                        "both" => Mode::Both,
                        other => bail!("unknown mode `{other}`: use core, agent or both"),
                    }
                }
                "--stop-after" => {
                    options.stop_after = Some(value()?.parse().context("--stop-after")?);
                }
                other => bail!("unknown flag `{other}`"),
            }
        }
        Ok(options)
    }
}

fn tool_definitions() -> Vec<rig::completion::ToolDefinition> {
    let mut tools = ToolSet::default();
    tools.add_tool(WriteFile::default());
    tools.add_tool(Add::default());
    tools.tool_definitions()
}

/// One completion streamed straight from the model: the stream rig's
/// decoders write, before any agent logic.
async fn core_stream(model: &DynModel<Completion>) -> Result<bool> {
    println!("\n=== core completion stream ===\n");
    let request = CompletionRequest::new(Message::user(PROMPT))
        .preamble(PREAMBLE)
        .tools(tool_definitions())
        .tool_choice(ToolChoice::Required)
        .max_tokens(4096);
    let mut stream = model.stream(request)?;
    let mut view = StreamView::new();
    while let Some(item) = stream.next().await {
        match item? {
            Item::Event(event) => view.event(&event),
            Item::Unknown(_) => {}
        }
    }
    let response = stream.finish().await?;
    println!("\nfinish reason: {:?}", response.finish_reason());
    Ok(view.summary())
}

/// A full agent run over the same prompt: the agent forwards each call's
/// fragments live, the hook sees each one first, and the tools run once
/// the turn's calls have ended.
async fn agent_stream(model: DynModel<Completion>, stop_after: Option<usize>) -> Result<bool> {
    println!("\n=== agent stream ===\n");
    let write_file = WriteFile::default();
    let executed: Executed = write_file.executed.clone();
    let add = Add {
        executed: executed.clone(),
    };
    let agent = AgentBuilder::new(model)
        .preamble(PREAMBLE)
        .tool(write_file.clone())
        .tool(add)
        .max_tokens(4096)
        .build();
    let hook = FragmentHook::new(stop_after);
    let mut stream = agent
        .prompt(PROMPT)
        .add_hook(hook.clone())
        .max_turns(3)
        .stream();

    let mut view = StreamView::new();
    let mut all_match = true;
    let mut stopped = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(event))) => view.event(&event),
            Ok(MultiTurnStreamItem::CompletionCall(call)) => {
                println!(
                    "\ncompletion call finished: {:?} output tokens",
                    call.usage.output_tokens
                );
                all_match &= view.summary();
                view = StreamView::new();
                println!();
            }
            Ok(MultiTurnStreamItem::ToolCall { tool_call }) => {
                println!(
                    "committed call `{}` ({})",
                    tool_call.function.name, tool_call.id
                );
            }
            Ok(MultiTurnStreamItem::ToolResult { tool_result }) => {
                let content: Vec<String> = tool_result
                    .content
                    .iter()
                    .map(|content| match content {
                        ToolResultContent::Text(text) => text.text.clone(),
                        ToolResultContent::Json { value } => value.to_string(),
                        ToolResultContent::Image(_) => "<image>".to_owned(),
                    })
                    .collect();
                println!(
                    "tool result for {}: {}",
                    tool_result.call,
                    content.join(" ")
                );
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                println!("\nfinal answer: {}", response.output());
            }
            Ok(_) => {}
            Err(error) => {
                stopped = Some(error.to_string());
                break;
            }
        }
    }

    println!("\nargument fragments the hook saw, by part and tool:");
    for ((part, tool), count) in hook.counts() {
        println!("  part {part} `{tool}`: {count}");
    }
    let executed = executed.entries();
    println!("tools executed: {executed:?}");
    if let Ok(files) = write_file.files.lock() {
        for (path, content) in files.iter() {
            println!("  {path}: {} bytes", content.len());
        }
    }
    match (stopped, stop_after) {
        (Some(reason), Some(_)) => {
            println!("run stopped: {reason}");
            if !executed.is_empty() {
                bail!("a tool ran although the run stopped mid-call");
            }
            println!("no tool ran: the run stopped before any call ended");
            Ok(view.summary() && all_match)
        }
        (Some(reason), None) => bail!("the run failed: {reason}"),
        (None, _) => Ok(all_match),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = Options::from_args()?;
    let model = options.provider.model(&options.model)?;
    println!("provider {:?}, model {}", options.provider, options.model);

    let mut all_match = true;
    if options.mode != Mode::Agent {
        all_match &= core_stream(&model).await?;
    }
    if options.mode != Mode::Core {
        all_match &= agent_stream(model, options.stop_after).await?;
    }
    if !all_match {
        bail!("a call's joined fragments differ from the arguments its end states");
    }
    println!("\nevery call's joined fragments are the arguments its end states");
    Ok(())
}
