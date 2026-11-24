//! Claude Code as a model: `claude --print` with streamed JSON in and out,
//! one process for a whole conversation. Every built-in tool, settings
//! file, hook, skill, command, memory and MCP server of the user's is left
//! out; the process sees the system prompt and the tools Rig gives it.
//!
//! Each line of input carries an id, and Claude Code says under that id
//! when it holds the line and when its model starts on it. A line written
//! while a turn runs is given to the model in that turn, after the tool
//! results it waits for.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rig_harness::rig_core::completion::{FinishReason, Usage};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::harness::{
    Event, Harness, Launch, Part, Prompt, Running, Stderr, Steer, effort_wire, note, spawn,
};

#[cfg(test)]
mod tests;

/// The MCP server name Rig's tools are under: the model sees a tool as
/// `mcp__rig__<name>`.
const SERVER: &str = "rig";

/// How long Claude Code waits for a tool, in milliseconds: Rig's tools set
/// their own limits, and a person may take a while to approve one.
const TOOL_TIMEOUT_MS: &str = "86400000";

/// Variables a child `claude` must not inherit: those that say this process
/// runs inside Claude Code, and those that would make it answer with an API
/// key or another provider instead of its own sign-in.
pub(crate) const INHERITED: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_ENV_FILE",
];

/// Claude Code.
pub(crate) struct Claude;

impl Harness for Claude {
    fn note(&self, project: Option<&Path>) -> String {
        note(&format!("mcp__{SERVER}__read"), "", Some(project))
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let Launch {
            mut command,
            dir,
            system,
            model,
            effort,
            mcp,
            tools: _,
        } = launch;
        let system_file = dir.join("system.md");
        let mcp_file = dir.join("mcp.json");
        let servers = json!({ "mcpServers": { SERVER: {
            "type": "http",
            "url": mcp.url(),
            "headers": { "Authorization": format!("Bearer {}", mcp.token()) },
        } } });
        std::fs::write(&system_file, system)
            .and_then(|()| std::fs::write(&mcp_file, servers.to_string()))
            .map_err(|why| format!("could not write the session's files: {why}"))?;
        let settings = json!({
            "disableAllHooks": true,
            "autoMemoryEnabled": false,
            "includeGitInstructions": false,
            "claudeMdExcludes": ["**/CLAUDE.md", "**/AGENTS.md", "**/.claude/rules/**"],
        });
        command
            .current_dir(dir)
            .args(["--print", "--verbose"])
            .args(["--input-format", "stream-json"])
            .args(["--output-format", "stream-json"])
            .arg("--include-partial-messages")
            .args(["--model", model])
            .arg("--system-prompt-file")
            .arg(&system_file)
            .args(["--tools", ""])
            .arg("--mcp-config")
            .arg(&mcp_file)
            .arg("--strict-mcp-config")
            .args(["--setting-sources", ""])
            .args(["--settings", &settings.to_string()])
            .arg("--disable-slash-commands")
            .arg("--no-session-persistence")
            // Without it newer models send their reasoning as empty text.
            .args(["--thinking-display", "summarized"])
            // Nothing is left to ask about but Rig's tools, which Rig gates.
            .args(["--permission-mode", "bypassPermissions"])
            .env("ENABLE_CLAUDEAI_MCP_SERVERS", "0")
            .env("DISABLE_AUTO_COMPACT", "1")
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            .env("CLAUDE_CODE_AUTO_CONNECT_IDE", "0")
            .env("MCP_TOOL_TIMEOUT", TOOL_TIMEOUT_MS)
            .env("MAX_MCP_OUTPUT_TOKENS", "1000000");
        if let Some(effort) = effort {
            command.args(["--effort", effort_wire(effort, true)]);
        }
        for name in INHERITED {
            command.env_remove(name);
        }
        let (process, pipes) = spawn(command)?;
        let (prompts, mut prompted) = mpsc::unbounded_channel::<Prompt>();
        let (steers, mut steered) = mpsc::unbounded_channel::<Steer>();
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        tokio::spawn(async move {
            let mut stdin = pipes.stdin;
            let mut reader = Reader::default();
            let mut lines = BufReader::new(pipes.stdout).lines();
            loop {
                tokio::select! {
                    prompt = prompted.recv() => {
                        let Some(prompt) = prompt else {
                            return;
                        };
                        let id = uuid::Uuid::new_v4().to_string();
                        let mut line = user_message(&prompt.parts, &id, false).to_string();
                        line.push('\n');
                        if stdin.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Some(steer) = steered.recv() => {
                        // A program that does not say when it holds a line
                        // cannot say whether its turn took this one.
                        if !reader.reports {
                            let why = "this Claude Code does not report its input".to_owned();
                            let _ = steer.taken.send(Err(why));
                            continue;
                        }
                        let id = uuid::Uuid::new_v4().to_string();
                        let mut line = user_message(&steer.parts, &id, true).to_string();
                        line.push('\n');
                        reader.offered.insert(id, steer.taken);
                        if stdin.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    line = lines.next_line() => match line {
                        Ok(Some(line)) => {
                            for event in reader.read(&line) {
                                if events.send(event).is_err() {
                                    return;
                                }
                            }
                        }
                        Ok(None) | Err(_) => break,
                    }
                }
            }
            // The program is gone: a prompt it left open failed.
            let why = stderr.explain("claude exited").await;
            let _ = events.send(Event::Done(Err(why)));
        });
        Ok(Running {
            prompts,
            steers: Some(steers),
            events: reported,
            _alive: Box::new(process),
        })
    }
}

/// A prompt as one line of streamed input, under `id`. One for the turn in
/// progress waits for the tool calls in flight and is read with their
/// results.
fn user_message(prompt: &[Part], id: &str, steers: bool) -> Value {
    let content: Vec<Value> = prompt
        .iter()
        .map(|part| match part {
            Part::Text(text) => json!({ "type": "text", "text": text }),
            Part::Image { media_type, data } => json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data },
            }),
        })
        .collect();
    let mut line = json!({
        "type": "user",
        "message": { "role": "user", "content": content },
        "uuid": id,
    });
    if let (true, Some(line)) = (steers, line.as_object_mut()) {
        line.insert("priority".to_owned(), "next".into());
    }
    line
}

/// Turns the lines of streamed output into events.
#[derive(Default)]
struct Reader {
    /// The ids of the API messages whose content arrived as stream events.
    streamed: HashSet<String>,
    /// The input side of the open API message's usage.
    input: Counted,
    /// Whether the program says what becomes of each line it is sent.
    reports: bool,
    /// Who waits to hear that the program holds a line, by the line's id.
    offered: HashMap<String, oneshot::Sender<Result<(), String>>>,
    /// The lines the program holds and its model has not started on.
    held: HashSet<String>,
    /// How the turn ended while the program held a line: the reply's end,
    /// unless the program starts on that line or drops it.
    ended: Option<Result<FinishReason, String>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Line {
    StreamEvent {
        event: Stream,
    },
    Assistant {
        message: Whole,
    },
    CommandLifecycle {
        command_uuid: String,
        state: String,
    },
    Result {
        #[serde(default)]
        is_error: bool,
        #[serde(default)]
        stop_reason: Option<String>,
        #[serde(default)]
        result: Option<String>,
        #[serde(default)]
        errors: Vec<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Stream {
    MessageStart {
        message: Started,
    },
    ContentBlockDelta {
        delta: Delta,
    },
    MessageDelta {
        #[serde(default)]
        usage: Counted,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct Started {
    id: String,
    #[serde(default)]
    usage: Counted,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(other)]
    Other,
}

/// A complete API message, as Claude Code repeats it after its stream.
#[derive(Deserialize)]
struct Whole {
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    content: Vec<Block>,
    #[serde(default)]
    usage: Counted,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    #[serde(other)]
    Other,
}

/// The counters of an API message. A later report leaves out what an
/// earlier one said.
#[derive(Deserialize, Default, Clone, Copy)]
struct Counted {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens_details: Details,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Details {
    thinking_tokens: Option<u64>,
}

impl Counted {
    /// These counters, with `earlier`'s where these say nothing.
    fn over(self, earlier: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.or(earlier.input_tokens),
            output_tokens: self.output_tokens.or(earlier.output_tokens),
            cache_read_input_tokens: self
                .cache_read_input_tokens
                .or(earlier.cache_read_input_tokens),
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .or(earlier.cache_creation_input_tokens),
            output_tokens_details: Details {
                thinking_tokens: self
                    .output_tokens_details
                    .thinking_tokens
                    .or(earlier.output_tokens_details.thinking_tokens),
            },
        }
    }

    /// Anthropic counts cached input apart from `input_tokens`; Rig's
    /// input counter holds all three.
    fn usage(self) -> Usage {
        let mut usage = Usage::new();
        let read = self.cache_read_input_tokens;
        let written = self.cache_creation_input_tokens;
        usage.input_tokens = self
            .input_tokens
            .map(|fresh| fresh + read.unwrap_or(0) + written.unwrap_or(0));
        usage.output_tokens = self.output_tokens;
        usage.cached_input_tokens = read;
        usage.cache_creation_input_tokens = written;
        usage.reasoning_tokens = self.output_tokens_details.thinking_tokens;
        usage.total_tokens = usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(input, output)| input + output);
        usage
    }
}

/// The model name of a message Claude Code wrote itself, such as a quota
/// notice. Its text is in the result that follows.
const SYNTHETIC: &str = "<synthetic>";

impl Reader {
    /// The events one line of output stands for. A line that is not a
    /// known message says nothing.
    fn read(&mut self, line: &str) -> Vec<Event> {
        let Ok(line) = serde_json::from_str::<Line>(line) else {
            return Vec::new();
        };
        match line {
            Line::StreamEvent { event } => match event {
                Stream::MessageStart { message } => {
                    self.streamed.insert(message.id);
                    self.input = message.usage;
                    vec![Event::Began]
                }
                Stream::ContentBlockDelta { delta } => match delta {
                    Delta::Text { text } if !text.is_empty() => vec![Event::Text(text)],
                    Delta::Thinking { thinking } if !thinking.is_empty() => {
                        vec![Event::Reasoning(thinking)]
                    }
                    _ => Vec::new(),
                },
                Stream::MessageDelta { usage } => {
                    vec![Event::Usage(usage.over(self.input).usage())]
                }
                Stream::Other => Vec::new(),
            },
            // A message that was not streamed: Claude Code's retry of a
            // stalled stream arrives whole.
            Line::Assistant { message }
                if message.model != SYNTHETIC && self.streamed.insert(message.id.clone()) =>
            {
                let mut events: Vec<Event> = message
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        Block::Text { text } if !text.is_empty() => Some(Event::Text(text)),
                        Block::Thinking { thinking } if !thinking.is_empty() => {
                            Some(Event::Reasoning(thinking))
                        }
                        _ => None,
                    })
                    .collect();
                events.push(Event::Usage(message.usage.usage()));
                events
            }
            Line::Assistant { .. } | Line::Other => Vec::new(),
            Line::CommandLifecycle {
                command_uuid,
                state,
            } => {
                self.reports = true;
                let (held, started) = (state == "queued", state == "started");
                let was_held = match held {
                    true => !self.held.insert(command_uuid.clone()),
                    false => self.held.remove(&command_uuid),
                };
                let unread = format!("Claude Code {state} a message of the user's unread");
                if let Some(taken) = self.offered.remove(&command_uuid) {
                    let _ = taken.send(match held || started {
                        true => Ok(()),
                        false => Err(unread.clone()),
                    });
                }
                // A line the program said it held and then dropped is lost:
                // the turn fails, so the message is the user's to send again.
                if was_held && !(held || started || state == "completed") {
                    self.ended = None;
                    return vec![Event::Done(Err(unread))];
                }
                // The program answers a line its turn did not get to as a
                // turn of its own, at once: the reply goes on into that one.
                match self.ended.take() {
                    Some(_) if started => Vec::new(),
                    Some(done) if self.held.is_empty() => vec![Event::Done(done)],
                    ended => {
                        self.ended = ended;
                        Vec::new()
                    }
                }
            }
            Line::Result {
                is_error,
                stop_reason,
                result,
                errors,
            } => {
                self.streamed.clear();
                let done = if is_error {
                    let why = result.unwrap_or_else(|| errors.join("; "));
                    Err(if why.is_empty() {
                        "Claude Code reported an error".to_owned()
                    } else {
                        why
                    })
                } else {
                    match stop_reason.as_deref() {
                        Some("max_tokens") => Ok(FinishReason::Length),
                        Some("refusal") => Err("the model refused to answer".to_owned()),
                        _ => Ok(FinishReason::Stop),
                    }
                };
                if done.is_ok() && !self.held.is_empty() {
                    self.ended = Some(done);
                    return Vec::new();
                }
                vec![Event::Done(done)]
            }
        }
    }
}
