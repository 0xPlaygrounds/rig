//! Reading another tool's session file into [`Steps`](Step): what was
//! said, called and answered, in order, with the source's timestamps,
//! mapped onto rig's messages and tools.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use rig_core::message::{
    AssistantContent, CallId, Reasoning, Text, ToolCall, ToolFunction, ToolName, ToolResult,
    ToolResultContent,
};
use serde_json::{Map, Value, json};

use crate::import_sessions::Source;

/// Most characters of an unknown entry kept in its note.
const NOTE_CHARS: usize = 2_000;

/// One thing that happened in the source session.
#[derive(Clone, Debug)]
pub(crate) struct Step {
    /// When, in milliseconds since the Unix epoch; 0 when the source did
    /// not say.
    pub(crate) at_ms: u64,
    pub(crate) kind: StepKind,
}

/// What a [`Step`] is.
#[derive(Clone, Debug)]
pub(crate) enum StepKind {
    /// The user's text.
    User(String),
    /// A model turn: text, reasoning and tool calls, in order.
    Assistant {
        content: Vec<AssistantContent>,
        model: String,
    },
    /// Tool results.
    Results(Vec<ToolResult>),
    /// An entry the importer does not understand, kept as a note.
    Note { kind: String, raw: String },
}

/// A source session, read.
#[derive(Clone, Debug, Default)]
pub(crate) struct Parsed {
    /// The source's own id of the session.
    pub(crate) id: String,
    /// The directory it ran in.
    pub(crate) cwd: Option<PathBuf>,
    /// The main conversation.
    pub(crate) main: Vec<Step>,
    /// Each subagent (sidechain) conversation, by the source's key.
    pub(crate) sidechains: Vec<(String, Vec<Step>)>,
    /// How many entries of each kind were read, such as `user`,
    /// `tool call → shell` or `unknown: summary`.
    pub(crate) counts: BTreeMap<String, usize>,
}

impl Parsed {
    /// The first thing the user said.
    pub(crate) fn first_message(&self) -> Option<&str> {
        self.main.iter().find_map(|step| match &step.kind {
            StepKind::User(text) => Some(text.as_str()),
            _ => None,
        })
    }

    /// When the session started and when it last changed.
    pub(crate) fn span(&self) -> (u64, u64) {
        let times = self.main.iter().map(|step| step.at_ms).filter(|&at| at > 0);
        let first = times.clone().min().unwrap_or_default();
        (first, times.max().unwrap_or_default())
    }

    fn count(&mut self, kind: impl Into<String>) {
        *self.counts.entry(kind.into()).or_default() += 1;
    }

    /// The steps of the conversation `chain`: the main one for `None`.
    fn steps(&mut self, chain: Option<&str>) -> &mut Vec<Step> {
        let Some(chain) = chain else {
            return &mut self.main;
        };
        let at = match self.sidechains.iter().position(|(key, _)| key == chain) {
            Some(at) => at,
            None => {
                self.sidechains.push((chain.to_owned(), Vec::new()));
                self.sidechains.len().saturating_sub(1)
            }
        };
        match self.sidechains.get_mut(at) {
            Some((_, steps)) => steps,
            None => &mut self.main,
        }
    }

    /// Adds `step` to `chain`, into its last step when both are model
    /// turns, as a streamed turn is logged in pieces.
    fn push(&mut self, chain: Option<&str>, step: Step) {
        let steps = self.steps(chain);
        if let (
            Some(Step {
                kind: StepKind::Assistant { content, .. },
                ..
            }),
            StepKind::Assistant { content: more, .. },
        ) = (steps.last_mut(), &step.kind)
        {
            content.extend(more.iter().cloned());
            return;
        }
        steps.push(step);
    }

    /// Adds a note about the entry `kind` the importer does not understand.
    fn note(&mut self, chain: Option<&str>, at_ms: u64, kind: &str, line: &str) {
        self.count(format!("unknown: {kind}"));
        let raw = rig_tools::shorten(line, NOTE_CHARS);
        let kind = kind.to_owned();
        self.push(
            chain,
            Step {
                at_ms,
                kind: StepKind::Note { kind, raw },
            },
        );
    }
}

/// Reads `text`, a session file of `source`.
pub(crate) fn parse(source: Source, text: &str) -> Parsed {
    match source {
        Source::ClaudeCode => claude_code(text),
        Source::Codex => codex(text),
    }
}

/// Milliseconds since the Unix epoch of an RFC 3339 time; 0 for none.
fn millis(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_str)
        .and_then(|text| chrono::DateTime::parse_from_rfc3339(text).ok())
        .and_then(|at| u64::try_from(at.timestamp_millis()).ok())
        .unwrap_or_default()
}

fn text_of(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// A call to `name` with `arguments`, mapped onto rig's tool of the same
/// job when there is one.
fn call(
    source: Source,
    parsed: &mut Parsed,
    id: &str,
    name: &str,
    arguments: Value,
) -> Option<ToolCall> {
    let (mapped, arguments) = map_tool(source, name, arguments);
    match &mapped {
        Some(mapped) => parsed.count(format!("tool call {name} → {mapped}")),
        None => parsed.count(format!("tool call {name} (imported as is)")),
    }
    let name = ToolName::new(mapped.unwrap_or_else(|| name.to_owned())).ok()?;
    Some(ToolCall::new(
        CallId::from_wire(id),
        ToolFunction::new(name, arguments),
    ))
}

/// rig's tool for the source's `name`, and the arguments it takes; `None`
/// keeps the call as it was.
fn map_tool(source: Source, name: &str, arguments: Value) -> (Option<String>, Value) {
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        other => other,
    };
    let get = |key: &str| arguments.get(key).cloned().unwrap_or(Value::Null);
    let mapped = match (source, name) {
        (Source::ClaudeCode, "Bash") => {
            let mut args = Map::new();
            args.insert("command".to_owned(), get("command"));
            if let Some(ms) = arguments.get("timeout").and_then(Value::as_u64) {
                args.insert("timeout_secs".to_owned(), json!(ms.div_ceil(1000)));
            }
            Some(("shell", Value::Object(args)))
        }
        (Source::ClaudeCode, "Edit") => {
            let edit = replacement(&arguments, "old_string", "new_string");
            Some(("edit", json!({ "path": get("file_path"), "edits": [edit] })))
        }
        (Source::ClaudeCode, "MultiEdit") => {
            let edits: Vec<Value> = arguments
                .get("edits")
                .and_then(Value::as_array)
                .map(|edits| {
                    let one = |edit| replacement(edit, "old_string", "new_string");
                    edits.iter().map(one).collect()
                })
                .unwrap_or_default();
            Some(("edit", json!({ "path": get("file_path"), "edits": edits })))
        }
        (Source::ClaudeCode, "Write") => Some((
            "write",
            json!({ "path": get("file_path"), "content": get("content") }),
        )),
        (Source::ClaudeCode, "Read") => {
            let mut args = Map::new();
            args.insert("path".to_owned(), get("file_path"));
            for key in ["offset", "limit"] {
                if let Some(n) = arguments.get(key).filter(|n| n.is_u64()) {
                    args.insert(key.to_owned(), n.clone());
                }
            }
            Some(("read", Value::Object(args)))
        }
        (Source::Codex, "shell" | "container.exec" | "shell_command" | "local_shell") => {
            Some(("shell", json!({ "command": command_line(&get("command")) })))
        }
        (Source::Codex, "exec_command") => {
            Some(("shell", json!({ "command": command_line(&get("cmd")) })))
        }
        _ => None,
    };
    match mapped {
        Some((name, arguments)) => (Some(name.to_owned()), arguments),
        None => (None, arguments),
    }
}

/// One replacement of rig's `edit` from the source's.
fn replacement(edit: &Value, old: &str, new: &str) -> Value {
    json!({
        "old_text": edit.get(old).cloned().unwrap_or(Value::Null),
        "new_text": edit.get(new).cloned().unwrap_or(Value::Null),
        "replace_all": edit.get("replace_all").and_then(Value::as_bool).unwrap_or(false),
    })
}

/// A command line from Codex's argument vector, such as
/// `["bash", "-lc", "ls"]`.
fn command_line(command: &Value) -> String {
    match command {
        Value::String(line) => line.clone(),
        Value::Array(words) => {
            let words: Vec<&str> = words.iter().filter_map(Value::as_str).collect();
            match words.as_slice() {
                [_, "-lc" | "-c", line] => (*line).to_owned(),
                words => words.join(" "),
            }
        }
        _ => String::new(),
    }
}

/// A tool result's text, from a string or a list of text blocks.
fn result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Object(object)) => object.get("content").or(object.get("output")).map_or_else(
            || Value::Object(object.clone()).to_string(),
            |inner| result_text(Some(inner)),
        ),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// The result of the call `id`, named after the call; a result of no call
/// seen is named `unknown`, and becomes a note when imported.
fn result(
    names: &HashMap<String, ToolName>,
    id: &str,
    text: String,
    is_error: bool,
) -> Option<ToolResult> {
    let name = match names.get(id) {
        Some(name) => name.clone(),
        None => ToolName::new("unknown").ok()?,
    };
    Some(ToolResult {
        call: CallId::from_wire(id),
        name,
        content: vec![ToolResultContent::text(text)],
        is_error,
    })
}

/// A Claude Code project log: one JSON entry per line, `user` and
/// `assistant` ones holding an Anthropic message; `isSidechain` ones are a
/// subagent's.
fn claude_code(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    let mut names: HashMap<String, ToolName> = HashMap::new();
    // The sidechain each entry is in, by its uuid.
    let mut chains: HashMap<String, String> = HashMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            parsed.note(None, 0, "unreadable line", line);
            continue;
        };
        let kind = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("untyped");
        let at_ms = millis(entry.get("timestamp"));
        if parsed.id.is_empty()
            && let Some(id) = entry.get("sessionId").and_then(Value::as_str)
        {
            parsed.id = id.to_owned();
        }
        if parsed.cwd.is_none() {
            parsed.cwd = entry.get("cwd").and_then(Value::as_str).map(PathBuf::from);
        }
        let uuid = text_of(entry.get("uuid"));
        let chain = entry
            .get("isSidechain")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .then(|| {
                let parent = text_of(entry.get("parentUuid"));
                let agent = entry.get("agentId").and_then(Value::as_str);
                let key = agent
                    .map(str::to_owned)
                    .or_else(|| chains.get(&parent).cloned())
                    .unwrap_or_else(|| uuid.clone());
                chains.insert(uuid.clone(), key.clone());
                key
            });
        let chain = chain.as_deref();
        let message = entry.get("message");
        let content = message.and_then(|message| message.get("content"));
        match (kind, content) {
            ("user", Some(Value::String(text))) => {
                parsed.count("user");
                parsed.push(
                    chain,
                    Step {
                        at_ms,
                        kind: StepKind::User(text.clone()),
                    },
                );
            }
            ("user", Some(Value::Array(blocks))) => {
                let mut results = Vec::new();
                let mut texts = Vec::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("tool_result") => {
                            let id = text_of(block.get("tool_use_id"));
                            let text = result_text(block.get("content"));
                            let error = block.get("is_error").and_then(Value::as_bool);
                            results.extend(result(&names, &id, text, error.unwrap_or(false)));
                        }
                        Some("text") => texts.push(text_of(block.get("text"))),
                        other => {
                            let kind = format!("user {}", other.unwrap_or("block"));
                            parsed.note(chain, at_ms, &kind, &block.to_string());
                        }
                    }
                }
                if !results.is_empty() {
                    parsed.count("tool result");
                    parsed.push(
                        chain,
                        Step {
                            at_ms,
                            kind: StepKind::Results(results),
                        },
                    );
                }
                if !texts.is_empty() {
                    parsed.count("user");
                    let text = texts.join("\n");
                    parsed.push(
                        chain,
                        Step {
                            at_ms,
                            kind: StepKind::User(text),
                        },
                    );
                }
            }
            ("assistant", Some(Value::Array(blocks))) => {
                parsed.count("assistant");
                let model = text_of(message.and_then(|message| message.get("model")));
                let mut content = Vec::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            content.push(AssistantContent::Text(Text::new(text_of(
                                block.get("text"),
                            ))));
                        }
                        Some("thinking") => {
                            let thinking = Reasoning::new(text_of(block.get("thinking")));
                            content.push(AssistantContent::Reasoning(thinking));
                        }
                        Some("tool_use") => {
                            let id = text_of(block.get("id"));
                            let name = text_of(block.get("name"));
                            let input = block.get("input").cloned().unwrap_or(Value::Null);
                            let Some(call) =
                                call(Source::ClaudeCode, &mut parsed, &id, &name, input)
                            else {
                                parsed.note(chain, at_ms, "unnamed tool_use", &block.to_string());
                                continue;
                            };
                            names.insert(id, call.function.name.clone());
                            content.push(AssistantContent::ToolCall(call));
                        }
                        other => {
                            let kind = format!("assistant {}", other.unwrap_or("block"));
                            parsed.note(chain, at_ms, &kind, &block.to_string());
                        }
                    }
                }
                if !content.is_empty() {
                    parsed.push(
                        chain,
                        Step {
                            at_ms,
                            kind: StepKind::Assistant { content, model },
                        },
                    );
                }
            }
            _ => parsed.note(chain, at_ms, kind, line),
        }
    }
    parsed
}

/// A Codex rollout: one `{timestamp, type, payload}` per line, the
/// conversation in its `response_item`s.
fn codex(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    let mut names: HashMap<String, ToolName> = HashMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            parsed.note(None, 0, "unreadable line", line);
            continue;
        };
        let at_ms = millis(entry.get("timestamp"));
        let payload = entry.get("payload").cloned().unwrap_or(Value::Null);
        let item = payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("untyped");
        let field = |key: &str| text_of(payload.get(key));
        match entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("untyped")
        {
            "session_meta" => {
                parsed.count("session_meta");
                parsed.id = field("id");
                parsed.cwd = payload
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(PathBuf::from);
            }
            // The events and turn settings repeat what the response items
            // hold: counted, not imported twice.
            kind @ ("event_msg" | "turn_context") => {
                parsed.count(format!("{kind} (repeats the items)"))
            }
            "response_item" => match item {
                "message" => {
                    let texts: Vec<&str> = payload
                        .get("content")
                        .and_then(Value::as_array)
                        .map(|blocks| {
                            blocks
                                .iter()
                                .filter_map(|block| block.get("text").and_then(Value::as_str))
                                .collect()
                        })
                        .unwrap_or_default();
                    let text = texts.join("\n");
                    match payload.get("role").and_then(Value::as_str) {
                        Some("user") => {
                            parsed.count("user");
                            parsed.push(
                                None,
                                Step {
                                    at_ms,
                                    kind: StepKind::User(text),
                                },
                            );
                        }
                        Some("assistant") => {
                            parsed.count("assistant");
                            let content = vec![AssistantContent::Text(Text::new(text))];
                            let model = String::new();
                            parsed.push(
                                None,
                                Step {
                                    at_ms,
                                    kind: StepKind::Assistant { content, model },
                                },
                            );
                        }
                        role => {
                            let kind = format!("{} message", role.unwrap_or("unknown"));
                            parsed.note(None, at_ms, &kind, line);
                        }
                    }
                }
                "reasoning" => {
                    parsed.count("reasoning");
                    let summary: Vec<&str> = payload
                        .get("summary")
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(|part| part.get("text")?.as_str())
                                .collect()
                        })
                        .unwrap_or_default();
                    let content = vec![AssistantContent::Reasoning(Reasoning::new(
                        summary.join("\n"),
                    ))];
                    parsed.push(
                        None,
                        Step {
                            at_ms,
                            kind: StepKind::Assistant {
                                content,
                                model: String::new(),
                            },
                        },
                    );
                }
                "function_call" | "custom_tool_call" | "local_shell_call" => {
                    let id = field("call_id");
                    let name = payload
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or(item)
                        .to_owned();
                    let arguments = match item {
                        "custom_tool_call" => json!({ "input": field("input") }),
                        "local_shell_call" => payload.get("action").cloned().unwrap_or(Value::Null),
                        _ => payload.get("arguments").cloned().unwrap_or(Value::Null),
                    };
                    let Some(call) = call(Source::Codex, &mut parsed, &id, &name, arguments) else {
                        parsed.note(None, at_ms, "unnamed call", line);
                        continue;
                    };
                    names.insert(id, call.function.name.clone());
                    let content = vec![AssistantContent::ToolCall(call)];
                    parsed.push(
                        None,
                        Step {
                            at_ms,
                            kind: StepKind::Assistant {
                                content,
                                model: String::new(),
                            },
                        },
                    );
                }
                "function_call_output" | "custom_tool_call_output" => {
                    parsed.count("tool result");
                    let id = field("call_id");
                    let text = result_text(payload.get("output"));
                    let results = result(&names, &id, text, false).into_iter().collect();
                    parsed.push(
                        None,
                        Step {
                            at_ms,
                            kind: StepKind::Results(results),
                        },
                    );
                }
                other => parsed.note(None, at_ms, other, line),
            },
            other => parsed.note(None, at_ms, other, line),
        }
    }
    parsed
}
