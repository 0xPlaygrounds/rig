//! CT-25: build a rig session from a chat transcript saved as Markdown.
//!
//! A plain program on rig's library crates (rig-core, rig-memory and the
//! `rig` crate's launcher protocol), no Bevy and no harness:
//! `transcript-to-session in.md [-o DIR] [--name N] [--model vendor/model]
//! [--user H] [--assistant H] [--tools]`. Speaker headings (`## User`,
//! `**Assistant:**`, …) split the transcript into messages kept verbatim;
//! text before the first heading becomes a system note; with `--tools`
//! a fence tagged `tool:<name>` in an assistant section becomes a recorded
//! tool call and its result. The session is written where `-o` says, or as
//! a new session of the rig home, so `rig --resume <name>` resumes it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rig_core::conversation::{Author, Conversation, Entry, Halt};
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, LocalCallId, Message, ToolCall, ToolFunction,
    ToolName, ToolResultContent, UserContent,
};
use rig_harness_protocol::{Home, SessionDir};
use rig_memory::{DirStore, Files, MODEL, SavedAgent, Session, SessionMeta, Store, title_of};
use serde::{Deserialize, Serialize};

/// The id of the one agent an imported session holds.
pub const AGENT: &str = "imported";

/// The file of the session's store that says where it was imported from
/// ([`Origin`]).
pub const ORIGIN: &str = "transcript-import/origin.json";

/// Who wrote the system note made of the text before the first heading.
const IMPORTER: &str = "transcript-import";

/// How a transcript is read.
#[derive(Clone, Debug)]
pub struct Options {
    /// The speaker name of the user's sections, matched case-insensitively.
    pub user: String,
    /// The speaker name of the assistant's sections.
    pub assistant: String,
    /// Whether `tool:<name>` fences in assistant sections become recorded
    /// tool calls and results.
    pub tools: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            user: "User".to_owned(),
            assistant: "Assistant".to_owned(),
            tools: false,
        }
    }
}

/// Who a section is from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Speaker {
    User,
    Assistant,
}

/// Why a transcript was not turned into a session.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// The command line cannot be read.
    #[error("{0}\n{USAGE}")]
    Usage(String),
    /// The transcript has no speaker heading outside a code fence.
    #[error(
        "no speaker headings (`## {user}` / `## {assistant}` or `**{user}:**` lines) outside code fences"
    )]
    NoHeadings {
        /// The user's heading.
        user: String,
        /// The assistant's heading.
        assistant: String,
    },
    /// The messages made do not pair each tool call with its result.
    #[error("the messages do not pair: {0}")]
    Unpaired(#[from] rig_core::conversation::pairing::PairingError),
    /// The target directory holds a session already.
    #[error("{0} holds a session already")]
    Occupied(PathBuf),
    /// Reading or writing failed.
    #[error("{path}: {failure}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// What failed.
        failure: std::io::Error,
    },
    /// The session store failed.
    #[error("{path}: {failure}")]
    Store {
        /// The session's directory.
        path: PathBuf,
        /// What failed.
        failure: rig_memory::SessionError,
    },
}

/// What reading a transcript gave.
#[derive(Clone, Debug)]
pub struct Imported {
    /// The conversation, in order: the system note, if any, then the
    /// messages.
    pub conversation: Conversation,
    /// The first line the user wrote, which names the session when no name
    /// is given.
    pub first_user_line: Option<String>,
    /// How many of each were made.
    pub counts: Counts,
    /// What was skipped or kept as text, with its line.
    pub skipped: Vec<String>,
}

/// How many messages of each kind a transcript gave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// System notes: 1 when the transcript has text before its first heading.
    pub system_notes: usize,
    /// User messages, tool results not counted.
    pub user: usize,
    /// Assistant messages.
    pub assistant: usize,
    /// Recorded tool call and result pairs.
    pub tool_pairs: usize,
}

/// Where an imported session came from, kept in its store at [`ORIGIN`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// The transcript file, as given.
    pub source: PathBuf,
    /// When it was imported, in milliseconds since the Unix epoch.
    pub imported_ms: u64,
}

impl Origin {
    /// The origin kept in `files`, if it has one that reads.
    pub fn read(files: &(impl Files + ?Sized)) -> Option<Self> {
        serde_json::from_slice(&files.read_file(ORIGIN).ok()?).ok()
    }
}

/// An open code fence: its character and how many of it.
#[derive(Clone, Copy)]
struct Fence {
    mark: char,
    len: usize,
}

/// The fence `line` opens or closes: its mark, length and info string.
fn fence_of(line: &str) -> Option<(Fence, &str)> {
    let trimmed = line.trim_start();
    let mark = trimmed.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let len = trimmed.chars().take_while(|c| *c == mark).count();
    let info = trimmed.get(len..)?.trim();
    (len >= 3).then_some((Fence { mark, len }, info))
}

/// Whether `line` closes `open`.
fn closes(open: Fence, line: &str) -> bool {
    fence_of(line).is_some_and(|(fence, info)| {
        fence.mark == open.mark && fence.len >= open.len && info.is_empty()
    })
}

impl Options {
    /// The speaker `label` names, if any.
    fn speaker(&self, label: &str) -> Option<Speaker> {
        let label = label.trim();
        if label.eq_ignore_ascii_case(self.user.trim()) {
            Some(Speaker::User)
        } else if label.eq_ignore_ascii_case(self.assistant.trim()) {
            Some(Speaker::Assistant)
        } else {
            None
        }
    }

    /// The speaker a heading line names, with the text after it on the
    /// line: `## User`, `## User:` or `**User:** text`.
    fn heading<'l>(&self, line: &'l str) -> Option<(Speaker, &'l str)> {
        let trimmed = line.trim();
        let hashes = trimmed.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) {
            let rest = trimmed.get(hashes..)?;
            if !rest.starts_with(char::is_whitespace) {
                return None;
            }
            let label = rest.trim().trim_end_matches('#').trim();
            let label = label.strip_suffix(':').unwrap_or(label);
            return Some((self.speaker(label)?, ""));
        }
        let inner = trimmed.strip_prefix("**")?;
        let (label, after) = inner.split_once("**")?;
        let (label, after) = match label.strip_suffix(':') {
            Some(label) => (label, after),
            None => (label, after.strip_prefix(':')?),
        };
        Some((self.speaker(label)?, after.trim_start()))
    }
}

/// `lines` joined, without the blank lines that start and end them.
fn joined(lines: &[&str]) -> String {
    let start = lines.iter().position(|line| !line.trim().is_empty());
    let end = lines.iter().rposition(|line| !line.trim().is_empty());
    match (start, end) {
        (Some(start), Some(end)) => lines.get(start..=end).unwrap_or_default().join("\n"),
        _ => String::new(),
    }
}

/// A section: who it is from, the line it starts on, its lines.
struct Section<'t> {
    speaker: Speaker,
    line: usize,
    lines: Vec<&'t str>,
}

/// Reads `markdown` as a transcript.
///
/// # Errors
///
/// When it has no speaker heading outside a code fence.
pub fn import(markdown: &str, options: &Options) -> Result<Imported, ImportError> {
    let mut preamble = Vec::new();
    let mut sections: Vec<Section> = Vec::new();
    let mut skipped = Vec::new();
    let mut open: Option<(Fence, usize)> = None;
    for (index, line) in markdown.lines().enumerate() {
        let number = index + 1;
        match open {
            Some((fence, _)) if closes(fence, line) => open = None,
            Some(_) => {}
            None => {
                if let Some((speaker, rest)) = options.heading(line) {
                    let lines = if rest.is_empty() { vec![] } else { vec![rest] };
                    sections.push(Section {
                        speaker,
                        line: number,
                        lines,
                    });
                    continue;
                }
                if let Some((fence, _)) = fence_of(line) {
                    open = Some((fence, number));
                }
            }
        }
        match sections.last_mut() {
            Some(section) => section.lines.push(line),
            None => preamble.push(line),
        }
    }
    if sections.is_empty() {
        return Err(ImportError::NoHeadings {
            user: options.user.clone(),
            assistant: options.assistant.clone(),
        });
    }
    if let Some((_, line)) = open {
        skipped.push(format!(
            "line {line}: a code fence that never closes runs to the end"
        ));
    }
    // Consecutive sections of one speaker are one message.
    let mut merged: Vec<(Speaker, usize, String)> = Vec::new();
    for section in sections {
        let text = joined(&section.lines);
        if text.is_empty() {
            skipped.push(format!("line {}: an empty section", section.line));
            continue;
        }
        match merged.last_mut() {
            Some((speaker, _, joined)) if *speaker == section.speaker => {
                joined.push_str("\n\n");
                joined.push_str(&text);
            }
            _ => merged.push((section.speaker, section.line, text)),
        }
    }
    let mut conversation = Conversation::new();
    let mut counts = Counts::default();
    let note = joined(&preamble);
    if !note.is_empty() {
        conversation.push(Entry::new(Message::system(note)).by(Author::plugin(IMPORTER)));
        counts.system_notes = 1;
    }
    let mut first_user_line = None;
    for (speaker, line, text) in merged {
        match speaker {
            Speaker::User => {
                if first_user_line.is_none() {
                    first_user_line = text
                        .lines()
                        .find(|line| !line.trim().is_empty())
                        .map(str::to_owned);
                }
                if options.tools && has_tool_fence(&text) {
                    skipped.push(format!(
                        "line {line}: a tool fence in a user section, kept as text"
                    ));
                }
                conversation.push(Message::user(text));
                counts.user += 1;
            }
            Speaker::Assistant if options.tools => {
                for part in assistant_parts(&text, line, &mut skipped) {
                    match part {
                        Part::Text(text) => {
                            conversation.push(Message::assistant(text));
                            counts.assistant += 1;
                        }
                        Part::Call(before, call, output) => {
                            let mut content: Vec<_> =
                                before.into_iter().map(AssistantContent::text).collect();
                            let result = call.result(vec![ToolResultContent::text(output)]);
                            content.push(AssistantContent::ToolCall(call));
                            conversation.push(Message::Assistant(AssistantMessage::new(content)));
                            conversation.push(Message::tool_results(vec![result]));
                            counts.assistant += 1;
                            counts.tool_pairs += 1;
                        }
                    }
                }
            }
            Speaker::Assistant => {
                conversation.push(Message::assistant(text));
                counts.assistant += 1;
            }
        }
    }
    let ends_with_user = matches!(conversation.messages().last().map(|message| &**message), Some(Message::User { content })
        if content.iter().all(|item| matches!(item, UserContent::Text(_))));
    if ends_with_user {
        // Resuming leaves the last request for the user to send on.
        conversation.halt(Halt::Kept);
    }
    conversation.validate()?;
    Ok(Imported {
        conversation,
        first_user_line,
        counts,
        skipped,
    })
}

/// The tool a fence's info string names: `tool:<name>`.
fn tool_of(info: &str) -> Option<ToolName> {
    let name = info.split_whitespace().next()?.strip_prefix("tool:")?;
    ToolName::new(name).ok()
}

/// Whether `text` holds a fence tagged as a tool's output.
fn has_tool_fence(text: &str) -> bool {
    text.lines()
        .filter_map(fence_of)
        .any(|(_, info)| tool_of(info).is_some())
}

/// A part of an assistant section read with `--tools`.
enum Part {
    /// Text that stays a message.
    Text(String),
    /// The text before a tool fence, the call it records and its output.
    Call(Option<String>, ToolCall, String),
}

/// `text`, an assistant section starting on line `first`, split at its
/// tool fences, each one closed; an unclosed one stays text.
fn assistant_parts(text: &str, first: usize, skipped: &mut Vec<String>) -> Vec<Part> {
    let lines: Vec<&str> = text.lines().collect();
    let mut parts = Vec::new();
    let mut before: Vec<&str> = Vec::new();
    let mut at = 0;
    while let Some(line) = lines.get(at) {
        let Some((fence, info)) = fence_of(line) else {
            before.push(line);
            at += 1;
            continue;
        };
        let rest = lines.get(at + 1..).unwrap_or_default();
        let end = rest.iter().position(|line| closes(fence, line));
        match (tool_of(info), end) {
            (Some(name), Some(end)) => {
                let body = rest.get(..end).unwrap_or_default();
                let (arguments, output) = match body.split_first() {
                    Some((command, output)) if command.starts_with("$ ") => (
                        serde_json::json!({ "command": command.trim_start_matches("$ ") }),
                        output.join("\n"),
                    ),
                    _ => (serde_json::json!({}), body.join("\n")),
                };
                let call = ToolCall::new(
                    CallId::from(LocalCallId::new()),
                    ToolFunction::new(name, arguments),
                );
                let text = joined(&before);
                parts.push(Part::Call((!text.is_empty()).then_some(text), call, output));
                before.clear();
                at += end + 2;
            }
            (tool, end) => {
                if tool.is_some() {
                    let line = first + at;
                    skipped.push(format!(
                        "line {line}: a tool fence that never closes, kept as text"
                    ));
                }
                // A fence of anything else stays text, closing line and all.
                let taken = end.map_or(rest.len(), |end| end + 1);
                before.extend(lines.get(at..=at + taken).unwrap_or_default());
                at += taken + 1;
            }
        }
    }
    let text = joined(&before);
    if !text.is_empty() {
        parts.push(Part::Text(text));
    }
    parts
}

/// What to write the session as.
#[derive(Clone, Debug)]
pub struct Target<'a> {
    /// The session's directory.
    pub dir: &'a Path,
    /// The name `rig --resume` finds it by.
    pub name: &'a str,
    /// The model of its agent, `vendor/model`; `None` leaves the harness's
    /// default to apply on resume.
    pub model: Option<&'a str>,
    /// The transcript it was imported from.
    pub source: &'a Path,
}

/// Writes `imported` as a session in `target.dir`, saved, named and listed
/// as the harness writes one, with its [`Origin`].
///
/// # Errors
///
/// When the directory holds a session already, or a write fails.
pub fn write(imported: &Imported, target: &Target) -> Result<(), ImportError> {
    let path = || target.dir.to_owned();
    let store_failed = |failure| ImportError::Store {
        path: path(),
        failure,
    };
    let io_failed = |failure| ImportError::Io {
        path: path(),
        failure,
    };
    let store = DirStore::open(target.dir).map_err(store_failed)?;
    if !store.logs().map_err(io_failed)?.is_empty() {
        return Err(ImportError::Occupied(path()));
    }
    let mut agent = SavedAgent {
        id: AGENT.to_owned(),
        conversation: imported.conversation.clone(),
        ..SavedAgent::default()
    };
    if let Some(model) = target.model {
        agent
            .saved
            .insert(MODEL.to_owned(), serde_json::Value::from(model));
    }
    let mut session = Session::default();
    session.agents = vec![agent];
    session.save(&store).map_err(store_failed)?;
    let now = rig_core::now_ms();
    let title = imported
        .first_user_line
        .as_deref()
        .map(title_of)
        .unwrap_or_default();
    let meta = SessionMeta {
        title,
        updated: now,
        ..SessionMeta::default()
    };
    let origin = Origin {
        source: target.source.to_owned(),
        imported_ms: now,
    };
    let dir = SessionDir::at(target.dir);
    let origin = serde_json::to_vec_pretty(&origin).map_err(|failure| io_failed(failure.into()))?;
    meta.write(&store)
        .and_then(|()| store.write_file(ORIGIN, &origin))
        .and_then(|()| dir.set_name(Some(target.name)))
        .and_then(|()| dir.mark_saved())
        .map_err(io_failed)
}

/// The command line, read.
#[derive(Clone, Debug, Default)]
struct Args {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    name: Option<String>,
    model: Option<String>,
    options: Options,
}

/// The usage line.
pub const USAGE: &str = "usage: transcript-to-session IN.md [-o DIR] [--name NAME] \
    [--model vendor/model] [--user HEADING] [--assistant HEADING] [--tools]";

fn parse_args(args: &[String]) -> Result<Args, ImportError> {
    let mut read = Args::default();
    let mut words = args.iter();
    while let Some(word) = words.next() {
        let mut value = || {
            words
                .next()
                .cloned()
                .ok_or_else(|| ImportError::Usage(format!("{word} takes a value")))
        };
        match word.as_str() {
            "-o" | "--output" => read.output = Some(value()?.into()),
            "--name" => read.name = Some(value()?),
            "--model" => read.model = Some(value()?),
            "--user" => read.options.user = value()?,
            "--assistant" => read.options.assistant = value()?,
            "--tools" => read.options.tools = true,
            flag if flag.starts_with('-') => {
                return Err(ImportError::Usage(format!("unknown option {flag}")));
            }
            input if read.input.is_none() => read.input = Some(input.into()),
            extra => {
                return Err(ImportError::Usage(format!("unexpected argument {extra}")));
            }
        }
    }
    if let Some(model) = &read.model
        && !model
            .split_once('/')
            .is_some_and(|(vendor, name)| !vendor.is_empty() && !name.is_empty())
    {
        let why = format!("--model {model}: use vendor/model");
        return Err(ImportError::Usage(why));
    }
    Ok(read)
}

/// Runs the program on `args` (without the program's name), with `home`
/// the rig home a session goes to when no `-o` is given: what it reports.
///
/// # Errors
///
/// What to say before exiting non-zero: a bad command line, an unreadable
/// or malformed transcript (nothing is written then), or a failed write.
pub fn run(args: &[String], home: &Home) -> Result<String, ImportError> {
    let args = parse_args(args)?;
    let input = args
        .input
        .ok_or_else(|| ImportError::Usage("no transcript named".to_owned()))?;
    let markdown = std::fs::read_to_string(&input).map_err(|failure| ImportError::Io {
        path: input.clone(),
        failure,
    })?;
    let imported = import(&markdown, &args.options)?;
    let name = args
        .name
        .or_else(|| imported.first_user_line.as_deref().map(title_of))
        .unwrap_or_else(|| title_of(&input.display().to_string()));
    let (dir, resume) = match &args.output {
        Some(dir) => (dir.clone(), None),
        None => {
            let (id, dir) = home.new_session().map_err(|failure| ImportError::Io {
                path: home.sessions(),
                failure,
            })?;
            (dir.path().to_owned(), Some(id))
        }
    };
    let target = Target {
        dir: &dir,
        name: &name,
        model: args.model.as_deref(),
        source: &input,
    };
    if let Err(why) = write(&imported, &target) {
        if resume.is_some() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        return Err(why);
    }
    let counts = imported.counts;
    let mut report = format!(
        "Wrote session `{name}` to {}: {} system note(s), {} user and {} assistant messages, \
         {} tool call(s).",
        dir.display(),
        counts.system_notes,
        counts.user,
        counts.assistant,
        counts.tool_pairs,
    );
    for skipped in &imported.skipped {
        let _ = write!(report, "\nSkipped: {skipped}");
    }
    match resume {
        Some(id) => {
            let _ = write!(report, "\nResume it with `rig --resume {id}`.");
        }
        None => {
            let _ = write!(
                report,
                "\nThe harness resumes sessions kept in its home ({}); move the directory \
                 there under a session id to resume it.",
                home.sessions().display()
            );
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests;
