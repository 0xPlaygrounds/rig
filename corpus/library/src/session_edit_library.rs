//! `session-edit`: a plain program on rig's library crates (rig-core,
//! rig-memory, and the std-only `RIG_HOME` layout of `rig`), with no Bevy,
//! no harness and no plugins, that opens a saved rig session and shows,
//! shrinks, cuts, renames or exports it.
//!
//! ```text
//! session-edit <session> list
//! session-edit <session> shrink --over 4000 [--in-place | --as <name>]
//! session-edit <session> drop <from>..<to>  [--in-place | --as <name>]
//! session-edit <session> rename <name>      [--in-place]
//! session-edit <session> export --markdown <file>
//! ```
//!
//! `<session>` is a session's directory, or its id or name in the default
//! session store (`$RIG_HOME/sessions`, found with [`Home::find`]). An
//! edit is saved as a new session (`--as`, the default), a copy of the
//! session's logs and files, or in place (`--in-place`), after a copy of
//! the session's directory to `<dir>.bak`. A shrink rewrites every saved
//! copy of each result ([`rewrite`]), so the edited session's history no
//! longer holds what was cut; a drop is a recorded splice of the main
//! conversation.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::conversation::{Author, Conversation, Entry, pairing};
use rig_core::message::{AssistantContent, CallId, Message, ToolResultContent, UserContent};
use rig_core::now_ms;
use rig_harness_protocol::{Home, SessionDir};
use rig_memory::{
    DirStore, Messages, Repair, Rewrite, SavedAgent, Session, SessionError, SessionMeta, Store,
    rewrite,
};
use serde_json::Value;

/// The lines a shrunk tool result keeps at each end.
pub const KEEP_LINES: usize = 20;

/// Who the edits are recorded as made by.
const AUTHOR: &str = "session-edit";

/// How the program is called.
pub const USAGE: &str = "usage: session-edit <session> list
       session-edit <session> shrink --over <chars> [--in-place | --as <name>]
       session-edit <session> drop <from>..<to> [--in-place | --as <name>]
       session-edit <session> rename <name> [--in-place]
       session-edit <session> export --markdown <file>
<session> is a session directory, a session id or a session name.";

/// Why the program stopped: said once, on stderr, with a non-zero exit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<SessionError> for Error {
    fn from(error: SessionError) -> Self {
        Self(error.to_string())
    }
}

impl From<String> for Error {
    fn from(why: String) -> Self {
        Self(why)
    }
}

impl From<&str> for Error {
    fn from(why: &str) -> Self {
        Self(why.to_owned())
    }
}

/// Where an edit goes.
enum Target {
    /// Over the session, after a copy to `<dir>.bak`.
    InPlace,
    /// A new session, named so, or after the session.
    As(Option<String>),
}

/// What the program was asked.
enum Command {
    List,
    Export(PathBuf),
    Edit(Edit),
}

/// A change of the session.
enum Edit {
    Shrink(usize),
    Drop(usize, usize),
    Rename(String),
}

/// Runs `session-edit` with `args` (without the program's name) against
/// the sessions of `home`, writing what it did to `out`.
///
/// # Errors
///
/// When the arguments are wrong, the session cannot be found or read
/// whole (missing, cut short, in another format), an edit does not fit
/// the conversation, or a write fails; nothing is written then.
pub fn run(args: &[String], home: &Home, out: &mut dyn Write) -> Result<(), Error> {
    let [session, command, rest @ ..] = args else {
        return Err(USAGE.into());
    };
    let mut target = None;
    let mut over = None;
    let mut markdown = None;
    let mut words = Vec::new();
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        let mut value = || {
            rest.next()
                .ok_or_else(|| Error(format!("{arg} needs a value")))
        };
        match arg.as_str() {
            "--in-place" => target = Some(Target::InPlace),
            "--as" => target = Some(Target::As(Some(value()?.clone()))),
            "--over" => {
                let chars = value()?;
                let chars = chars
                    .parse()
                    .map_err(|_| format!("--over {chars}: not a number"))?;
                over = Some(chars);
            }
            "--markdown" => markdown = Some(PathBuf::from(value()?)),
            _ => words.push(arg.as_str()),
        }
    }
    let command = match (command.as_str(), words.as_slice()) {
        ("list", []) => Command::List,
        ("shrink", []) => Command::Edit(Edit::Shrink(over.ok_or("shrink needs --over <chars>")?)),
        ("drop", [range]) => {
            let (from, to) = parse_range(range)?;
            Command::Edit(Edit::Drop(from, to))
        }
        ("rename", [name]) => Command::Edit(Edit::Rename((*name).to_owned())),
        ("export", []) => Command::Export(markdown.ok_or("export needs --markdown <file>")?),
        _ => return Err(USAGE.into()),
    };
    let opened = open(locate(home, session)?)?;
    let write = |out: &mut dyn Write, text: &str| writeln!(out, "{text}").map_err(Error::from);
    match command {
        Command::List => write(out, &list(&opened)),
        Command::Export(file) => {
            fs::write(&file, markdown_of(&opened))
                .map_err(|why| format!("cannot write {}: {why}", file.display()))?;
            write(out, &format!("exported to {}", file.display()))
        }
        Command::Edit(edit) => {
            let target = target.unwrap_or(Target::As(None));
            let report = save(home, opened, target, &edit)?;
            write(out, &report.join("\n"))
        }
    }
}

/// `<from>..<to>`, both kept in the range, or one index.
fn parse_range(range: &str) -> Result<(usize, usize), Error> {
    let number = |text: &str| {
        text.trim()
            .parse::<usize>()
            .map_err(|_| Error(format!("{range}: not a range like 2..4")))
    };
    match range.split_once("..") {
        Some((from, to)) => Ok((number(from)?, number(to.trim_start_matches('='))?)),
        None => number(range).map(|at| (at, at)),
    }
}

/// A session found and read whole.
struct Opened {
    dir: SessionDir,
    store: DirStore,
    session: Session,
    /// Its listing ([`SessionMeta`]); empty when it has none.
    meta: SessionMeta,
}

impl Opened {
    /// What the session is listed as: its name ([`SessionDir::name`]),
    /// else the start of its first message.
    fn shown(&self) -> Option<String> {
        let title = Some(self.meta.title.clone()).filter(|title| !title.is_empty());
        self.dir.name().or(title)
    }
}

/// The session `arg` names: a directory, or the id or name of a saved
/// session under `home`.
fn locate(home: &Home, arg: &str) -> Result<SessionDir, Error> {
    let path = Path::new(arg);
    if path.is_dir() {
        return Ok(SessionDir::at(path));
    }
    let id = home.find(arg).map_err(|why| {
        format!(
            "no session {arg}: it is no directory, and {why} in {}",
            home.sessions().display()
        )
    })?;
    Ok(home.session(&id))
}

/// Reads the session in `dir` whole, strictly: a session cut short, in
/// another format or holding no conversation is refused.
fn open(dir: SessionDir) -> Result<Opened, Error> {
    let shown = dir.path().display().to_string();
    let refuse = |why: String| Error(format!("cannot read the session {shown}: {why}"));
    if dir.is_deleting() {
        return Err(refuse("it is being deleted".to_owned()));
    }
    let store = DirStore::open(dir.path()).map_err(|why| refuse(why.to_string()))?;
    let session = Session::read(&store).map_err(|why| refuse(why.to_string()))?;
    if let Some(torn) = session
        .repairs
        .iter()
        .find(|repair| matches!(repair, Repair::TornTail { .. }))
    {
        return Err(refuse(format!("{torn}; the file is truncated")));
    }
    if session.agents.is_empty() {
        return Err(refuse("it holds no saved conversation".to_owned()));
    }
    let meta = SessionMeta::read(&store).unwrap_or_default();
    Ok(Opened {
        dir,
        store,
        session,
        meta,
    })
}

/// The main agent's conversation, to change it.
fn main_of(session: &mut Session) -> Result<&mut Conversation, Error> {
    let id = session.main_agent().map(|agent| agent.id.clone());
    let agent = id.and_then(|id| session.agent_mut(&id));
    let agent = agent.ok_or("the session has no main conversation")?;
    Ok(&mut agent.conversation)
}

/// What a spawned agent is called: its title, else its id.
fn agent_name(agent: &SavedAgent) -> String {
    agent.title().unwrap_or_else(|| agent.id.clone())
}

/// The `list` output.
fn list(opened: &Opened) -> String {
    let session = &opened.session;
    let main = session.main_agent();
    let messages = main
        .map(|agent| agent.conversation.messages())
        .unwrap_or_default();
    let model = main.and_then(SavedAgent::model);
    let mut lines = vec![
        format!("session: {}", opened.dir.path().display()),
        format!("name: {}", opened.shown().as_deref().unwrap_or("(unnamed)")),
        format!("model: {}", model.as_deref().unwrap_or("(unknown)")),
        format!("messages: {}", messages.len()),
    ];
    for (at, message) in messages.iter().enumerate() {
        let tools = tool_names(message);
        let tools = if tools.is_empty() {
            String::new()
        } else {
            format!("  {}", tools.join(", "))
        };
        let (role, size) = (role_of(message), size_of(message));
        lines.push(format!("{at:>4}  {role:<11} {size:>8} chars{tools}"));
    }
    let subagents: Vec<&SavedAgent> = session
        .agents
        .iter()
        .filter(|agent| agent.parent.is_some())
        .collect();
    if !subagents.is_empty() {
        lines.push(format!("subagents: {}", subagents.len()));
    }
    for agent in subagents {
        let count = agent.conversation.messages().len();
        let gone = if agent.retired { ", retired" } else { "" };
        lines.push(format!("  {}: {count} messages{gone}", agent_name(agent)));
    }
    lines.join("\n")
}

/// What `message` is: `user`, `assistant`, `system`, or `tool result` for
/// a user message of tool results alone.
fn role_of(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::Assistant(_) => "assistant",
        Message::User { content }
            if !content.is_empty()
                && content
                    .iter()
                    .all(|part| matches!(part, UserContent::ToolResult(_))) =>
        {
            "tool result"
        }
        Message::User { .. } => "user",
    }
}

/// The tools `message` calls, or whose results it holds.
fn tool_names(message: &Message) -> Vec<String> {
    match message {
        Message::Assistant(turn) => turn
            .tool_calls()
            .map(|call| call.function.name.as_str().to_owned())
            .collect(),
        Message::User { content } => content
            .iter()
            .filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result.name.as_str().to_owned()),
                _ => None,
            })
            .collect(),
        Message::System { .. } => Vec::new(),
    }
}

/// The characters of what `message` says: its text, its tool calls' names
/// and arguments, its tool results; media count none.
fn size_of(message: &Message) -> usize {
    let chars = |text: &str| text.chars().count();
    match message {
        Message::System { content } => chars(content),
        Message::User { content } => content
            .iter()
            .map(|part| match part {
                UserContent::Text(text) => chars(&text.text),
                UserContent::ToolResult(result) => result_size(&result.content),
                _ => 0,
            })
            .sum(),
        Message::Assistant(turn) => turn
            .content
            .iter()
            .map(|part| match part {
                AssistantContent::Text(text) => chars(&text.text),
                AssistantContent::Reasoning(reasoning) => chars(&reasoning.text),
                AssistantContent::ToolCall(call) => {
                    let arguments = Value::Object(call.function.arguments.clone());
                    chars(call.function.name.as_str()) + chars(&arguments.to_string())
                }
                _ => 0,
            })
            .sum(),
    }
}

/// The characters of a tool result's text and JSON.
fn result_size(content: &[ToolResultContent]) -> usize {
    content
        .iter()
        .map(|item| match item {
            ToolResultContent::Text(text) => text.text.chars().count(),
            ToolResultContent::Json { value } => value.to_string().chars().count(),
            ToolResultContent::Image(_) => 0,
        })
        .sum()
}

/// What a shrink of every tool result over `over` characters will do to
/// each agent's conversation as it is now.
fn shrink_report(session: &Session, over: usize) -> Vec<String> {
    let mut report = Vec::new();
    for agent in &session.agents {
        let (results, lines) = agent
            .conversation
            .messages()
            .iter()
            .map(|message| shrink(&mut Message::clone(message), over))
            .fold((0, 0), |(results, lines), (r, l)| (results + r, lines + l));
        if results > 0 {
            let whose = match agent.parent {
                None => "the conversation".to_owned(),
                Some(_) => format!("subagent {}", agent_name(agent)),
            };
            report.push(format!(
                "shrank {results} tool results of {whose}, removing {lines} lines"
            ));
        }
    }
    if report.is_empty() {
        report.push(format!(
            "no tool result is over {over} characters and {} lines",
            2 * KEEP_LINES
        ));
    }
    report
}

/// The shrink as a rewrite of every message a session keeps, in every
/// record of every log.
fn shrinker(over: usize) -> impl Rewrite {
    Messages(move |message: &mut Message| shrink(message, over).0 > 0)
}

/// Replaces each text of each tool result of `message` over `over`
/// characters with its first and last [`KEEP_LINES`] lines and a marker;
/// the results shrunk and the lines removed.
fn shrink(message: &mut Message, over: usize) -> (usize, usize) {
    let (mut results, mut lines) = (0, 0);
    let Message::User { content } = message else {
        return (results, lines);
    };
    for part in content {
        let UserContent::ToolResult(result) = part else {
            continue;
        };
        if result_size(&result.content) <= over {
            continue;
        }
        let mut cut_any = false;
        for item in &mut result.content {
            if let ToolResultContent::Text(text) = item
                && let Some((short, removed)) = cut_lines(&text.text)
            {
                text.text = short;
                lines += removed;
                cut_any = true;
            }
        }
        results += usize::from(cut_any);
    }
    (results, lines)
}

/// `text` as its first and last [`KEEP_LINES`] lines around a marker of
/// how many were removed; `None` when it has no more lines than kept.
pub fn cut_lines(text: &str) -> Option<(String, usize)> {
    let lines: Vec<&str> = text.split('\n').collect();
    let removed = lines.len().checked_sub(2 * KEEP_LINES).filter(|n| *n > 0)?;
    let marker = format!("[{removed} lines removed by session-edit]");
    let head = lines.iter().take(KEEP_LINES).copied();
    let tail = lines.iter().skip(KEEP_LINES + removed).copied();
    let kept: Vec<&str> = head.chain([marker.as_str()]).chain(tail).collect();
    Some((kept.join("\n"), removed))
}

/// The ids of the calls `message` makes or answers.
fn call_ids(message: &Message) -> Vec<CallId> {
    match message {
        Message::Assistant(turn) => turn.tool_calls().map(|call| call.id.clone()).collect(),
        Message::User { content } => content
            .iter()
            .filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result.call.clone()),
                _ => None,
            })
            .collect(),
        Message::System { .. } => Vec::new(),
    }
}

/// `message` without the calls and results of `calls`, and what was taken
/// out of it; `None` when it has none of them.
fn strip(message: &Message, calls: &HashSet<CallId>) -> Option<(Message, Vec<String>)> {
    let mut taken = Vec::new();
    let stripped = match message {
        Message::Assistant(turn) => {
            let content = turn.content.iter().filter(|part| match part {
                AssistantContent::ToolCall(call) if calls.contains(&call.id) => {
                    let name = call.function.name.as_str();
                    taken.push(format!("tool call {name} ({})", call.id.wire()));
                    false
                }
                _ => true,
            });
            let content = content.cloned().collect();
            Message::Assistant(turn.clone().with_content(content))
        }
        Message::User { content } => {
            let content = content.iter().filter(|part| match part {
                UserContent::ToolResult(result) if calls.contains(&result.call) => {
                    let name = result.name.as_str();
                    taken.push(format!("result of {name} ({})", result.call.wire()));
                    false
                }
                _ => true,
            });
            Message::User {
                content: content.cloned().collect(),
            }
        }
        Message::System { .. } => return None,
    };
    (!taken.is_empty()).then_some((stripped, taken))
}

/// Whether `message` says nothing.
fn is_empty(message: &Message) -> bool {
    match message {
        Message::Assistant(turn) => turn.content.is_empty(),
        Message::User { content } => content.is_empty(),
        Message::System { content } => content.is_empty(),
    }
}

/// Removes messages `from..=to` of `conversation`, and every call whose
/// result they held and every result of a call they held, wherever it
/// is, as one change ([`Conversation::set_entries`], so the entries kept
/// keep their ids); what it removed. Refused, changing nothing, when the
/// range is outside the conversation or the rest would not pair.
fn drop_range(
    conversation: &mut Conversation,
    from: usize,
    to: usize,
) -> Result<Vec<String>, Error> {
    let messages = conversation.messages();
    let len = messages.len();
    if from > to || to >= len {
        return Err(Error(format!(
            "{from}..{to} is not a range of the conversation's {len} messages (0..{})",
            len.saturating_sub(1)
        )));
    }
    let dropped = messages.get(from..=to).unwrap_or_default();
    let calls: HashSet<CallId> = dropped
        .iter()
        .flat_map(|message| call_ids(message))
        .collect();
    let mut report = Vec::new();
    let mut kept = Vec::with_capacity(len);
    for (at, entry) in (0..len).filter_map(|at| Some((at, conversation.entry(at)?))) {
        if (from..=to).contains(&at) {
            let tools = tool_names(&entry.message).join(", ");
            let tools = if tools.is_empty() {
                tools
            } else {
                format!(" of {tools}")
            };
            let role = role_of(&entry.message);
            report.push(format!("removed message {at} ({role}{tools})"));
        } else if let Some((stripped, taken)) = strip(&entry.message, &calls) {
            report.extend(
                taken
                    .iter()
                    .map(|what| format!("removed {what} from message {at}")),
            );
            if is_empty(&stripped) {
                report.push(format!("removed message {at}, which held nothing else"));
            } else {
                kept.push(Entry {
                    message: stripped.into(),
                    ..entry
                });
            }
        } else {
            kept.push(entry);
        }
    }
    let rest: Vec<&Message> = kept.iter().map(|entry| &*entry.message).collect();
    pairing::validate(&rest).map_err(|why| {
        format!(
            "dropping {from}..{to} would leave the conversation broken ({why}); drop a wider range"
        )
    })?;
    conversation.set_entries(kept, Author::plugin(AUTHOR));
    Ok(report)
}

/// Saves `edit` of `opened` to `target`; what it did, then where it
/// went. Nothing is written when the edit is refused.
fn save(home: &Home, opened: Opened, target: Target, edit: &Edit) -> Result<Vec<String>, Error> {
    let shown = opened.shown();
    let Opened {
        dir,
        store,
        mut session,
        meta,
    } = opened;
    // Done on the session as read first, so a refused edit writes nothing.
    let mut report = match edit {
        Edit::Shrink(over) => shrink_report(&session, *over),
        Edit::Drop(from, to) => drop_range(main_of(&mut session)?, *from, *to)?,
        Edit::Rename(name) => vec![format!("named the session {name}")],
    };
    match target {
        Target::InPlace => {
            if dir.runs_elsewhere() {
                return Err(
                    "rig is running this session; quit it before editing it in place".into(),
                );
            }
            let backup = backup(dir.path())?;
            match edit {
                Edit::Shrink(over) => {
                    rewrite(&store, Author::plugin(AUTHOR), &shrinker(*over))?;
                }
                Edit::Drop(..) => session.save(&store)?,
                Edit::Rename(name) => dir.set_name(Some(name))?,
            }
            write_meta(&store, meta)?;
            report.push(format!(
                "saved in place; the session as it was is in {}",
                backup.display()
            ));
        }
        Target::As(name) => {
            let name = match edit {
                Edit::Rename(renamed) => renamed.clone(),
                _ => name.unwrap_or_else(|| {
                    let base = shown.or_else(|| dir.id().map(|id| id.to_string()));
                    format!("{} (edited)", base.unwrap_or_else(|| "session".to_owned()))
                }),
            };
            let (id, new) = home.new_session()?;
            let made = (|| {
                let to = DirStore::open(new.path())?;
                copy_store(&store, &to)?;
                if let Edit::Shrink(over) = edit {
                    rewrite(&to, Author::plugin(AUTHOR), &shrinker(*over))?;
                }
                if let Edit::Drop(from, until) = edit {
                    let mut copy = Session::read(&to)?;
                    drop_range(main_of(&mut copy)?, *from, *until)?;
                    copy.save(&to)?;
                }
                if let Ok(directory) = fs::read(dir.directory()) {
                    fs::write(new.directory(), directory)?;
                }
                write_meta(&to, meta)?;
                new.set_name(Some(&name))?;
                new.mark_saved()?;
                Ok::<_, Error>(())
            })();
            if let Err(why) = made {
                fs::remove_dir_all(new.path()).ok();
                return Err(why);
            }
            report.push(format!(
                "saved as session {id} ({name}); resume it with: rig --resume {id}"
            ));
        }
    }
    Ok(report)
}

/// Copies every log and file (blobs among them) of `from` to `to`.
fn copy_store(from: &dyn Store, to: &dyn Store) -> io::Result<()> {
    for log in from.logs()? {
        to.replace_log(&log, 0, &from.read_log(&log)?)?;
    }
    for file in from.files()? {
        to.write_file(&file, &from.read_file(&file)?)?;
    }
    Ok(())
}

/// Keeps `meta` as the session's listing, updated now.
fn write_meta(store: &DirStore, mut meta: SessionMeta) -> Result<(), Error> {
    meta.updated = now_ms();
    meta.write(store)?;
    Ok(())
}

/// Copies the directory `dir` to `<dir>.bak`, replacing an older copy;
/// the copy's path.
fn backup(dir: &Path) -> Result<PathBuf, Error> {
    let mut name = dir.as_os_str().to_owned();
    name.push(".bak");
    let backup = PathBuf::from(name);
    if backup.exists() {
        fs::remove_dir_all(&backup)?;
    }
    copy_dir(dir, &backup).map_err(|why| {
        format!(
            "cannot copy {} to {}: {why}",
            dir.display(),
            backup.display()
        )
    })?;
    Ok(backup)
}

/// Copies the directory `from`, and everything in it, to `to`.
fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// The session as Markdown: a heading per message, tool calls as fenced
/// JSON and tool results as fenced text; the main conversation, then each
/// subagent's.
fn markdown_of(opened: &Opened) -> String {
    let session = &opened.session;
    let mut text = String::new();
    let name = opened.shown().unwrap_or_else(|| "Session".to_owned());
    let main = session.main_agent();
    text.push_str(&format!("# {name}\n\n"));
    if let Some(model) = main.and_then(SavedAgent::model) {
        text.push_str(&format!("Model: `{model}`\n\n"));
    }
    if let Some(main) = main {
        conversation_markdown(&mut text, main.conversation.messages());
    }
    for agent in session.agents.iter().filter(|agent| agent.parent.is_some()) {
        text.push_str(&format!("# Subagent: {}\n\n", agent_name(agent)));
        conversation_markdown(&mut text, agent.conversation.messages());
    }
    text
}

/// Appends `messages` to `text`, a heading each.
fn conversation_markdown(text: &mut String, messages: &[Arc<Message>]) {
    for (at, message) in messages.iter().map(Arc::as_ref).enumerate() {
        let role = role_of(message);
        let mut role_title = role.to_owned();
        if let Some(first) = role_title.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        text.push_str(&format!("## {at} · {role_title}\n\n"));
        match message {
            Message::System { content } => paragraph(text, content),
            Message::User { content } => {
                for part in content {
                    match part {
                        UserContent::Text(said) => paragraph(text, &said.text),
                        UserContent::ToolResult(result) => {
                            let error = if result.is_error { " (failed)" } else { "" };
                            paragraph(
                                text,
                                &format!("Result of `{}`{error}:", result.name.as_str()),
                            );
                            let body: Vec<String> = result
                                .content
                                .iter()
                                .map(|item| match item {
                                    ToolResultContent::Text(said) => said.text.clone(),
                                    ToolResultContent::Json { value } => {
                                        serde_json::to_string_pretty(value).unwrap_or_default()
                                    }
                                    ToolResultContent::Image(_) => "(image)".to_owned(),
                                })
                                .collect();
                            fenced(text, "text", &body.join("\n"));
                        }
                        _ => paragraph(text, "*(media)*"),
                    }
                }
            }
            Message::Assistant(turn) => {
                for part in &turn.content {
                    match part {
                        AssistantContent::Text(said) => paragraph(text, &said.text),
                        AssistantContent::Reasoning(reasoning) => {
                            paragraph(text, &format!("*Reasoning:* {}", reasoning.text));
                        }
                        AssistantContent::ToolCall(call) => {
                            paragraph(
                                text,
                                &format!("Tool call `{}`:", call.function.name.as_str()),
                            );
                            let arguments = Value::Object(call.function.arguments.clone());
                            let json = serde_json::to_string_pretty(&arguments).unwrap_or_default();
                            fenced(text, "json", &json);
                        }
                        _ => paragraph(text, "*(provider item)*"),
                    }
                }
            }
        }
    }
}

fn paragraph(text: &mut String, said: &str) {
    text.push_str(said);
    text.push_str("\n\n");
}

/// Appends `body` fenced as `language`, with a fence longer than any run
/// of backticks in it.
fn fenced(text: &mut String, language: &str, body: &str) {
    let longest = body
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or_default();
    let fence = "`".repeat(longest.max(2) + 1);
    text.push_str(&format!("{fence}{language}\n{body}\n{fence}\n\n"));
}

#[cfg(test)]
mod tests;
