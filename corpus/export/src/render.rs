//! A saved session as Markdown, HTML or JSON. Markdown and HTML walk the
//! same tree, each agent a section, a subagent's nested under the `task`
//! call that spawned it; JSON is each agent's messages as rig-core
//! serializes them.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::conversation::short_id;
use rig_core::message::{DocumentData, DocumentSourceKind, Image, MimeType, ToolResultContent};
use rig_harness::prelude::*;
use rig_harness::rig_memory::summary_author;
use serde_json::{Value, json};

use super::{Export, ExportedAgent, ModelCall};

/// The `format` field of an export's JSON. Its shape:
///
/// ```json
/// { "format": "rig-export/2",
///   "session": { "id": "…", "name": "…", "started_ms": 1760000000000,
///                "left_out": ["a log that could not be read (why)"] },
///   "models": ["vendor/model"],
///   "agents": [{
///     "id": "…", "parent": "…" | null,
///     "task_call": "the parent's task call id" | null, "title": "…" | null,
///     "model": "vendor/model" | null, "started_ms": 0, "retired": false,
///     "usage": rig-core's UsageTotals (tokens, cost) of `calls`,
///     "calls": [{ "effect": 7, "purpose": "turn" | null, "turn": 0 | null,
///                 "model": "provider/model",
///                 "usage": rig-core Usage (tokens, cost), "response_id": "…" | null,
///                 "disposition": "ran" | "superseded" | … }],
///     "messages": [rig-core Message, …],
///     "authors": [{ "message": 0, "author": Author }] }] }
/// ```
pub const JSON_FORMAT: &str = "rig-export/2";

/// Most characters of a tool result shown in Markdown and HTML.
const RESULT_CHARS: usize = 2000;

/// What an export is written as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Html,
    Json,
}

impl Format {
    pub(crate) fn parse(word: &str) -> Option<Self> {
        match word {
            "md" | "markdown" => Some(Self::Markdown),
            "html" => Some(Self::Html),
            "json" => Some(Self::Json),
            _ => None,
        }
    }

    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Html => "html",
            Self::Json => "json",
        }
    }
}

/// The files of `session` exported as `format` to `path`: the export, and
/// for Markdown the images it links to.
pub(crate) fn render(
    session: &Export,
    format: Format,
    thinking: bool,
    path: &Path,
) -> Vec<(PathBuf, Vec<u8>)> {
    match format {
        Format::Json => {
            let text = serde_json::to_string_pretty(&to_json(session)).unwrap_or_default();
            vec![(path.to_owned(), text.into_bytes())]
        }
        Format::Markdown => {
            let mut sink = Markdown {
                out: String::new(),
                open: Vec::new(),
                thinking,
                images: Vec::new(),
                dir: PathBuf::from(format!("{}.files", path.display())),
            };
            walk_session(&mut sink, session);
            let mut files = vec![(path.to_owned(), sink.out.into_bytes())];
            files.extend(sink.images);
            files
        }
        Format::Html => {
            let mut sink = Html(String::new());
            walk_session(&mut sink, session);
            let page = format!(
                "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{}</title>\
                 <style>{STYLE}</style></head><body>\n{}</body></html>\n",
                escape(&session.name),
                sink.0
            );
            vec![(path.to_owned(), page.into_bytes())]
        }
    }
}

const STYLE: &str = "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;\
padding:0 1rem;line-height:1.5}section{border-left:3px solid #ccc;padding-left:1rem;margin:1rem 0}\
pre{background:#f4f4f4;padding:.5rem;overflow-x:auto;white-space:pre-wrap}\
.summary{background:#fff8dc;padding:.5rem}img{max-width:100%}";

/// What Markdown and HTML write, in the order the session is walked.
trait Sink {
    fn header(&mut self, title: &str, rows: &[(&str, String)]);
    fn open(&mut self, depth: usize, title: &str);
    fn close(&mut self);
    fn text(&mut self, who: &str, text: &str);
    fn reasoning(&mut self, text: &str);
    fn summary(&mut self, text: &str);
    fn call(&mut self, name: &str, id: &str, args: &str);
    fn result(&mut self, name: &str, text: &str, error: bool);
    fn image(&mut self, image: &Image);
}

fn walk_session(sink: &mut dyn Sink, session: &Export) {
    let mut rows = vec![("Models", models(session).join(", "))];
    if let Some(id) = &session.id {
        rows.insert(0, ("Session", id.clone()));
    }
    if let Some(started) = session.started_ms {
        rows.push(("Started", utc(started / 1000)));
    }
    if !session.left_out.is_empty() {
        rows.push(("Left out", session.left_out.join("; ")));
    }
    sink.header(&session.name, &rows);
    let mut walked = HashSet::new();
    let roots = session.agents.iter().filter(|agent| {
        agent
            .saved
            .parent
            .as_ref()
            .is_none_or(|parent| !session.agents.iter().any(|other| &other.saved.id == parent))
    });
    for root in roots {
        walk(sink, session, root, 0, &mut walked);
    }
}

/// `agent`'s section, its subagents' nested in it.
fn walk<'s>(
    sink: &mut dyn Sink,
    session: &'s Export,
    agent: &'s ExportedAgent,
    depth: usize,
    walked: &mut HashSet<&'s str>,
) {
    if !walked.insert(agent.saved.id.as_str()) {
        return;
    }
    let mut title = match (&agent.saved.parent, spawned_by(agent)) {
        (Some(_), Some((_, Some(task)))) => {
            format!("Subagent {} “{task}”", short_id(&agent.saved.id))
        }
        (Some(_), _) => format!("Subagent {}", short_id(&agent.saved.id)),
        (None, _) => format!("Agent {}", short_id(&agent.saved.id)),
    };
    if let Some(model) = &agent.model {
        title.push_str(&format!(" ({model})"));
    }
    sink.open(depth, &title);
    let children: Vec<&ExportedAgent> = session
        .agents
        .iter()
        .filter(|child| child.saved.parent.as_ref() == Some(&agent.saved.id))
        .collect();
    let conversation = &agent.saved.conversation;
    for (at, message) in conversation.messages().iter().enumerate() {
        match &**message {
            message::Message::User { content } => {
                let author = conversation.author(at);
                for content in content {
                    user_content(sink, content, author);
                    // A subagent follows the result of the call that
                    // spawned it.
                    if let UserContent::ToolResult(result) = content {
                        let id = result.call.to_string();
                        let spawned = children
                            .iter()
                            .filter(|child| spawned_by(child).is_some_and(|(call, _)| call == id));
                        for child in spawned {
                            walk(sink, session, child, depth + 1, walked);
                        }
                    }
                }
            }
            message::Message::Assistant(assistant) => {
                for content in &assistant.content {
                    match content {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            sink.text("Assistant", &text.text);
                        }
                        AssistantContent::Reasoning(reasoning) if !reasoning.redacted => {
                            sink.reasoning(&reasoning.text);
                        }
                        AssistantContent::ToolCall(call) => {
                            let id = call.id.to_string();
                            let args = serde_json::to_string_pretty(&call.function.arguments);
                            sink.call(call.function.name.as_str(), &id, &args.unwrap_or_default());
                        }
                        AssistantContent::Image(image) => sink.image(image),
                        _ => {}
                    }
                }
            }
            message::Message::System { content } => sink.text("System", content),
        }
    }
    // Subagents no result of this conversation names, such as those of a
    // call still running.
    for child in children {
        walk(sink, session, child, depth + 1, walked);
    }
    sink.close();
}

fn user_content(sink: &mut dyn Sink, content: &UserContent, author: Option<&Author>) {
    let summary = author.is_some_and(|author| *author == summary_author());
    match content {
        // A compaction's summary, which stands for the messages it replaced.
        UserContent::Text(text) if summary => sink.summary(&text.text),
        UserContent::Text(text) => sink.text(&sender(author), &text.text),
        UserContent::ToolResult(result) => {
            let mut text = String::new();
            for part in &result.content {
                match part {
                    ToolResultContent::Text(part) => text.push_str(&part.text),
                    ToolResultContent::Json { value } => text.push_str(&value.to_string()),
                    ToolResultContent::Image(image) => sink.image(image),
                }
            }
            let text = rig_tools::shorten(&text, RESULT_CHARS);
            sink.result(result.name.as_str(), &text, result.is_error);
        }
        UserContent::Image(image) => sink.image(image),
        UserContent::Document(document) => match &document.data {
            DocumentData::Text(text) => sink.text("Attached document", text),
            DocumentData::File(source) => sink.text("Attached document", &source.to_string()),
        },
        UserContent::Audio(_) => sink.text(&sender(author), "[audio]"),
        UserContent::Video(_) => sink.text(&sender(author), "[video]"),
    }
}

/// Who a user message is from.
fn sender(author: Option<&Author>) -> String {
    let Some(author) = author else {
        return "User".to_owned();
    };
    let from = author.from.as_deref().map(|from| short_id(from).to_owned());
    match (&author.kind, from) {
        (AuthorKind::User, _) => "User".to_owned(),
        (AuthorKind::Agent, Some(from)) => format!("Agent {from}"),
        (AuthorKind::Agent, None) => "An agent".to_owned(),
        (AuthorKind::Plugin(name), _) => format!("Plugin {name}"),
        (AuthorKind::Rig, _) => "Rig".to_owned(),
    }
}

/// The call that spawned `agent` and its task's title: the request and
/// title of its brief, among the user messages it starts with, when its
/// parent sent it.
fn spawned_by(agent: &ExportedAgent) -> Option<(String, Option<String>)> {
    let conversation = &agent.saved.conversation;
    let first = conversation
        .entries()
        .take_while(|entry| matches!(entry.message, message::Message::User { .. }));
    first.filter_map(|entry| entry.author).find_map(|author| {
        let from_parent = author.from.is_some() && author.from == agent.saved.parent;
        let request = author.request.as_ref().filter(|_| from_parent)?;
        Some((request.0.clone(), author.title.clone()))
    })
}

/// Every model the session's agents use or answered with.
fn models(session: &Export) -> Vec<String> {
    let mut models: Vec<String> = Vec::new();
    for agent in &session.agents {
        let answered = agent
            .saved
            .conversation
            .messages()
            .iter()
            .filter_map(|message| {
                let message::Message::Assistant(assistant) = &**message else {
                    return None;
                };
                let origin = assistant.origin.as_ref()?;
                Some(format!("{}/{}", origin.provider, origin.model))
            });
        for model in agent.model.iter().cloned().chain(answered) {
            if !models.contains(&model) {
                models.push(model);
            }
        }
    }
    models
}

fn to_json(session: &Export) -> Value {
    let agents: Vec<Value> = session
        .agents
        .iter()
        .map(|agent| {
            let spawned = spawned_by(agent);
            let messages = agent.saved.conversation.messages();
            let authors: Vec<Value> = agent
                .saved
                .conversation
                .entries()
                .filter_map(|entry| {
                    let author = entry.author?;
                    Some(json!({ "message": entry.index, "author": author }))
                })
                .collect();
            json!({
                "id": agent.saved.id,
                "parent": agent.saved.parent,
                "task_call": spawned.as_ref().map(|(call, _)| call),
                "title": spawned.and_then(|(_, title)| title),
                "model": agent.model,
                "started_ms": agent.saved.started_ms,
                "retired": agent.saved.retired,
                "usage": agent.totals(),
                "calls": agent.calls.iter().map(call_json).collect::<Vec<_>>(),
                "messages": messages,
                "authors": authors,
            })
        })
        .collect();
    json!({
        "format": JSON_FORMAT,
        "session": {
            "id": session.id, "name": session.name, "started_ms": session.started_ms,
            "left_out": session.left_out,
        },
        "models": models(session),
        "agents": agents,
    })
}

fn call_json(call: &ModelCall) -> Value {
    json!({
        "effect": call.effect,
        "purpose": call.purpose,
        "turn": call.turn,
        "model": call.model,
        "usage": call.usage,
        "response_id": call.response_id,
        "disposition": call.disposition,
    })
}

/// `seconds` since the Unix epoch as a UTC date and time.
fn utc(seconds: u64) -> String {
    let days = seconds / 86_400;
    let rest = seconds % 86_400;
    // Howard Hinnant's civil-from-days, for days since 1970-01-01.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
        rest / 3600,
        rest % 3600 / 60
    )
}

/// An image's MIME type, PNG when unknown.
fn mime(image: &Image) -> &'static str {
    image
        .media_type
        .as_ref()
        .map_or("image/png", MimeType::to_mime_type)
}

struct Markdown {
    out: String,
    /// The titles of the sections open.
    open: Vec<String>,
    thinking: bool,
    /// The images written beside the export, which it links to.
    images: Vec<(PathBuf, Vec<u8>)>,
    dir: PathBuf,
}

/// A Markdown heading's hashes for a section `depth` deep.
fn hashes(depth: usize) -> String {
    "#".repeat((depth + 2).min(6))
}

/// A fence longer than any backtick run in `text`.
fn fence(text: &str) -> String {
    let longest = text
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or_default();
    "`".repeat(longest.max(2) + 1)
}

impl Sink for Markdown {
    fn header(&mut self, title: &str, rows: &[(&str, String)]) {
        let _ = writeln!(self.out, "# {title}\n");
        for (key, value) in rows {
            let _ = writeln!(self.out, "- **{key}:** {value}");
        }
        self.out.push('\n');
    }

    fn open(&mut self, depth: usize, title: &str) {
        let _ = writeln!(self.out, "{} {title}\n", hashes(depth));
        self.open.push(title.to_owned());
    }

    /// Markdown has no end of a section: a nested one says where it ends.
    fn close(&mut self) {
        let closed = self.open.pop();
        if let (Some(closed), false) = (closed, self.open.is_empty()) {
            let _ = writeln!(self.out, "*End of {closed}.*\n\n---\n");
        }
    }

    fn text(&mut self, who: &str, text: &str) {
        let _ = writeln!(self.out, "**{who}:**\n\n{text}\n");
    }

    fn reasoning(&mut self, text: &str) {
        if self.thinking {
            let quoted: Vec<String> = text.lines().map(|line| format!("> {line}")).collect();
            let _ = writeln!(self.out, "*Thinking:*\n\n{}\n", quoted.join("\n"));
        }
    }

    fn summary(&mut self, text: &str) {
        let _ = writeln!(
            self.out,
            "**Compaction summary** (what the model sees in place of the earlier messages):\n\n{text}\n"
        );
    }

    fn call(&mut self, name: &str, id: &str, args: &str) {
        let fence = fence(args);
        let _ = writeln!(
            self.out,
            "**Tool call `{name}`** ({id}):\n\n{fence}json\n{args}\n{fence}\n"
        );
    }

    fn result(&mut self, name: &str, text: &str, error: bool) {
        let what = if error { "failed" } else { "result" };
        let fence = fence(text);
        let _ = writeln!(self.out, "*`{name}` {what}:*\n\n{fence}\n{text}\n{fence}\n");
    }

    fn image(&mut self, image: &Image) {
        let link = match &image.data {
            DocumentSourceKind::Url(url) => Some(url.clone()),
            DocumentSourceKind::Base64(data) => STANDARD
                .decode(data)
                .ok()
                .map(|bytes| self.keep(image, bytes)),
            DocumentSourceKind::Raw(bytes) => Some(self.keep(image, bytes.clone())),
            DocumentSourceKind::FileId(_) | DocumentSourceKind::Unknown => None,
        };
        let _ = match link {
            Some(link) => writeln!(self.out, "![image]({link})\n"),
            None => writeln!(self.out, "*[an image not kept]*\n"),
        };
    }
}

impl Markdown {
    /// Writes `bytes` beside the export; the relative link to them.
    fn keep(&mut self, image: &Image, bytes: Vec<u8>) -> String {
        let extension = mime(image).trim_start_matches("image/");
        let name = format!("image-{}.{extension}", self.images.len() + 1);
        let path = self.dir.join(&name);
        self.images.push((path, bytes));
        let dir = self
            .dir
            .file_name()
            .map(|dir| dir.to_string_lossy().into_owned());
        format!("{}/{name}", dir.unwrap_or_default())
    }
}

struct Html(String);

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Sink for Html {
    fn header(&mut self, title: &str, rows: &[(&str, String)]) {
        let _ = writeln!(self.0, "<h1>{}</h1>\n<ul>", escape(title));
        for (key, value) in rows {
            let _ = writeln!(self.0, "<li><b>{key}:</b> {}</li>", escape(value));
        }
        self.0.push_str("</ul>\n");
    }

    fn open(&mut self, depth: usize, title: &str) {
        let level = (depth + 2).min(6);
        let _ = writeln!(self.0, "<section><h{level}>{}</h{level}>", escape(title));
    }

    fn close(&mut self) {
        self.0.push_str("</section>\n");
    }

    fn text(&mut self, who: &str, text: &str) {
        let _ = writeln!(
            self.0,
            "<p><b>{}:</b></p><pre>{}</pre>",
            escape(who),
            escape(text)
        );
    }

    fn reasoning(&mut self, text: &str) {
        let _ = writeln!(
            self.0,
            "<details><summary>Thinking</summary><pre>{}</pre></details>",
            escape(text)
        );
    }

    fn summary(&mut self, text: &str) {
        let _ = writeln!(
            self.0,
            "<div class=\"summary\"><b>Compaction summary</b> (what the model sees in place of \
             the earlier messages)<pre>{}</pre></div>",
            escape(text)
        );
    }

    fn call(&mut self, name: &str, id: &str, args: &str) {
        let _ = writeln!(
            self.0,
            "<p><b>Tool call <code>{}</code></b> ({})</p><pre>{}</pre>",
            escape(name),
            escape(id),
            escape(args)
        );
    }

    fn result(&mut self, name: &str, text: &str, error: bool) {
        let what = if error { "failed" } else { "result" };
        let _ = writeln!(
            self.0,
            "<details><summary><code>{}</code> {what}</summary><pre>{}</pre></details>",
            escape(name),
            escape(text)
        );
    }

    fn image(&mut self, image: &Image) {
        let source = match &image.data {
            DocumentSourceKind::Url(url) => Some(url.clone()),
            DocumentSourceKind::Base64(data) => Some(format!("data:{};base64,{data}", mime(image))),
            DocumentSourceKind::Raw(bytes) => Some(format!(
                "data:{};base64,{}",
                mime(image),
                STANDARD.encode(bytes)
            )),
            DocumentSourceKind::FileId(_) | DocumentSourceKind::Unknown => None,
        };
        let _ = match source {
            Some(source) => writeln!(
                self.0,
                "<p><img src=\"{}\" alt=\"image\"></p>",
                escape(&source)
            ),
            None => writeln!(self.0, "<p><i>[an image not kept]</i></p>"),
        };
    }
}
