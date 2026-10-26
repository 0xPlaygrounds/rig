//! Context inspector (spec `57-context-inspector.md`): `/why` shows what an
//! agent's last model request held, part by part, and how it differed
//! from what the agent's state alone would have sent.
//!
//! A turn's model call is an effect entity whose [`ModelRequest`] the
//! kernel builds in Admit, its system prompt in parts and its messages as
//! a view of the agent's conversation beside it; Prepare systems then edit
//! it, each message edit recorded with its author, and Start renders the
//! prompt in front of the messages and sends it. The kernel's own prompt
//! and tools are kept as the call's [`Built`] when the `ModelRequest` is
//! inserted, before any Prepare system ran. After Prepare, the prompt in
//! parts as Start will send it ([`ModelRequest::prompt`]), who wrote each
//! message and the attributed edits of the messages are kept as the
//! call's [`Prepared`]. When the call is `Started`, the request as sent
//! goes to the agent's [`LastRequest`] with the model it went to after
//! every adapter ([`SentTo`]) and, for each prompt section, the plugin that
//! spawned it ([`ProvidedBy`]).
//!
//! `/why [full] [agent]` (or `/context --last`) then shows the request's
//! estimated tokens, a bar of the model's input limit by part, the system
//! prompt by section and plugin, the tools with their schema sizes, the
//! messages by kind, the options, and how the sent request differs from
//! the built one. `/why full` writes the request as JSON in the session's
//! directory. Nothing calls a model.

use std::fmt::Write as _;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{
    self, CompletionRequest, Section, ToolDefinition, tokens_label,
};
use rig_memory::{HeuristicTokenCounter, TokenCounter, summary_author};

use message::Message;

/// The width of the token bar, in characters.
const BAR: u64 = 40;

/// Said of a change to the prompt or tools: the kernel records no author
/// for those.
const UNNAMED: &str = ", by a Prepare system the kernel does not name";

/// Adds `/why`, `/context --last`, and what they read.
#[derive(Default)]
pub struct ContextInspectorPlugin;

/// The plugin's stage systems.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextInspectorSystems {
    /// After Prepare: a turn's prompt in parts and messages, as Start will
    /// send them, and who edited the messages.
    Prepared,
    /// `/context --last` taken over as `/why`, before the harness runs the
    /// line as `/context`.
    Alias,
}

impl Plugin for ContextInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            AgentPass,
            (
                ContextInspectorSystems::Prepared
                    .in_set(Stage::Prepare)
                    .after(PrepareSystems::Fit),
                ContextInspectorSystems::Alias
                    .in_set(Stage::Gate)
                    .before(CommandSystems),
            ),
        )
        .add_systems(
            AgentPass,
            (
                keep_prepared.in_set(ContextInspectorSystems::Prepared),
                alias.in_set(ContextInspectorSystems::Alias),
            ),
        )
        .add_observer(keep_built)
        .add_observer(keep_sent)
        .add_command(
            "why",
            "Show what the last model request held: `/why [full] [agent]`",
            why,
        );
    }
}

/// On a turn's model call: its prompt and tools as the kernel built them
/// from the agent, before Prepare systems changed them.
#[derive(Component, Clone, Debug)]
pub struct Built {
    /// The system prompt in parts.
    pub prompt: Option<completion::SystemPrompt>,
    /// The tools offered.
    pub tools: Vec<Arc<ToolDefinition>>,
}

/// On a turn's model call, after Prepare: what Start sends, and who
/// changed its messages.
#[derive(Component, Clone, Debug)]
pub struct Prepared {
    /// The system prompt in parts as Start sends it.
    pub prompt: completion::SystemPrompt,
    /// Who wrote each message, in order.
    pub authors: Vec<Option<Author>>,
    /// The edits Prepare systems made to the messages, each with its
    /// author.
    pub edits: Vec<ChangeRecord>,
}

/// An agent's last turn request, as it was sent.
#[derive(Component, Clone)]
pub struct LastRequest {
    /// The model, `vendor/model`.
    pub model: String,
    /// The model's input limit, when the catalog knows it.
    pub limit: Option<u32>,
    /// The request as sent.
    pub sent: Arc<CompletionRequest>,
    /// The prompt and tools as the kernel built them, if it built it.
    pub built: Option<Built>,
    /// What Prepare left, if the kernel built it.
    pub prepared: Option<Prepared>,
    /// Each part of the system prompt sent, in order.
    pub notes: Vec<SectionNote>,
}

/// A part of a system prompt as sent, and where it came from.
#[derive(Clone, Debug)]
pub struct SectionNote {
    /// The section's tag.
    pub tag: String,
    /// The name of the entity that holds it, if any.
    pub name: Option<String>,
    /// Its estimated tokens, as sent.
    pub tokens: u64,
    /// Where it came from, such as `from SomePlugin`.
    pub from: String,
}

/// `text` between `<tag>` and `</tag>`, as rig-core renders a section in a
/// system prompt; empty when `text` is blank.
fn tagged(tag: &str, text: &str) -> String {
    let prompt = completion::SystemPrompt {
        role: String::new(),
        sections: vec![Section::new(tag, text)],
    };
    prompt.text()
}

/// Keeps the prompt and tools of each turn's call as they are first put
/// on the call.
fn keep_built(
    inserted: On<Insert<ModelRequest>>,
    calls: Query<&ModelRequest, (With<TurnCall>, Without<Built>)>,
    mut commands: Commands,
) {
    if let Ok(call) = calls.get(inserted.entity) {
        let built = Built {
            prompt: call.prompt.clone(),
            tools: call.request.tools.clone(),
        };
        commands.entity(inserted.entity).try_insert_if_new(built);
    }
}

/// After Prepare, keeps what each admitted turn call will send and who
/// edited its messages, before Start takes its prompt and history.
fn keep_prepared(
    calls: Query<(Entity, &ModelRequest), (Added<Admitted>, With<TurnCall>)>,
    mut commands: Commands,
) {
    for (call, model) in &calls {
        let (Some(prompt), Some(history)) = (model.prompt.clone(), &model.history) else {
            continue;
        };
        let prepared = Prepared {
            prompt,
            authors: history
                .entries()
                .map(|entry| entry.author.cloned())
                .collect(),
            edits: history.unsaved().to_vec(),
        };
        commands.entity(call).try_insert(prepared);
    }
}

/// Copies each turn's call, as it starts, to its agent's [`LastRequest`].
fn keep_sent(
    started: On<Add<Started>>,
    calls: Query<
        (
            &ModelRequest,
            &SentTo,
            Option<&Built>,
            Option<&Prepared>,
            &ForAgent,
        ),
        With<TurnCall>,
    >,
    sections: Query<(
        &PromptSection,
        Option<&SectionOf>,
        Option<&Name>,
        Option<&ProvidedBy>,
    )>,
    names: Query<&Name>,
    mut commands: Commands,
) {
    let Ok((call, SentTo(sent_to), built, prepared, &ForAgent(agent))) = calls.get(started.entity)
    else {
        return;
    };
    let note = |section: &Section| {
        let block = tagged(&section.tag, &section.text);
        let spawned = sections.iter().find(|(spawned, of, ..)| {
            spawned.tag == section.tag && of.is_none_or(|of| of.0 == agent)
        });
        let (name, from) = match spawned {
            None => (None, "added before sending".to_owned()),
            Some((spawned, of, name, by)) => {
                let plugin = by.and_then(|by| names.get(by.0).ok()).map(short);
                let mut from = match (plugin, of) {
                    (Some(plugin), _) => format!("from {plugin}"),
                    (None, Some(_)) => "this agent's own".to_owned(),
                    (None, None) => "from no known plugin".to_owned(),
                };
                if spawned.text.trim() != section.text.trim() {
                    from.push_str(", rewritten before sending");
                }
                (name.map(|name| name.as_str().to_owned()), from)
            }
        };
        (!block.is_empty()).then(|| SectionNote {
            tag: section.tag.clone(),
            name,
            tokens: tokens(&block),
            from,
        })
    };
    let notes = prepared.map_or_else(Vec::new, |prepared| {
        prepared.prompt.sections.iter().filter_map(note).collect()
    });
    let last = LastRequest {
        model: sent_to.spec.reference(),
        limit: sent_to.spec.input_limit(),
        sent: call.request.clone(),
        built: built.cloned(),
        prepared: prepared.cloned(),
        notes,
    };
    commands.entity(agent).try_insert(last);
}

/// `/context --last [full] [agent]` runs as `/why [full] [agent]`.
fn alias(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (ReadyToStart, IncomingMessages)>,
    mut commands: Commands,
) {
    for (typed, &ForAgent(agent), mut decision) in &mut typed {
        let Some(rest) = typed
            .command()
            .and_then(|line| line.strip_prefix("context --last"))
        else {
            continue;
        };
        if (rest.is_empty() || rest.starts_with(' '))
            && decision.decide(
                Decider::plugin("context-inspector"),
                Outcome::custom("command"),
            )
        {
            let mode = typed.mode;
            commands.trigger(RunCommand {
                mode,
                ..RunCommand::new(agent, format!("why{rest}"))
            });
        }
    }
}

/// `/why [full] [agent]`.
fn why(
    In(args): In<CommandArgs>,
    agents: Query<(Entity, &Name, &AgentId, Option<&LastRequest>, &LastUsage), With<Agent>>,
    tools: Query<(&ToolDef, Option<&ProvidedBy>)>,
    names: Query<&Name>,
    session: Option<Res<SessionPaths>>,
    mut notices: MessageWriter<Notice>,
) {
    let wanted = args.args.trim();
    let (full, wanted) = match wanted.strip_prefix("full") {
        Some(rest) if rest.is_empty() || rest.starts_with(' ') => (true, rest.trim()),
        _ => (false, wanted),
    };
    let found = agents.iter().find(|(entity, name, id, ..)| {
        if wanted.is_empty() {
            *entity == args.agent
        } else {
            name.as_str() == wanted || id.0 == wanted
        }
    });
    let Some((_, name, id, last, usage)) = found else {
        notices.write(Notice::error(args.agent, format!("No agent `{wanted}`.")));
        return;
    };
    let Some(last) = last else {
        let text = format!("{name} has sent no model request yet.");
        notices.write(Notice::info(args.agent, text));
        return;
    };
    if full {
        notices.write(match write_full(session.as_deref(), &id.0, &last.sent) {
            Ok(path) => Notice::info(
                args.agent,
                format!("Wrote {name}'s last request to {path}."),
            ),
            Err(why) => Notice::error(args.agent, format!("Writing the request failed: {why}")),
        });
        return;
    }
    let providers: Vec<(String, Option<String>)> = tools
        .iter()
        .map(|(def, by)| {
            let plugin = by.and_then(|by| names.get(by.0).ok()).map(short);
            (def.0.name.as_str().to_owned(), plugin)
        })
        .collect();
    let text = report(name.as_str(), last, &providers, usage.context());
    notices.write(Notice::info(args.agent, text));
}

/// Writes `request` as JSON in the session's `requests` directory; its
/// path.
fn write_full(
    session: Option<&SessionPaths>,
    agent: &str,
    request: &CompletionRequest,
) -> Result<String, String> {
    let dir = session
        .ok_or("this run has no session directory")?
        .path()
        .join("requests");
    let path = dir.join(format!("{agent}-last-request.json"));
    let json = serde_json::to_string_pretty(request).map_err(|error| error.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    std::fs::write(&path, json).map_err(|error| error.to_string())?;
    Ok(path.display().to_string())
}

/// A plugin's name without its module path.
fn short(name: &Name) -> String {
    let name = name.as_str();
    name.rsplit("::").next().unwrap_or(name).to_owned()
}

/// The estimated tokens of a message.
fn message_tokens(message: &Message) -> u64 {
    HeuristicTokenCounter::default().count(message) as u64
}

/// The estimated tokens of a text.
fn tokens(text: &str) -> u64 {
    message_tokens(&Message::user(text))
}

/// The estimated tokens of a tool's name, description and schema.
fn tool_tokens(tool: &ToolDefinition) -> u64 {
    let schema = serde_json::to_string(&tool.parameters).unwrap_or_default();
    tokens(&format!("{} {} {schema}", tool.name, tool.description))
}

/// The system prompt and the messages of a request.
fn split(request: &CompletionRequest) -> (&str, &[Arc<Message>]) {
    match request
        .chat_history
        .split_first()
        .map(|(first, rest)| (&**first, rest))
    {
        Some((Message::System { content }, rest)) => (content.as_str(), rest),
        _ => ("", request.chat_history.as_slice()),
    }
}

/// Messages of one kind: how many, and their tokens.
#[derive(Default, Clone, Copy)]
struct Tally(usize, u64);

/// A kind of message, as `/why` groups them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MessageKind {
    User,
    Assistant,
    ToolResults,
    Summaries,
    System,
}

impl MessageKind {
    /// Every kind, in the order the report lists them.
    const ALL: [Self; 5] = [
        Self::User,
        Self::Assistant,
        Self::ToolResults,
        Self::Summaries,
        Self::System,
    ];

    /// The kind of `message`, written by `author`.
    fn of(message: &Message, author: Option<&Author>) -> Self {
        match message {
            _ if author == Some(&summary_author()) => Self::Summaries,
            Message::Assistant(_) => Self::Assistant,
            Message::System { .. } => Self::System,
            Message::User { content } => {
                if content
                    .iter()
                    .any(|part| matches!(part, UserContent::ToolResult(_)))
                {
                    Self::ToolResults
                } else {
                    Self::User
                }
            }
        }
    }

    /// Its name in the report.
    fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolResults => "tool results",
            Self::Summaries => "summaries",
            Self::System => "system",
        }
    }

    /// Its mark in the bar.
    fn mark(self) -> char {
        match self {
            Self::User => 'U',
            Self::Assistant => 'A',
            Self::ToolResults => 'R',
            Self::Summaries => 'C',
            Self::System => 'S',
        }
    }
}

/// The `/why` report of `agent`'s `last` request; `providers` says which
/// plugin added each tool, `reported` the context the provider counted.
fn report(
    agent: &str,
    last: &LastRequest,
    providers: &[(String, Option<String>)],
    reported: Option<u64>,
) -> String {
    let (preamble, messages) = split(&last.sent);
    let mut changes = Vec::new();

    let system_tokens = tokens(preamble);
    let mut lines = vec![format!(
        "System prompt: ~{} tokens",
        tokens_label(system_tokens)
    )];
    lines.extend(prompt_lines(last));

    let tools = &last.sent.tools;
    let tool_total: u64 = tools.iter().map(|tool| tool_tokens(tool)).sum();
    lines.push(format!(
        "Tools offered: {} (~{} tokens)",
        tools.len(),
        tokens_label(tool_total)
    ));
    for tool in tools {
        let name = tool.name.as_str();
        let plugin = providers
            .iter()
            .find(|(provided, _)| provided == name)
            .and_then(|(_, plugin)| plugin.as_deref())
            .unwrap_or("no known plugin");
        let size = tokens_label(tool_tokens(tool));
        lines.push(format!("  {name}: ~{size} ({plugin})"));
    }

    let authors = last
        .prepared
        .iter()
        .flat_map(|prepared| prepared.authors.iter().map(Option::as_ref))
        .chain(std::iter::repeat(None));
    let mut kinds = MessageKind::ALL.map(|kind| (kind, Tally::default()));
    for (message, author) in messages.iter().zip(authors) {
        let kind = MessageKind::of(message, author);
        if let Some((_, tally)) = kinds.iter_mut().find(|(each, _)| *each == kind) {
            tally.0 += 1;
            tally.1 += message_tokens(message);
        }
    }
    let message_total: u64 = kinds.iter().map(|(_, tally)| tally.1).sum();
    lines.push(format!(
        "Messages: {} (~{} tokens)",
        messages.len(),
        tokens_label(message_total)
    ));
    for (kind, tally) in kinds {
        if tally.0 > 0 {
            lines.push(format!(
                "  {}: {} (~{})",
                kind.label(),
                tally.0,
                tokens_label(tally.1)
            ));
        }
    }

    let request = &last.sent;
    let reasoning = request.options.reasoning.as_ref().map_or_else(
        || "default".to_owned(),
        |reasoning| format!("{reasoning:?}"),
    );
    let max = request
        .max_tokens
        .map_or_else(|| "the model's default".to_owned(), |max| max.to_string());
    lines.push(format!(
        "Options: model {}, reasoning {reasoning}, max tokens {max}",
        last.model
    ));

    if let Some(built) = &last.built {
        if let (Some(built), Some(prepared)) = (&built.prompt, &last.prepared) {
            changes.extend(prompt_changes(built, &prepared.prompt));
        }
        changes.extend(tool_changes(&built.tools, tools));
        for change in &mut changes {
            change.push_str(UNNAMED);
        }
    }
    if let Some(prepared) = &last.prepared {
        changes.extend(prepared.edits.iter().flat_map(message_changes));
    }
    lines.push(match (&last.built, changes.is_empty()) {
        (None, _) => "Changes by plugins: unknown (the kernel did not build it).".to_owned(),
        (Some(_), true) => "Changes by plugins: none.".to_owned(),
        (Some(_), false) => format!("Changes by plugins:\n  {}", changes.join("\n  ")),
    });

    let total = system_tokens + tool_total + message_total;
    let mut head = format!(
        "Last request of {agent} to {}: ~{} tokens",
        last.model,
        tokens_label(total)
    );
    if let Some(limit) = last.limit.filter(|limit| *limit > 0) {
        let share = total * 100 / u64::from(limit);
        let limit = tokens_label(u64::from(limit));
        write!(head, " of a {limit} input limit ({share}%)").ok();
    }
    if let Some(reported) = reported {
        let reported = tokens_label(reported);
        write!(head, "; the provider counted {reported} in the last reply").ok();
    }
    head.push('.');
    let mut parts = vec![('S', system_tokens), ('T', tool_total)];
    parts.extend(kinds.map(|(kind, tally)| (kind.mark(), tally.1)));
    let scale = last.limit.map_or(total, u64::from).max(total).max(1);
    let legend = "S system, T tools, U user, A assistant, R tool results, C summaries, . free";
    format!(
        "{head}\n{}\n{legend}\n{}",
        bar(&parts, scale),
        lines.join("\n")
    )
}

/// The parts of the system prompt sent, in order: the role, then each
/// section with where it came from.
fn prompt_lines(last: &LastRequest) -> Vec<String> {
    let role = last
        .prepared
        .as_ref()
        .map(|prepared| prepared.prompt.role.trim());
    let role = role
        .filter(|role| !role.is_empty())
        .map(|role| format!("  the role: ~{}", size(role)));
    let sections = last.notes.iter().map(|note| {
        let named = note
            .name
            .as_deref()
            .map(|name| format!(" \"{name}\""))
            .unwrap_or_default();
        let tokens = tokens_label(note.tokens);
        format!("  <{}>{named}: ~{tokens}, {}", note.tag, note.from)
    });
    role.into_iter().chain(sections).collect()
}

/// How the `sent` prompt's parts differ from the `built` ones: the role,
/// and sections added, rewritten or left out.
fn prompt_changes(
    built: &completion::SystemPrompt,
    sent: &completion::SystemPrompt,
) -> Vec<String> {
    let mut changes = Vec::new();
    if built.role.trim() != sent.role.trim() {
        changes.push(format!("~ the role was set (~{} tokens)", size(&sent.role)));
    }
    let shown = |prompt: &completion::SystemPrompt, tag: &str| {
        prompt
            .section(tag)
            .map(|section| section.text.trim().to_owned())
            .filter(|text| !text.is_empty())
    };
    let tags = sent.sections.iter().chain(&built.sections);
    let mut seen: Vec<&str> = Vec::new();
    for tag in tags.map(|section| section.tag.as_str()) {
        if seen.contains(&tag) {
            continue;
        }
        seen.push(tag);
        match (shown(built, tag), shown(sent, tag)) {
            (None, Some(text)) => {
                changes.push(format!("+ section <{tag}> (~{} tokens)", size(&text)));
            }
            (Some(_), None) => changes.push(format!("- section <{tag}> left out")),
            (Some(built), Some(sent)) if built != sent => {
                changes.push(format!("~ section <{tag}> rewritten"));
            }
            _ => {}
        }
    }
    changes
}

/// The estimated tokens of a text, as a label.
fn size(text: &str) -> String {
    tokens_label(tokens(text))
}

/// The tools added to and left out of the `built` ones.
fn tool_changes(built: &[Arc<ToolDefinition>], sent: &[Arc<ToolDefinition>]) -> Vec<String> {
    let has =
        |tools: &[Arc<ToolDefinition>], name: &str| tools.iter().any(|tool| tool.name == name);
    let added = sent.iter().filter(|tool| !has(built, tool.name.as_str()));
    let dropped = built.iter().filter(|tool| !has(sent, tool.name.as_str()));
    added
        .map(|tool| format!("+ tool {}", tool.name))
        .chain(dropped.map(|tool| format!("- tool {} left out", tool.name)))
        .collect()
}

/// What one attributed edit of a request's messages did, naming who made
/// it: messages added, rewritten (spliced in again under their own ids) or
/// left out. Asides and halts never reach the model.
fn message_changes(edit: &ChangeRecord) -> Vec<String> {
    let by = edit
        .by
        .as_ref()
        .map_or_else(|| "rig".to_owned(), Author::label);
    let (at, removed, with): (_, &[MessageId], &[Entry]) = match &edit.change {
        Change::Push { entries } => ("at the end".to_owned(), &[], entries),
        Change::Splice {
            from,
            removed,
            with,
            ..
        } => {
            let at = match from {
                0 => "before the conversation".to_owned(),
                from => format!("after message {from}"),
            };
            (at, removed, with)
        }
        Change::Reset { .. } => return vec![format!("~ the messages replaced, by {by}")],
        Change::Halt { .. } | Change::Aside { .. } => return Vec::new(),
    };
    let (rewritten, added): (Vec<&Entry>, Vec<&Entry>) = with
        .iter()
        .partition(|entry| entry.id.is_some_and(|id| removed.contains(&id)));
    let mut changes = Vec::new();
    if !rewritten.is_empty() {
        changes.push(format!("~ {} messages rewritten {at}", rewritten.len()));
    }
    if !added.is_empty() {
        let roles: Vec<&str> = added.iter().map(|entry| role(&entry.message)).collect();
        let size: u64 = added
            .iter()
            .map(|entry| message_tokens(&entry.message))
            .sum();
        changes.push(format!(
            "+ {} messages added {at} ({}; ~{} tokens)",
            added.len(),
            roles.join(", "),
            tokens_label(size)
        ));
    }
    let dropped = removed.len().saturating_sub(rewritten.len());
    if dropped > 0 {
        changes.push(format!("- {dropped} messages left out {at}"));
    }
    changes
        .into_iter()
        .map(|change| format!("{change}, by {by}"))
        .collect()
}

/// A message's role.
fn role(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant(_) => "assistant",
    }
}

/// A bar of `parts` (each a mark and its tokens) out of `scale` tokens.
fn bar(parts: &[(char, u64)], scale: u64) -> String {
    let mut bar = String::new();
    let mut used = 0;
    for &(mark, tokens) in parts {
        let width = (tokens * BAR).div_ceil(scale).min(BAR - used);
        used += width;
        bar.extend((0..width).map(|_| mark));
    }
    bar.extend((used..BAR).map(|_| '.'));
    format!("[{bar}]")
}

#[cfg(test)]
mod tests;
