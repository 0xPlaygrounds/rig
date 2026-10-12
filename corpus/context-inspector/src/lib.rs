//! Context inspector (spec `57-context-inspector.md`): `/why` shows what an
//! agent's last model request held, part by part, and how it differed
//! from what the agent's state alone would have sent.
//!
//! A turn's model call is an effect entity whose [`ModelRequest`] the
//! kernel builds in Admit, Prepare systems then edit, and Start fits to the
//! model and sends. The kernel's own request is kept as the call's
//! [`Built`] when the `ModelRequest` is inserted, before any Prepare
//! system ran, its system prompt rendered in front; the request as sent is copied to the agent's
//! [`LastRequest`] when the call is `Started`, with the model it went to
//! after every adapter ([`SentTo`]), the prompt sections that applied and
//! the plugin that spawned each ([`ProvidedBy`]). Each
//! part of the system prompt is rendered as the kernel renders it
//! (rig-core's `SystemPrompt::text`) and found in the prompt sent.
//!
//! `/why [full] [agent]` (or `/context --last`) then shows the request's
//! estimated tokens, a bar of the context window by part, the system
//! prompt by section and plugin, the tools with their schema sizes, the
//! messages by kind, the options, and how the sent request differs from
//! the built one. `/why full` writes the request as JSON in the session's
//! directory. Nothing calls a model.

use std::fmt::Write as _;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{
    self, CompletionRequest, Section, ToolDefinition, tokens_label,
};
use rig_memory::{HeuristicTokenCounter, TokenCounter};

use message::Message;

/// The width of the token bar, in characters.
const BAR: u64 = 40;

/// Adds `/why`, `/context --last`, and what they read.
#[derive(Default)]
pub struct ContextInspectorPlugin;

/// The plugin's stage system: `/context --last` taken over as `/why`,
/// before the harness runs the line as `/context`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextInspectorSystems;

impl Plugin for ContextInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            AgentPass,
            ContextInspectorSystems
                .in_set(Stage::Gate)
                .before(CommandSystems),
        )
        .add_systems(AgentPass, alias.in_set(ContextInspectorSystems))
        .add_observer(keep_built)
        .add_observer(keep_sent)
        .add_command(
            "why",
            "Show what the last model request held: `/why [full] [agent]`",
            why,
        );
    }
}

/// On a turn's model call: its request as the kernel built it from the
/// agent, before Prepare systems changed it.
#[derive(Component, Clone, Debug)]
pub struct Built(pub CompletionRequest);

/// An agent's last turn request, as it was sent.
#[derive(Component, Clone, Debug)]
pub struct LastRequest {
    /// The model, `vendor/model`.
    pub model: String,
    /// The model's context window, when the catalog knows it.
    pub window: Option<u32>,
    /// The request as sent.
    pub sent: CompletionRequest,
    /// The request as the kernel built it, if it built it.
    pub built: Option<CompletionRequest>,
    /// The agent's own prompt.
    pub role: String,
    /// The `<tool_rules>` block of the tools sent, as the kernel renders it.
    pub rules: String,
    /// The prompt sections that applied.
    pub sections: Vec<SectionNote>,
    /// The texts of the summaries that replaced older messages.
    pub summaries: Vec<String>,
}

/// A prompt section that applied to a request, and who spawned it.
#[derive(Clone, Debug)]
pub struct SectionNote {
    /// The section.
    pub section: PromptSection,
    /// Its entity's name, if any.
    pub name: Option<String>,
    /// The plugin that spawned it, if known.
    pub plugin: Option<String>,
    /// Whether it is the agent's own (`SectionOf`).
    pub own: bool,
}

impl SectionNote {
    /// The section as the kernel writes it in a system prompt.
    fn block(&self) -> String {
        tagged(&self.section.tag, &self.section.text)
    }
}

/// `text` between `<tag>` and `</tag>`, as the kernel writes a section in a
/// system prompt; empty when `text` is blank.
fn tagged(tag: &str, text: &str) -> String {
    let prompt = completion::SystemPrompt {
        role: String::new(),
        sections: vec![Section::new(tag, text)],
    };
    prompt.text()
}

/// The `tool_rules` section of tools with `rules`, as the kernel writes
/// it: a line per rule, a rule two tools share once; empty without rules.
fn tool_rules<'a>(rules: impl IntoIterator<Item = &'a ToolRules>) -> String {
    let mut lines: Vec<String> = Vec::new();
    for rule in rules.into_iter().flat_map(|rules| rules.0.iter()) {
        let line = format!("- {}", rule.trim());
        if !rule.trim().is_empty() && !lines.contains(&line) {
            lines.push(line);
        }
    }
    tagged("tool_rules", &lines.join("\n"))
}

/// Keeps the request of each turn's call as it is first put on the call.
fn keep_built(
    inserted: On<Insert<ModelRequest>>,
    calls: Query<&ModelRequest, (With<TurnCall>, Without<Built>)>,
    mut commands: Commands,
) {
    if let Ok(call) = calls.get(inserted.entity) {
        let request = call.request.clone();
        let built = Built(match &call.prompt {
            Some(prompt) => request.with_system(prompt),
            None => request,
        });
        commands.entity(inserted.entity).try_insert_if_new(built);
    }
}

/// Copies each turn's call, as it starts, to its agent's [`LastRequest`].
fn keep_sent(
    started: On<Add<Started>>,
    calls: Query<(&ModelRequest, &SentTo, Option<&Built>, &ForAgent), With<TurnCall>>,
    agents: Query<(&SystemPrompt, &Conversation)>,
    sections: Query<(
        &PromptSection,
        Option<&SectionOf>,
        Option<&Name>,
        Option<&ProvidedBy>,
    )>,
    tools: Query<(&ToolDef, &ToolRules)>,
    names: Query<&Name>,
    mut commands: Commands,
) {
    let Ok((call, SentTo(sent_to), built, &ForAgent(agent))) = calls.get(started.entity) else {
        return;
    };
    let Ok((prompt, conversation)) = agents.get(agent) else {
        return;
    };
    let rules = call.request.tools.iter().filter_map(|sent| {
        let rules = tools.iter().find(|(def, _)| def.0.name == sent.name);
        rules.map(|(_, rules)| rules)
    });
    let sections = sections
        .iter()
        .filter(|(_, of, ..)| of.is_none_or(|of| of.0 == agent))
        .map(|(section, of, name, by)| SectionNote {
            section: section.clone(),
            name: name.map(|name| name.as_str().to_owned()),
            plugin: by.and_then(|by| names.get(by.0).ok()).map(short),
            own: of.is_some(),
        })
        .filter(|note| !note.block().is_empty())
        .collect();
    let last = LastRequest {
        model: sent_to.spec.reference(),
        window: sent_to.spec.context_window,
        sent: call.request.clone(),
        built: built.map(|built| built.0.clone()),
        role: prompt.0.trim().to_owned(),
        rules: tool_rules(rules),
        sections,
        summaries: summaries(conversation),
    };
    commands.entity(agent).try_insert(last);
}

/// The texts of the compaction summaries in `conversation`, each
/// spliced in place of the messages it replaced.
fn summaries(conversation: &Conversation) -> Vec<String> {
    // `rig_compaction::summary_origin()`, which this crate does not depend on.
    let summary = Origin::plugin("compaction");
    let messages = conversation.messages().iter().enumerate();
    let items = messages.filter_map(|(at, message)| match message {
        Message::User { content } => Some(
            content
                .iter()
                .enumerate()
                .map(move |(item, part)| (at, item, part)),
        ),
        _ => None,
    });
    items
        .flatten()
        .filter_map(|(at, item, part)| match part {
            UserContent::Text(text) if conversation.origin(at, item) == Some(&summary) => {
                Some(text.text.clone())
            }
            _ => None,
        })
        .collect()
}

/// `/context --last [full] [agent]` runs as `/why [full] [agent]`, taken
/// over before the harness would run it as `/context`.
fn alias(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
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
fn split(request: &CompletionRequest) -> (&str, &[Message]) {
    match request.chat_history.split_first() {
        Some((Message::System { content }, rest)) => (content.as_str(), rest),
        _ => ("", request.chat_history.as_slice()),
    }
}

/// Messages of one kind: how many, and their tokens.
#[derive(Default, Clone, Copy)]
struct Tally(usize, u64);

/// The kind of a message, as `/why` groups them, given the summaries'
/// texts.
fn kind(message: &Message, summaries: &[String]) -> usize {
    match message {
        Message::Assistant(_) => 1,
        Message::System { .. } => 4,
        Message::User { content } => {
            let parts = content.iter();
            if parts
                .clone()
                .any(|part| matches!(part, UserContent::ToolResult(_)))
            {
                2
            } else if parts.clone().any(
                |part| matches!(part, UserContent::Text(text) if summaries.contains(&text.text)),
            ) {
                3
            } else {
                0
            }
        }
    }
}

/// The names of the kinds [`kind`] returns, and their marks in the bar.
const KINDS: [(&str, char); 5] = [
    ("user", 'U'),
    ("assistant", 'A'),
    ("tool results", 'R'),
    ("summaries", 'C'),
    ("system", 'U'),
];

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
    lines.extend(prompt_parts(last, preamble, &mut changes));

    let tools = &last.sent.tools;
    let tool_total: u64 = tools.iter().map(tool_tokens).sum();
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

    let mut kinds = [Tally::default(); KINDS.len()];
    for message in messages {
        if let Some(tally) = kinds.get_mut(kind(message, &last.summaries)) {
            tally.0 += 1;
            tally.1 += message_tokens(message);
        }
    }
    let message_total: u64 = kinds.iter().map(|tally| tally.1).sum();
    lines.push(format!(
        "Messages: {} (~{} tokens)",
        messages.len(),
        tokens_label(message_total)
    ));
    for ((label, _), tally) in KINDS.iter().zip(kinds) {
        if tally.0 > 0 {
            lines.push(format!(
                "  {label}: {} (~{})",
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
        changes.extend(tool_changes(&built.tools, tools));
        changes.extend(message_changes(split(built).1, messages));
    }
    lines.push(match (&last.built, changes.is_empty()) {
        (None, _) => "Changes by plugins: unknown (the kernel did not build it).".to_owned(),
        (Some(_), true) => "Changes by plugins: none.".to_owned(),
        (Some(_), false) => format!(
            "Changes by plugins' Prepare systems (the kernel does not say whose):\n  {}",
            changes.join("\n  ")
        ),
    });

    let total = system_tokens + tool_total + message_total;
    let mut head = format!(
        "Last request of {agent} to {}: ~{} tokens",
        last.model,
        tokens_label(total)
    );
    if let Some(window) = last.window.filter(|window| *window > 0) {
        let share = total * 100 / u64::from(window);
        let window = tokens_label(u64::from(window));
        write!(head, " of a {window} window ({share}%)").ok();
    }
    if let Some(reported) = reported {
        let reported = tokens_label(reported);
        write!(head, "; the provider counted {reported} in the last reply").ok();
    }
    head.push('.');
    let mut parts = vec![('S', system_tokens), ('T', tool_total)];
    parts.extend(
        KINDS
            .iter()
            .zip(kinds)
            .map(|((_, mark), tally)| (*mark, tally.1)),
    );
    let scale = last.window.map_or(total, u64::from).max(total).max(1);
    let legend = "S system, T tools, U user, A assistant, R tool results, C summaries, . free";
    format!(
        "{head}\n{}\n{legend}\n{}",
        bar(&parts, scale),
        lines.join("\n")
    )
}

/// The system prompt's parts, as the kernel renders each, found in the
/// sent `preamble` in the order it holds them: the agent's own prompt, the
/// tool rules and each section, with what was rewritten, left out or added
/// in `changes`.
fn prompt_parts(last: &LastRequest, preamble: &str, changes: &mut Vec<String>) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = preamble.to_owned();
    if !last.role.is_empty() {
        if preamble.starts_with(&last.role) {
            found.push((
                0,
                format!("  the agent's own prompt: ~{}", size(&last.role)),
            ));
            rest = rest.replacen(&last.role, "", 1);
        } else {
            changes.push("~ the agent's own prompt was rewritten".to_owned());
        }
    }
    if let Some(at) = preamble
        .find(&last.rules)
        .filter(|_| !last.rules.is_empty())
    {
        let line = format!(
            "  <tool_rules> of the tools offered: ~{}",
            size(&last.rules)
        );
        found.push((at, line));
        rest = rest.replacen(&last.rules, "", 1);
    }
    for note in &last.sections {
        let tag = &note.section.tag;
        let block = note.block();
        let named = note
            .name
            .as_deref()
            .map(|name| format!(" \"{name}\""))
            .unwrap_or_default();
        if let Some(at) = preamble.find(&block) {
            let from = match (&note.plugin, note.own) {
                (Some(plugin), _) => format!("from {plugin}"),
                (None, true) => "this agent's own".to_owned(),
                (None, false) => "from no known plugin".to_owned(),
            };
            found.push((at, format!("  <{tag}>{named}: ~{}, {from}", size(&block))));
            rest = rest.replacen(&block, "", 1);
        } else if preamble.contains(&format!("<{tag}>")) {
            changes.push(format!("~ section <{tag}>{named} rewritten"));
        } else {
            changes.push(format!("- section <{tag}>{named} left out"));
        }
    }
    found.sort_by_key(|(at, _)| *at);
    let mut parts: Vec<String> = found.into_iter().map(|(_, line)| line).collect();
    let added = rest.trim();
    if !added.is_empty() {
        let shown: String = added.chars().take(60).collect();
        parts.push(format!("  added before sending: ~{}", size(added)));
        changes.push(format!(
            "+ system prompt text \"{shown}\" (~{} tokens)",
            size(added)
        ));
    }
    parts
}

/// The estimated tokens of a text, as a label.
fn size(text: &str) -> String {
    tokens_label(tokens(text))
}

/// The tools added to and left out of the `built` ones.
fn tool_changes(built: &[ToolDefinition], sent: &[ToolDefinition]) -> Vec<String> {
    let has = |tools: &[ToolDefinition], name: &str| tools.iter().any(|tool| tool.name == name);
    let added = sent.iter().filter(|tool| !has(built, tool.name.as_str()));
    let dropped = built.iter().filter(|tool| !has(sent, tool.name.as_str()));
    added
        .map(|tool| format!("+ tool {}", tool.name))
        .chain(dropped.map(|tool| format!("- tool {} left out", tool.name)))
        .collect()
}

/// How the `sent` messages differ from the `built` ones: the run between
/// their common start and end, as messages added, rewritten or left out.
fn message_changes(built: &[Message], sent: &[Message]) -> Vec<String> {
    let start = built.iter().zip(sent).take_while(|(a, b)| a == b).count();
    let built = built.get(start..).unwrap_or_default();
    let sent = sent.get(start..).unwrap_or_default();
    let end = built
        .iter()
        .rev()
        .zip(sent.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let built = built.get(..built.len() - end).unwrap_or_default();
    let sent = sent.get(..sent.len() - end).unwrap_or_default();
    let rewritten = built.len().min(sent.len());
    let at = match start {
        0 => "before the conversation".to_owned(),
        _ => format!("after message {start}"),
    };
    let mut changes = Vec::new();
    if rewritten > 0 {
        changes.push(format!("~ {rewritten} messages rewritten {at}"));
    }
    if let Some(added) = sent.get(rewritten..).filter(|added| !added.is_empty()) {
        let roles: Vec<&str> = added.iter().map(role).collect();
        let size: u64 = added.iter().map(message_tokens).sum();
        changes.push(format!(
            "+ {} messages added {at} ({}; ~{} tokens)",
            added.len(),
            roles.join(", "),
            tokens_label(size)
        ));
    }
    if let Some(dropped) = built.get(rewritten..).filter(|dropped| !dropped.is_empty()) {
        changes.push(format!("- {} messages left out {at}", dropped.len()));
    }
    changes
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
