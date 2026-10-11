//! Context inspector (spec `57-context-inspector.md`): `/why` shows what an
//! agent's last model request held, part by part, and how it differed
//! from what the agent's state alone would have sent.
//!
//! Two records make it. When [`PrepareRequest`] is triggered, the plugin
//! notes what the agent's state says the request holds: its own prompt,
//! the prompt sections that apply (with the plugin that spawned each, from
//! [`ProvidedBy`]), the tools it may use and its messages ([`Baseline`]).
//! The request as sent, after every `PrepareRequest` observer changed it,
//! is not kept by the kernel, so each agent's and turn's [`Connection`] is
//! wrapped in a passive rig-core layer ([`LAYER`]) that hands every
//! completion it dispatches to the world as the agent's [`LastRequest`].
//!
//! `/why [full] [agent]` (or `/context --last`) then shows the request's
//! estimated tokens, a bar of the context window by part, the system
//! prompt by section and plugin, the tools with their schema sizes, the
//! messages by kind, the options, and the differences from the baseline,
//! attributed to the plugins that observe `PrepareRequest`. `/why full`
//! writes the whole request as JSON under [`RequestDumps`]. Nothing calls
//! a model.

use std::fmt::Write as _;
use std::path::PathBuf;

use bevy_ecs::event::EventKey;
use crossbeam_channel::{Receiver, Sender};
use rig_core::completion::{CompletionRequest, Message, ToolDefinition, tokens_label};
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::ErrorReport;
use rig_core::serve::{Decision, Intercept, Verdict};
use rig_ecs::effects::Handler;
use rig_harness::prelude::*;
use rig_memory::{HeuristicTokenCounter, TokenCounter};

/// The name of the layer on each connection, as the effect log records it.
pub const LAYER: &str = "context-inspector";

/// The width of the token bar, in characters.
const BAR: u64 = 40;

/// Adds `/why`, `/context --last`, and what they read.
#[derive(Default)]
pub struct ContextInspectorPlugin;

impl Plugin for ContextInspectorPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let dumps = app
            .world()
            .get_resource::<SessionPaths>()
            .map_or_else(|| PathBuf::from(".rig"), |paths| paths.path().to_owned())
            .join("requests");
        let prepare = app.world_mut().register_event_key::<PrepareRequest>();
        app.insert_resource(Sent { sender, receiver })
            .insert_resource(RequestDumps(dumps))
            .insert_resource(PrepareKey(prepare))
            .add_observer(tap)
            .add_observer(alias)
            .add_systems(Update, take_sent)
            .add_command(
                "why",
                "Show what the last model request held: `/why [full] [agent]`",
                why,
            );
        // Marked, so `/why` does not count it among the plugins that
        // change requests.
        app.world_mut().add_observer(baseline).insert(Inspector);
    }
}

/// Where `/why full` writes requests: the session's `requests` directory.
#[derive(Resource, Clone, Debug)]
pub struct RequestDumps(pub PathBuf);

/// An agent's last model request as its connection sent it.
#[derive(Component, Clone, Debug)]
pub struct LastRequest {
    /// The model, `vendor/model`.
    pub model: String,
    /// The model's context window, when the catalog knows it.
    pub window: Option<u32>,
    /// The request.
    pub request: CompletionRequest,
    /// What the agent's state said the request holds.
    baseline: Option<Baseline>,
}

/// What the agent's state alone puts in a request, noted when the request
/// is prepared.
#[derive(Component, Clone, Debug, Default)]
struct Baseline {
    /// The agent's own prompt.
    role: String,
    /// The prompt sections that apply, in prompt order.
    sections: Vec<Section>,
    /// The names of the tools it is offered.
    tools: Vec<String>,
    /// The conversation as the request sends it.
    messages: Vec<Message>,
    /// The text of the summary that replaces older messages, if any.
    summary: Option<String>,
}

/// A prompt section and who spawned it.
#[derive(Clone, Debug)]
struct Section {
    tag: String,
    text: String,
    name: Option<String>,
    plugin: Option<String>,
    own: bool,
}

/// The observer of this plugin's own.
#[derive(Component)]
struct Inspector;

/// The event key of [`PrepareRequest`], to find who observes it.
#[derive(Resource)]
struct PrepareKey(EventKey);

/// Requests the connections' layers saw, on their way to the world.
#[derive(Resource)]
struct Sent {
    sender: Sender<Tapped>,
    receiver: Receiver<Tapped>,
}

/// One completion a layer saw.
struct Tapped {
    agent: Entity,
    model: String,
    window: Option<u32>,
    request: CompletionRequest,
}

/// The passive layer: hands each completion to the world and lets it pass.
struct Tap {
    agent: Entity,
    model: String,
    window: Option<u32>,
    sender: Sender<Tapped>,
    wake: Wake,
}

impl Intercept for Tap {
    fn name(&self) -> String {
        LAYER.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> Decision {
        if let EffectKind::Completion { request, .. } = kind {
            let tapped = Tapped {
                agent: self.agent,
                model: self.model.clone(),
                window: self.window,
                request: request.clone(),
            };
            // The world may be gone; the request still goes.
            self.sender.send(tapped).ok();
            self.wake.wake();
        }
        Decision::Proceed
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// Wraps an agent's or a turn's new connection in the [`Tap`], once.
fn tap(
    inserted: On<Insert<Connection>>,
    owners: Query<(&Connection, Option<&TurnOf>, Has<Agent>)>,
    sent: Res<Sent>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((connection, turn, is_agent)) = owners.get(inserted.entity) else {
        return;
    };
    let agent = match (turn, is_agent) {
        (Some(TurnOf(agent)), _) => *agent,
        (None, true) => inserted.entity,
        (None, false) => return,
    };
    let handler = &connection.handler.0;
    if handler
        .descriptor()
        .layers
        .iter()
        .any(|layer| layer == LAYER)
    {
        return;
    }
    let tap = Tap {
        agent,
        model: connection.spec.reference(),
        window: connection.spec.context_window,
        sender: sent.sender.clone(),
        wake: wake.clone(),
    };
    let tapped = Connection {
        spec: connection.spec.clone(),
        handler: Handler(handler.clone().layered(tap)),
    };
    if let Ok(mut owner) = commands.get_entity(inserted.entity) {
        owner.try_insert(tapped);
    }
}

/// Notes what the agent's state puts in the request being prepared. It
/// reads the world, not the request, so the observers' order does not
/// matter.
fn baseline(
    prepare: On<PrepareRequest>,
    turns: Query<Option<&Connection>, With<TurnOf>>,
    agents: Query<(
        &Conversation,
        Option<&Condensed>,
        &SystemPrompt,
        &ToolAccess,
        Option<&Connection>,
    )>,
    sections: Query<(
        &PromptSection,
        Option<&SectionOf>,
        Option<&Name>,
        Option<&ProvidedBy>,
    )>,
    tools: Query<&ToolDef>,
    plugins: Query<&Name>,
    mut commands: Commands,
) {
    let agent = prepare.agent;
    let Ok((conversation, condensed, prompt, access, own)) = agents.get(agent) else {
        return;
    };
    let routed = turns.get(prepare.entity).ok().flatten();
    let native_tools = routed
        .or(own)
        .is_some_and(|connection| connection.spec.tools);
    let mut applying: Vec<(&PromptSection, Section)> = sections
        .iter()
        .filter(|(section, of, ..)| {
            !section.text.trim().is_empty() && of.is_none_or(|of| of.0 == agent)
        })
        .map(|(section, of, name, by)| {
            let noted = Section {
                tag: section.tag.clone(),
                text: section.text.trim().to_owned(),
                name: name.map(|name| name.as_str().to_owned()),
                plugin: by.and_then(|by| plugins.get(by.0).ok()).map(short),
                own: of.is_some(),
            };
            (section, noted)
        })
        .collect();
    applying.sort_by(|(a, _), (b, _)| a.order.cmp(&b.order).then_with(|| a.tag.cmp(&b.tag)));
    let mut offered: Vec<String> = tools
        .iter()
        .map(|def| def.0.name.as_str().to_owned())
        .filter(|name| native_tools && access.allows(name))
        .collect();
    offered.sort();
    let messages = match condensed {
        Some(condensed) => condensed.request(conversation.messages()),
        None => conversation.messages().to_vec(),
    };
    let noted = Baseline {
        role: prompt.0.trim().to_owned(),
        sections: applying.into_iter().map(|(_, section)| section).collect(),
        tools: offered,
        messages,
        summary: condensed.map(|condensed| condensed.summary.clone()),
    };
    if let Ok(mut agent) = commands.get_entity(agent) {
        agent.try_insert(noted);
    }
}

/// Keeps each request a layer saw as its agent's [`LastRequest`], with the
/// baseline noted for it.
fn take_sent(sent: Res<Sent>, baselines: Query<&Baseline>, mut commands: Commands) {
    for tapped in sent.receiver.try_iter() {
        let last = LastRequest {
            model: tapped.model,
            window: tapped.window,
            request: tapped.request,
            baseline: baselines.get(tapped.agent).ok().cloned(),
        };
        if let Ok(mut agent) = commands.get_entity(tapped.agent) {
            agent.try_insert(last);
        }
    }
}

/// `/context --last` is `/why`.
fn alias(mut typed: On<Deliver>) {
    if typed.origin.kind != OriginKind::User {
        return;
    }
    if let Some(rest) = typed.text.trim().strip_prefix("/context --last") {
        typed.text = format!("/why{rest}");
    }
}

/// `/why [full] [agent]`.
#[expect(clippy::too_many_arguments, reason = "a command system's queries")]
fn why(
    In(args): In<CommandArgs>,
    agents: Query<(Entity, &Name, &AgentId, Option<&LastRequest>, &LastUsage), With<Agent>>,
    tools: Query<(&ToolDef, Option<&ProvidedBy>)>,
    names: Query<&Name>,
    observers: Query<(&Observer, &ProvidedBy), Without<Inspector>>,
    key: Res<PrepareKey>,
    dumps: Res<RequestDumps>,
    mut notices: MessageWriter<Notice>,
) {
    let mut words: Vec<&str> = args.args.split_whitespace().collect();
    let full = words.first() == Some(&"full");
    if full {
        words.remove(0);
    }
    let wanted = words.join(" ");
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
        let path = dumps.0.join(format!("{}-last-request.json", id.0));
        let written = serde_json::to_string_pretty(&last.request)
            .map_err(|error| error.to_string())
            .and_then(|json| {
                std::fs::create_dir_all(&dumps.0).map_err(|error| error.to_string())?;
                std::fs::write(&path, json).map_err(|error| error.to_string())
            });
        notices.write(match written {
            Ok(()) => Notice::info(
                args.agent,
                format!("Wrote {name}'s last request to {}.", path.display()),
            ),
            Err(why) => Notice::error(args.agent, format!("Writing the request failed: {why}")),
        });
        return;
    }
    let mut changers: Vec<String> = observers
        .iter()
        .filter(|(observer, _)| observer.descriptor().event_keys().contains(&key.0))
        .filter_map(|(_, by)| names.get(by.0).ok().map(short))
        .collect();
    changers.sort();
    changers.dedup();
    let by = match changers.as_slice() {
        [] => "a plugin".to_owned(),
        [one] => one.clone(),
        many => format!("one of {}", many.join(", ")),
    };
    let providers: Vec<(String, Option<String>)> = tools
        .iter()
        .map(|(def, by)| {
            let plugin = by.and_then(|by| names.get(by.0).ok()).map(short);
            (def.0.name.as_str().to_owned(), plugin)
        })
        .collect();
    let text = report(name.as_str(), last, &providers, &by, usage.context());
    notices.write(Notice::info(args.agent, text));
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

/// Messages of one kind and their tokens.
#[derive(Default, Clone, Copy)]
struct Tally {
    count: usize,
    tokens: u64,
}

impl Tally {
    fn add(&mut self, tokens: u64) {
        self.count += 1;
        self.tokens += tokens;
    }
}

/// The request's messages by kind.
#[derive(Default)]
struct Kinds {
    user: Tally,
    assistant: Tally,
    results: Tally,
    summaries: Tally,
    system: Tally,
}

impl Kinds {
    fn of(messages: &[Message], summary: Option<&str>) -> Self {
        let mut kinds = Self::default();
        for message in messages {
            let tokens = message_tokens(message);
            let tally = match message {
                Message::System { .. } => &mut kinds.system,
                Message::Assistant(_) => &mut kinds.assistant,
                Message::User { content } => {
                    let result = content
                        .iter()
                        .any(|part| matches!(part, UserContent::ToolResult(_)));
                    let summarized = content.iter().any(|part| {
                        matches!(part, UserContent::Text(text) if Some(text.text.as_str()) == summary)
                    });
                    match (result, summarized) {
                        (true, _) => &mut kinds.results,
                        (false, true) => &mut kinds.summaries,
                        (false, false) => &mut kinds.user,
                    }
                }
            };
            tally.add(tokens);
        }
        kinds
    }
}

/// The `/why` report of `agent`'s `last` request; `by` names the plugins
/// that change requests, `reported` the context the provider counted.
fn report(
    agent: &str,
    last: &LastRequest,
    providers: &[(String, Option<String>)],
    by: &str,
    reported: Option<u64>,
) -> String {
    let request = &last.request;
    let (preamble, messages) = match request.chat_history.split_first() {
        Some((Message::System { content }, rest)) => (content.as_str(), rest),
        _ => ("", request.chat_history.as_slice()),
    };
    let baseline = last.baseline.as_ref();
    let mut changes: Vec<String> = Vec::new();

    let system_tokens = tokens(preamble);
    let mut prompt = vec![format!(
        "System prompt: ~{} tokens",
        tokens_label(system_tokens)
    )];
    let mut rest = preamble.to_owned();
    if let Some(baseline) = baseline {
        if !baseline.role.is_empty() {
            if rest.starts_with(&baseline.role) {
                prompt.push(format!(
                    "  the agent's own prompt: ~{}",
                    tokens_label(tokens(&baseline.role))
                ));
                rest = rest.replacen(&baseline.role, "", 1);
            } else {
                changes.push(format!("~ the agent's own prompt was rewritten, by {by}"));
            }
        }
        if let (Some(start), Some(end)) = (rest.find("<tool_rules>"), rest.find("</tool_rules>"))
            && let Some(block) = rest.get(start..end + "</tool_rules>".len())
        {
            let block = block.to_owned();
            prompt.push(format!(
                "  <tool_rules> of the tools offered: ~{}",
                tokens_label(tokens(&block))
            ));
            rest = rest.replacen(&block, "", 1);
        }
        for section in &baseline.sections {
            let block = format!("<{0}>\n{1}\n</{0}>", section.tag, section.text);
            let named = section
                .name
                .as_deref()
                .map(|name| format!(" \"{name}\""))
                .unwrap_or_default();
            if rest.contains(&block) {
                let from = match (&section.plugin, section.own) {
                    (Some(plugin), _) => format!("from {plugin}"),
                    (None, true) => "this agent's own".to_owned(),
                    (None, false) => "from no known plugin".to_owned(),
                };
                prompt.push(format!(
                    "  <{}>{named}: ~{}, {from}",
                    section.tag,
                    tokens_label(tokens(&block))
                ));
                rest = rest.replacen(&block, "", 1);
            } else if rest.contains(&format!("<{}>", section.tag)) {
                changes.push(format!(
                    "~ section <{}>{named} rewritten, by {by}",
                    section.tag
                ));
            } else {
                changes.push(format!(
                    "- section <{}>{named} left out, by {by}",
                    section.tag
                ));
            }
        }
        let added = rest.trim();
        if !added.is_empty() {
            let shown: String = added.chars().take(60).collect();
            prompt.push(format!(
                "  added before sending: ~{}",
                tokens_label(tokens(added))
            ));
            changes.push(format!(
                "+ system prompt text \"{shown}\" (~{} tokens), by {by}",
                tokens_label(tokens(added))
            ));
        }
    }

    let tool_tokens_total: u64 = request.tools.iter().map(tool_tokens).sum();
    let mut tools = vec![format!(
        "Tools offered: {} (~{} tokens)",
        request.tools.len(),
        tokens_label(tool_tokens_total)
    )];
    for tool in &request.tools {
        let name = tool.name.as_str();
        let plugin = providers
            .iter()
            .find(|(provided, _)| provided == name)
            .and_then(|(_, plugin)| plugin.as_deref())
            .unwrap_or("no known plugin");
        tools.push(format!(
            "  {name}: ~{} ({plugin})",
            tokens_label(tool_tokens(tool))
        ));
    }
    if let Some(baseline) = baseline {
        for tool in &request.tools {
            if !baseline.tools.iter().any(|name| name == tool.name.as_str()) {
                changes.push(format!("+ tool {}, by {by}", tool.name));
            }
        }
        for name in &baseline.tools {
            if !request.tools.iter().any(|tool| tool.name.as_str() == name) {
                changes.push(format!("- tool {name} left out, by {by}"));
            }
        }
    }

    let summary = baseline.and_then(|baseline| baseline.summary.as_deref());
    let kinds = Kinds::of(messages, summary);
    let message_tokens_total: u64 = messages.iter().map(message_tokens).sum();
    let mut listed = vec![format!(
        "Messages: {} (~{} tokens)",
        messages.len(),
        tokens_label(message_tokens_total)
    )];
    for (label, tally) in [
        ("user", kinds.user),
        ("assistant", kinds.assistant),
        ("tool results", kinds.results),
        ("summaries", kinds.summaries),
        ("system", kinds.system),
    ] {
        if tally.count > 0 {
            listed.push(format!(
                "  {label}: {} (~{})",
                tally.count,
                tokens_label(tally.tokens)
            ));
        }
    }
    if let Some(baseline) = baseline {
        changes.extend(message_changes(&baseline.messages, messages, by));
    }

    let total = system_tokens + tool_tokens_total + message_tokens_total;
    let options = &request.options;
    let mut text = format!(
        "Last request of {agent} to {}: ~{} tokens",
        last.model,
        tokens_label(total)
    );
    if let Some(window) = last.window.filter(|window| *window > 0) {
        let share = total * 100 / u64::from(window);
        write!(
            text,
            " of a {} window ({share}%)",
            tokens_label(u64::from(window))
        )
        .ok();
    }
    if let Some(reported) = reported {
        write!(
            text,
            "; the provider counted {} in the last reply",
            tokens_label(reported)
        )
        .ok();
    }
    text.push('.');
    let parts = [
        ('S', system_tokens),
        ('T', tool_tokens_total),
        ('U', kinds.user.tokens + kinds.system.tokens),
        ('A', kinds.assistant.tokens),
        ('R', kinds.results.tokens),
        ('C', kinds.summaries.tokens),
    ];
    let scale = last.window.map_or(total, u64::from).max(total).max(1);
    text.push('\n');
    text.push_str(&bar(&parts, scale));
    text.push_str("\nS system, T tools, U user, A assistant, R tool results, C summaries, . free");
    for block in [prompt, tools, listed] {
        text.push('\n');
        text.push_str(&block.join("\n"));
    }
    let reasoning = options.reasoning.as_ref().map_or_else(
        || "default".to_owned(),
        |reasoning| format!("{reasoning:?}"),
    );
    let max = request
        .max_tokens
        .map_or_else(|| "the model's default".to_owned(), |max| max.to_string());
    write!(
        text,
        "\nOptions: model {}, reasoning {reasoning}, max tokens {max}",
        last.model
    )
    .ok();
    if let Some(temperature) = request.temperature {
        write!(text, ", temperature {temperature}").ok();
    }
    match (baseline, changes.is_empty()) {
        (None, _) => text.push_str("\nChanges by plugins: unknown (the request was not prepared)."),
        (Some(_), true) => text.push_str("\nChanges by plugins: none."),
        (Some(_), false) => {
            text.push_str("\nChanges by plugins before sending:\n  ");
            text.push_str(&changes.join("\n  "));
        }
    }
    text
}

/// How the `sent` messages differ from the `noted` ones: the run between
/// their common start and end, as messages added, rewritten or left out.
fn message_changes(noted: &[Message], sent: &[Message], by: &str) -> Vec<String> {
    let start = noted.iter().zip(sent).take_while(|(a, b)| a == b).count();
    let (noted_rest, sent_rest) = (
        noted.get(start..).unwrap_or_default(),
        sent.get(start..).unwrap_or_default(),
    );
    let end = noted_rest
        .iter()
        .rev()
        .zip(sent_rest.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let noted_mid = noted_rest
        .get(..noted_rest.len().saturating_sub(end))
        .unwrap_or_default();
    let sent_mid = sent_rest
        .get(..sent_rest.len().saturating_sub(end))
        .unwrap_or_default();
    let rewritten = noted_mid.len().min(sent_mid.len());
    let where_ = if start == 0 {
        "before the conversation".to_owned()
    } else {
        format!("after message {start}")
    };
    let mut changes = Vec::new();
    if rewritten > 0 {
        changes.push(format!(
            "~ {rewritten} messages rewritten {where_}, by {by}"
        ));
    }
    if let Some(added) = sent_mid.get(rewritten..).filter(|added| !added.is_empty()) {
        let roles: Vec<&str> = added.iter().map(role).collect();
        let tokens: u64 = added.iter().map(message_tokens).sum();
        changes.push(format!(
            "+ {} messages added {where_} ({}; ~{} tokens), by {by}",
            added.len(),
            roles.join(", "),
            tokens_label(tokens)
        ));
    }
    if let Some(dropped) = noted_mid
        .get(rewritten..)
        .filter(|dropped| !dropped.is_empty())
    {
        changes.push(format!(
            "- {} messages left out {where_}, by {by}",
            dropped.len()
        ));
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
        bar.extend(std::iter::repeat_n(
            mark,
            usize::try_from(width).unwrap_or(0),
        ));
    }
    bar.extend(std::iter::repeat_n(
        '.',
        usize::try_from(BAR - used).unwrap_or(0),
    ));
    format!("[{bar}]")
}

#[cfg(test)]
mod tests;
