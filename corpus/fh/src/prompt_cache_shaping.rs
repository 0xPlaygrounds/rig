//! Prompt cache shaping: lays every turn's model request out so the
//! provider's prompt cache hits as often as it can, and shows how often it
//! did.
//!
//! - **Stable order.** A request's tools are sorted by name, after every
//!   other plugin's edits. The kernel already keeps the prompt sections in
//!   a fixed order and sends a cache key per agent for providers that cache
//!   prefixes on their own.
//! - **Volatile parts move to the end.** A prompt section that changes (a
//!   clock, a status block) is taken out of the system prompt and its text
//!   added to the request's last message, after every cache marker. A
//!   section counts as changing once its [`PromptSection`] changed after it
//!   was added, or once its text differed from the agent's previous request
//!   while the tools stayed the same. One in a request without a prompt in
//!   parts is reported, never dropped.
//! - **Markers.** Every request asks for cache points at the end of the
//!   tools and of the system prompt, and one after the first, at the end of
//!   the previous request's messages. A provider that caches by marker
//!   places them within its budget; others ignore them.
//! - **Breaks.** Each agent keeps the prefix of its last request; where the
//!   next one differs, the first change (a tool, a prompt section by name,
//!   a message index) is kept, and `/cache why` lists the last ten.
//! - **Hit rate.** Each turn call's cache reads over its input tokens; the
//!   agent's status line shows the last request's and the session's rate.
//!
//! Without the plugin the requests are laid out as the kernel builds them;
//! the conversation itself is never changed, only the requests.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{
    CachePoint, Message, Section, SystemPrompt, ToolDefinition, Usage, message::UserContent,
};
use rig_harness::rig_core::effect;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// After the cached input of `rig_basics::telemetry`, before the cost.
const RATE: StatusItem = StatusItem::new("fh-prompt-cache-shaping", Side::Right, 25, 2);
/// The breaks `/cache why` keeps per agent.
const KEPT_BREAKS: usize = 10;

/// The Prepare system that lays requests out: after every request edit,
/// before the request is fitted to its model.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheShapingSystems;

/// The plugin.
#[derive(Default)]
pub struct CacheShapingPlugin;

impl Plugin for CacheShapingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SessionCache>()
            .register_required_components::<Agent, CacheHistory>()
            .configure_sets(
                AgentPass,
                CacheShapingSystems
                    .in_set(Stage::Prepare)
                    .after(PrepareSystems::Edit)
                    .before(PrepareSystems::Fit),
            )
            .add_systems(AgentPass, shape.in_set(CacheShapingSystems))
            .add_systems(PostUpdate, show.in_set(StatusSystems::Show))
            .add_observer(record_usage)
            .add_command(
                "cache",
                "Prompt cache hit rate; `/cache why` lists the last cache breaks",
                cache,
            );
    }
}
/// Input tokens and how many of them the cache read or wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hits {
    /// The input tokens, cached or not.
    pub input: u64,
    /// The input tokens read from the cache.
    pub read: u64,
    /// The input tokens written to the cache.
    pub written: u64,
}

impl Hits {
    /// What `usage` reports.
    pub fn of(usage: &Usage) -> Self {
        Self {
            input: usage.input_tokens.unwrap_or(0),
            read: usage.cached_input_tokens.unwrap_or(0),
            written: usage.cache_creation_input_tokens.unwrap_or(0),
        }
    }

    /// The share of the input read from the cache, in percent, rounded;
    /// none without input.
    pub fn rate(&self) -> Option<u64> {
        (self.input > 0).then(|| (self.read * 100 + self.input / 2) / self.input)
    }

    fn add(&mut self, other: Hits) {
        self.input += other.input;
        self.read += other.read;
        self.written += other.written;
    }
}

/// Every agent's turn calls together.
#[derive(Resource, Default, Debug)]
pub struct SessionCache(pub Hits);

/// A request whose cacheable prefix differed from its agent's previous
/// one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheBreak {
    /// The agent's request it happened in, from 1.
    pub request: u64,
    /// The first part that changed, such as "tool list changed: `x`
    /// added".
    pub cause: String,
}

/// On every agent: its requests' prefix history and cache hits. A
/// subagent is an agent of its own, so its requests never count against
/// its parent's.
#[derive(Component, Default, Debug)]
pub struct CacheHistory {
    /// The agent's turn requests laid out so far.
    pub requests: u64,
    /// The last breaks, oldest first.
    pub breaks: VecDeque<CacheBreak>,
    /// The tags of the prompt sections that change, moved out of the
    /// system prompt.
    pub volatile: BTreeSet<String>,
    /// The sections that change but could not be moved.
    pub unmovable: BTreeSet<String>,
    /// The last turn call's hits.
    pub last: Option<Hits>,
    /// The prefix of the last request.
    prefix: Option<Prefix>,
}

/// What a request's cache keys on, in the order it is sent.
#[derive(Debug, Clone)]
struct Prefix {
    tools: Vec<Arc<ToolDefinition>>,
    /// The system prompt without what was moved; a whole one's text as
    /// its role.
    prompt: SystemPrompt,
    /// The conversation after the system prompt, without what was moved.
    messages: Vec<Message>,
}

/// Lays each turn call out as it is admitted, after every other edit.
fn shape(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    sections: Query<(Ref<PromptSection>, Option<&SectionOf>)>,
    mut agents: Query<&mut CacheHistory>,
    mut notices: MessageWriter<Notice>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok(mut history) = agents.get_mut(agent) else {
            continue;
        };
        let history = &mut *history;
        let changed = sections.iter().filter(|(section, of)| {
            of.is_none_or(|of| of.0 == agent) && section.last_changed() != section.added()
        });
        history
            .volatile
            .extend(changed.map(|(section, _)| section.tag.clone()));

        let ModelRequest {
            request,
            prompt,
            history: conversation,
            ..
        } = &mut *call;
        let request = Arc::make_mut(request);
        request
            .tools
            .sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
        let (moved, prompt) = match prompt {
            Some(prompt) => {
                let (moved, kept): (Vec<Section>, _) = std::mem::take(&mut prompt.sections)
                    .into_iter()
                    .partition(|section| history.volatile.contains(&section.tag));
                prompt.sections = kept;
                (moved, prompt.clone())
            }
            None => {
                for tag in &history.volatile {
                    if history.unmovable.insert(tag.clone()) {
                        let text = format!(
                            "Prompt section `{tag}` changes but this request's system prompt \
                             is not in parts; it stays where it is."
                        );
                        notices.write(Notice::warn(agent, text));
                    }
                }
                let role = request.system_instructions().unwrap_or_default().to_owned();
                (
                    Vec::new(),
                    SystemPrompt {
                        role,
                        sections: Vec::new(),
                    },
                )
            }
        };
        // A request the kernel built sends its conversation after the
        // system prompt; one a plugin brought, its messages.
        let messages = match conversation {
            Some(conversation) => conversation.messages(),
            None => &request.chat_history,
        };
        let lead = usize::from(matches!(
            messages.first().map(|message| &**message),
            Some(Message::System { .. })
        ));
        let now = Prefix {
            tools: request.tools.clone(),
            prompt,
            messages: messages
                .iter()
                .skip(lead)
                .map(|message| (**message).clone())
                .collect(),
        };

        request
            .cache_points
            .extend([CachePoint::Tools, CachePoint::System]);
        history.requests += 1;
        if let Some(previous) = &history.prefix {
            let common = previous
                .messages
                .iter()
                .zip(&now.messages)
                .take_while(|(was, is)| was == is)
                .count();
            // Counted as cache points count: without system messages.
            let turns = now
                .messages
                .iter()
                .take(common)
                .filter(|message| !matches!(message, Message::System { .. }))
                .count();
            if turns > 0 && common < now.messages.len() {
                request.cache_points.push(CachePoint::Messages(turns));
            }
            if previous.tools == now.tools {
                let differs = |section: &&Section| {
                    previous
                        .prompt
                        .section(&section.tag)
                        .is_some_and(|was| was.text != section.text)
                };
                let changing = now.prompt.sections.iter().filter(differs);
                history
                    .volatile
                    .extend(changing.map(|section| section.tag.clone()));
            }
            if let Some(cause) = first_change(previous, &now) {
                let request = history.requests;
                history.breaks.push_back(CacheBreak { request, cause });
                if history.breaks.len() > KEPT_BREAKS {
                    history.breaks.pop_front();
                }
            }
        }
        if !moved.is_empty() {
            let text = SystemPrompt {
                role: String::new(),
                sections: moved,
            }
            .text();
            match conversation {
                Some(conversation) => {
                    let by = Author::plugin("prompt-cache-shaping");
                    let last = conversation.messages().len().saturating_sub(1);
                    match conversation.entry(last) {
                        Some(mut entry) if matches!(*entry.message, Message::User { .. }) => {
                            if let Message::User { content } =
                                std::sync::Arc::make_mut(&mut entry.message)
                            {
                                content.push(UserContent::text(text));
                            }
                            conversation.splice(last.., [entry], by);
                        }
                        _ => {
                            conversation.push(Entry::new(Message::user(text)).by(by));
                        }
                    }
                }
                None => match request
                    .chat_history
                    .last_mut()
                    .map(std::sync::Arc::make_mut)
                {
                    Some(Message::User { content }) => content.push(UserContent::text(text)),
                    _ => request.chat_history.push(Message::user(text).into()),
                },
            }
        }
        history.prefix = Some(now);
    }
}

/// The first part of `now`'s prefix that differs from `was`'s, in the
/// order they are sent.
fn first_change(was: &Prefix, now: &Prefix) -> Option<String> {
    if was.tools != now.tools {
        let named = |tools: &[Arc<ToolDefinition>], name: &str| {
            tools.iter().any(|t| t.name.as_str() == name)
        };
        if let Some(tool) = now
            .tools
            .iter()
            .find(|t| !named(&was.tools, t.name.as_str()))
        {
            return Some(format!("tool list changed: `{}` added", tool.name.as_str()));
        }
        if let Some(tool) = was
            .tools
            .iter()
            .find(|t| !named(&now.tools, t.name.as_str()))
        {
            return Some(format!(
                "tool list changed: `{}` removed",
                tool.name.as_str()
            ));
        }
        let changed = now.tools.iter().zip(&was.tools).find(|(a, b)| a != b);
        let name = changed.map_or("?", |(tool, _)| tool.name.as_str());
        return Some(format!("tool `{name}` changed"));
    }
    if was.prompt.role != now.prompt.role {
        return Some("system prompt changed outside its sections".to_owned());
    }
    // Blank sections are not sent.
    let sent = |prompt: &SystemPrompt| -> Vec<(String, String)> {
        let sections = prompt.sections.iter();
        let sent = sections.filter(|section| !section.text.trim().is_empty());
        sent.map(|section| (section.tag.clone(), section.text.trim().to_owned()))
            .collect()
    };
    let (was_sections, now_sections) = (sent(&was.prompt), sent(&now.prompt));
    for (at, (tag, text)) in now_sections.iter().enumerate() {
        match was_sections.get(at) {
            Some((old, old_text)) if old == tag && old_text != text => {
                return Some(format!("system prompt section `{tag}` changed"));
            }
            Some((old, _)) if old != tag => {
                let added = !was_sections.iter().any(|(was, _)| was == tag);
                let (tag, how) = if added {
                    (tag, "added")
                } else {
                    (old, "removed")
                };
                return Some(format!("system prompt section `{tag}` {how}"));
            }
            None => return Some(format!("system prompt section `{tag}` added")),
            Some(_) => {}
        }
    }
    if let Some((tag, _)) = was_sections.get(now_sections.len()) {
        return Some(format!("system prompt section `{tag}` removed"));
    }
    if let Some(at) = was
        .messages
        .iter()
        .zip(&now.messages)
        .position(|(was, is)| was != is)
    {
        return Some(format!("message {at} changed"));
    }
    (now.messages.len() < was.messages.len())
        .then(|| format!("message {} removed", now.messages.len()))
}

/// Keeps each answered turn call's cache hits.
fn record_usage(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &ForAgent), (With<TurnCall>, With<ModelRequest>)>,
    mut agents: Query<&mut CacheHistory>,
    mut session: ResMut<SessionCache>,
) {
    let Ok((Outcome(Ok(effect::Answered::Completion(reply))), &ForAgent(agent))) =
        calls.get(settled.entity)
    else {
        return;
    };
    let hits = Hits::of(&reply.usage);
    if hits.input == 0 {
        return;
    }
    session.0.add(hits);
    if let Ok(mut history) = agents.get_mut(agent) {
        history.last = Some(hits);
    }
}

/// `cache 92% / 87%`: the agent's last request and the session.
fn show(mut agents: Query<(Ref<CacheHistory>, &mut StatusItems)>, session: Res<SessionCache>) {
    for (history, mut items) in &mut agents {
        if !history.is_changed() && !session.is_changed() {
            continue;
        }
        let Some(last) = history.last.and_then(|last| last.rate()) else {
            continue;
        };
        let all = session.0.rate().unwrap_or(last);
        items.show(RATE.says(format!("cache {last}% / {all}%"), Tone::Dim));
    }
}

/// `/cache`: the hit rates; `/cache why`: the last breaks.
fn cache(
    In(args): In<CommandArgs>,
    agents: Query<&CacheHistory>,
    session: Res<SessionCache>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(history) = agents.get(args.agent) else {
        return;
    };
    let percent = |rate: Option<u64>| rate.map_or("–".to_owned(), |rate| format!("{rate}%"));
    let text = match args.args.trim() {
        "" => format!(
            "Cache hits: last request {}, session {} ({} of {} input tokens read, {} written).",
            percent(history.last.and_then(|last| last.rate())),
            percent(session.0.rate()),
            session.0.read,
            session.0.input,
            session.0.written,
        ),
        "why" => {
            let mut lines: Vec<String> = history
                .breaks
                .iter()
                .map(|cut| format!("request {}: {}", cut.request, cut.cause))
                .collect();
            if lines.is_empty() {
                lines.push("No cache breaks.".to_owned());
            }
            lines.extend(
                history
                    .unmovable
                    .iter()
                    .map(|tag| format!("prompt section `{tag}` changes but could not be moved")),
            );
            lines.join("\n")
        }
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/cache takes nothing or `why`, not `{other}`"),
            ));
            return;
        }
    };
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;
