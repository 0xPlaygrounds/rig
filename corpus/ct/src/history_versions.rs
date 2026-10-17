//! CT-11: undo for conversation changes. Every change to an agent's
//! conversation that is not simply appending (a cut, a summary, an edit, a
//! reorder, a branch switch), by any plugin or command,
//! is kept as a numbered [`Version`] with its time, its author, a one-line
//! description, and the conversation as it was before it. The versions
//! are saved with the session ([`ConvVersions`]) and trimmed to the
//! `[conv-history]` table's `keep` count and `max_age_days`, oldest first.
//!
//! - `/conv-log` lists them newest first, with message and token counts
//!   before and after.
//! - `/conv-diff <n>` shows what version `n` removed, added and replaced.
//! - `/conv-restore <n>` makes the conversation what it was just before
//!   version `n`, then puts back on top, verbatim, the entries appended
//!   since; the restore is a version too.
//!
//! The changes are read from the kernel's [`ConversationChanged`] messages. A splice
//! names the ids it removed, not their messages, so the plugin keeps a copy
//! of each conversation ([`Shadow`]) folded from the same messages, and
//! takes the "before" from it.

use std::collections::{HashMap, HashSet};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::conversation::Conversation as Chat;
use rig_harness::rig_ecs::WriteLogs;
use rig_harness::rig_memory::{HeuristicTokenCounter, TokenCounter, summary_in};
use serde::{Deserialize, Serialize};

/// The plugin's name: its settings table and the author of its restores.
pub const NAME: &str = "conv-history";

/// Keeps a history of every rewrite of each agent's conversation, with
/// `/conv-log`, `/conv-diff` and `/conv-restore`.
#[derive(Default)]
pub struct HistoryVersionsPlugin;

impl Plugin for HistoryVersionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<HistorySettings>(NAME)
            .register_required_components::<Agent, ConvVersions>()
            .keeps_saved::<ConvVersions>()
            .add_systems(Last, (record, prune).chain().after(WriteLogs))
            .add_command("conv-log", "List the versions of the conversation, newest first", log)
            .add_command("conv-diff", "What version N of the conversation changed: /conv-diff N", diff)
            .add_command(
                "conv-restore",
                "Make the conversation what it was before version N, keeping turns appended since: /conv-restore N",
                restore,
            );
    }
}

/// The `[conv-history]` table: how many versions are kept (`keep`, 50 by
/// default) and for how long (`max_age_days`, forever by default).
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct HistorySettings {
    /// The most versions kept per agent; older ones go first.
    pub keep: usize,
    /// Versions older than this many days go.
    pub max_age_days: Option<u64>,
}

impl Default for HistorySettings {
    fn default() -> Self {
        Self {
            keep: 50,
            max_age_days: None,
        }
    }
}

/// One rewrite of a conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Version {
    /// Its number, rising, never reused.
    pub n: u32,
    /// When it was made, in milliseconds since the Unix epoch.
    pub at: u64,
    /// Who made it.
    pub by: Author,
    /// What it did, in one line.
    pub description: String,
    /// Messages before it and after it.
    pub counts: (usize, usize),
    /// Estimated tokens before it and after it.
    pub tokens: (usize, usize),
    /// The conversation's entries just before it.
    pub before: Vec<Entry>,
    /// The range of `before` it replaced.
    pub replaced: (usize, usize),
    /// What it put there.
    pub with: Vec<Entry>,
    /// The ids it made: entries new with it, such as a summary.
    pub made: Vec<MessageId>,
    /// The highest id the conversation had made once it was made; an
    /// entry with a higher one came later.
    pub last_id: Option<MessageId>,
}

/// An agent's versions, oldest first, saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct ConvVersions {
    /// The versions kept.
    pub versions: Vec<Version>,
    /// The number of the newest version made; the first is 1.
    pub next: u32,
    /// The highest message id seen so far.
    pub last_id: Option<MessageId>,
    /// What the restore in progress is, for its version's description.
    #[serde(skip)]
    pending: Option<String>,
}

/// The plugin's copy of an agent's conversation, folded from its
/// [`ConversationChanged`] changes, as it was before each.
#[derive(Component, Debug)]
pub struct Shadow(Chat);

/// An agent as recording reads it.
type Recording<'a> = (
    Entity,
    Ref<'a, Conversation>,
    &'a mut ConvVersions,
    Option<&'a mut Shadow>,
);

/// Folds each agent's saved changes into its shadow, keeping a version of
/// each that is not an append.
fn record(
    mut committed: MessageReader<ConversationChanged>,
    mut agents: Query<Recording>,
    mut commands: Commands,
) {
    let mut changes: HashMap<Entity, Vec<&ChangeRecord>> = HashMap::new();
    for ConversationChanged { agent, change } in committed.read() {
        changes.entry(*agent).or_default().push(change);
    }
    for (agent, conversation, mut versions, shadow) in &mut agents {
        let Some(mut shadow) = shadow else {
            if conversation.unsaved().is_empty() {
                versions.last_id = versions.last_id.max(highest(&conversation));
                commands.entity(agent).insert(Shadow(conversation.view()));
            }
            continue;
        };
        for change in changes.remove(&agent).unwrap_or_default() {
            fold(&mut shadow.0, &mut versions, change);
        }
        let settled = conversation.is_changed() && conversation.unsaved().is_empty();
        if settled && !same_entries(&shadow.0, &conversation) {
            // A restore or a reread brings a conversation back unannounced.
            versions.last_id = versions.last_id.max(highest(&conversation));
            shadow.0 = conversation.view();
        }
    }
}

/// Applies `change` to `shadow`, first keeping a version of it when it
/// rewrites rather than appends.
fn fold(shadow: &mut Chat, versions: &mut ConvVersions, change: &ChangeRecord) {
    let len = shadow.messages().len();
    let rewrites = match &change.change {
        Change::Splice { from, removed, .. } => !(removed.is_empty() && *from == len),
        Change::Reset { state } => !same_entries(shadow, state),
        Change::Push { .. } | Change::Halt { .. } | Change::Aside { .. } => false,
    };
    let before: Vec<Entry> = (0..len).filter_map(|at| shadow.entry(at)).collect();
    let seen = versions.last_id;
    if shadow.apply(change).is_err() {
        return;
    }
    versions.last_id = versions.last_id.max(highest(shadow));
    if !rewrites {
        return;
    }
    let (replaced, with) = match &change.change {
        Change::Splice { from, to, with, .. } => ((*from, *to), with.clone()),
        _ => (
            (0, len),
            (0..shadow.messages().len())
                .filter_map(|at| shadow.entry(at))
                .collect(),
        ),
    };
    let by = change.by.clone().unwrap_or_else(Author::rig);
    let mine = by.kind == AuthorKind::Plugin(NAME.to_owned());
    let pending = versions.pending.take().filter(|_| mine);
    let described = pending.unwrap_or_else(|| describe(&change.change, &before, replaced, &with));
    let made = with
        .iter()
        .filter_map(|entry| entry.id)
        .filter(|id| Some(*id) > seen)
        .collect();
    let version = Version {
        n: versions.next,
        at: change.at.unwrap_or_else(now_ms),
        by,
        description: described,
        counts: (before.len(), shadow.messages().len()),
        tokens: (
            tokens(before.iter().map(|entry| &*entry.message)),
            tokens(shadow.messages().iter().map(|message| &**message)),
        ),
        before,
        replaced,
        with,
        made,
        last_id: versions.last_id,
    };
    push(versions, version);
}

/// Keeps `version`, numbering it.
fn push(versions: &mut ConvVersions, mut version: Version) {
    versions.next = versions.next.saturating_add(1);
    version.n = versions.next;
    versions.versions.push(version);
}

/// Drops versions beyond the settings' count and age, oldest first.
fn prune(mut agents: Query<&mut ConvVersions>, settings: Res<HistorySettings>) {
    let day_ms = 24 * 60 * 60 * 1000;
    let oldest = settings
        .max_age_days
        .map(|days| now_ms().saturating_sub(days.saturating_mul(day_ms)));
    for mut versions in &mut agents {
        let old = versions
            .versions
            .iter()
            .take_while(|version| oldest.is_some_and(|oldest| version.at < oldest))
            .count();
        let over = versions.versions.len().saturating_sub(settings.keep);
        let drop = old.max(over);
        if drop > 0 {
            versions.versions.drain(..drop);
        }
    }
}

/// Whether `a` and `b` hold the same messages with the same ids.
fn same_entries(a: &Chat, b: &Chat) -> bool {
    a.messages() == b.messages()
        && a.entries()
            .map(|entry| entry.id)
            .eq(b.entries().map(|entry| entry.id))
}

/// The highest id among `conversation`'s entries.
fn highest(conversation: &Chat) -> Option<MessageId> {
    conversation.entries().map(|entry| entry.id).max()
}

/// The estimated tokens of `messages`.
fn tokens<'m>(messages: impl IntoIterator<Item = &'m Message>) -> usize {
    let counter = HeuristicTokenCounter::default();
    messages
        .into_iter()
        .map(|message| counter.count(message))
        .sum()
}

/// One line on what `change` did to `before`.
fn describe(
    change: &Change,
    before: &[Entry],
    (from, to): (usize, usize),
    with: &[Entry],
) -> String {
    if matches!(change, Change::Reset { .. }) {
        return format!(
            "replaced the whole conversation ({} → {} messages)",
            before.len(),
            with.len()
        );
    }
    let old = before.get(from..to).unwrap_or_default();
    let old_ids: HashSet<MessageId> = old.iter().filter_map(|entry| entry.id).collect();
    let new_ids: HashSet<MessageId> = with.iter().filter_map(|entry| entry.id).collect();
    if summary_in(change).is_some() {
        let condensed = old_ids.difference(&new_ids).count();
        return format!("compaction: condensed {condensed} message(s) into a summary");
    }
    let removed = old_ids.difference(&new_ids).count();
    let added = new_ids.difference(&old_ids).count();
    let edited = old
        .iter()
        .filter(|was| {
            with.iter()
                .any(|is| is.id == was.id && is.message != was.message)
        })
        .count();
    let order = |entries: &[Entry]| {
        entries
            .iter()
            .filter_map(|entry| entry.id)
            .collect::<Vec<_>>()
    };
    let kept = |ids: Vec<MessageId>, other: &HashSet<MessageId>| -> Vec<MessageId> {
        ids.into_iter().filter(|id| other.contains(id)).collect()
    };
    let reordered = kept(order(old), &new_ids) != kept(order(with), &old_ids);
    let mut parts = Vec::new();
    if removed > 0 {
        parts.push(format!("removed {removed}"));
    }
    if added > 0 {
        parts.push(format!("added {added}"));
    }
    if edited > 0 {
        parts.push(format!("edited {edited}"));
    }
    if reordered {
        parts.push("reordered".to_owned());
    }
    if parts.is_empty() {
        return "rewrote messages without changing them".to_owned();
    }
    format!("{} message(s)", parts.join(", "))
}

/// A message, abbreviated to a line.
fn preview(message: &Message) -> String {
    let text = match message {
        Message::System { content } => format!("system: {content}"),
        Message::User { content } => {
            let text = content.iter().find_map(|part| match part {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            });
            format!(
                "user: {}",
                text.unwrap_or_else(|| "[tool results]".to_owned())
            )
        }
        Message::Assistant(assistant) => {
            let calls: Vec<&str> = assistant
                .tool_calls()
                .map(|call| call.function.name.as_str())
                .collect();
            match assistant.text() {
                text if !text.is_empty() => format!("assistant: {text}"),
                _ => format!("assistant: [calls {}]", calls.join(", ")),
            }
        }
    };
    let line = text.lines().next().unwrap_or_default();
    match line.char_indices().nth(60) {
        Some((at, _)) => format!("{}…", line.get(..at).unwrap_or(line)),
        None => line.to_owned(),
    }
}

/// How long ago `at` was.
fn ago(at: u64) -> String {
    let secs = now_ms().saturating_sub(at) / 1000;
    match secs {
        0..60 => format!("{secs}s ago"),
        60..3600 => format!("{}m ago", secs / 60),
        3600..86_400 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86_400),
    }
}

/// `/conv-log`.
fn log(
    In(args): In<CommandArgs>,
    agents: Query<&ConvVersions>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(versions) = agents.get(args.agent) else {
        return;
    };
    if versions.versions.is_empty() {
        notices.write(Notice::info(
            args.agent,
            "No versions: the conversation was only appended to.",
        ));
        return;
    }
    let mut lines = vec![format!(
        "{} version(s), newest first:",
        versions.versions.len()
    )];
    for version in versions.versions.iter().rev() {
        let (before, after) = version.counts;
        let (tokens_before, tokens_after) = version.tokens;
        lines.push(format!(
            "#{} {} by {}: {}; {before} → {after} messages, ~{tokens_before} → ~{tokens_after} tokens",
            version.n,
            ago(version.at),
            version.by.label(),
            version.description,
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// The version numbered by `args`, or a notice why not.
fn find<'v>(
    args: &CommandArgs,
    versions: &'v ConvVersions,
    notices: &mut MessageWriter<Notice>,
) -> Option<&'v Version> {
    let n = args.args.trim().parse::<u32>().ok();
    let found = n.and_then(|n| versions.versions.iter().find(|version| version.n == n));
    if found.is_none() {
        let text = format!("No version {:?}; /conv-log lists them.", args.args.trim());
        notices.write(Notice::error(args.agent, text));
    }
    found
}

/// `/conv-diff N`.
fn diff(
    In(args): In<CommandArgs>,
    agents: Query<&ConvVersions>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(versions) = agents.get(args.agent) else {
        return;
    };
    let Some(version) = find(&args, versions, &mut notices) else {
        return;
    };
    let mut lines = vec![format!(
        "#{} by {}: {}",
        version.n,
        version.by.label(),
        version.description
    )];
    let (from, to) = version.replaced;
    let old = version.before.get(from..to).unwrap_or_default();
    for was in old {
        match version.with.iter().find(|is| is.id == was.id) {
            None => lines.push(format!("- {}", preview(&was.message))),
            Some(is) if is.message != was.message => {
                lines.push(format!("~ {}", preview(&was.message)));
                lines.push(format!("  → {}", preview(&is.message)));
            }
            Some(_) => {}
        }
    }
    for is in &version.with {
        if !old.iter().any(|was| was.id == is.id) {
            lines.push(format!("+ {}", preview(&is.message)));
        }
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// `/conv-restore N`: the entries before version `N`, then those appended
/// since, as they are now.
fn restore(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, &mut ConvVersions), Without<ActiveTurn>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((mut conversation, mut versions)) = agents.get_mut(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "/conv-restore waits until the turn ends.",
        ));
        return;
    };
    let Some(version) = find(&args, &versions, &mut notices) else {
        return;
    };
    let later = versions.versions.iter().filter(|later| later.n > version.n);
    let made_later: HashSet<MessageId> =
        later.flat_map(|later| later.made.iter().copied()).collect();
    let appended = conversation
        .entries()
        .filter(|entry| Some(entry.id) > version.last_id && !made_later.contains(&entry.id))
        .filter_map(|entry| conversation.entry(entry.index))
        .collect::<Vec<_>>();
    let entries = version.before.iter().cloned().chain(appended).collect();
    let n = version.n;
    versions.pending = Some(format!("restored the conversation to before version {n}"));
    let by = Author::plugin(NAME).titled(format!("/conv-restore {n}"));
    let spliced = conversation.set_entries(entries, by);
    if spliced == Spliced::default() {
        versions.pending = None;
        notices.write(Notice::info(
            args.agent,
            format!("The conversation is already as before version {n}."),
        ));
    } else {
        notices.write(Notice::info(
            args.agent,
            format!("Restored to before version {n}."),
        ));
    }
}

#[cfg(test)]
mod tests;
