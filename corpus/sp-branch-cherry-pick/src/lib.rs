//! SP-13: cherry-pick turns between branches.
//!
//! `/cherry-pick` opens a picker of the turns kept in the agent's other
//! branches (rig-core's `rig.branch` asides, so not on the active line),
//! grouped by branch, each showing its branch's title, its first prompt
//! line and where the branch's leaf is. One turn picked, or a run of turns
//! of one branch ticked, is copied to the end of the active line: first a
//! note telling the model the exchange comes from another attempt whose
//! file changes were not applied, then each message, its tool calls with
//! fresh ids answered by their copied results. Nothing runs again and no
//! file changes; the source branch is only read. A note in the transcript
//! marks each copied turn `picked from <branch>`, and an aside after the
//! copy records it, so `/cherry-pick undo` takes the last pick back while
//! nothing came after it, after a restart too.

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{self, Kept};
use rig_harness::rig_core::message::{CallId, LocalCallId, Message as Said};
use rig_harness::rig_memory::{COMPACTED, summary_author};

/// The command's name, and the author of what it adds.
pub const NAME: &str = "cherry-pick";

/// The kind of the aside recording a pick, after its last copied entry:
/// `{"branch": title, "entries": [ids]}`, the note's id first.
pub const PICK: &str = "cherry-pick.pick";

/// `/cherry-pick`, `/cherry-pick undo` and `/cherry-pick take <branch>
/// <first> <last>`, which the picker runs.
#[derive(Default)]
pub struct CherryPickPlugin;

impl Plugin for CherryPickPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            NAME,
            "Copy turns from another branch to the end of this one: /cherry-pick [undo]",
            cherry_pick,
        );
    }
}

/// A turn of a kept branch that the picker offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The kept branch's aside.
    pub branch: AsideId,
    /// The branch's title.
    pub title: String,
    /// The turn's first entry.
    pub first: MessageId,
    /// The turn's last entry.
    pub last: MessageId,
    /// The first line of the turn's prompt.
    pub prompt: String,
    /// How many entries the branch's line has: where its leaf is.
    pub leaf: usize,
}

/// On a picker of turns: what each of its items stands for.
#[derive(Component)]
struct Candidates(Vec<Candidate>);

/// What a pick did or why it was refused.
type Done = Result<String, String>;

/// Runs `/cherry-pick` and its subcommands for the agent.
fn cherry_pick(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let Ok((mut conversation, running)) = agents.get_mut(agent) else {
        notices.write(Notice::error(agent, "That agent has no conversation."));
        return;
    };
    if running {
        let why = "Picking waits for the running turn to end; stop it first.";
        notices.write(Notice::error(agent, why));
        return;
    }
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let done: Done = match words.as_slice() {
        [] => open_picker(&conversation, agent, &mut commands),
        ["undo"] => undo(&mut conversation),
        ["take", branch, first, last] => parse_take(branch, first, last)
            .and_then(|(branch, first, last)| take(&mut conversation, branch, first, last)),
        _ => Err("Usage: /cherry-pick [undo]".to_owned()),
    };
    match done {
        Ok(text) if text.is_empty() => {}
        Ok(text) => {
            notices.write(Notice::info(agent, text));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Reads `take`'s arguments.
fn parse_take(
    branch: &str,
    first: &str,
    last: &str,
) -> Result<(AsideId, MessageId, MessageId), String> {
    let bad = |what: &str| format!("{what} is not an id");
    Ok((
        branch.parse().map_err(|_| bad(branch))?,
        first.parse().map_err(|_| bad(first))?,
        last.parse().map_err(|_| bad(last))?,
    ))
}

/// Opens the picker of every kept branch's turns, by branch.
fn open_picker(conversation: &Conversation, agent: Entity, commands: &mut Commands) -> Done {
    let candidates = candidates(conversation);
    if candidates.is_empty() {
        return Err("No other branch has a turn to pick.".to_owned());
    }
    let body = "Enter picks one turn; Tab ticks a run of turns of one branch.";
    let mut picker = Picker::new("Copy turns from another branch")
        .for_agent(agent)
        .body(body)
        .multiple();
    for candidate in &candidates {
        let label = format!("[{}] {}", candidate.title, candidate.prompt);
        picker = picker.item(label, "");
        if let Some(item) = picker.items.last_mut() {
            item.detail = Some(format!("leaf at #{}", candidate.leaf));
        }
    }
    commands
        .spawn((picker, Candidates(candidates)))
        .observe(picked);
    Ok(String::new())
}

/// Runs `take` for the turns picked: one, or a run of one branch's.
fn picked(
    picked: On<Picked>,
    pickers: Query<(&Picker, &Candidates)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let (Some(chosen), Ok((picker, Candidates(candidates)))) =
        (&picked.chosen, pickers.get(picked.entity))
    else {
        return;
    };
    let Some(agent) = picker.agent else {
        return;
    };
    let mut items = chosen.items.clone();
    items.sort_unstable();
    items.dedup();
    let run: Vec<&Candidate> = items.iter().filter_map(|at| candidates.get(*at)).collect();
    let (Some(first), Some(last)) = (run.first(), run.last()) else {
        return;
    };
    let one_branch = run.iter().all(|turn| turn.branch == first.branch);
    let in_a_row = items
        .windows(2)
        .all(|pair| matches!(pair, [a, b] if b.checked_sub(*a) == Some(1)));
    if !one_branch || !in_a_row {
        let why = "Pick one turn, or turns in a row of one branch.";
        notices.write(Notice::error(agent, why));
        return;
    }
    let line = format!("{NAME} take {} {} {}", first.branch, first.first, last.last);
    commands.trigger(RunCommand::new(agent, line));
}

/// The turns of every kept branch, by branch: the entries of each that the
/// active line does not hold, from each prompt to the next. Entries before
/// a branch's first prompt make a turn of their own.
pub fn candidates(conversation: &conversation::Conversation) -> Vec<Candidate> {
    let active: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    let mut found = Vec::new();
    for kept in conversation.branches() {
        let own = own_entries(&kept, &active);
        let leaf = kept.at.saturating_add(kept.branch.entries.len());
        for turn in turns(&own) {
            let (Some(first), Some(last)) = (turn.first(), turn.last()) else {
                continue;
            };
            let (Some(first_id), Some(last_id)) = (first.id, last.id) else {
                continue;
            };
            found.push(Candidate {
                branch: kept.id,
                title: kept.branch.title.clone(),
                first: first_id,
                last: last_id,
                prompt: turn
                    .iter()
                    .find_map(|entry| prompt_line(entry))
                    .unwrap_or_else(|| "(a turn's continuation)".to_owned()),
                leaf,
            });
        }
    }
    found
}

/// The entries of `kept` that are its own and not on the active line.
fn own_entries(kept: &Kept, active: &[MessageId]) -> Vec<conversation::Entry> {
    let entries = kept.branch.entries.iter();
    let own = entries.filter(|entry| entry.id.is_none_or(|id| !active.contains(&id)));
    own.cloned().collect()
}

/// `entries` cut before each prompt.
fn turns(entries: &[conversation::Entry]) -> Vec<&[conversation::Entry]> {
    let mut turns = Vec::new();
    let mut start = 0;
    for (at, entry) in entries.iter().enumerate() {
        if at > start && prompt_line(entry).is_some() {
            turns.extend(entries.get(start..at));
            start = at;
        }
    }
    turns.extend(entries.get(start..).filter(|rest| !rest.is_empty()));
    turns
}

/// The first line of `entry` when it is a prompt: a user message with text
/// by the user or an agent, not a plugin or rig.
fn prompt_line(entry: &conversation::Entry) -> Option<String> {
    let asked = entry
        .author
        .as_ref()
        .is_none_or(|author| matches!(author.kind, AuthorKind::User | AuthorKind::Agent));
    let Said::User { content } = &entry.message else {
        return None;
    };
    let mut texts = content.iter().filter_map(|item| match item {
        UserContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    });
    let text = texts.next().filter(|_| asked)?;
    let line = text.lines().map(str::trim).find(|line| !line.is_empty());
    Some(line.unwrap_or_default().to_owned())
}

/// Copies the entries `first..=last` of kept branch `branch` to the end of
/// the active line, after the note; what it did.
fn take(
    conversation: &mut Conversation,
    branch: AsideId,
    first: MessageId,
    last: MessageId,
) -> Done {
    let kept = conversation
        .branches()
        .into_iter()
        .find(|kept| kept.id == branch);
    let kept = kept.ok_or_else(|| format!("There is no kept branch {branch}."))?;
    let active: Vec<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    let own = own_entries(&kept, &active);
    let at = |id: MessageId| own.iter().position(|entry| entry.id == Some(id));
    let (Some(start), Some(end)) = (at(first), at(last)) else {
        return Err(format!(
            "Those turns are not branch {}'s own.",
            kept.branch.title
        ));
    };
    let range = own.get(start..=end).unwrap_or_default();
    if range.is_empty() {
        return Err("Nothing to pick there.".to_owned());
    }
    let ids: Vec<MessageId> = range.iter().filter_map(|entry| entry.id).collect();
    let summary = summary_author();
    let summarized = range
        .iter()
        .any(|entry| entry.author.as_ref() == Some(&summary));
    let compacted = kept.branch.asides.iter().any(|(_, aside)| {
        aside.kind == COMPACTED && aside.after.is_some_and(|after| ids.contains(&after))
    });
    if summarized || compacted {
        return Err("Those turns include a compaction point; pick turns after it.".to_owned());
    }
    let copies = copy(range);
    let messages: Vec<Said> = copies.iter().map(|entry| entry.message.clone()).collect();
    if conversation::pairing::validate(&messages).is_err()
        || !matches!(messages.last(), Some(Said::Assistant(_)))
    {
        return Err("Those turns end without their reply or a call's result.".to_owned());
    }
    if conversation.awaits_model() || !conversation.pending_calls().is_empty() {
        return Err("This branch waits for an answer; pick once it has one.".to_owned());
    }
    let title = kept.branch.title.clone();
    let by = Author::plugin(NAME);
    let note = Entry::new(Said::user(note_text(&title))).by(by.clone());
    let from = conversation.messages().len();
    let count = copies.len().saturating_add(1);
    conversation.splice(
        from..,
        std::iter::once(note).chain(copies.clone()),
        by.clone(),
    );
    let new: Vec<MessageId> = (from..from.saturating_add(count))
        .filter_map(|at| conversation.id(at))
        .collect();
    for (copy, id) in copies.iter().zip(new.iter().skip(1)) {
        if prompt_line(copy).is_some() || Some(id) == new.get(1) {
            let mark = Note::new(
                format!("picked from {title}"),
                format!("Copied from branch {title} ({branch}); nothing was run again."),
            );
            conversation.add_aside(mark.aside(by.clone(), Some(*id)));
        }
    }
    let record = serde_json::json!({ "branch": title, "entries": new });
    conversation.add_aside(Aside::new(PICK, by, record).after(new.last().copied()));
    Ok(format!(
        "Picked {} messages from {title}.",
        count.saturating_sub(1)
    ))
}

/// What the model reads before the copies.
fn note_text(title: &str) -> String {
    format!(
        "The messages that follow are copied from an alternative attempt (the branch \
         \"{title}\" of this conversation). Their file changes were NOT applied to the \
         working tree: the files are as they are now, not as those messages leave them. \
         No tool was run again."
    )
}

/// Copies of `entries` as new entries: no ids, each tool call with a fresh
/// id its result follows, each reply without what binds it to the
/// provider call that made it (reasoning, provider items and their
/// origin), so any provider reads it. A reply left empty goes.
fn copy(entries: &[conversation::Entry]) -> Vec<Entry> {
    let mut ids: Vec<(CallId, CallId)> = Vec::new();
    let mut fresh = |old: &CallId| -> CallId {
        if let Some((_, new)) = ids.iter().find(|(known, _)| known == old) {
            return new.clone();
        }
        let new = CallId::Local(LocalCallId::new());
        ids.push((old.clone(), new.clone()));
        new
    };
    let mut copies = Vec::new();
    for entry in entries {
        let message = match &entry.message {
            Said::Assistant(reply) => {
                let mut content = Vec::new();
                for block in &reply.content {
                    match block.canonical() {
                        AssistantContent::Reasoning(_) | AssistantContent::Opaque(_) => {}
                        AssistantContent::ToolCall(mut call) => {
                            call.id = fresh(&call.id);
                            content.push(AssistantContent::ToolCall(call));
                        }
                        other => content.push(other),
                    }
                }
                if content.is_empty() {
                    continue;
                }
                let mut copied = AssistantMessage::new(content);
                copied.stop = reply.stop.clone();
                Said::Assistant(copied)
            }
            Said::User { content } => {
                let content = content.iter().cloned().map(|item| match item {
                    UserContent::ToolResult(mut result) => {
                        result.call = fresh(&result.call);
                        UserContent::ToolResult(result)
                    }
                    other => other,
                });
                Said::User {
                    content: content.collect(),
                }
            }
            other => other.clone(),
        };
        copies.push(Entry {
            message,
            author: entry.author.clone(),
            id: None,
            made: None,
        });
    }
    copies
}

/// Takes the last pick out, while nothing came after it.
fn undo(conversation: &mut Conversation) -> Done {
    let by = Author::plugin(NAME);
    let record = conversation
        .asides()
        .rev()
        .find(|(_, aside)| aside.kind == PICK && aside.author == by)
        .map(|(id, aside)| (id, aside.value.clone()));
    let Some((record, value)) = record else {
        return Err("There is no pick to undo on this branch.".to_owned());
    };
    let picked: Vec<MessageId> = value
        .get("entries")
        .and_then(|ids| serde_json::from_value(ids.clone()).ok())
        .unwrap_or_default();
    let len = conversation.messages().len();
    let from = len.saturating_sub(picked.len());
    let tail: Vec<MessageId> = (from..len).filter_map(|at| conversation.id(at)).collect();
    if picked.is_empty() || tail != picked {
        return Err("Something was added after the last pick; it stays.".to_owned());
    }
    let marks: Vec<AsideId> = conversation
        .asides()
        .filter(|(_, aside)| aside.author == by)
        .filter(|(_, aside)| aside.after.is_some_and(|after| picked.contains(&after)))
        .map(|(id, _)| id)
        .collect();
    for id in marks.into_iter().chain([record]) {
        conversation.remove_aside(id);
    }
    conversation.splice(from.., [], by);
    Ok(format!(
        "Took back the last pick ({} messages).",
        picked.len()
    ))
}

#[cfg(test)]
mod tests;
