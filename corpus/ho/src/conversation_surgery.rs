//! Plugin corpus (held-out): H20: Cut messages out of a conversation (spec
//! `H20-conversation-surgery.md`).
//!
//! `/history` lists the focused agent's messages, numbered, with their
//! rough size. `/stub <n>` replaces the content of the tool results in
//! message `n` with a stub naming their size. `/drop <n>` or
//! `/drop <a>-<b>` cuts messages out, taking each tool call and its result
//! together and joining the messages the cut leaves side by side.
//! `/drop undo` puts back what the agent's last edit changed. Every edit
//! is a logged splice, so the session log keeps the edited conversation;
//! it is refused while the agent's turn runs.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::tokens_label;
use rig_harness::rig_core::message::{CallId, Message, ToolResultContent};
use rig_harness::rig_core::serde_json;
use rig_memory::{HeuristicTokenCounter, TokenCounter};

/// Who the edits are recorded as made by.
const PLUGIN: &str = "conversation-surgery";

#[derive(Default)]
pub struct ConversationSurgeryPlugin;

impl Plugin for ConversationSurgeryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LastEdits>()
            .add_command(
                "history",
                "List the agent's messages, numbered, with their size",
                history,
            )
            .add_command(
                "stub",
                "Replace the tool results of message n with a stub: /stub <n>",
                stub,
            )
            .add_command(
                "drop",
                "Cut messages out of the conversation: /drop <n>, /drop <a>-<b>, /drop undo",
                drop_messages,
            );
    }
}

/// The ids of the calls a message makes or answers.
fn call_ids(message: &Message) -> Vec<CallId> {
    match message {
        Message::Assistant(turn) => turn.tool_calls().map(|call| call.id.clone()).collect(),
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some(result.call.clone()),
                _ => None,
            })
            .collect(),
        Message::System { .. } => Vec::new(),
    }
}

/// `entry` without the calls and results of `ids`; `None` when nothing
/// worth sending is left.
fn without(mut entry: Entry, ids: &HashSet<CallId>) -> Option<Entry> {
    let left = match std::sync::Arc::make_mut(&mut entry.message) {
        Message::User { content } => {
            content.retain(
                |item| !matches!(item, UserContent::ToolResult(result) if ids.contains(&result.call)),
            );
            !content.is_empty()
        }
        Message::Assistant(turn) => {
            turn.content.retain(|block| {
                !matches!(block, AssistantContent::ToolCall(call) if ids.contains(&call.id))
            });
            turn.content.iter().any(AssistantContent::is_answer)
        }
        Message::System { .. } => true,
    };
    left.then_some(entry)
}

/// Messages `range` of `conversation`, with their ids and authors.
fn entries(conversation: &Conversation, range: Range<usize>) -> Vec<Entry> {
    range.filter_map(|at| conversation.entry(at)).collect()
}

/// An edit of an agent's conversation, kept for `/drop undo`: the
/// messages it replaced, those it put in, and where they began.
#[derive(Clone, Debug)]
struct Edit {
    at: usize,
    before: Vec<Entry>,
    after: Vec<Entry>,
}

/// Each agent's last edit in this run.
#[derive(Resource, Default)]
struct LastEdits(HashMap<Entity, Edit>);

/// An edit to make: messages `range` become `with`.
struct Plan {
    range: Range<usize>,
    with: Vec<Entry>,
    said: String,
    /// Kept for undo; an undo is not.
    keep: bool,
}

/// What an edit changes: the agent's conversation, whose log keeps it.
#[derive(SystemParam)]
struct Surgery<'w, 's> {
    agents: Query<'w, 's, (&'static mut Conversation, Has<ActiveTurn>)>,
    edits: ResMut<'w, LastEdits>,
    notices: MessageWriter<'w, Notice>,
}

impl Surgery<'_, '_> {
    /// Makes the edit `plan` gives for `agent`'s conversation and its last
    /// edit; refused while a turn runs.
    fn edit(
        &mut self,
        agent: Entity,
        command: &str,
        plan: impl FnOnce(&Conversation, Option<&Edit>) -> Result<Plan, String>,
    ) {
        let Ok((mut conversation, busy)) = self.agents.get_mut(agent) else {
            return;
        };
        if busy {
            let why = format!(
                "/{command}: the agent's turn is running; wait for the turn to end or interrupt it."
            );
            self.notices.write(Notice::error(agent, why));
            return;
        }
        let plan = match plan(&conversation, self.edits.0.get(&agent)) {
            Ok(plan) => plan,
            Err(why) => {
                self.notices
                    .write(Notice::error(agent, format!("/{command}: {why}")));
                return;
            }
        };
        let before = entries(&conversation, plan.range.clone());
        let at = plan.range.start;
        let halted = conversation.halted();
        let spliced = conversation.splice(plan.range, plan.with, Author::plugin(PLUGIN));
        // A splice of the end clears the halt; a user message left last
        // still waits for the next one rather than going to the model on a
        // restore, as the user stopped or kept it.
        conversation.halt(halted.unwrap_or(Halt::Kept));
        let after: Vec<Entry> = spliced
            .changed
            .iter()
            .filter_map(|id| conversation.index_of(*id))
            .filter_map(|at| conversation.entry(at))
            .collect();
        if plan.keep {
            self.edits.0.insert(agent, Edit { at, before, after });
        } else {
            self.edits.0.remove(&agent);
        }
        self.notices.write(Notice::info(agent, plan.said));
    }
}

/// `/history`.
fn history(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(conversation) = agents.get(args.agent) else {
        return;
    };
    let counter = HeuristicTokenCounter::default();
    let mut lines = vec![format!("{} messages:", conversation.messages().len())];
    for (at, message) in conversation.messages().iter().enumerate() {
        let (role, items) = describe(message);
        let tokens = u64::try_from(counter.count(message)).unwrap_or(u64::MAX);
        lines.push(format!(
            "{:>4}  {role:<9}  ~{:<5} {}",
            at + 1,
            tokens_label(tokens),
            items.join(", ")
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// A message's role and what it holds.
fn describe(message: &Message) -> (&'static str, Vec<String>) {
    match message {
        Message::System { .. } => ("system", vec!["text".to_owned()]),
        Message::User { content } => {
            let items = content.iter().map(|item| match item {
                UserContent::Text(_) => "text".to_owned(),
                UserContent::ToolResult(result) => {
                    format!("result of {} ({})", result.name.as_str(), result.call)
                }
                UserContent::Image(_) => "image".to_owned(),
                UserContent::Audio(_) => "audio".to_owned(),
                UserContent::Video(_) => "video".to_owned(),
                UserContent::Document(_) => "document".to_owned(),
            });
            ("user", items.collect())
        }
        Message::Assistant(turn) => {
            let items = turn.content.iter().map(|block| match block {
                AssistantContent::Text(_) => "text".to_owned(),
                AssistantContent::ToolCall(call) => {
                    format!("call {} ({})", call.function.name.as_str(), call.id)
                }
                AssistantContent::Reasoning(_) => "reasoning".to_owned(),
                AssistantContent::Image(_) => "image".to_owned(),
                _ => "provider item".to_owned(),
            });
            ("assistant", items.collect())
        }
    }
}

/// The message numbered `text`, from 1, as an index.
fn number(text: &str, len: usize) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if (1..=len).contains(&n) => Ok(n - 1),
        _ => Err(format!(
            "`{text}` is no message number; the conversation has {len} (see /history)"
        )),
    }
}

/// `n` bytes, rounded.
fn size(bytes: usize) -> String {
    match bytes {
        0..1_000 => format!("{bytes} bytes"),
        1_000..1_000_000 => format!("{} KB", bytes.div_ceil(1_000)),
        _ => format!("{:.1} MB", bytes as f64 / 1e6),
    }
}

/// How a stubbed result begins.
const STUB: &str = "[removed by the user: ";

/// `/stub <n>`.
fn stub(In(args): In<CommandArgs>, mut surgery: Surgery) {
    surgery.edit(args.agent, "stub", |conversation, _| {
        let at = number(&args.args, conversation.messages().len())?;
        let mut entry = conversation.entry(at);
        let Some(Message::User { content }) = entry
            .as_mut()
            .map(|entry| std::sync::Arc::make_mut(&mut entry.message))
        else {
            return Err(format!("message {} holds no tool result", at + 1));
        };
        let mut stubbed = Vec::new();
        for item in content {
            let UserContent::ToolResult(result) = item else {
                continue;
            };
            if let [only] = result.content.as_slice()
                && only.as_text().is_some_and(|text| text.starts_with(STUB))
            {
                continue;
            }
            let bytes = result
                .content
                .iter()
                .map(|content| match content {
                    ToolResultContent::Text(text) => text.text.len(),
                    other => serde_json::to_string(other).map_or(0, |json| json.len()),
                })
                .sum();
            result.content = vec![ToolResultContent::text(format!("{STUB}{}]", size(bytes)))];
            stubbed.push(format!(
                "{} ({}): {}",
                result.name.as_str(),
                result.call,
                size(bytes)
            ));
        }
        if stubbed.is_empty() {
            return Err(format!("message {} holds no tool result to stub", at + 1));
        }
        Ok(Plan {
            range: at..at + 1,
            with: entry.into_iter().collect(),
            said: format!(
                "Stubbed the results in message {}: {}.",
                at + 1,
                stubbed.join(", ")
            ),
            keep: true,
        })
    });
}

/// `/drop <n>`, `/drop <a>-<b>` or `/drop undo`.
fn drop_messages(In(args): In<CommandArgs>, mut surgery: Surgery) {
    let text = args.args.trim();
    surgery.edit(args.agent, "drop", |conversation, last| {
        if text == "undo" {
            return undo(conversation, last);
        }
        let len = conversation.messages().len();
        let (a, b) = match text.split_once('-') {
            Some((a, b)) => (number(a, len)?, number(b, len)?),
            None => (number(text, len)?, number(text, len)?),
        };
        if a > b {
            return Err(format!("{} comes after {}", a + 1, b + 1));
        }
        Ok(plan_drop(conversation, a..b + 1))
    });
}

/// Drops messages `chosen` and every call or result paired with one of
/// theirs, then joins the messages left side by side: two of the model's,
/// or two user messages by the same author (one entry has one author).
fn plan_drop(conversation: &Conversation, chosen: Range<usize>) -> Plan {
    let messages = conversation.messages();
    let ids: HashSet<CallId> = messages
        .get(chosen.clone())
        .unwrap_or_default()
        .iter()
        .flat_map(|message| call_ids(message))
        .collect();
    let paired = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| call_ids(message).iter().any(|id| ids.contains(id)))
        .map(|(at, _)| at);
    let (first, last) = paired.fold((chosen.start, chosen.end), |(first, last), at| {
        (first.min(at), last.max(at + 1))
    });
    // With a neighbour on each side, so what the cut leaves side by side
    // is joined.
    let range = first.saturating_sub(1)..last.saturating_add(1).min(messages.len());
    let mut notes = Vec::new();
    // Each kept message, and the one it joins into.
    let mut kept: Vec<(usize, Entry)> = Vec::new();
    for (at, entry) in entries(conversation, range.clone()).into_iter().enumerate() {
        let at = range.start + at;
        if chosen.contains(&at) {
            continue;
        }
        let was = entry.clone();
        let Some(mut entry) = without(entry, &ids) else {
            notes.push(format!(
                "removed message {}, whose calls or results went with them",
                at + 1
            ));
            continue;
        };
        if entry != was {
            notes.push(format!(
                "took the paired calls or results out of message {}",
                at + 1
            ));
        }
        let Some((into, last)) = kept.last_mut() else {
            kept.push((at, entry));
            continue;
        };
        match (
            std::sync::Arc::make_mut(&mut last.message),
            std::sync::Arc::make_mut(&mut entry.message),
        ) {
            (Message::User { content }, Message::User { content: more })
                if last.author == entry.author =>
            {
                notes.push(format!(
                    "joined message {} to message {}",
                    at + 1,
                    *into + 1
                ));
                content.append(more);
            }
            (Message::Assistant(turn), Message::Assistant(more)) => {
                notes.push(format!(
                    "joined message {} to message {}",
                    at + 1,
                    *into + 1
                ));
                if turn.origin != more.origin {
                    turn.origin = None;
                }
                turn.content.append(&mut more.content);
                turn.stop = more.stop.take();
            }
            _ => kept.push((at, entry)),
        }
    }
    let span = if chosen.len() == 1 {
        format!("message {}", chosen.end)
    } else {
        format!("messages {}-{}", chosen.start + 1, chosen.end)
    };
    let mut said = format!("Dropped {span}.");
    for note in notes {
        said.push_str(&format!(" Also {note}."));
    }
    said.push_str(" /drop undo puts it back.");
    Plan {
        range,
        with: kept.into_iter().map(|(_, entry)| entry).collect(),
        said,
        keep: true,
    }
}

/// Puts back what the agent's last edit changed.
fn undo(conversation: &Conversation, last: Option<&Edit>) -> Result<Plan, String> {
    let Some(edit) = last else {
        return Err("nothing to undo in this run".to_owned());
    };
    let start = edit
        .after
        .first()
        .and_then(|entry| entry.id)
        .map_or(Some(edit.at), |id| conversation.index_of(id));
    let range = start.map(|start| start..start + edit.after.len());
    let Some(range) = range.filter(|range| entries(conversation, range.clone()) == edit.after)
    else {
        return Err("the edited messages changed since; nothing to undo".to_owned());
    };
    Ok(Plan {
        range,
        with: edit.before.clone(),
        said: "Put back the messages the last edit changed.".to_owned(),
        keep: false,
    })
}
