//! Plugin corpus (held-out): H20: Cut messages out of a conversation (spec
//! `H20-conversation-surgery.md`).
//!
//! `/history` lists the focused agent's messages, numbered, with their
//! rough size. `/stub <n>` replaces the content of the tool results in
//! message `n` with a stub naming their size. `/drop <n>` or
//! `/drop <a>-<b>` cuts messages out, taking each tool call and its result
//! together and joining the messages the cut leaves side by side.
//! `/drop undo` puts back what the agent's last edit changed. An edit
//! cuts the conversation back to its first changed message and commits
//! the rest again, so the session log keeps the edited conversation; it
//! is refused while the agent's turn runs.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Usage, tokens_label};
use rig_harness::rig_core::message::{CallId, Message, ToolResultContent};
use rig_harness::rig_core::serde_json;
use rig_harness::rig_ecs::agent::Halt;
use rig_memory::{HeuristicTokenCounter, TokenCounter};

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

/// A message as it is edited: a user message keeps where each of its
/// items came from.
#[derive(Clone, PartialEq, Debug)]
enum Piece {
    User(Vec<(UserContent, Option<Origin>)>),
    Other(Message),
}

impl Piece {
    /// The piece without the calls and results of `ids`; `None` when
    /// nothing worth sending is left.
    fn without(self, ids: &HashSet<CallId>) -> Option<Self> {
        match self {
            Self::User(mut items) => {
                items.retain(|(item, _)| {
                    !matches!(item, UserContent::ToolResult(result) if ids.contains(&result.call))
                });
                (!items.is_empty()).then_some(Self::User(items))
            }
            Self::Other(Message::Assistant(mut turn)) => {
                turn.content.retain(|block| {
                    !matches!(block, AssistantContent::ToolCall(call) if ids.contains(&call.id))
                });
                let answers = turn.content.iter().any(AssistantContent::is_answer);
                answers.then_some(Self::Other(Message::Assistant(turn)))
            }
            other => Some(other),
        }
    }

    /// Joins `next` into this piece when both are the user's or the
    /// model's, else hands it back.
    fn join(&mut self, next: Self) -> Option<Self> {
        match (self, next) {
            (Self::User(items), Self::User(more)) => {
                items.extend(more);
                None
            }
            (Self::Other(Message::Assistant(turn)), Self::Other(Message::Assistant(more))) => {
                if turn.origin != more.origin {
                    turn.origin = None;
                }
                turn.content.extend(more.content);
                turn.stop = more.stop;
                None
            }
            (_, next) => Some(next),
        }
    }

    fn is_user(&self) -> bool {
        matches!(self, Self::User(_))
    }
}

/// The messages of `conversation` from `from` on.
fn pieces(conversation: &Conversation, from: usize) -> Vec<Piece> {
    let messages = conversation.messages().iter().enumerate().skip(from);
    messages
        .map(|(at, message)| match message {
            Message::User { content } => Piece::User(
                content
                    .iter()
                    .enumerate()
                    .map(|(item, content)| {
                        (content.clone(), conversation.origin(at, item).cloned())
                    })
                    .collect(),
            ),
            other => Piece::Other(other.clone()),
        })
        .collect()
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

fn estimate<'a>(messages: impl IntoIterator<Item = &'a Message>) -> u64 {
    let counter = HeuristicTokenCounter::default();
    let tokens: usize = messages
        .into_iter()
        .map(|message| counter.count(message))
        .sum();
    u64::try_from(tokens).unwrap_or(u64::MAX)
}

/// An edit of an agent's conversation, kept for `/drop undo`: its
/// messages from `from` on before and after it.
#[derive(Clone, Debug)]
struct Edit {
    from: usize,
    before: Vec<Piece>,
    after: Vec<Piece>,
}

/// Each agent's last edit in this run.
#[derive(Resource, Default)]
struct LastEdits(HashMap<Entity, Edit>);

/// An edit to make: the messages from `from` on become `after`.
struct Plan {
    from: usize,
    after: Vec<Piece>,
    said: String,
    /// Kept for undo; an undo is not.
    keep: bool,
}

/// What an edit changes: the agent's conversation and its last usage,
/// through the commit that logs it.
#[derive(SystemParam)]
struct Surgery<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            &'static mut Conversation,
            Option<&'static Condensed>,
            Has<ActiveTurn>,
            Option<&'static mut LastUsage>,
        ),
    >,
    commit: Commit<'w, 's>,
    edits: ResMut<'w, LastEdits>,
    notices: MessageWriter<'w, Notice>,
}

impl Surgery<'_, '_> {
    /// Makes the edit `plan` gives for `agent`'s conversation, from its
    /// first message requests still send, and its last edit; refused while
    /// a turn runs.
    fn edit(
        &mut self,
        agent: Entity,
        command: &str,
        plan: impl FnOnce(&Conversation, usize, Option<&Edit>) -> Result<Plan, String>,
    ) {
        let Ok((mut conversation, condensed, busy, usage)) = self.agents.get_mut(agent) else {
            return;
        };
        if busy {
            let why = format!(
                "/{command}: the agent's turn is running; wait for the turn to end or interrupt it."
            );
            self.notices.write(Notice::error(agent, why));
            return;
        }
        let first = condensed.map_or(0, |condensed| condensed.upto);
        let plan = plan(&conversation, first, self.edits.0.get(&agent))
            .and_then(|plan| joins_compacted(&conversation, first, &plan).map_or(Ok(plan), Err));
        let plan = match plan {
            Ok(plan) => plan,
            Err(why) => {
                self.notices
                    .write(Notice::error(agent, format!("/{command}: {why}")));
                return;
            }
        };
        let from = plan.from;
        let before = pieces(&conversation, from);
        let old = estimate(conversation.messages().get(from..).unwrap_or_default());
        self.commit.truncate(agent, &mut conversation, from);
        for piece in plan.after {
            commit_piece(&mut self.commit, agent, &mut conversation, piece);
        }
        // A user message left last waits for the next one, as after a
        // failed turn, rather than going to the model on a restore.
        self.commit.halt(agent, &mut conversation, Halt::Kept);
        let new = estimate(conversation.messages().get(from..).unwrap_or_default());
        if let Some(mut usage) = usage
            && let Some(context) = usage.context()
        {
            let left = (context + new).saturating_sub(old);
            *usage = LastUsage(Some(Usage::new().total_tokens(left)));
        }
        if plan.keep {
            let after = pieces(&conversation, from);
            self.edits.0.insert(
                agent,
                Edit {
                    from,
                    before,
                    after,
                },
            );
        } else {
            self.edits.0.remove(&agent);
        }
        self.notices.write(Notice::info(agent, plan.said));
    }
}

/// Why `plan` cannot be made: it would change or join a message a summary
/// replaced, which requests no longer send.
fn joins_compacted(conversation: &Conversation, first: usize, plan: &Plan) -> Option<String> {
    if plan.from < first {
        return Some(format!(
            "message {} was compacted; requests no longer send it",
            plan.from + 1
        ));
    }
    let previous = plan.from.checked_sub(1).filter(|_| plan.from == first)?;
    let previous = conversation.messages().get(previous)?;
    let next = plan.after.first()?;
    let same = matches!(previous, Message::User { .. }) == next.is_user();
    same.then(|| "the edit would join a compacted message".to_owned())
}

/// Commits `piece` to `agent`'s conversation, a user message's items with
/// where they came from.
fn commit_piece(commit: &mut Commit, agent: Entity, conversation: &mut Conversation, piece: Piece) {
    let items = match piece {
        Piece::User(items) => items,
        Piece::Other(message) => {
            commit.message(agent, conversation, message);
            return;
        }
    };
    let mut groups: Vec<(Option<Origin>, Vec<UserContent>)> = Vec::new();
    for (item, origin) in items {
        match groups.last_mut() {
            Some((from, group)) if *from == origin => group.push(item),
            _ => groups.push((origin, vec![item])),
        }
    }
    for (origin, content) in groups {
        commit.message_from(agent, conversation, Message::User { content }, origin);
    }
}

/// `/history`.
fn history(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&Condensed>)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((conversation, condensed)) = agents.get(args.agent) else {
        return;
    };
    let compacted = condensed.map_or(0, |condensed| condensed.upto);
    let counter = HeuristicTokenCounter::default();
    let mut lines = vec![format!("{} messages:", conversation.messages().len())];
    for (at, message) in conversation.messages().iter().enumerate() {
        let (role, items) = describe(message);
        let tokens = u64::try_from(counter.count(message)).unwrap_or(u64::MAX);
        let mark = if at < compacted { "  (compacted)" } else { "" };
        lines.push(format!(
            "{:>4}  {role:<9}  ~{:<5} {}{mark}",
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
    surgery.edit(args.agent, "stub", |conversation, _, _| {
        let at = number(&args.args, conversation.messages().len())?;
        let mut pieces = pieces(conversation, at);
        let Some(Piece::User(items)) = pieces.first_mut() else {
            return Err(format!("message {} holds no tool result", at + 1));
        };
        let mut stubbed = Vec::new();
        for (item, _) in items {
            let UserContent::ToolResult(result) = item else {
                continue;
            };
            let bytes = result
                .content
                .iter()
                .map(|content| match content {
                    ToolResultContent::Text(text) => text.text.len(),
                    other => serde_json::to_string(other).map_or(0, |json| json.len()),
                })
                .sum();
            if let [only] = result.content.as_slice()
                && only.as_text().is_some_and(|text| text.starts_with(STUB))
            {
                continue;
            }
            let text = format!("{STUB}{}]", size(bytes));
            result.content = vec![ToolResultContent::text(text)];
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
        let said = format!(
            "Stubbed the results in message {}: {}.",
            at + 1,
            stubbed.join(", ")
        );
        Ok(Plan {
            from: at,
            after: pieces,
            said,
            keep: true,
        })
    });
}

/// `/drop <n>`, `/drop <a>-<b>` or `/drop undo`.
fn drop_messages(In(args): In<CommandArgs>, mut surgery: Surgery) {
    let text = args.args.trim();
    surgery.edit(args.agent, "drop", |conversation, first, last| {
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
        plan_drop(conversation, first, (a, b))
    });
}

/// Drops messages `a` to `b` and every call or result paired with one of
/// theirs, then joins the messages left side by side.
fn plan_drop(
    conversation: &Conversation,
    first: usize,
    (a, b): (usize, usize),
) -> Result<Plan, String> {
    let messages = conversation.messages();
    let chosen = a..=b;
    let ids: HashSet<CallId> = messages
        .get(a..=b)
        .unwrap_or_default()
        .iter()
        .flat_map(call_ids)
        .collect();
    let lowest = messages
        .iter()
        .position(|message| call_ids(message).iter().any(|id| ids.contains(id)))
        .map_or(a, |paired| paired.min(a));
    if lowest < first {
        return Err(format!(
            "message {} was compacted; requests no longer send it",
            lowest + 1
        ));
    }
    // From the message before, so what the cut leaves side by side is
    // joined.
    let from = if lowest > first { lowest - 1 } else { lowest };
    let mut notes = Vec::new();
    let mut kept: Vec<(usize, Piece)> = Vec::new();
    for (at, piece) in (from..).zip(pieces(conversation, from)) {
        if chosen.contains(&at) {
            continue;
        }
        let was = piece.clone();
        let Some(piece) = piece.without(&ids) else {
            notes.push(format!(
                "removed message {}, whose calls or results went with them",
                at + 1
            ));
            continue;
        };
        if piece != was {
            notes.push(format!(
                "took the paired calls or results out of message {}",
                at + 1
            ));
        }
        match kept.last_mut() {
            Some((into, last)) => {
                if let Some(piece) = last.join(piece) {
                    kept.push((at, piece));
                } else {
                    notes.push(format!(
                        "joined message {} to message {}",
                        at + 1,
                        *into + 1
                    ));
                }
            }
            None => kept.push((at, piece)),
        }
    }
    let span = if a == b {
        format!("message {}", a + 1)
    } else {
        format!("messages {}-{}", a + 1, b + 1)
    };
    let mut said = format!("Dropped {span}.");
    for note in notes {
        said.push_str(&format!(" Also {note}."));
    }
    said.push_str(" /drop undo puts it back.");
    Ok(Plan {
        from,
        after: kept.into_iter().map(|(_, piece)| piece).collect(),
        said,
        keep: true,
    })
}

/// Puts back what the agent's last edit changed, with what came after it.
fn undo(conversation: &Conversation, last: Option<&Edit>) -> Result<Plan, String> {
    let Some(edit) = last else {
        return Err("nothing to undo in this run".to_owned());
    };
    let now = pieces(conversation, edit.from);
    if edit.from > conversation.messages().len() || !now.starts_with(&edit.after) {
        return Err("the edited messages changed since; nothing to undo".to_owned());
    }
    let mut after = edit.before.clone();
    after.extend(
        now.get(edit.after.len()..)
            .unwrap_or_default()
            .iter()
            .cloned(),
    );
    Ok(Plan {
        from: edit.from,
        after,
        said: "Put back the messages the last edit changed.".to_owned(),
        keep: false,
    })
}
