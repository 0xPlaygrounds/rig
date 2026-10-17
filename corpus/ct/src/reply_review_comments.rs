//! CT-08: review comments on the agent's earlier replies, like a pull
//! request's.
//!
//! `/comment` opens a picker over the agent's replies (the ones still in
//! its conversation, and the ones an edit took out, read back from the
//! session's log), then a picker over the chosen reply's lines (tick a
//! range, or type a passage), then asks for the comment. The terminal view's
//! selected reply (Alt+Up/Down) skips the first picker. Each comment is a
//! draft until sent; `/comments` lists the drafts to edit or delete them,
//! and `/comments send` sends them alone.
//!
//! The next message the user sends carries every draft, as one block
//! before the user's text: each comment with its quote, the turn of the
//! reply it is about, and a note when that reply is no longer in the
//! agent's conversation (compacted or removed). Once the message is
//! delivered, the drafts are marked sent.
//!
//! Each comment is kept as two asides after the reply it is about: its data
//! (kind [`COMMENT`]) and a [`Note`] the fronts draw next to the reply,
//! marked draft or sent. Asides are saved with the session.

use std::collections::HashSet;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{
    Aside, AsideId, Change, Conversation as Thread, MessageId,
};
use rig_harness::rig_core::message::{Message, UserContent};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::{self, LogRecord, Store};
use rig_tui::{MainPane, TranscriptItem, TranscriptView};
use serde::{Deserialize, Serialize};

/// The plugin's name, the author of its asides.
const NAME: &str = "review-comments";

/// The kind of the aside holding a [`Comment`].
pub const COMMENT: &str = "review-comment";

/// The label of the block of comments a message carries.
const LABEL: &str = "review comments";

/// Review comments on earlier replies: `/comment`, `/comments`, and their
/// delivery with the user's next message.
#[derive(Default)]
pub struct ReviewCommentsPlugin;

/// The plugin's Prepare system that adds the drafts to the user's next
/// message, in `PrepareSystems::Edit` (after the files a message names are
/// attached, in `Compose`), for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReviewCommentSystems;

impl Plugin for ReviewCommentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_harness::pick::PickPlugin>() {
            app.add_plugins(rig_harness::pick::PickPlugin);
        }
        if !app.is_plugin_added::<rig_harness::ask::AskPlugin>() {
            app.add_plugins(rig_harness::ask::AskPlugin);
        }
        app.configure_sets(
            AgentPass,
            ReviewCommentSystems.in_set(PrepareSystems::Edit),
        )
        .add_systems(AgentPass, attach.in_set(ReviewCommentSystems))
        .add_observer(answered)
        .add_observer(delivered)
        .add_command(
            "comment",
            "Comment on a passage of an earlier reply: pick the reply, the lines, then type the comment",
            comment,
        )
        .add_command(
            "comments",
            "List the draft comments to edit or delete them; /comments send sends them",
            comments,
        );
    }
}

/// A comment on a passage of a reply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comment {
    /// The reply it is about.
    pub reply: MessageId,
    /// The turn of that reply, counted from 1.
    pub turn: u64,
    /// The reply's lines quoted, counted from 1, when picked as lines.
    pub lines: Option<(usize, usize)>,
    /// The passage quoted.
    pub quote: String,
    /// The comment.
    pub text: String,
    /// Whether it was sent to the model.
    pub sent: bool,
    /// The aside of the note that shows it.
    pub note: Option<AsideId>,
}

impl Comment {
    /// The comments of `conversation`, oldest first, with their asides.
    pub fn of(conversation: &Thread) -> Vec<(AsideId, Comment)> {
        let ours = conversation
            .asides()
            .filter(|(_, aside)| aside.kind == COMMENT && aside.author == author());
        ours.filter_map(|(id, aside)| Some((id, Comment::deserialize(&aside.value).ok()?)))
            .collect()
    }

    /// The drafts of `conversation`, in the order of the replies and lines
    /// they are about.
    pub fn drafts(conversation: &Thread) -> Vec<(AsideId, Comment)> {
        let mut drafts: Vec<_> = Self::of(conversation)
            .into_iter()
            .filter(|(_, comment)| !comment.sent)
            .collect();
        drafts.sort_by_key(|(_, comment)| (comment.turn, comment.lines.map(|(from, _)| from)));
        drafts
    }

    /// Where it is, as the model reads it: the turn and the lines.
    fn place(&self) -> String {
        match self.lines {
            Some((from, to)) if from == to => format!("turn {}, line {from}", self.turn),
            Some((from, to)) => format!("turn {}, lines {from}-{to}", self.turn),
            None => format!("turn {}", self.turn),
        }
    }

    /// The note that shows it in the transcript.
    fn note(&self) -> Note {
        let state = if self.sent { "sent" } else { "draft" };
        let quote: Vec<String> = self.quote.lines().map(|line| format!("> {line}")).collect();
        let title = format!("Comment ({state}) on {}: {}", self.place(), self.text);
        Note::new(title, format!("{}\n\n{}", quote.join("\n"), self.text))
    }

    /// Puts it in `conversation` as aside `id`, or a new one, with its
    /// note; the comment's aside.
    fn save(mut self, conversation: &mut Thread, id: Option<AsideId>) -> AsideId {
        let after = Some(self.reply);
        let note = self.note().aside(author(), after);
        match self.note {
            Some(at) if conversation.replace_aside(at, note.clone()) => {}
            _ => self.note = Some(conversation.add_aside(note)),
        }
        let value = serde_json::to_value(&self).unwrap_or_default();
        let aside = Aside::new(COMMENT, author(), value).after(after);
        match id {
            Some(id) if conversation.replace_aside(id, aside.clone()) => id,
            _ => conversation.add_aside(aside),
        }
    }

    /// Takes comment `id` and its note out of `conversation`; whether it
    /// was there.
    fn remove(conversation: &mut Thread, id: AsideId) -> bool {
        let Some(comment) = Self::of(conversation)
            .into_iter()
            .find_map(|(at, comment)| (at == id).then_some(comment))
        else {
            return false;
        };
        if let Some(note) = comment.note {
            conversation.remove_aside(note);
        }
        conversation.remove_aside(id).is_some()
    }
}

/// The author of the plugin's asides.
fn author() -> Author {
    Author::plugin(NAME)
}

/// A reply the user may comment on.
#[derive(Clone, Debug)]
struct Reply {
    /// Its entry.
    id: MessageId,
    /// Its text.
    text: String,
    /// Its turn, counted from 1.
    turn: u64,
    /// Whether it is still in the conversation.
    in_context: bool,
}

impl Reply {
    /// Entry `id` of `conversation`, when it is a reply with text.
    fn at(conversation: &Thread, id: MessageId, in_context: bool) -> Option<Self> {
        let at = conversation.index_of(id)?;
        let text = reply_text(conversation, id)?;
        let turn = conversation
            .turns()
            .filter(|(_, turn)| conversation.start_of(turn.start) <= at)
            .map(|(_, turn)| turn.index.saturating_add(1))
            .next_back()
            .unwrap_or(1);
        Some(Self {
            id,
            text,
            turn,
            in_context,
        })
    }

    /// Its first line, for a picker.
    fn label(&self) -> String {
        let first = self.text.lines().find(|line| !line.trim().is_empty());
        format!("Turn {}: {}", self.turn, first.unwrap_or_default().trim())
    }
}

/// The text of reply `id` in `conversation`, when it is there and says
/// something.
fn reply_text(conversation: &Thread, id: MessageId) -> Option<String> {
    let at = conversation.index_of(id)?;
    let Some(Message::Assistant(reply)) = conversation.messages().get(at).map(|message| &**message)
    else {
        return None;
    };
    let text = reply.text();
    (!text.trim().is_empty()).then_some(text)
}

/// The replies of `conversation`, oldest first, then those an edit took
/// out of it, read back from its log `log` in `store`.
fn replies(conversation: &Thread, saved: Option<(&dyn Store, &str)>) -> Vec<Reply> {
    let mut replies: Vec<Reply> = conversation
        .entries()
        .filter_map(|entry| Reply::at(conversation, entry.id, true))
        .collect();
    if let Some((store, log)) = saved {
        replies.extend(removed_replies(conversation, store, log));
    }
    replies
}

/// The replies the log `log` of `store` once held that `conversation` no
/// longer does, as they were just before the edit that took them out.
fn removed_replies(conversation: &Thread, store: &dyn Store, log: &str) -> Vec<Reply> {
    let Ok(records) = rig_memory::history(store, log) else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut gone = Vec::new();
    for (stamp, record) in records {
        let LogRecord::Change(recorded) = record else {
            continue;
        };
        let Change::Splice { removed, .. } = recorded.change else {
            continue;
        };
        let removed: Vec<MessageId> = removed
            .into_iter()
            .filter(|id| conversation.index_of(*id).is_none() && seen.insert(*id))
            .collect();
        if removed.is_empty() {
            continue;
        }
        let Ok(then) = rig_memory::as_of(store, log, stamp.n) else {
            continue;
        };
        gone.extend(
            removed
                .into_iter()
                .filter_map(|id| Reply::at(&then, id, false)),
        );
    }
    gone
}

/// Reply `id` of an agent: in its conversation, else in its log.
fn find_reply(
    conversation: &Thread,
    saved: Option<(&dyn Store, &str)>,
    id: MessageId,
) -> Option<Reply> {
    Reply::at(conversation, id, true).or_else(|| {
        let (store, log) = saved?;
        removed_replies(conversation, store, log)
            .into_iter()
            .find(|reply| reply.id == id)
    })
}

/// What `/comment` was asked.
enum CommentArgs {
    /// Pick the reply.
    Pick,
    /// Pick the passage of a reply.
    Reply(MessageId),
    /// Comment on lines of a reply, counted from 1.
    Lines(MessageId, usize, usize),
    /// Comment on a passage of a reply.
    Quote(MessageId, String),
}

impl CommentArgs {
    /// `/comment`, `/comment <reply>`, `/comment <reply> lines <a>[-<b>]`
    /// or `/comment <reply> quote <text>`.
    fn parse(args: &str) -> Option<Self> {
        let mut words = args.splitn(3, ' ');
        let Some(reply) = words.next().filter(|word| !word.is_empty()) else {
            return Some(Self::Pick);
        };
        let reply: MessageId = serde_json::from_str(reply).ok()?;
        match (words.next(), words.next().map(str::trim)) {
            (None, _) => Some(Self::Reply(reply)),
            (Some("lines"), Some(range)) => {
                let (from, to) = range.split_once('-').unwrap_or((range, range));
                let (from, to) = (from.trim().parse().ok()?, to.trim().parse().ok()?);
                (1 <= from && from <= to).then_some(Self::Lines(reply, from, to))
            }
            (Some("quote"), Some(text)) if !text.is_empty() => {
                Some(Self::Quote(reply, text.to_owned()))
            }
            _ => None,
        }
    }
}

/// A picker of a reply's lines, with the line each item shows.
#[derive(Component)]
struct PassagePicker {
    /// The reply.
    reply: MessageId,
    /// The line, counted from 1, of each item.
    lines: Vec<usize>,
}

/// On a question asking for a comment's text: the comment, and the
/// aside it replaces when it edits one.
#[derive(Component)]
struct Commenting {
    /// The comment, its text not typed yet.
    comment: Comment,
    /// The comment edited.
    editing: Option<AsideId>,
}

/// On a message that carries draft comments: their asides.
#[derive(Component)]
struct Carries(Vec<AsideId>);

/// The session's store and an agent's log in it.
fn saved<'a>(store: Option<&'a SessionStore>, id: &'a AgentId) -> Option<(&'a dyn Store, &'a str)> {
    store.map(|store| (&*store.0, id.0.as_str()))
}

/// `/comment`: picks a reply, a passage of it, then asks for the comment.
fn comment(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, &AgentId)>,
    store: Option<Res<SessionStore>>,
    view: Option<Single<&TranscriptView, With<MainPane>>>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, id)) = agents.get(agent) else {
        return;
    };
    let saved = saved(store.as_deref(), id);
    let Some(asked) = CommentArgs::parse(&args.args) else {
        let usage = "Use /comment, /comment <reply>, /comment <reply> lines <a>-<b> or /comment <reply> quote <text>.";
        notices.write(Notice::error(agent, usage));
        return;
    };
    let selected = view.and_then(|view| match view.selected {
        Some(TranscriptItem::Message(id)) => Reply::at(conversation, id, true).map(|_| id),
        _ => None,
    });
    let asked = match (asked, selected) {
        (CommentArgs::Pick, Some(id)) => CommentArgs::Reply(id),
        (asked, _) => asked,
    };
    let reply_id = match &asked {
        CommentArgs::Pick => {
            let replies = replies(conversation, saved);
            if replies.is_empty() {
                notices.write(Notice::error(agent, "There is no reply to comment on."));
                return;
            }
            let mut picker = Picker::new("Comment on which reply?").for_agent(agent);
            for reply in replies {
                let line = format!(
                    "comment {}",
                    serde_json::to_string(&reply.id).unwrap_or_default()
                );
                picker = picker.item(reply.label(), line);
                if let Some(item) = picker.items.last_mut() {
                    item.detail = (!reply.in_context).then(|| "no longer in context".to_owned());
                }
            }
            commands.spawn(picker);
            return;
        }
        CommentArgs::Reply(id) | CommentArgs::Lines(id, ..) | CommentArgs::Quote(id, _) => *id,
    };
    let Some(reply) = find_reply(conversation, saved, reply_id) else {
        notices.write(Notice::error(
            agent,
            "That reply is not in this conversation or its log.",
        ));
        return;
    };
    let (lines, quote) = match asked {
        CommentArgs::Pick | CommentArgs::Reply(_) => {
            commands
                .spawn(passage_picker(agent, &reply))
                .observe(passage_picked);
            return;
        }
        CommentArgs::Lines(_, from, to) => {
            let count = to.saturating_sub(from).saturating_add(1);
            let picked: Vec<&str> = reply
                .text
                .lines()
                .skip(from.saturating_sub(1))
                .take(count)
                .collect();
            if picked.len() != count {
                notices.write(Notice::error(
                    agent,
                    format!("The reply has no lines {from}-{to}."),
                ));
                return;
            }
            (Some((from, to)), picked.join("\n"))
        }
        CommentArgs::Quote(_, text) => {
            if !reply.text.contains(&text) {
                notices.write(Notice::error(agent, "The reply does not say that."));
                return;
            }
            (None, text)
        }
    };
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::error(agent, "Nobody can type the comment here."));
        return;
    }
    let comment = Comment {
        reply: reply.id,
        turn: reply.turn,
        lines,
        quote,
        text: String::new(),
        sent: false,
        note: None,
    };
    commands.spawn(comment_question(agent, comment, None, None));
}

/// The question for a comment's text, proposing `initial`.
fn comment_question(
    agent: Entity,
    comment: Comment,
    editing: Option<AsideId>,
    initial: Option<String>,
) -> impl Bundle {
    let quote: Vec<String> = comment
        .quote
        .lines()
        .map(|line| format!("> {line}"))
        .collect();
    let mut ask = Ask::new(format!("Your comment on {}", comment.place()))
        .body(quote.join("\n"))
        .free_text();
    if let Some(initial) = initial {
        ask = ask.initial(initial);
    }
    (ask, ForAgent(agent), Commenting { comment, editing })
}

/// A picker of `reply`'s lines: tick a range, or type a passage.
fn passage_picker(agent: Entity, reply: &Reply) -> (Picker, PassagePicker) {
    let mut picker = Picker::new(format!(
        "Comment on turn {}: tick the lines, or type a passage",
        reply.turn
    ))
    .for_agent(agent)
    .multiple()
    .free_text();
    let mut lines = Vec::new();
    for (number, line) in (1..).zip(reply.text.lines()) {
        if line.trim().is_empty() {
            continue;
        }
        picker = picker.item(line.trim(), "");
        if let Some(item) = picker.items.last_mut() {
            item.detail = Some(format!("line {number}"));
        }
        lines.push(number);
    }
    (
        picker,
        PassagePicker {
            reply: reply.id,
            lines,
        },
    )
}

/// Goes on with the lines ticked, else the passage typed.
fn passage_picked(
    picked: On<Picked>,
    pickers: Query<(&PassagePicker, &Picker)>,
    mut commands: Commands,
) {
    let (Some(chosen), Ok((passage, picker))) = (&picked.chosen, pickers.get(picked.entity)) else {
        return;
    };
    let Some(agent) = picker.agent else {
        return;
    };
    let reply = serde_json::to_string(&passage.reply).unwrap_or_default();
    let ticked: Vec<usize> = chosen
        .items
        .iter()
        .filter_map(|item| passage.lines.get(*item).copied())
        .collect();
    let line = match (ticked.iter().min(), ticked.iter().max(), &chosen.text) {
        (Some(from), Some(to), _) => format!("comment {reply} lines {from}-{to}"),
        (_, _, Some(text)) if !text.trim().is_empty() => {
            format!("comment {reply} quote {}", text.trim())
        }
        _ => return,
    };
    commands.trigger(RunCommand::new(agent, line));
}

/// Keeps each comment typed as a draft, or its edit.
fn answered(
    added: On<Add<Outcome>>,
    asks: Query<(&Commenting, &Outcome, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((commenting, outcome, &ForAgent(agent))) = asks.get(added.entity) else {
        return;
    };
    let text = Answer::of(outcome).and_then(|answer| answer.text);
    let Some(text) = text
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
    else {
        notices.write(Notice::info(agent, "No comment kept."));
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let comment = Comment {
        text,
        ..commenting.comment.clone()
    };
    let place = comment.place();
    comment.save(&mut conversation, commenting.editing);
    let drafts = Comment::drafts(&conversation).len();
    notices.write(Notice::info(
        agent,
        format!("Comment on {place} kept; {drafts} draft(s) go with your next message."),
    ));
}

/// `/comments`: lists the drafts to edit or delete them;
/// `/comments send`, `/comments edit <id> [text]`, `/comments delete <id>`.
fn comments(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let drafts = Comment::drafts(&conversation);
    let mut words = args.args.splitn(3, ' ');
    let (verb, id, text) = (words.next().unwrap_or_default(), words.next(), words.next());
    let id: Option<AsideId> = id.and_then(|id| serde_json::from_str(id).ok());
    let draft = id.and_then(|id| drafts.iter().find(|(at, _)| *at == id).cloned());
    match (verb, draft) {
        ("", _) => {
            if drafts.is_empty() {
                notices.write(Notice::info(agent, "No draft comments."));
                return;
            }
            let mut picker = Picker::new("Draft comments").for_agent(agent);
            let mut listed = Vec::new();
            for (number, (id, comment)) in (1..).zip(&drafts) {
                let id = serde_json::to_string(id).unwrap_or_default();
                let quote = comment.quote.lines().next().unwrap_or_default();
                listed.push(format!(
                    "{number}. {} \"{quote}\": {}",
                    comment.place(),
                    comment.text
                ));
                picker = picker
                    .item(
                        format!("Edit {number}: {}", comment.text),
                        format!("comments edit {id}"),
                    )
                    .item(
                        format!("Delete {number}: {}", comment.text),
                        format!("comments delete {id}"),
                    );
            }
            notices.write(Notice::info(agent, listed.join("\n")));
            commands.spawn(picker);
        }
        ("send", _) if drafts.is_empty() => {
            notices.write(Notice::error(agent, "No draft comments to send."));
        }
        ("send", _) => {
            commands.spawn((Incoming::new("", args.mode), ForAgent(agent)));
        }
        ("edit", Some((id, comment))) => {
            match text.map(str::trim).filter(|text| !text.is_empty()) {
                Some(text) => {
                    let edited = Comment {
                        text: text.to_owned(),
                        ..comment
                    };
                    edited.save(&mut conversation, Some(id));
                }
                None if answerers.can_answer(Some(agent)) => {
                    let initial = Some(comment.text.clone());
                    commands.spawn(comment_question(agent, comment, Some(id), initial));
                }
                None => {
                    notices.write(Notice::error(agent, "Use /comments edit <id> <text>."));
                }
            }
        }
        ("delete", Some((id, _))) => {
            Comment::remove(&mut conversation, id);
            notices.write(Notice::info(agent, "Comment deleted."));
        }
        _ => {
            notices.write(Notice::error(
                agent,
                "Use /comments, /comments send, /comments edit <id> or /comments delete <id>.",
            ));
        }
    }
}

/// Puts every draft not carried yet before the text of the user's message
/// as it comes in, unless it is a command.
fn attach(
    mut messages: Query<
        (Entity, &mut Incoming, &ForAgent),
        (Added<Admitted>, IncomingMessages, Without<Started>),
    >,
    carried: Query<&Carries, Without<Outcome>>,
    agents: Query<&Conversation>,
    mut commands: Commands,
) {
    let mut taken: HashSet<AsideId> = carried
        .iter()
        .flat_map(|carries| carries.0.clone())
        .collect();
    for (message, mut incoming, &ForAgent(agent)) in &mut messages {
        if incoming.author.kind != AuthorKind::User || incoming.command().is_some() {
            continue;
        }
        let Ok(conversation) = agents.get(agent) else {
            continue;
        };
        let drafts: Vec<_> = Comment::drafts(conversation)
            .into_iter()
            .filter(|(id, _)| taken.insert(*id))
            .collect();
        if drafts.is_empty() {
            continue;
        }
        let block = Attachment {
            label: LABEL.to_owned(),
            content: UserContent::text(block(conversation, &drafts)),
        };
        incoming.parts.insert(0, MessagePart::Attachment(block));
        let ids = drafts.into_iter().map(|(id, _)| id).collect();
        commands.entity(message).insert(Carries(ids));
    }
}

/// The block the model reads `drafts` in.
fn block(conversation: &Thread, drafts: &[(AsideId, Comment)]) -> String {
    let mut block = format!(
        "The user reviewed your earlier replies and left {} comment(s). Answer each one, in order.\n",
        drafts.len()
    );
    for (number, (_, comment)) in (1..).zip(drafts) {
        block.push_str(&format!(
            "\n{number}. On your reply in {}:\n",
            comment.place()
        ));
        for line in comment.quote.lines() {
            block.push_str(&format!("> {line}\n"));
        }
        let in_view = reply_text(conversation, comment.reply)
            .is_some_and(|text| text.contains(&comment.quote));
        if !in_view {
            block.push_str("(The original reply is no longer in context: this quote is all that is left of it.)\n");
        }
        block.push_str(&format!("Comment: {}\n", comment.text));
    }
    block
}

/// Marks the drafts a message carried sent once it is delivered.
fn delivered(
    added: On<Add<DeliveredIn>>,
    messages: Query<(&Carries, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((carries, &ForAgent(agent))) = messages.get(added.entity) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    for (id, comment) in Comment::of(&conversation) {
        if carries.0.contains(&id) && !comment.sent {
            Comment {
                sent: true,
                ..comment
            }
            .save(&mut conversation, Some(id));
        }
    }
}

#[cfg(test)]
mod tests;
