//! Reviewable, reversible compaction. rig-compaction's `/compact [focus]`
//! summarizes everything before the newest turns (two by default,
//! `[reviewable-compaction] keep_turns`), steered by the focus, and this
//! plugin holds the summary for review before it replaces anything:
//! Accept, Edit (the summary as an editable answer, or in `$EDITOR` with
//! the `tui` feature), Regenerate or Cancel. A run nobody answers, and an
//! automatic compaction, are not held.
//!
//! Every compaction keeps what it replaced in the session (rig-memory's
//! [`Compacted`]), so `/uncompact` lists them and `/uncompact N` puts back
//! what compaction N summarized, with the message ids it had, undoing the
//! later ones that folded it in, after checking the result fits the
//! model's window. Messages added after the summary stay after them.

use rig_compaction::{
    CompactionDefault, Summarizing, Uncompact, summarized_prefix, summary_author, summary_in,
    summary_state,
};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Message, tokens_label};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::{
    CompactReason, Compacted, LogRecord, SummaryState, estimate_tokens, undo_summary,
};
use serde::Deserialize;

/// The plugin's settings table.
/// user asked for makes: before the newest turns.
pub const NAME: &str = "reviewable-compaction";

/// The review of `/compact`'s summary, its cut, and `/uncompact`.
#[derive(Default)]
pub struct ReviewableCompactionPlugin;

impl Plugin for ReviewableCompactionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_compaction::CompactionPlugin>() {
            app.add_plugins(rig_compaction::CompactionPlugin);
        }
        app.add_setting::<ReviewSettings>(NAME)
            .add_systems(
                AgentPass,
                (
                    keep_newest_turns.in_set(PrepareSystems::Edit),
                    review.in_set(ReviewSystems::Act),
                ),
            )
            .add_observer(reviewed)
            .add_observer(edited)
            .add_command(
                "uncompact",
                "List this session's compactions, or put back what compaction N summarized",
                uncompact,
            );
        #[cfg(feature = "tui")]
        app.add_systems(Update, editor::take_edit);
    }
}

/// The `[reviewable-compaction]` table.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ReviewSettings {
    /// The newest turns `/compact` keeps as they are.
    pub keep_turns: usize,
}

impl Default for ReviewSettings {
    fn default() -> Self {
        Self { keep_turns: 2 }
    }
}

/// On a summary call this plugin holds for review.
#[derive(Component, Clone, Copy, Debug)]
#[component(storage = "SparseSet")]
struct UnderReview;

/// On a summary call of a compaction the user asked for: how many
/// messages it replaces and their estimated tokens, as it was planned; a
/// call Regenerate sends again is planned and measured anew.
#[derive(Component, Clone, Copy, Debug)]
struct Replaces(usize, u64);

/// On the review question, for the summary call.
#[derive(Component, Clone, Copy, Debug)]
struct Reviewing(Entity);

/// On the question that edits the summary, for the summary call.
#[derive(Component, Clone, Copy, Debug)]
struct Editing(Entity);

/// Where the newest `keep` turns of `messages` start, the first `from`
/// being summaries: at the `keep`-th newest message the user typed.
/// `None` when no turn before them would be summarized.
fn newest_turns(messages: &[std::sync::Arc<Message>], from: usize, keep: usize) -> Option<usize> {
    if keep == 0 {
        return Some(messages.len());
    }
    let typed = |message: &Message| match message {
        Message::User { content } => content
            .iter()
            .any(|item| !matches!(item, UserContent::ToolResult(_))),
        _ => false,
    };
    let starts: Vec<usize> = (from..messages.len())
        .filter(|&at| messages.get(at).is_some_and(|message| typed(message)))
        .collect();
    let first_kept = starts.len().checked_sub(keep)?;
    starts.get(first_kept).copied()
}

/// Plans each summary call of a compaction the user asked for again, by
/// the default policy cut before the newest turns, and measures what it
/// replaces. One with no turn before them is planned as rig-compaction
/// planned it.
fn keep_newest_turns(
    mut calls: Query<
        (Entity, &mut ModelRequest, &mut Summarizing, &ForAgent),
        (Added<Admitted>, ModelCalls),
    >,
    agents: Query<(&Conversation, Option<&Connection>)>,
    (default, settings): (Res<CompactionDefault>, Res<ReviewSettings>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent)) in &mut calls {
        if !matches!(summarizing.reason, CompactReason::Asked { .. }) {
            continue;
        }
        let Ok((conversation, connection)) = agents.get(agent) else {
            continue;
        };
        let messages = conversation.messages();
        let from = summarized_prefix(conversation);
        let upto = newest_turns(messages, from, settings.keep_turns)
            .filter(|&upto| upto > from && upto <= messages.len());
        if let (Some(connection), Some(upto)) = (connection, upto) {
            let reason = summarizing.reason.clone();
            let state = summary_state(conversation);
            match (default.0).plan(&state, messages, from, upto, &connection.spec, &reason) {
                Ok((state, request)) => {
                    *summarizing = Summarizing {
                        reason,
                        keep_from: conversation.id(upto),
                        state,
                        summary: None,
                    };
                    model.request = request.into();
                }
                Err(why) => {
                    notices.write(Notice::warn(agent, format!("Cannot compact: {why}.")));
                }
            }
        }
        let messages = conversation.messages();
        let end = summarizing
            .keep_from
            .and_then(|upto| conversation.index_of(upto))
            .unwrap_or(messages.len());
        let older = messages.get(..end).unwrap_or_default();
        let replaces = Replaces(older.len(), estimate_tokens(older));
        commands.entity(call).try_insert(replaces);
    }
}

/// Holds each summary of a compaction the user asked for, once the model
/// wrote it, for review by someone who can answer.
fn review(
    calls: Query<
        (Entity, &Summarizing, &Replaces, &ForAgent),
        (ReadyToSettle, Without<UnderReview>),
    >,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, summarizing, &replaces, &ForAgent(agent)) in &calls {
        if summarizing.summary.is_none() || !answerers.can_answer(Some(agent)) {
            continue;
        }
        commands.entity(call).insert(UnderReview);
        present(&mut commands, (call, agent), summarizing, replaces);
    }
}

/// Asks about the summary of `call`, holding it until answered.
fn present(
    commands: &mut Commands,
    (call, agent): (Entity, Entity),
    summarizing: &Summarizing,
    Replaces(replaced, before): Replaces,
) {
    let summary = summarizing.summary.clone().unwrap_or_default();
    let state = SummaryState {
        summary: summary.clone(),
        ..summarizing.state.clone()
    };
    let after = state
        .message()
        .map(|text| estimate_tokens(&[Message::user(text)]))
        .unwrap_or_default();
    let body = format!(
        "{summary}\n\nreplaces {replaced} messages, ~{} tokens → ~{}",
        tokens_label(before),
        tokens_label(after)
    );
    let ask = Ask::new("Compact the conversation into this summary?")
        .body(body)
        .choice("accept", "Accept")
        .choice("edit", "Edit");
    #[cfg(feature = "tui")]
    let ask = ask.choice("editor", "Edit in $EDITOR");
    let ask = ask
        .choice("regenerate", "Regenerate")
        .choice("cancel", "Cancel");
    commands.spawn((ask, ForAgent(agent), Reviewing(call), HoldOf(call)));
}

/// Asks for the summary of `call` as editable text.
fn ask_edit(commands: &mut Commands, call: Entity, agent: Entity, summary: &str) {
    let ask = Ask::new("Edit the summary").initial(summary).multiline();
    commands.spawn((ask, ForAgent(agent), Editing(call), HoldOf(call)));
}

/// Acts on the review's answer; the summary call goes on when nothing
/// holds it any more.
fn reviewed(
    settled: On<Add<Settled>>,
    asks: Query<(&Reviewing, &Outcome)>,
    mut calls: Query<(&mut Summarizing, &ForAgent)>,
    #[cfg(feature = "tui")] scratch: Option<Res<Scratch>>,
    mut commands: Commands,
    #[cfg(feature = "tui")] mut notices: MessageWriter<Notice>,
) {
    let Ok((&Reviewing(call), outcome)) = asks.get(settled.entity) else {
        return;
    };
    commands.entity(settled.entity).remove::<HoldOf>();
    let Ok((mut summarizing, &ForAgent(agent))) = calls.get_mut(call) else {
        return;
    };
    let summary = summarizing.summary.clone().unwrap_or_default();
    match Answer::of(outcome) {
        Some(answer) if answer.is("accept") => {}
        Some(answer) if answer.is("edit") => ask_edit(&mut commands, call, agent, &summary),
        #[cfg(feature = "tui")]
        Some(answer) if answer.is("editor") => {
            let scratch = scratch.as_deref();
            editor::open(&mut commands, call, agent, &summary, scratch, &mut notices);
        }
        // The successor is reviewed when its summary is in.
        Some(answer) if answer.is("regenerate") => commands.queue(resend(call)),
        // Cancelled: rig-compaction replaces nothing.
        _ => summarizing.summary = None,
    }
}

/// Takes the edited summary back to the review; without an edit, the
/// summary as it was.
fn edited(
    settled: On<Add<Settled>>,
    asks: Query<(&Editing, &Outcome)>,
    mut calls: Query<(&mut Summarizing, &Replaces, &ForAgent)>,
    mut commands: Commands,
) {
    let Ok((&Editing(call), outcome)) = asks.get(settled.entity) else {
        return;
    };
    commands.entity(settled.entity).remove::<HoldOf>();
    let Ok((mut summarizing, &replaces, &ForAgent(agent))) = calls.get_mut(call) else {
        return;
    };
    let text = Answer::of(outcome).and_then(|answer| answer.text);
    edit(&mut summarizing, text);
    present(&mut commands, (call, agent), &summarizing, replaces);
}

/// Makes `text` the summary, unless it is empty.
fn edit(summarizing: &mut Summarizing, text: Option<String>) {
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        summarizing.summary = Some(text.trim().to_owned());
    }
}

/// `HH:MM` UTC of `ms` since the Unix epoch.
fn clock(ms: u64) -> String {
    let minutes = ms / 60_000;
    format!("{:02}:{:02} UTC", (minutes / 60) % 24, minutes % 60)
}

/// `/uncompact`: lists the compactions to choose from (a picker, or
/// notices for a run nobody answers); `/uncompact N` puts back what the
/// `N`th summarized, undoing the later ones first.
fn uncompact(
    In(args): In<CommandArgs>,
    agents: Query<(
        &AgentId,
        &Conversation,
        Option<&Connection>,
        Has<ActiveTurn>,
    )>,
    store: Option<Res<SessionStore>>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((id, conversation, connection, busy)) = agents.get(agent) else {
        return;
    };
    let store = store.as_deref().map(|store| &*store.0);
    let compactions: Vec<Compacted> = Compacted::of(conversation).map(|(_, it)| it).collect();
    if args.args.is_empty() {
        let lines = list(store, &id.0, &compactions);
        if lines.is_empty() {
            let text = "No compaction of this conversation to undo.";
            notices.write(Notice::info(agent, text));
        } else if answerers.can_answer(Some(agent)) {
            let picker = Picker::new("Undo a compaction").for_agent(agent);
            let picker = lines
                .into_iter()
                .enumerate()
                .fold(picker, |picker, (at, line)| {
                    picker.item(line, format!("uncompact {}", at + 1))
                });
            commands.spawn(picker);
        } else {
            for line in lines {
                notices.write(Notice::info(agent, line));
            }
            let text = "/uncompact N puts back what compaction N summarized.";
            notices.write(Notice::info(agent, text));
        }
        return;
    }
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let chosen = args.args.trim().parse::<usize>().ok();
    let Some(undo) = chosen
        .filter(|&n| n >= 1)
        .and_then(|n| compactions.len().checked_sub(n - 1))
        .filter(|&undo| undo > 0)
    else {
        let text = format!(
            "/uncompact takes a compaction from 1 to {}",
            compactions.len()
        );
        notices.write(Notice::error(agent, text));
        return;
    };
    let Some(store) = store else {
        let text = "Cannot uncompact: the session is not saved.";
        notices.write(Notice::error(agent, text));
        return;
    };
    // What the conversation would be, measured before anything changes.
    let mut probe = conversation.0.clone();
    for _ in 0..undo {
        if let Err(why) = undo_summary(&mut probe, store, &id.0, summary_author()) {
            notices.write(Notice::error(agent, format!("Cannot uncompact: {why}.")));
            return;
        }
    }
    let tokens = estimate_tokens(probe.messages());
    let spec = connection.map(|connection| &*connection.spec);
    if let Some((spec, window)) = spec.and_then(|spec| Some((spec, spec.context_window?)))
        && tokens > u64::from(window)
    {
        let text = format!(
            "Cannot uncompact: the conversation would be about {} tokens, more than {}'s \
             context window of {}.",
            tokens_label(tokens),
            spec.reference(),
            tokens_label(u64::from(window))
        );
        notices.write(Notice::error(agent, text));
        return;
    }
    for _ in 0..undo {
        commands.trigger(Uncompact { entity: agent });
    }
}

/// A line for each of `compactions`, oldest first: when the session's log
/// says it was made, how many messages it replaced and its summary's first
/// line.
fn list(store: Option<&dyn Store>, log: &str, compactions: &[Compacted]) -> Vec<String> {
    let records = store
        .and_then(|store| history(store, log).ok())
        .unwrap_or_default();
    let made = |summary| {
        records.iter().find_map(|(stamp, record)| match record {
            LogRecord::Change(recorded) if summary_in(&recorded.change) == Some(summary) => {
                Some(clock(stamp.t))
            }
            _ => None,
        })
    };
    let line = |(at, compacted): (usize, &Compacted)| {
        let first = compacted.state.summary.lines().next().unwrap_or_default();
        let when = made(compacted.summary).unwrap_or_else(|| "unsaved".to_owned());
        let replaced = compacted.replaced.len();
        format!("{}. {when}, {replaced} messages replaced: {first}", at + 1)
    };
    compactions.iter().enumerate().map(line).collect()
}

#[cfg(feature = "tui")]
mod editor {
    //! The review's "Edit in $EDITOR": the summary in a file of the
    //! session's scratch directory, edited with the terminal.

    use std::path::PathBuf;
    use std::process::Command;

    use rig_compaction::Summarizing;
    use rig_harness::prelude::*;
    use rig_tui::{RunInTerminal, TerminalExit};

    use super::{Replaces, ask_edit, edit, present};

    /// On the editor's run: the summary call and the file.
    #[derive(Component, Clone, Debug)]
    pub(super) struct EditorFor(Entity, PathBuf);

    /// Opens the summary of `call` in `$EDITOR`, else asks for it here.
    pub(super) fn open(
        commands: &mut Commands,
        call: Entity,
        agent: Entity,
        summary: &str,
        scratch: Option<&Scratch>,
        notices: &mut MessageWriter<Notice>,
    ) {
        let dir = scratch.and_then(|scratch| scratch.0.clone());
        let path = dir.map(|dir| dir.join("reviewable-compaction-summary.md"));
        let Some(path) = path.filter(|path| std::fs::write(path, summary).is_ok()) else {
            let why = "The summary cannot be written for the editor; edit it here.";
            notices.write(Notice::error(agent, why));
            ask_edit(commands, call, agent, summary);
            return;
        };
        let program = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_owned());
        let mut editor = Command::new(program);
        editor.arg(&path);
        commands.spawn((RunInTerminal(editor), EditorFor(call, path), HoldOf(call)));
    }

    /// Takes the summary as the editor left it, unless it failed, back to
    /// the review.
    pub(super) fn take_edit(
        done: Query<(Entity, &TerminalExit, &EditorFor), Added<TerminalExit>>,
        mut calls: Query<(&mut Summarizing, &Replaces, &ForAgent)>,
        mut commands: Commands,
    ) {
        for (run, exit, EditorFor(call, path)) in &done {
            commands.entity(run).despawn();
            let Ok((mut summarizing, &replaces, &ForAgent(agent))) = calls.get_mut(*call) else {
                continue;
            };
            let text = std::fs::read_to_string(path).ok();
            edit(&mut summarizing, text.filter(|_| exit.success()));
            let call = (*call, agent);
            present(&mut commands, call, &summarizing, replaces);
        }
    }
}

#[cfg(test)]
mod tests;
