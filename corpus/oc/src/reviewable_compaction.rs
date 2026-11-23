//! Reviewable compaction. `rig_basics::compaction`'s `/compact [focus]`
//! summarizes everything before the newest turns (two by default,
//! `[reviewable-compaction] keep_turns`), steered by the focus, and this
//! plugin holds the summary for review before it replaces anything:
//! Accept, Edit (the summary as an editable answer, or in `$VISUAL` or
//! `$EDITOR` with the `tui` feature), Regenerate or Cancel. A run nobody
//! answers, and an automatic compaction, are not held.

use rig_basics::compaction::{CompactReason, Summarizing, summarized_prefix};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Message, tokens_label};
use rig_harness::rig_memory::{SummaryState, estimate_tokens};
use serde::Deserialize;

/// The plugin's settings table.
/// user asked for makes: before the newest turns.
pub const NAME: &str = "reviewable-compaction";

/// The review of `/compact`'s summary, and its cut.
#[derive(Default)]
pub struct ReviewableCompactionPlugin;

impl Plugin for ReviewableCompactionPlugin {
    fn build(&self, app: &mut App) {
        rig_harness::load::<rig_basics::compaction::CompactionPlugin>(app, "rig-basics", "");
        app.add_setting::<ReviewSettings>(NAME)
            .add_systems(
                AgentPass,
                (
                    keep_newest_turns.in_set(PrepareSystems::Edit),
                    review.in_set(ReviewSystems::Act),
                ),
            )
            .add_observer(reviewed)
            .add_observer(edited);
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

/// Moves the cut of each summary call of a compaction the user asked for
/// to before the newest turns, so `rig_basics::compaction` asks it there, and
/// measures what it replaces. One with no turn before them keeps
/// `rig_basics::compaction`'s cut.
fn keep_newest_turns(
    mut calls: Query<
        (Entity, &mut Summarizing, &mut ModelRequest, &ForAgent),
        (Added<Admitted>, ModelCalls),
    >,
    agents: Query<(&Conversation, Option<&Connection>)>,
    settings: Res<ReviewSettings>,
    mut commands: Commands,
) {
    for (call, mut summarizing, mut model, &ForAgent(agent)) in &mut calls {
        let asked = matches!(summarizing.reason, CompactReason::Asked { .. });
        if !asked || summarizing.history.is_some() {
            continue;
        }
        let Ok((conversation, own)) = agents.get(agent) else {
            continue;
        };
        let messages = conversation.messages();
        let from = summarized_prefix(conversation);
        let upto = newest_turns(messages, from, settings.keep_turns)
            .filter(|&upto| upto > from && upto <= messages.len());
        let spec = model.connection.as_ref().or(own).map(|it| it.spec.clone());
        if let (Some(upto), Some(spec)) = (upto, spec) {
            let planned = Summarizing::plan_at(conversation, upto, summarizing.reason.clone());
            if let Some((planned, Ok(request))) =
                planned.map(|planned| (planned.clone(), planned.request(&spec, None)))
            {
                *summarizing = planned;
                *model = ModelRequest {
                    connection: model.connection.clone(),
                    ..ModelRequest::new(request)
                };
            }
        }
        let older = &summarizing.replaces;
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
    let ask = ask.choice("editor", "Edit in the editor");
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
        // Cancelled: `rig_basics::compaction` replaces nothing.
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

#[cfg(feature = "tui")]
mod editor {
    //! The review's "Edit in the editor": the summary in a file of the
    //! session's scratch directory, edited with the terminal.

    use std::path::PathBuf;

    use rig_basics::compaction::Summarizing;
    use rig_harness::prelude::*;
    use rig_tui::{RunInTerminal, TerminalExit};

    use super::{Replaces, ask_edit, edit, present};

    /// On the editor's run: the summary call and the file.
    #[derive(Component, Clone, Debug)]
    pub(super) struct EditorFor(Entity, PathBuf);

    /// Opens the summary of `call` in the user's editor, else asks for it here.
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
        let line = corpus_shared::editor::line();
        let editor = corpus_shared::editor::command(line.as_deref(), &[&path]);
        commands.spawn((RunInTerminal(editor), EditorFor(call, path), HoldOf(call)));
    }

    /// Takes the summary as the editor left it back to the review, unless
    /// it failed, which a notice says.
    pub(super) fn take_edit(
        done: Query<(Entity, &TerminalExit, &EditorFor), Added<TerminalExit>>,
        mut calls: Query<(&mut Summarizing, &Replaces, &ForAgent)>,
        mut commands: Commands,
        mut notices: MessageWriter<Notice>,
    ) {
        for (run, exit, EditorFor(call, path)) in &done {
            commands.entity(run).despawn();
            let Ok((mut summarizing, &replaces, &ForAgent(agent))) = calls.get_mut(*call) else {
                continue;
            };
            let text = match exit.failure() {
                Some(why) => {
                    let why = format!("The editor failed ({why}); the summary is unchanged.");
                    notices.write(Notice::error(agent, why));
                    None
                }
                None => std::fs::read_to_string(path).ok(),
            };
            edit(&mut summarizing, text);
            let call = (*call, agent);
            present(&mut commands, call, &summarizing, replaces);
        }
    }
}

#[cfg(test)]
mod tests;
