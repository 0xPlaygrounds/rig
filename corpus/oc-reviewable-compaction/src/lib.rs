//! Reviewable, reversible compaction. `/compact [focus]` asks the agent's
//! model for a summary of everything before the newest turns (two by
//! default, `[reviewable-compaction] keep_turns`), steered by the focus,
//! and shows it for review before it replaces anything: Accept, Edit (the
//! summary as an editable answer, or in `$EDITOR` with the `tui` feature),
//! Regenerate or Cancel. A run nobody answers accepts it unreviewed.
//!
//! Every compaction, rig-compaction's automatic ones included, is listed in
//! the agent's saved [`CompactionHistory`]. The summarized messages stay in
//! the session's log, where the compaction's splice marks them, and
//! `/uncompact [N]` puts them back from there in place of the summary,
//! after checking they fit the model's window. Messages added after the
//! summary stay after them.

use std::sync::Arc;

use rig_compaction::{Compaction, Summarized, plan, summary_author, summary_in};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message, tokens_label};
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness::rig_memory::{
    CompactReason, CompactionPolicy, Cut, Record, Summarizer, SummaryState, apply_summary,
};
use serde::{Deserialize, Serialize};

/// The author of this plugin's edits of a conversation: a restore.
pub const NAME: &str = "reviewable-compaction";

/// `/compact` with a review, and `/uncompact`.
#[derive(Default)]
pub struct ReviewableCompactionPlugin;

impl Plugin for ReviewableCompactionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_compaction::CompactionPlugin>() {
            app.add_plugins(rig_compaction::CompactionPlugin);
        }
        app.add_setting::<ReviewSettings>(NAME)
            .register_required_components::<Agent, CompactionHistory>()
            .add_observer(drafted)
            .add_observer(reviewed)
            .add_observer(edited)
            .add_observer(follow_successor::<Drafting>)
            .add_systems(PostUpdate, record);
        #[cfg(feature = "tui")]
        app.add_systems(Update, editor::take_edit);
    }

    /// Takes `/compact` over from rig-compaction, whose automatic
    /// compaction stays.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut commands = world.query_filtered::<(Entity, &Name), With<SlashCommand>>();
        let theirs: Vec<Entity> = commands
            .iter(world)
            .filter(|(_, name)| name.as_str() == "/compact")
            .map(|(entity, _)| entity)
            .collect();
        for command in theirs {
            world.entity_mut(command).insert(Disabled);
        }
        app.add_command(
            "compact",
            "Summarize all but the newest turns, after you review it; /compact <focus> says what to keep",
            compact,
        )
        .add_command(
            "uncompact",
            "List this session's compactions, or put back what compaction N summarized",
            uncompact,
        );
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

/// The compactions of an agent's conversation that can be undone, oldest
/// first. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompactionHistory(pub Vec<CompactionRecord>);

/// One compaction of [`CompactionHistory`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionRecord {
    /// The summary's id as the compaction made it: how the session's log
    /// finds the splice that put it in.
    pub made: MessageId,
    /// The summary's id now, which a restore of a later compaction renews.
    pub now: MessageId,
    /// When, in milliseconds since the Unix epoch.
    pub at: u64,
    /// How many entries it replaced, an earlier summary included.
    pub replaced: usize,
    /// The summary state after it, which rig-compaction's next summary
    /// builds on.
    pub state: SummaryState,
}

/// On a compaction's own turn: what the review is about.
#[derive(Component, Clone, Debug)]
struct Review {
    agent: Entity,
    /// The first message kept.
    upto: Option<MessageId>,
    /// The state the summary goes into.
    state: SummaryState,
    /// The summary request, for Regenerate.
    request: CompletionRequest,
    /// The entries replaced and their estimated tokens.
    replaced: (usize, u64),
    /// The summary under review.
    summary: String,
}

/// On a summary call; its turn has the [`Review`].
#[derive(Component, Clone, Copy, Debug)]
struct Drafting;

/// On the review question, for the turn under review.
#[derive(Component, Clone, Copy, Debug)]
struct Reviewing(Entity);

/// On the question that edits the summary, for the turn under review.
#[derive(Component, Clone, Copy, Debug)]
struct Editing(Entity);

/// Where the newest `keep` turns of `messages` start, the first `from`
/// being summaries: before the `keep`-th newest message the user typed.
/// `None` when no turn before them would be summarized.
fn newest_turns(messages: &[Message], from: usize, keep: usize) -> Option<usize> {
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
        .filter(|&at| messages.get(at).is_some_and(typed))
        .collect();
    let first_kept = starts.len().checked_sub(keep)?;
    starts.get(first_kept).copied().filter(|&at| at > from)
}

/// `/compact [focus]`: plans the summary and asks the model for it, in a
/// turn of its own that ends once the review does.
fn compact(
    In(args): In<CommandArgs>,
    agents: Query<
        (
            &Conversation,
            Option<&Summarized>,
            Option<&Connection>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    policy: Option<Res<Compaction>>,
    settings: Res<ReviewSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, summarized, connection, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::no_model(agent));
        return;
    };
    let keep = settings.keep_turns;
    let policy = CompactionPolicy {
        cut: Cut::With(Arc::new(move |messages, from, _, _| {
            newest_turns(messages, from, keep)
        })),
        ..policy.map(|policy| policy.0.clone()).unwrap_or_default()
    };
    let reason = CompactReason::Asked { focus: args.args };
    let planned = plan(conversation, summarized, &policy, reason, &connection.spec);
    let (summarizing, request) = match planned {
        Ok(Some(planned)) => planned,
        Ok(None) => {
            let why = format!("Nothing to compact before the newest {keep} turns.");
            notices.write(Notice::info(agent, why));
            return;
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
            return;
        }
    };
    let messages = conversation.messages();
    let end = summarizing
        .upto
        .and_then(|upto| conversation.index_of(upto))
        .unwrap_or(messages.len());
    let older = messages.get(..end).unwrap_or_default();
    let review = Review {
        agent,
        upto: summarizing.upto,
        state: summarizing.state,
        request: request.clone(),
        replaced: (older.len(), policy.estimate(older)),
        summary: String::new(),
    };
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent), review))
        .id();
    draft(&mut commands, turn, agent, request);
}

/// Asks the model of `agent` for a summary, a call of `turn`.
fn draft(commands: &mut Commands, turn: Entity, agent: Entity, request: CompletionRequest) {
    commands.spawn((
        Name::new("compaction summary"),
        ModelRequest::new(request),
        Effect::default().with_purpose("compaction"),
        ForAgent(agent),
        CallOf(turn),
        Drafting,
    ));
}

/// Ends the compaction's `turn`, saying `why`.
fn end(commands: &mut Commands, turn: Entity, why: &str) {
    let outcome = TurnOutcome::Stopped(Some(why.to_owned()));
    commands.entity(turn).try_insert(outcome);
}

/// The text that stands for the summarized messages, with `summary`.
fn summary_message(review: &Review) -> Option<String> {
    let state = SummaryState {
        summary: review.summary.clone(),
        ..review.state.clone()
    };
    state.message()
}

/// Shows the summary of `turn` for review, or accepts it when nobody can
/// answer.
fn present(
    commands: &mut Commands,
    turn: Entity,
    review: &Review,
    answerers: &Answerers,
    agents: &mut Query<&mut Conversation>,
    notices: &mut MessageWriter<Notice>,
) {
    if !answerers.can_answer(Some(review.agent)) {
        accept(commands, turn, review, agents, notices);
        return;
    }
    let (replaced, before) = review.replaced;
    let after = summary_message(review)
        .map(|text| CompactionPolicy::default().estimate(&[Message::user(text)]))
        .unwrap_or_default();
    let body = format!(
        "{}\n\nreplaces {replaced} messages, ~{} tokens → ~{}",
        review.summary,
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
    let agent = review.agent;
    commands.spawn((ask, ForAgent(agent), Reviewing(turn), HoldOf(turn)));
}

/// Puts the summary of `turn` in place of the messages it summarizes and
/// ends the turn.
fn accept(
    commands: &mut Commands,
    turn: Entity,
    review: &Review,
    agents: &mut Query<&mut Conversation>,
    notices: &mut MessageWriter<Notice>,
) {
    let agent = review.agent;
    let state = SummaryState {
        summary: review.summary.clone(),
        ..review.state.clone()
    };
    let applied = agents
        .get_mut(agent)
        .ok()
        .and_then(|mut conversation| apply_summary(&mut conversation, review.upto, &state));
    match applied {
        Some(_) => {
            commands.entity(agent).insert(Summarized(state));
            let text = format!(
                "Compacted {} messages into the summary; /uncompact puts them back.",
                review.replaced.0
            );
            notices.write(Notice::info(agent, text));
            end(commands, turn, "compacted");
        }
        None => {
            let why = "Compaction failed: the messages it kept were taken out.";
            notices.write(Notice::error(agent, why));
            end(commands, turn, "compaction failed");
        }
    }
}

/// Takes a summary call's answer for review.
fn drafted(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &Outcome, Has<SupersededBy>), With<Drafting>>,
    mut turns: Query<&mut Review>,
    mut agents: Query<&mut Conversation>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CallOf(turn), outcome, superseded)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut review) = turns.get_mut(turn) else {
        return;
    };
    if superseded {
        return;
    }
    let summary = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(response)) => {
            Summarizer::summary_text(response).map_err(|why| why.to_string())
        }
    };
    match summary {
        Ok(summary) => {
            review.summary = summary;
            present(
                &mut commands,
                turn,
                &review,
                &answerers,
                &mut agents,
                &mut notices,
            );
        }
        Err(why) => {
            let text = format!("Compaction failed: {why}.");
            notices.write(Notice::error(review.agent, text));
            end(&mut commands, turn, "compaction failed");
        }
    }
}

/// Acts on the review's answer.
fn reviewed(
    settled: On<Add<Settled>>,
    asks: Query<(&Reviewing, &Outcome)>,
    turns: Query<&Review>,
    mut agents: Query<&mut Conversation>,
    #[cfg(feature = "tui")] scratch: Option<Res<Scratch>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Reviewing(turn), outcome)) = asks.get(settled.entity) else {
        return;
    };
    commands.entity(settled.entity).remove::<HoldOf>();
    let Ok(review) = turns.get(turn) else {
        return;
    };
    let agent = review.agent;
    match Answer::of(outcome) {
        Some(answer) if answer.is("accept") => {
            accept(&mut commands, turn, review, &mut agents, &mut notices);
        }
        Some(answer) if answer.is("edit") => {
            let ask = Ask::new("Edit the summary")
                .initial(review.summary.clone())
                .multiline();
            commands.spawn((ask, ForAgent(agent), Editing(turn), HoldOf(turn)));
        }
        #[cfg(feature = "tui")]
        Some(answer) if answer.is("editor") => {
            editor::open(
                &mut commands,
                turn,
                review,
                scratch.as_deref(),
                &mut notices,
            );
        }
        Some(answer) if answer.is("regenerate") => {
            draft(&mut commands, turn, agent, review.request.clone());
        }
        _ => {
            let text = "Compaction cancelled; the conversation is unchanged.";
            notices.write(Notice::info(agent, text));
            end(&mut commands, turn, "compaction cancelled");
        }
    }
}

/// Takes the edited summary back to the review; without an edit, the
/// summary as it was.
fn edited(
    settled: On<Add<Settled>>,
    asks: Query<(&Editing, &Outcome)>,
    mut turns: Query<&mut Review>,
    mut agents: Query<&mut Conversation>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Editing(turn), outcome)) = asks.get(settled.entity) else {
        return;
    };
    commands.entity(settled.entity).remove::<HoldOf>();
    let Ok(mut review) = turns.get_mut(turn) else {
        return;
    };
    if let Some(text) = Answer::of(outcome).and_then(|answer| answer.text)
        && !text.trim().is_empty()
    {
        review.summary = text.trim().to_owned();
    }
    present(
        &mut commands,
        turn,
        &review,
        &answerers,
        &mut agents,
        &mut notices,
    );
}

/// Lists each compaction of an agent's conversation, its own restores
/// left out, with the summary state it left.
fn record(
    mut committed: MessageReader<Committed>,
    mut agents: Query<(Option<&Summarized>, &mut CompactionHistory)>,
) {
    let mine = Author::plugin(NAME);
    for Committed { agent, change } in committed.read() {
        let Change::Splice { removed, .. } = &change.change else {
            continue;
        };
        let Some(summary) =
            summary_in(&change.change).filter(|_| change.by.as_ref() != Some(&mine))
        else {
            continue;
        };
        let Ok((summarized, mut compactions)) = agents.get_mut(*agent) else {
            continue;
        };
        compactions.0.push(CompactionRecord {
            made: summary,
            now: summary,
            at: change.at.unwrap_or_else(now_ms),
            replaced: removed.len(),
            state: summarized.map(|it| it.0.clone()).unwrap_or_default(),
        });
    }
}

/// `HH:MM` UTC of `ms` since the Unix epoch.
fn clock(ms: u64) -> String {
    let minutes = ms / 60_000;
    format!("{:02}:{:02} UTC", (minutes / 60) % 24, minutes % 60)
}

/// `/uncompact`: lists the compactions to choose from (a picker, or
/// notices for a run nobody answers); `/uncompact N` restores the `N`th.
#[allow(clippy::type_complexity)]
fn uncompact(
    In(args): In<CommandArgs>,
    mut agents: Query<(
        &AgentId,
        &mut Conversation,
        &mut CompactionHistory,
        Option<&Connection>,
        Has<ActiveTurn>,
    )>,
    store: Option<Res<SessionStore>>,
    policy: Option<Res<Compaction>>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((id, mut conversation, mut compactions, connection, busy)) = agents.get_mut(agent)
    else {
        return;
    };
    if args.args.is_empty() {
        list(
            agent,
            &conversation,
            &compactions,
            &answerers,
            &mut commands,
            &mut notices,
        );
        return;
    }
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(at) = args
        .args
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
    else {
        notices.write(Notice::error(
            agent,
            "/uncompact takes the number of a compaction",
        ));
        return;
    };
    let Some(store) = store else {
        notices.write(Notice::error(
            agent,
            "Cannot uncompact: the session is not saved.",
        ));
        return;
    };
    let policy = policy.map(|policy| policy.0.clone()).unwrap_or_default();
    let restoring = Restoring {
        store: &*store.0,
        log: &id.0,
        policy: &policy,
        spec: connection.map(|connection| &*connection.spec),
    };
    match restoring.restore(&mut conversation, &mut compactions, at) {
        Ok(restored) => {
            let summarized = conversation
                .entries()
                .next()
                .filter(|first| first.author == Some(&summary_author()))
                .and_then(|first| compactions.0.iter().find(|record| record.now == first.id))
                .map(|record| record.state.clone())
                .unwrap_or_default();
            commands.entity(agent).insert(Summarized(summarized));
            let text = format!(
                "Put back the {restored} messages compaction {} summarized.",
                at + 1
            );
            notices.write(Notice::info(agent, text));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot uncompact: {why}.")));
        }
    }
}

/// Shows the compactions of `agent`, numbered from the oldest.
fn list(
    agent: Entity,
    conversation: &Conversation,
    compactions: &CompactionHistory,
    answerers: &Answerers,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    if compactions.0.is_empty() {
        notices.write(Notice::info(
            agent,
            "No compaction of this conversation to undo.",
        ));
        return;
    }
    let lines = compactions.0.iter().enumerate().map(|(at, record)| {
        let first = record.state.summary.lines().next().unwrap_or_default();
        let folded = match conversation.index_of(record.now) {
            Some(_) => "",
            None => " (a later compaction summarized it)",
        };
        let label = format!(
            "{}. {}, {} messages replaced: {first}{folded}",
            at + 1,
            clock(record.at),
            record.replaced
        );
        (label, format!("uncompact {}", at + 1))
    });
    if answerers.can_answer(Some(agent)) {
        let picker = lines.fold(
            Picker::new("Undo a compaction").for_agent(agent),
            |picker, (label, command)| picker.item(label, command),
        );
        commands.spawn(picker);
    } else {
        for (label, _) in lines {
            notices.write(Notice::info(agent, label));
        }
        notices.write(Notice::info(
            agent,
            "/uncompact N puts back what compaction N summarized.",
        ));
    }
}

/// What a restore reads: the session's log of the agent, and the policy
/// and model that say whether the restored conversation fits.
struct Restoring<'a> {
    store: &'a dyn Store,
    log: &'a str,
    policy: &'a CompactionPolicy,
    spec: Option<&'a rig_harness::rig_core::catalog::ModelSpec>,
}

impl Restoring<'_> {
    /// Puts back, in place of the summary of compaction `at`, the messages
    /// it replaced, as the session's log holds them, and forgets the
    /// compaction. How many messages came back.
    fn restore(
        &self,
        conversation: &mut Conversation,
        compactions: &mut CompactionHistory,
        at: usize,
    ) -> Result<usize, String> {
        let record = compactions
            .0
            .get(at)
            .cloned()
            .ok_or_else(|| format!("there is no compaction {}", at + 1))?;
        let Some(summary_at) = conversation.index_of(record.now) else {
            return Err(
                "a later compaction summarized its summary; undo that one first".to_owned(),
            );
        };
        let records = history(self.store, self.log).map_err(|why| why.to_string())?;
        let made = records.iter().find_map(|(stamp, saved)| match saved {
            Record::Change(recorded) if summary_in(&recorded.change) == Some(record.made) => {
                match &recorded.change {
                    Change::Splice { removed, .. } => Some((stamp.n, removed.clone())),
                    _ => None,
                }
            }
            _ => None,
        });
        let Some((n, removed)) = made else {
            return Err("the session's log does not hold it".to_owned());
        };
        let before = as_of(self.store, self.log, n).map_err(|why| why.to_string())?;
        let originals: Vec<Entry> = removed
            .iter()
            .filter_map(|id| before.index_of(*id).and_then(|at| before.entry(at)))
            .collect();
        if originals.len() != removed.len() {
            return Err("the session's log lacks some of its messages".to_owned());
        }
        let messages = conversation.messages();
        let summary = self
            .policy
            .estimate(messages.get(summary_at..=summary_at).unwrap_or_default());
        let back: Vec<Message> = originals
            .iter()
            .map(|entry| entry.message.clone())
            .collect();
        let tokens =
            (self.policy.estimate(messages) + self.policy.estimate(&back)).saturating_sub(summary);
        if let Some(spec) = self.spec
            && self.policy.over_threshold(tokens, spec)
        {
            return Err(format!(
                "the conversation would be about {} tokens, more than {}'s context window leaves room for",
                tokens_label(tokens),
                spec.reference()
            ));
        }
        let restored = originals.len();
        let spliced = conversation.splice(summary_at..=summary_at, originals, Author::plugin(NAME));
        // An earlier summary that came back has a new id.
        for other in &mut compactions.0 {
            let new = removed
                .iter()
                .position(|id| *id == other.now)
                .and_then(|at| spliced.changed.get(at));
            if let Some(new) = new {
                other.now = *new;
            }
        }
        compactions.0.remove(at);
        Ok(restored)
    }
}

#[cfg(feature = "tui")]
mod editor {
    //! The review's "Edit in $EDITOR": the summary in a file of the
    //! session's scratch directory, edited with the terminal.

    use std::path::PathBuf;
    use std::process::Command;

    use rig_harness::prelude::*;
    use rig_tui::{RunInTerminal, TerminalExit};

    use super::{Review, present};

    /// On the editor's run: the turn under review and the file.
    #[derive(Component, Clone, Debug)]
    pub(super) struct EditorFor(Entity, PathBuf);

    /// Opens the summary of `turn` in `$EDITOR`, else back to the review.
    pub(super) fn open(
        commands: &mut Commands,
        turn: Entity,
        review: &Review,
        scratch: Option<&Scratch>,
        notices: &mut MessageWriter<Notice>,
    ) {
        let dir = scratch.and_then(|scratch| scratch.0.clone());
        let path = dir.map(|dir| dir.join("reviewable-compaction-summary.md"));
        let Some(path) = path.filter(|path| std::fs::write(path, &review.summary).is_ok()) else {
            let why = "The summary cannot be written for the editor; edit it here.";
            notices.write(Notice::error(review.agent, why));
            let ask = Ask::new("Edit the summary")
                .initial(review.summary.clone())
                .multiline();
            commands.spawn((
                ask,
                ForAgent(review.agent),
                super::Editing(turn),
                HoldOf(turn),
            ));
            return;
        };
        let program = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_owned());
        let mut editor = Command::new(program);
        editor.arg(&path);
        commands.spawn((RunInTerminal(editor), EditorFor(turn, path), HoldOf(turn)));
    }

    /// Takes the summary as the editor left it, unless it failed, back to
    /// the review.
    pub(super) fn take_edit(
        done: Query<(Entity, &TerminalExit, &EditorFor), Added<TerminalExit>>,
        mut turns: Query<&mut Review>,
        mut agents: Query<&mut Conversation>,
        answerers: Answerers,
        mut commands: Commands,
        mut notices: MessageWriter<Notice>,
    ) {
        for (run, exit, EditorFor(turn, path)) in &done {
            commands.entity(run).despawn();
            let Ok(mut review) = turns.get_mut(*turn) else {
                continue;
            };
            if let (true, Ok(text)) = (exit.success(), std::fs::read_to_string(path))
                && !text.trim().is_empty()
            {
                review.summary = text.trim().to_owned();
            }
            present(
                &mut commands,
                *turn,
                &review,
                &answerers,
                &mut agents,
                &mut notices,
            );
        }
    }
}

#[cfg(test)]
mod tests;
