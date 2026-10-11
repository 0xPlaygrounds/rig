//! Plugin corpus (held-out): H14: Session notes written on exit (spec `H14-exit-notes.md`).
//!
//! When the app is about to exit for good (`/quit`, Ctrl-D, the end of a
//! `--print` run, a signal), a session that changed files or had at least
//! three turns gets a note in `.rig/notes/<date>-<session>.md`: what the
//! main agent's conversation was about, summarized by a model call of the
//! plugin's own, and the files its tool calls changed. The exit waits for
//! it at most 15 s; Esc, or a second quit, skips the wait. `/notes-now`
//! writes the note without quitting.
//!
//! The harness has no way to hold an exit: `/quit`, `--print` and signals
//! write `AppExit` directly. So the plugin takes the `AppExit` messages out
//! in `Last`, before Bevy's `OnAppExitSystems` see them, and writes the
//! first one again once the note is written.

use std::path::{Path, PathBuf};

use bevy_app::OnAppExitSystems;
use bevy_ecs::message::Messages;
use rig_compaction::CompactionSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{AssistantContent, Message};
use rig_harness::rig_core::tool::FilesChanged;
use rig_sessions::SessionTitle;
use rig_tui::{AppKeyBindingsExt, KeyBinding, KeyPress, run_command};
use serde::Deserialize;

/// How long the exit waits for the summary.
const WAIT: Duration = Duration::from_secs(15);

/// The turns of the main agent after which a session gets a note even
/// when it changed no file.
const MIN_TURNS: u64 = 3;

/// What the summary call asks, after the main agent's conversation.
const ASK: &str = "The session ends here. Write a short note for whoever picks this work up \
tomorrow, you or the user, in Markdown with three sections: `## Goal` (what the session set out \
to do), `## Decisions` (what was decided or done, and why) and `## Open questions` (what is \
left). Be brief. Do not call tools; the changed files are listed separately.";

/// What stands in for a summary the model did not give.
const UNAVAILABLE: &str = "(summary unavailable)";

/// The `[exit-notes]` settings: `model = "vendor/name"` writes the summary
/// on that model instead of the main agent's.
#[derive(Resource, Deserialize, Default, Debug, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct ExitNotesSettings {
    /// The model that summarizes; the main agent's when unset.
    pub model: Option<String>,
}

/// What the note says beside the summary, kept with the session across
/// `/reload` and `/resume`: the main agent's turns, the files the tool
/// calls changed (relative to the project when inside it), and the file
/// the note went to, which writing again replaces.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SessionNotes {
    /// The main agent's turns that ended.
    pub turns: u64,
    /// The changed files, in the order first changed.
    pub files: Vec<String>,
    /// The note's file name in `.rig/notes`, once one was written.
    pub file: Option<String>,
}

impl SessionNotes {
    /// Whether quitting writes a note.
    fn worth_a_note(&self) -> bool {
        !self.files.is_empty() || self.turns >= MIN_TURNS
    }
}

/// On the plugin's summary call.
#[derive(Component)]
pub struct NotesCall;

/// The status shown while the exit waits, and the wait's deadline; both go
/// once the note is written.
#[derive(Component)]
struct WaitingUi;

/// On a hold the plugin keeps on the agents' work while the exit waits.
#[derive(Component)]
struct ExitHold;

/// The note being written, and the exit it holds back.
#[derive(Resource, Default)]
struct Writing {
    /// The summary call, until it settled.
    call: Option<Entity>,
    /// The exit held back until the note is written.
    held: Option<AppExit>,
    /// The exit went through: later ones are not held.
    released: bool,
}

/// Writes session notes on exit and on `/notes-now`.
#[derive(Default)]
pub struct ExitNotesPlugin;

impl Plugin for ExitNotesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ExitNotesSettings>("exit-notes")
            .init_resource::<ProjectRoot>()
            .init_resource::<Writing>()
            .insert_resource(SessionNotes::default())
            .add_command(
                "notes-now",
                "Write this session's notes to .rig/notes now",
                notes_now,
            )
            .add_systems(
                AgentPass,
                (
                    ask_for_notes
                        .in_set(Stage::Prepare)
                        .before(CompactionSystems::Prepare),
                    hold_work.in_set(Stage::Gate),
                ),
            )
            .add_systems(Last, hold_exit.before(OnAppExitSystems))
            .add_observer(count_turns)
            .add_observer(count_files)
            .add_observer(take_notes)
            .add_observer(time_out);
        app.add_key_binding(
            KeyBinding::new(KeyPress::ctrl('d'), "Quit"),
            run_command("quit"),
        );
    }
}

/// `/notes-now`: writes the note without quitting.
fn notes_now(
    In(args): In<CommandArgs>,
    notes: Res<SessionNotes>,
    mut writing: ResMut<Writing>,
    agents: PrimaryQuery,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if notes.turns == 0 {
        notices.write(Notice::info(args.agent, "No turns yet: nothing to note."));
        return;
    }
    if writing.call.is_some() {
        notices.write(Notice::info(
            args.agent,
            "The session notes are being written.",
        ));
        return;
    }
    if let Some(agent) = primary(&agents) {
        writing.call = Some(summarize(agent, &mut commands));
        notices.write(Notice::info(args.agent, "Writing session notes…"));
    }
}

/// Spawns the summary call of `agent`'s conversation: built by the kernel
/// as a turn's request would be, but outside any turn, so its answer never
/// reaches the conversation; its usage counts as `notes`.
fn summarize(agent: Entity, commands: &mut Commands) -> Entity {
    commands
        .spawn((
            Name::new("session notes"),
            NotesCall,
            BuildFrom(agent),
            ForAgent(agent),
            Effect::default().with_purpose("notes"),
        ))
        .id()
}

/// Asks the summary call for the note, on the configured model if any.
fn ask_for_notes(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<NotesCall>)>,
    settings: Res<ExitNotesSettings>,
    models: Res<Models>,
    mut notices: MessageWriter<Notice>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        call.request.chat_history.push(Message::user(ASK));
        let Some(model) = &settings.model else {
            continue;
        };
        match models.connect(model) {
            Ok(connection) => call.connection = Some(connection),
            Err(why) => {
                let why = format!("Session notes on the agent's model: {model}: {why}");
                notices.write(Notice::error(agent, why));
            }
        }
    }
}

/// While an exit waits for the note, nothing else starts: the agents'
/// work stays as the exit would leave it, for a restart to carry on.
fn hold_work(
    starting: Query<Entity, (Undecided, Without<NotesCall>)>,
    writing: Res<Writing>,
    mut commands: Commands,
) {
    if writing.held.is_none() {
        return;
    }
    for effect in &starting {
        commands.spawn((ExitHold, HoldOf(effect)));
    }
}

/// Holds back the app's exit while the note is written, and lets it go
/// once it is: takes every `AppExit` out before `OnAppExitSystems` read
/// it, and writes the first one again.
#[expect(clippy::too_many_arguments, reason = "one system decides the exit")]
fn hold_exit(
    mut exits: ResMut<Messages<AppExit>>,
    mut writing: ResMut<Writing>,
    reason: Option<Res<ExitReason>>,
    notes: Res<SessionNotes>,
    agents: PrimaryQuery,
    holds: Query<Entity, With<ExitHold>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if writing.released {
        return;
    }
    let asked: Vec<AppExit> = exits.drain().collect();
    if let Some(held) = writing.held.clone() {
        if writing.call.is_none() {
            exits.write(held);
            writing.released = true;
            for hold in &holds {
                commands.entity(hold).despawn();
            }
        } else if asked
            .iter()
            .any(|exit| ExitReason::of(exit) != ExitReason::Signal)
            && let Some(call) = writing.call
        {
            // A second quit; a signal repeats on its own every frame.
            commands
                .entity(call)
                .insert_if_new(Outcome::refused("skipped"));
        }
        return;
    }
    let Some(first) = asked.first().cloned() else {
        return;
    };
    let reason = reason.map_or_else(|| ExitReason::of(&first), |reason| reason.clone());
    let leaving = !matches!(reason, ExitReason::Reload | ExitReason::Switch(_));
    let agent = primary(&agents);
    let Some(agent) = agent.filter(|_| leaving && notes.worth_a_note()) else {
        for exit in asked {
            exits.write(exit);
        }
        writing.released = true;
        return;
    };
    writing.held = Some(first);
    if writing.call.is_none() {
        writing.call = Some(summarize(agent, &mut commands));
    }
    notices.write(Notice::info(None, "Writing session notes… (Esc to skip)"));
    let status = Picker::new("Writing session notes… (Esc to skip)").item("Skip", "");
    commands.spawn((WaitingUi, status)).observe(skip);
    commands.spawn((WaitingUi, WakeAt::after(WAIT)));
}

/// Esc, or `Skip`, on the status: the note goes without a summary.
fn skip(_: On<Picked>, writing: Res<Writing>, mut commands: Commands) {
    if let Some(call) = writing.call {
        commands
            .entity(call)
            .insert_if_new(Outcome::refused("skipped"));
    }
}

/// The wait is over: the note goes without a summary.
fn time_out(
    due: On<Add<Due>>,
    deadlines: Query<(), (With<WaitingUi>, With<WakeAt>)>,
    writing: Res<Writing>,
    mut commands: Commands,
) {
    if let (Ok(()), Some(call)) = (deadlines.get(due.entity), writing.call) {
        let why = format!("no summary after {} s", WAIT.as_secs());
        commands.entity(call).insert_if_new(Outcome::refused(why));
    }
}

/// Counts the main agent's turns as they end.
fn count_turns(
    settled: On<Add<Settled>>,
    turns: Query<&TurnOf>,
    agents: PrimaryQuery,
    mut notes: ResMut<SessionNotes>,
) {
    if let Ok(&TurnOf(agent)) = turns.get(settled.entity)
        && primary(&agents) == Some(agent)
    {
        notes.turns += 1;
    }
}

/// Keeps the files each settled tool call says it changed.
fn count_files(
    settled: On<Add<Settled>>,
    calls: Query<&Published, With<ToolRequest>>,
    root: Res<ProjectRoot>,
    mut notes: ResMut<SessionNotes>,
) {
    let Ok(published) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(changed)) = published.0.get::<FilesChanged>() else {
        return;
    };
    for path in changed.paths() {
        let shown = Path::new(path)
            .strip_prefix(&root.0)
            .map_or_else(|_| path.to_owned(), |inside| inside.display().to_string());
        if !notes.files.contains(&shown) {
            notes.files.push(shown);
        }
    }
}

/// Writes the note once the summary call settled, with its summary or
/// without one.
#[expect(
    clippy::too_many_arguments,
    reason = "the note reads the whole session"
)]
fn take_notes(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, With<NotesCall>>,
    ui: Query<Entity, With<WaitingUi>>,
    mut writing: ResMut<Writing>,
    mut notes: ResMut<SessionNotes>,
    root: Res<ProjectRoot>,
    (title, paths): (Option<Res<SessionTitle>>, Option<Res<SessionPaths>>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(outcome) = calls.get(settled.entity) else {
        return;
    };
    writing.call = None;
    for entity in &ui {
        commands.entity(entity).despawn();
    }
    let summary = outcome.completion().and_then(|response| {
        let text: Vec<&str> = response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        let text = text.join("").trim().to_owned();
        (response.error.is_none() && !text.is_empty()).then_some(text)
    });
    let name = session_name(title.as_deref(), paths.as_deref());
    let file = notes.file.clone().unwrap_or_else(|| {
        let date = chrono::Local::now().format("%Y-%m-%d");
        format!("{date}-{}.md", slug(&name))
    });
    let dir = root.0.join(".rig").join("notes");
    let path: PathBuf = dir.join(&file);
    let text = note(&name, &notes, summary.as_deref());
    let written = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, text));
    match written {
        Ok(()) => {
            notes.file = Some(file);
            let said = format!("Session notes written to {}.", path.display());
            notices.write(Notice::info(None, said));
        }
        Err(why) => {
            let said = format!("Could not write {}: {why}", path.display());
            notices.write(Notice::error(None, said));
        }
    }
}

/// The session's name: the one `/name` gave it, else the start of its
/// first message, else its id.
fn session_name(title: Option<&SessionTitle>, paths: Option<&SessionPaths>) -> String {
    let named = title.and_then(|title| title.name.clone());
    let typed = title
        .map(|title| title.typed.clone())
        .filter(|typed| !typed.trim().is_empty());
    let id = paths
        .and_then(|paths| paths.path().file_name())
        .map(|id| id.to_string_lossy().into_owned());
    named
        .or(typed)
        .or(id)
        .unwrap_or_else(|| "session".to_owned())
}

/// `name` as a file name: lower-case words joined by `-`, at most 48
/// characters.
fn slug(name: &str) -> String {
    let words: Vec<String> = name
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    let slug: String = words.join("-").chars().take(48).collect();
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "session".to_owned()
    } else {
        slug.to_owned()
    }
}

/// The note's text.
fn note(name: &str, notes: &SessionNotes, summary: Option<&str>) -> String {
    let files = if notes.files.is_empty() {
        "- none\n".to_owned()
    } else {
        notes
            .files
            .iter()
            .map(|file| format!("- {file}\n"))
            .collect()
    };
    let date = chrono::Local::now().format("%Y-%m-%d %H:%M");
    format!(
        "# Session notes: {name}\n\n{date}, {turns} turns.\n\n## Files changed\n\n{files}\n{summary}\n",
        turns = notes.turns,
        summary = summary.unwrap_or(UNAVAILABLE),
    )
}

#[cfg(test)]
mod tests;
