//! Plugin corpus (held-out): H14: Session notes written on exit (spec `H14-exit-notes.md`).
//!
//! When the app is about to exit for good (`/quit`, Ctrl-D, the end of a
//! `--print` run, a signal), a session that changed files or had at least
//! three turns gets a note in `.rig/notes/<date>-<session>.md`: what the
//! main agent's conversation was about, summarized by a model call of the
//! plugin's own, and the files its tool calls changed. The plugin holds the
//! exit effect until the note is written, at most 15 s; Esc, or a second
//! quit, skips the summary. `/notes-now` writes the note without quitting.

use std::path::Path;

use rig_basics::sessions::SessionTitle;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{AssistantContent, Message};
use rig_harness::rig_core::tool::FilesChanged;
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

/// What the exit waits for, on the hold and the status.
const WRITING: &str = "Writing session notes… (Esc to skip)";

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
    /// The main agent's turns, over the whole session.
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

/// The plugin's stage systems.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExitNotesSystems {
    /// Prepare: asks the summary call for the note.
    Ask,
    /// Gate: holds the exit while the note is written.
    HoldExit,
}

/// On the plugin's summary call.
#[derive(Component)]
pub struct NotesCall;

/// On the plugin's holds of exits, and on its status picker; all go once
/// the note is written.
#[derive(Component)]
struct NotesWait;

/// Where the app's exit is, for the note.
#[derive(Resource, Default, PartialEq, Eq)]
enum Leaving {
    /// No exit asked for, or none that needs a note.
    #[default]
    No,
    /// The exits are held until the note is written.
    Waiting,
    /// The exit goes: nothing more to hold.
    Released,
}

/// Writes session notes on exit and on `/notes-now`.
#[derive(Default)]
pub struct ExitNotesPlugin;

impl Plugin for ExitNotesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ExitNotesSettings>("exit-notes")
            .init_resource::<ProjectRoot>()
            .init_resource::<Leaving>()
            .insert_resource(SessionNotes::default())
            .add_command(
                "notes-now",
                "Write this session's notes to .rig/notes now",
                notes_now,
            )
            .configure_sets(
                AgentPass,
                (
                    ExitNotesSystems::Ask.in_set(PrepareSystems::Edit),
                    ExitNotesSystems::HoldExit.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    ask_for_notes.in_set(ExitNotesSystems::Ask),
                    hold_exit.in_set(ExitNotesSystems::HoldExit),
                ),
            )
            .add_observer(follow_successor::<NotesCall>)
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

/// The summary call in flight, if any.
type InFlight<'w, 's> =
    Query<'w, 's, &'static mut Decision, (With<NotesCall>, Without<Settled>, ModelCalls)>;

/// `/notes-now`: writes the note without quitting.
fn notes_now(
    In(args): In<CommandArgs>,
    notes: Res<SessionNotes>,
    calls: InFlight,
    agents: MainAgentQuery,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let said = match main_agent(&agents) {
        _ if notes.turns == 0 => "No turns yet: nothing to note.",
        _ if !calls.is_empty() => "The session notes are being written.",
        Some(agent) => {
            summarize(agent, &mut commands);
            "Writing session notes…"
        }
        None => return,
    };
    notices.write(Notice::info(args.agent, said));
}

/// Spawns the summary call of `agent`'s conversation: built by the kernel
/// as a turn's request would be, but outside any turn, so its answer never
/// reaches the conversation; its usage counts as `notes`.
fn summarize(agent: Entity, commands: &mut Commands) {
    commands.spawn((
        Name::new("session notes"),
        NotesCall,
        SideCall(agent),
        ForAgent(agent),
        Effect::default().with_purpose("notes"),
    ));
}

/// Asks the summary call for the note, on the configured model if any. A
/// summary call is no turn's ([`TurnCall`]), so plugins that edit a turn's
/// requests need no order against it.
fn ask_for_notes(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent),
        (Added<Admitted>, With<NotesCall>, Without<TurnCall>),
    >,
    settings: Res<ExitNotesSettings>,
    models: Res<Models>,
    mut notices: MessageWriter<Notice>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if let Some(history) = &mut call.history {
            history.push(Entry::new(Message::user(ASK)).by(Author::plugin("exit-notes")));
        }
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

/// Holds each exit about to go while the note is written. The first exit
/// that leaves for good, of a session worth a note, starts the summary and
/// the deadline; one asked for while waiting skips the summary.
fn hold_exit(
    exits: Query<Entity, (With<ExitRequest>, ReadyToStart, PluginEffects)>,
    reason: Option<Res<ExitReason>>,
    notes: Res<SessionNotes>,
    agents: MainAgentQuery,
    mut leaving: ResMut<Leaving>,
    mut calls: InFlight,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for exit in &exits {
        if *leaving == Leaving::Released {
            continue;
        }
        let hold = (Name::new(WRITING), NotesWait, HoldOf(exit));
        if *leaving == Leaving::Waiting {
            // A second quit: the note goes without a summary.
            commands.spawn(hold);
            for mut call in &mut calls {
                call.decide(
                    Decider::plugin("ho-exit-notes"),
                    Outcome::refused("skipped"),
                );
            }
            continue;
        }
        let for_good = !matches!(
            reason.as_deref(),
            Some(ExitReason::Reload | ExitReason::Switch(_))
        );
        let agent = main_agent(&agents).filter(|_| for_good && notes.worth_a_note());
        let Some(agent) = agent else {
            *leaving = Leaving::Released;
            continue;
        };
        *leaving = Leaving::Waiting;
        commands.spawn((hold, WakeAt::after(WAIT)));
        if calls.is_empty() {
            summarize(agent, &mut commands);
        }
        notices.write(Notice::info(None, WRITING));
        let status = Picker::new(WRITING).item("Skip", "");
        commands.spawn((NotesWait, status)).observe(skip);
    }
}

/// Esc, or `Skip`, on the status: the note goes without a summary.
fn skip(_: On<Picked>, mut calls: InFlight) {
    for mut call in &mut calls {
        call.decide(
            Decider::plugin("ho-exit-notes"),
            Outcome::refused("skipped"),
        );
    }
}

/// The wait is over: the note goes without a summary.
fn time_out(due: On<Add<Due>>, holds: Query<(), With<NotesWait>>, mut calls: InFlight) {
    if holds.contains(due.entity) {
        for mut call in &mut calls {
            call.decide(
                Decider::plugin("ho-exit-notes"),
                Outcome::refused(format!("no summary after {} s", WAIT.as_secs())),
            );
        }
    }
}

/// Keeps the main agent's count of turns, which the kernel keeps over the
/// whole session, as each starts.
fn count_turns(
    started: On<Add<Turn>>,
    turns: Query<(&Turn, &TurnOf)>,
    agents: MainAgentQuery,
    mut notes: ResMut<SessionNotes>,
) {
    if let Ok((turn, &TurnOf(agent))) = turns.get(started.entity)
        && main_agent(&agents) == Some(agent)
    {
        notes.turns = notes.turns.max(turn.index.saturating_add(1));
    }
}

/// Keeps the files each settled tool call says it changed.
fn count_files(
    settled: On<Add<Settled>>,
    calls: Query<&Published, ToolCalls>,
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
/// without one, and lets the exit go if it waits.
#[expect(
    clippy::too_many_arguments,
    reason = "the note reads the whole session"
)]
fn take_notes(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, (With<NotesCall>, Without<SupersededBy>)>,
    waiting: Query<Entity, With<NotesWait>>,
    mut leaving: ResMut<Leaving>,
    mut notes: ResMut<SessionNotes>,
    root: Res<ProjectRoot>,
    (title, paths): (Option<Res<SessionTitle>>, Option<Res<SessionPaths>>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(outcome) = calls.get(settled.entity) else {
        return;
    };
    if *leaving == Leaving::Waiting {
        *leaving = Leaving::Released;
        for entity in &waiting {
            commands.entity(entity).try_despawn();
        }
    }
    let summary = outcome.completion().and_then(|response| {
        let text: String = response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        let text = text.trim();
        (response.error.is_none() && !text.is_empty()).then(|| text.to_owned())
    });
    let name = session_name(title.as_deref(), paths.as_deref());
    let file = notes.file.clone().unwrap_or_else(|| {
        let date = chrono::Local::now().format("%Y-%m-%d");
        format!("{date}-{}.md", slug(&name))
    });
    let dir = root.0.join(".rig").join("notes");
    let path = dir.join(&file);
    let text = note(&name, &notes, summary.as_deref());
    match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, text)) {
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
