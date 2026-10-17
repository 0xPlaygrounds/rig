//! Pinned notes: short notes an agent always keeps in mind, sent with
//! every request as a section of its system prompt, so no compaction or
//! summary can drop them. `/pin <text>`, `/pins` and `/unpin <n>` change
//! and show the shown agent's notes; the model does the same with the
//! `pin_note` tool. The notes are a saved, immutable component of the
//! agent: each change inserts a new value, which the session log records
//! and `/resume` and `/reload` bring back, and whose insertion rewrites the
//! agent's own [`PromptSection`]. A compaction's summary call is told the
//! notes in Prepare, so the summary does not repeat them, and a `task` call
//! with `inherit_pins: true` (an argument this plugin adds to `task`, which
//! hands it on in its request's `extra`) gives the subagent a copy of its
//! parent's.

use rig_basics::compaction::Summarizing;
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, TASK};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Message, Section};
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool with which the model pins and unpins notes.
pub const PIN_NOTE: &str = "pin_note";

/// The argument this plugin adds to `task`.
pub const INHERIT_PINS: &str = "inherit_pins";

/// Where the notes go in the system prompt: after the project's
/// instructions, which change less often, and before the environment.
pub const ORDER_PINS: i32 = 250;

/// The tag around the notes in the system prompt.
const TAG: &str = "pinned_notes";

/// Adds `/pin`, `/pins`, `/unpin`, the `pin_note` tool, the agent's
/// pinned-notes prompt section, the notes in compaction summaries,
/// `inherit_pins` on `task` and, with the `tui` feature, the "📌 n" status
/// item.
#[derive(Default)]
pub struct PinnedNotesPlugin;

impl Plugin for PinnedNotesPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Pins>()
            .add_observer(write_section)
            .add_command(
                "pin",
                "Pin a note the agent always keeps in mind: /pin <text>",
                pin,
            )
            .add_command("pins", "List the agent's pinned notes", list)
            .add_command("unpin", "Unpin a note by its number: /unpin <n>", unpin)
            .add_open_tool(
                PIN_NOTE,
                "Pin a note you must always keep in mind (`text`), unpin one by its number \
                 (`unpin`), or, with neither, list them. Pinned notes stay in your system \
                 prompt, even after the conversation is summarized. Use it for a decision or \
                 constraint that must outlive the conversation, such as one the user asks you to \
                 remember.",
                ToolOptions {
                    footprint: Some(Touches::Independent),
                },
                on_pin_note,
            )
            .configure_sets(
                AgentPass,
                (
                    PinnedNotesSystems::Prepare.in_set(PrepareSystems::Edit),
                    PinnedNotesSystems::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    tell_summaries.in_set(PinnedNotesSystems::Prepare),
                    inherit.in_set(PinnedNotesSystems::Review),
                ),
            );
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, tui::show.in_set(rig_tui::StatusSystems::Show));
    }

    /// Offers `inherit_pins` on `task`, once `rig_basics::subagents` added it; a
    /// `task` call puts it in its request's `extra`.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<&mut ToolDef>();
        for mut def in tools.iter_mut(world) {
            if def.0.name.as_str() != TASK {
                continue;
            }
            let def = std::sync::Arc::make_mut(&mut def.0);
            if let Some(properties) = def.parameters.get_mut("properties")
                && let Some(properties) = properties.as_object_mut()
            {
                let argument = serde_json::json!({
                    "type": "boolean",
                    "description": "When true, the subagent starts with a copy of your pinned \
                        notes. Off when absent.",
                });
                properties.insert(INHERIT_PINS.to_owned(), argument);
            }
            def.description.push_str(
                " Give `inherit_pins: true` when the subagent must keep your pinned notes in \
                 mind.",
            );
        }
    }
}

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinnedNotesSystems {
    /// Prepare: tells a compaction's summary call the agent's notes, after
    /// the default prompt went in.
    Prepare,
    /// Review: gives a subagent spawned with `inherit_pins` its parent's
    /// notes.
    Review,
}

/// An agent's pinned notes, oldest first. Saved with the session; each
/// change is a new value inserted on the agent.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Pins(pub Vec<String>);

impl Pins {
    /// The notes, numbered from 1, one a line.
    pub fn listed(&self) -> String {
        let lines = self.0.iter().enumerate();
        let lines: Vec<String> = lines
            .map(|(at, note)| format!("{}. {note}", at + 1))
            .collect();
        lines.join("\n")
    }

    /// The system prompt section's text; empty without notes.
    fn section(&self) -> String {
        if self.0.is_empty() {
            return String::new();
        }
        format!(
            "Pinned notes: keep each in mind in everything you do, whatever the conversation \
             says or leaves out.\n{}",
            self.listed()
        )
    }

    /// These notes with `note` added.
    fn with(&self, note: &str) -> Result<Self, String> {
        let note = note.trim();
        if note.is_empty() {
            return Err("A pinned note needs a text.".to_owned());
        }
        let mut notes = self.0.clone();
        notes.push(note.to_owned());
        Ok(Self(notes))
    }

    /// These notes without the `n`th (from 1), and the note taken out.
    fn without(&self, n: usize) -> Result<(Self, String), String> {
        let mut notes = self.0.clone();
        let at = n.checked_sub(1).filter(|&at| at < notes.len());
        let Some(at) = at else {
            return Err(format!(
                "There is no pinned note {n}; there are {}.",
                notes.len()
            ));
        };
        let note = notes.remove(at);
        Ok((Self(notes), note))
    }

    /// What `/pins` and a listing `pin_note` say.
    fn report(&self) -> String {
        match self.0.len() {
            0 => "No pinned notes.".to_owned(),
            n => format!("Pinned notes ({n}):\n{}", self.listed()),
        }
    }
}

/// On the [`PromptSection`] of an agent's pinned notes.
#[derive(Component)]
struct PinsSection;

/// Keeps the agent's pinned-notes section the text of its [`Pins`], each
/// time a value is inserted: a change, a new agent, or a restored one.
fn write_section(
    insert: On<Insert<Pins>>,
    pins: Query<&Pins>,
    mut sections: Query<(&SectionOf, &mut PromptSection), With<PinsSection>>,
    mut commands: Commands,
) {
    let agent = insert.entity;
    let Ok(pins) = pins.get(agent) else {
        return;
    };
    let section = PromptSection::new(ORDER_PINS, TAG, pins.section());
    match sections.iter_mut().find(|(of, _)| of.0 == agent) {
        Some((_, mut mine)) => {
            mine.set_if_neq(section);
        }
        None if !section.text.is_empty() => {
            commands.spawn((
                Name::new("pinned notes"),
                PinsSection,
                SectionOf(agent),
                section,
            ));
        }
        None => {}
    }
}

/// `/pin <text>`.
fn pin(
    In(args): In<CommandArgs>,
    pins: Query<&Pins>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let current = pins.get(agent).cloned().unwrap_or_default();
    match current.with(&args.args) {
        Ok(pins) => {
            let said = format!("Pinned note {}.", pins.0.len());
            commands.entity(agent).insert(pins);
            notices.write(Notice::info(agent, said));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// `/pins`.
fn list(In(args): In<CommandArgs>, pins: Query<&Pins>, mut notices: MessageWriter<Notice>) {
    let current = pins.get(args.agent).cloned().unwrap_or_default();
    notices.write(Notice::info(args.agent, current.report()));
}

/// `/unpin <n>`.
fn unpin(
    In(args): In<CommandArgs>,
    pins: Query<&Pins>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let current = pins.get(agent).cloned().unwrap_or_default();
    let n = args
        .args
        .parse::<usize>()
        .map_err(|_| "/unpin needs a note's number.".to_owned());
    match n.and_then(|n| current.without(n)) {
        Ok((pins, note)) => {
            commands.entity(agent).insert(pins);
            notices.write(Notice::info(agent, format!("Unpinned \"{note}\".")));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// The arguments of a `pin_note` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PinArgs {
    /// A note to pin: a decision or constraint to keep in mind, in one
    /// sentence.
    text: Option<String>,
    /// The number of a pinned note to unpin, as the listing numbers them.
    unpin: Option<usize>,
}

/// Pins, unpins or lists the calling agent's notes, and answers with
/// them.
fn on_pin_note(
    called: On<ToolCalled<PinArgs>>,
    pins: Query<&Pins>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (agent, args) = (called.agent, &called.args);
    let mut current = pins.get(agent).cloned().unwrap_or_default();
    let mut changed = Ok(());
    if let Some(n) = args.unpin {
        changed = current.without(n).map(|(pins, _)| current = pins);
    }
    if let (Ok(()), Some(text)) = (&changed, &args.text) {
        changed = current.with(text).map(|pins| current = pins);
    }
    let result = match changed {
        Ok(()) => {
            let report = current.report();
            if args.text.is_some() || args.unpin.is_some() {
                commands.entity(agent).insert(current);
            }
            ToolResult::success(report.into())
        }
        Err(why) => {
            ToolResult::failed(ToolExecutionError::other(format!("{why} Nothing changed.")))
        }
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("pinned-notes"), result);
    }
}

/// Tells each compaction's summary call the agent's pinned notes, so the
/// summary leaves them out.
fn tell_summaries(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent),
        (
            Added<Admitted>,
            ModelCalls,
            With<Summarizing>,
            Without<TurnCall>,
        ),
    >,
    pins: Query<&Pins>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Some(pins) = pins.get(agent).ok().filter(|pins| !pins.0.is_empty()) else {
            continue;
        };
        let note = format!(
            "The agent keeps these pinned notes in its system prompt, apart from the \
             conversation; they survive this compaction, so do not repeat them in the \
             summary:\n{}",
            pins.listed()
        );
        if let Some(prompt) = &mut call.prompt {
            prompt
                .sections
                .push(Section::new("pinned_notes_kept", note));
            continue;
        }
        match call
            .request_mut()
            .chat_history
            .first_mut()
            .map(std::sync::Arc::make_mut)
        {
            Some(Message::System { content }) => {
                content.push_str("\n\n");
                content.push_str(&note);
            }
            _ => call
                .request_mut()
                .chat_history
                .insert(0, Message::system(note).into()),
        }
    }
}

/// Gives the subagent of a `task` call with `inherit_pins: true` a copy
/// of its parent's pinned notes, as it is spawned.
fn inherit(
    spawned: Query<
        (&SpawnSubagent, &SpawnedAgent, &ForAgent),
        (ReadyToSettle, Added<Outcome>, PluginEffects),
    >,
    pins: Query<&Pins>,
    mut commands: Commands,
) {
    for (request, &SpawnedAgent(child), &ForAgent(parent)) in &spawned {
        let asked = request
            .extra
            .get(INHERIT_PINS)
            .and_then(serde_json::Value::as_bool);
        if asked == Some(true)
            && let Ok(pins) = pins.get(parent)
        {
            commands.entity(child).insert(pins.clone());
        }
    }
}

#[cfg(feature = "tui")]
mod tui {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::Pins;

    /// After the agent's subagents at work; gone before them.
    const PINNED: StatusItem = StatusItem::new("pinned-notes", Side::Left, 65, 9);

    /// "📌 n" while the agent has pinned notes.
    pub(super) fn show(mut agents: Query<(&Pins, &mut StatusItems), Changed<Pins>>) {
        for (pins, mut items) in &mut agents {
            let text = match pins.0.len() {
                0 => String::new(),
                n => format!("📌 {n}"),
            };
            items.show(PINNED.says(text, Tone::Dim));
        }
    }
}

#[cfg(test)]
mod tests;
