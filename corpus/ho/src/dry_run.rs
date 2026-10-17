//! Dry-run mode: see what an agent would do before it does it.
//!
//! `/dry-run on` puts the agent in dry run; a subagent it spawns starts in
//! it too. In dry run, `edit` and `write` are answered without running: a
//! unified diff of what would change, and "(dry run: not applied)". The
//! would-be content goes into the session's [`Overlay`], so a later edit
//! of the same file diffs against it, and a `read` of a file in the
//! overlay runs but answers with the overlay's content. `shell` runs only a
//! read-only command (`ls`, `cat`, `git status`, `git diff`, `rg`, …), and
//! any other tool only when it only reads ([`Touches::reads_only`]); the
//! rest are answered "(dry run: not executed)".
//!
//! `/dry-run apply` writes the overlay to disk file by file, refusing (and
//! naming) a file changed on disk since it was previewed; `/dry-run
//! discard` drops it. `/dry-run off` with previewed files asks whether to
//! keep, apply or discard them, or keeps them and says so when nobody can
//! answer. The mode and the overlay are saved with the session, and the
//! status line says `DRY RUN`.

use std::path::Path;

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent};
use rig_harness::prelude::*;
use rig_harness::rig_core::serde_json::{Map, Value, from_value};
use rig_harness::rig_core::tool::ToolOutput;
use rig_tools::edit::apply;
use rig_tools::fs::{self, file_change, read_text, write_atomic};
use rig_tools::{EditArgs, ReadArgs, ShellArgs, WriteArgs};
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::de::DeserializeOwned;

#[cfg(test)]
mod tests;

/// What a previewed change says after its diff.
pub const NOT_APPLIED: &str = "(dry run: not applied)";
/// What a call that would change something is answered in dry run.
pub const NOT_EXECUTED: &str = "(dry run: not executed)";

/// The commands `shell` runs in dry run: each part of a pipeline starts
/// with one of these.
const READ_ONLY: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "pwd",
    "rg",
    "grep",
    "git status",
    "git diff",
    "git log",
    "git show",
];

/// After the permission mode; kept long, it says nothing is applied.
const DRY_RUN_ITEM: StatusItem = StatusItem::new("ho-dry-run", Side::Left, 46, 17);

/// On an agent in dry run. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct DryRun;

/// A file as dry run would leave it.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub struct Previewed {
    /// Its absolute path.
    pub path: String,
    /// Its would-be content.
    pub content: String,
    /// What was on disk when it was first previewed; `None` when there was
    /// no file. Apply refuses a file whose disk content is no longer this.
    pub base: Option<String>,
}

/// The files dry run changed and did not apply, for every agent of the
/// session. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct Overlay(pub Vec<Previewed>);

impl Overlay {
    /// The would-be content of the file at the absolute `path`, if previewed.
    pub fn content(&self, path: &str) -> Option<&str> {
        let file = self.0.iter().find(|file| file.path == path);
        file.map(|file| file.content.as_str())
    }

    /// The file at `path` as dry run sees it: previewed, else on disk, read
    /// as the file tools read it.
    fn current(&self, path: &str) -> Result<String, ToolExecutionError> {
        self.content(path)
            .map_or_else(|| read_text(path), |content| Ok(content.to_owned()))
    }

    /// Previews `content` for the file at `path`, remembering what is on
    /// disk the first time.
    fn set(&mut self, path: &str, content: String) {
        match self.0.iter_mut().find(|file| file.path == path) {
            Some(file) => file.content = content,
            None => self.0.push(Previewed {
                path: path.to_owned(),
                content,
                base: std::fs::read_to_string(path).ok(),
            }),
        }
    }

    /// Writes every file whose disk content is still its base, and keeps
    /// the others: the paths applied, and each refused one with why.
    fn apply(&mut self) -> (Vec<String>, Vec<String>) {
        let (mut applied, mut refused, mut kept) = (Vec::new(), Vec::new(), Vec::new());
        for file in std::mem::take(&mut self.0) {
            if std::fs::read_to_string(&file.path).ok() != file.base {
                refused.push(format!("{} (changed on disk since the dry run)", file.path));
                kept.push(file);
                continue;
            }
            match write_atomic(Path::new(&file.path), file.content.as_bytes()) {
                Ok(()) => applied.push(file.path),
                Err(error) => {
                    refused.push(format!("{} ({error})", file.path));
                    kept.push(file);
                }
            }
        }
        self.0 = kept;
        (applied, refused)
    }
}

/// A file's would-be content, at its absolute path.
struct Change {
    path: String,
    content: String,
}

/// On a previewed `edit` or `write`: the change it goes into the overlay
/// with once it settled as previewed.
#[derive(Component)]
struct Pending(Change);

/// On a `read` whose result the overlay replaced: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Overlaid;

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DryRunSystems {
    /// Gate: decides the calls of an agent in dry run that would change
    /// something.
    Gate,
    /// Review: answers a `read` of a previewed file with its overlay.
    Review,
}

/// Dry-run mode for every agent.
#[derive(Default)]
pub struct DryRunPlugin;

impl Plugin for DryRunPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<DryRun>()
            .register_type::<Overlay>()
            .init_resource::<Overlay>()
            .add_command(
                "dry-run",
                "Preview changes instead of making them: /dry-run on|off|apply|discard",
                dry_run,
            )
            .configure_sets(
                AgentPass,
                (
                    DryRunSystems::Gate.in_set(Stage::Gate),
                    DryRunSystems::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    preview.in_set(DryRunSystems::Gate),
                    read_overlay.in_set(DryRunSystems::Review),
                ),
            )
            .add_observer(keep_previewed)
            .add_observer(inherit)
            .add_systems(PostUpdate, show_dry_run.in_set(StatusSystems::Show));
    }
}

/// Decides each call of an agent in dry run that would change something:
/// `edit` and `write` with the diff they would make, `shell` and other
/// tools that do not only read with "not executed".
fn preview(
    mut calls: Query<
        (Entity, &ToolRequest, &Footprint, &ForAgent, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    agents: Query<Option<&WorkDir>, With<DryRun>>,
    overlay: Res<Overlay>,
    mut commands: Commands,
) {
    for (call, request, footprint, &ForAgent(agent), mut decision) in &mut calls {
        let Ok(dir) = agents.get(agent) else {
            continue;
        };
        let arguments = &request.call.function.arguments;
        let previewed = match request.call.function.name.as_str() {
            "edit" => edited(&overlay, dir, arguments),
            "write" => written(&overlay, dir, arguments),
            "shell" => {
                let command = parse::<ShellArgs>(arguments).map(|args| args.command);
                let command = command.unwrap_or_default();
                if !read_only(&command) {
                    let skipped = ToolResult::skipped(format!("{NOT_EXECUTED} {command}"));
                    decision.decide(Decider::plugin("ho-dry-run"), skipped);
                }
                continue;
            }
            _ => {
                if !footprint.0.reads_only() {
                    decision.decide(
                        Decider::plugin("ho-dry-run"),
                        ToolResult::skipped(NOT_EXECUTED),
                    );
                }
                continue;
            }
        };
        match previewed {
            Ok((change, text)) => {
                let text = format!("{text}\n{NOT_APPLIED}");
                if decision.decide(
                    Decider::plugin("ho-dry-run"),
                    ToolResult::success(ToolOutput::text(text)),
                ) {
                    commands.entity(call).insert(Pending(change));
                }
            }
            Err(error) => {
                decision.decide(Decider::plugin("ho-dry-run"), ToolResult::failed(error));
            }
        }
    }
}

/// A call's arguments as `A`, or why they do not fit.
fn parse<A: DeserializeOwned>(arguments: &Map<String, Value>) -> Result<A, ToolExecutionError> {
    from_value(Value::Object(arguments.clone()))
        .map_err(|error| ToolExecutionError::invalid_args(format!("{error}; nothing ran")))
}

/// The absolute path the file tools of an agent working in `dir` open for
/// `path`: confined to `dir`, as they are for an agent with a [`WorkDir`].
fn place(dir: Option<&WorkDir>, path: &str) -> Result<String, ToolExecutionError> {
    let base = dir.map(|WorkDir(dir)| dir.as_path());
    let path = fs::place(base, base.is_some(), path)?;
    Ok(path.to_string_lossy().into_owned())
}

/// What a `write` would leave, and its diff against the overlay.
fn written(
    overlay: &Overlay,
    dir: Option<&WorkDir>,
    arguments: &Map<String, Value>,
) -> Result<(Change, String), ToolExecutionError> {
    let args = parse::<WriteArgs>(arguments)?;
    let path = place(dir, &args.path)?;
    let before = overlay.current(&path).ok();
    let diff = file_change(&path, before.as_deref(), Some(&args.content)).unified();
    let change = Change {
        path,
        content: args.content,
    };
    Ok((change, diff))
}

/// What an `edit` would leave, made against the overlay as `edit` makes
/// it, and its diff with `edit`'s notes.
fn edited(
    overlay: &Overlay,
    dir: Option<&WorkDir>,
    arguments: &Map<String, Value>,
) -> Result<(Change, String), ToolExecutionError> {
    let args = parse::<EditArgs>(arguments)?;
    let path = place(dir, &args.path)?;
    let text = overlay.current(&path)?;
    let applied = apply(&path, &text, &args.edits)?;
    let mut shown = applied.change.unified();
    for note in applied.notes {
        shown.push_str(&format!("Note: {note}\n"));
    }
    let change = Change {
        path,
        content: applied.text,
    };
    Ok((change, shown))
}

/// Whether `command` only reads: no redirection, sequencing or
/// substitution, and every part of its pipeline an allowed command.
fn read_only(command: &str) -> bool {
    let forbidden = ['>', '<', ';', '&', '`', '$', '\n', '('];
    if command.trim().is_empty() || command.contains(forbidden) {
        return false;
    }
    command.split('|').all(|part| {
        let part = part.trim();
        let allowed = READ_ONLY.iter().any(|allowed| {
            part.strip_prefix(allowed)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
        });
        // `git diff --output=…` writes a file; `rg --pre` runs a command.
        allowed && !part.contains("--output") && !part.contains("--pre")
    })
}

/// A previewed change goes into the overlay once its call settled with
/// the preview as its answer.
fn keep_previewed(
    settled: On<Add<Settled>>,
    calls: Query<(&Pending, &Outcome)>,
    mut overlay: ResMut<Overlay>,
) {
    if let Ok((Pending(change), outcome)) = calls.get(settled.entity)
        && outcome.tool_result().is_success()
    {
        overlay.set(&change.path, change.content.clone());
    }
}

/// A `read` by an agent in dry run of a previewed file answers with the
/// overlay's content, shown as `read` shows a file.
fn read_overlay(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut Outcome),
        (ReadyToSettle, ToolCalls, Without<Overlaid>),
    >,
    agents: Query<Option<&WorkDir>, With<DryRun>>,
    overlay: Res<Overlay>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), mut outcome) in &mut calls {
        if request.call.function.name.as_str() != "read" {
            continue;
        }
        let Ok(dir) = agents.get(agent) else {
            continue;
        };
        let Ok(args) = parse::<ReadArgs>(&request.call.function.arguments) else {
            continue;
        };
        let Some(content) = place(dir, &args.path)
            .ok()
            .and_then(|path| overlay.content(&path))
        else {
            continue;
        };
        *outcome = Outcome::from(ToolResult::success(ToolOutput::text(args.show(content))));
        commands.entity(call).insert(Overlaid);
    }
}

/// A subagent of an agent in dry run starts in dry run. A restored one
/// keeps its saved mode: it is not spawned through a request again.
fn inherit(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &ForAgent), With<SpawnSubagent>>,
    dry: Query<(), With<DryRun>>,
    mut commands: Commands,
) {
    if let Ok((&SpawnedAgent(child), &ForAgent(parent))) = requests.get(spawned.entity)
        && dry.contains(parent)
    {
        commands.entity(child).insert(DryRun);
    }
}

/// `/dry-run on|off|apply|discard`, or alone: what is previewed.
fn dry_run(
    In(args): In<CommandArgs>,
    dry: Query<(), With<DryRun>>,
    mut overlay: ResMut<Overlay>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let files: Vec<&str> = overlay.0.iter().map(|file| file.path.as_str()).collect();
    let listed = files.join(", ");
    match args.args.as_str() {
        "on" => {
            commands.entity(agent).insert(DryRun);
            let text = "Dry run on: edits, writes and commands are previewed, not made.";
            notices.write(Notice::info(agent, text));
        }
        "off" => {
            commands.entity(agent).remove::<DryRun>();
            if files.is_empty() {
                notices.write(Notice::info(agent, "Dry run off."));
            } else if answerers.can_answer(Some(agent)) {
                let picker = Picker::new("Dry run off: the previewed changes?")
                    .for_agent(agent)
                    .body(listed)
                    .item("Keep them for later", "")
                    .item("Apply them", "dry-run apply")
                    .item("Discard them", "dry-run discard");
                commands.spawn(picker);
            } else {
                let text = format!(
                    "Dry run off; kept the previewed changes to {listed}: /dry-run apply or \
                     /dry-run discard."
                );
                notices.write(Notice::info(agent, text));
            }
        }
        "apply" => {
            let (applied, refused) = overlay.apply();
            if applied.is_empty() && refused.is_empty() {
                notices.write(Notice::info(agent, "Nothing previewed to apply."));
            }
            if !applied.is_empty() {
                let text = format!("Applied {}.", applied.join(", "));
                notices.write(Notice::info(agent, text));
            }
            if !refused.is_empty() {
                let text = format!("Not applied, still previewed: {}.", refused.join(", "));
                notices.write(Notice::info(agent, text));
            }
        }
        "discard" => {
            let text = format!("Discarded {} previewed files.", files.len());
            notices.write(Notice::info(agent, text));
            overlay.0.clear();
        }
        "" => {
            let mode = if dry.contains(agent) { "on" } else { "off" };
            let previewed = if files.is_empty() { "nothing" } else { &listed };
            let text = format!("Dry run {mode}; previewed: {previewed}.");
            notices.write(Notice::info(agent, text));
        }
        other => {
            let text = format!("/dry-run takes on, off, apply or discard, not `{other}`.");
            notices.write(Notice::error(agent, text));
        }
    }
}

/// Says `DRY RUN` on the status line of each agent in dry run.
fn show_dry_run(
    mut agents: Query<(Entity, Option<Ref<DryRun>>, &mut StatusItems), With<Agent>>,
    mut removed: RemovedComponents<DryRun>,
) {
    let removed: Vec<Entity> = removed.read().collect();
    for (agent, dry, mut items) in &mut agents {
        let added = dry.as_ref().is_some_and(Ref::is_added);
        if added || items.is_added() || removed.contains(&agent) {
            let text = if dry.is_some() { "DRY RUN" } else { "" };
            items.show(DRY_RUN_ITEM.says(text, Tone::Yellow));
        }
    }
}
