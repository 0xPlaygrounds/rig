//! Architect and editor models (held-out spec 15). `/architect on` (or
//! `--architect`) puts the shown agent in architect mode: its turns go to
//! the architect model of the `[architect]` settings table, offer only the
//! read-only tools and `implement`, and any other call is refused. An
//! `implement` call starts a one-job editor helper on the editor model,
//! with the file tools, a short system prompt, the change description and
//! the listed files, and none of the architect's conversation. The call
//! answers with the files the editor changed and their diffs, how its
//! checks ended and its final message. The editor's work is recorded under
//! the call, so the terminal view nests it there, and `rig_basics::telemetry` counts
//! each model's usage apart. `/architect off` gives the agent back its own
//! model and tools.
//!
//! ```toml
//! [architect]
//! architect = "anthropic/claude-opus-4-1"
//! editor = "deepseek/deepseek-flash"
//! max_rounds = 3
//! ```

use std::path::{Path, PathBuf};

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Section;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::tool::{ExitStatus, FileChange, FilesChanged};
use rig_harness::rig_tools::workspace::{Stat, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The tool the architect hands a change to the editor with.
pub const IMPLEMENT: &str = "implement";

/// The editor's system prompt.
pub const EDITOR_PROMPT: &str = "Apply exactly this change; do not redesign. \
    Make the edits described, run a check only if the description asks for one, \
    and end with one short message saying what you did.";

/// What every request of an architect adds to its system prompt.
pub const ARCHITECT_PROMPT: &str = "Architect mode: you read and plan, and an editor model makes \
    the edits. Explore with the read-only tools, then call `implement` with a precise, \
    plain-language description of the change and the files it touches. Check its result, \
    and call `implement` again to correct it if needed.";

/// Why a call an architect may not make is refused.
pub const REFUSED: &str = "architect mode: only reading tools and `implement` may be used";

/// The tools an editor may get, those the app has.
const EDITOR_TOOLS: [&str; 8] = [
    "read", "search", "list", "edit", "write", "delete", "move", "shell",
];

/// The most diff lines of one file an `implement` result shows.
const DIFF_LINES: usize = 200;

/// The `[architect]` settings table.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ArchitectSettings {
    /// The model an architect's turns go to, as `vendor/model`; the
    /// agent's own when unset.
    pub architect: Option<String>,
    /// The model the editor runs on; the architect agent's own when unset.
    pub editor: Option<String>,
    /// How many `implement` calls one turn of the architect may make.
    pub max_rounds: u32,
}

impl Default for ArchitectSettings {
    fn default() -> Self {
        Self {
            architect: None,
            editor: None,
            max_rounds: 3,
        }
    }
}

/// On an agent in architect mode; saved with the session, and its removal
/// too.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved, Default)]
pub struct ArchitectMode;

/// On an architect's turn: the `implement` calls it made.
#[derive(Component, Default)]
struct Rounds(u32);

/// On an `implement` call while its editor works: what it changed and the
/// checks it ran, and where its paths are relative to.
#[derive(Component)]
struct Implementing {
    base: PathBuf,
    changes: Vec<FileChange>,
    checks: Vec<String>,
}

/// On an editor agent: the `implement` call it works for.
#[derive(Component)]
struct EditorFor(Entity);

/// A listed file as the architect named it, and its text; `None` when
/// there is no file there yet.
type Listed = (String, Option<String>);

/// The listed files read off the main thread, on an entity that is a
/// child of the `implement` call.
#[derive(Component)]
struct Briefed(Result<Vec<Listed>, ToolExecutionError>);

impl From<Result<Vec<Listed>, ToolExecutionError>> for Briefed {
    fn from(read: Result<Vec<Listed>, ToolExecutionError>) -> Self {
        Self(read)
    }
}

/// What an editor is told, on the entity [`Briefed`] lands on.
#[derive(Component)]
struct Briefing {
    architect: Entity,
    effect: EffectId,
    description: String,
}

/// The arguments of `implement`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImplementArgs {
    /// The change to make, in plain language, precise enough to apply
    /// without redesigning anything.
    pub description: String,
    /// The files the change touches, relative to the project; their
    /// contents go to the editor.
    #[serde(default)]
    pub files: Vec<String>,
}

/// The architect plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchitectSystems {
    /// In [`PrepareSystems::Edit`]: an architect's turn calls go to the
    /// architect model with the read-only tools and `implement`; no other
    /// agent is offered `implement`.
    Prepare,
    /// In [`Stage::Gate`]: refuses an architect's calls of other tools, and
    /// `implement` from any other agent.
    Gate,
}

/// `/architect`, `--architect`, the `implement` tool and the editors it
/// starts.
#[derive(Default)]
pub struct ArchitectEditorPlugin;

impl Plugin for ArchitectEditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Writes(Scope::Anything)),
        };
        app.register_type::<ArchitectMode>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_setting::<ArchitectSettings>("architect")
            .add_flag(Flag::new(
                "architect",
                "Start in architect mode: a strong model plans, a cheap one edits",
            ))
            .add_command(
                "architect",
                "Plan on the architect model, edit on the editor model: /architect on|off",
                architect,
            )
            .add_open_tool(
                IMPLEMENT,
                "Have the editor model apply a change you describe to the listed files, and \
                 get back the files it changed with their diffs and its final message. The \
                 editor sees only your description and the files you list.",
                options,
                implement,
            )
            .configure_sets(
                AgentPass,
                (
                    ArchitectSystems::Prepare.in_set(PrepareSystems::Edit),
                    ArchitectSystems::Gate.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    architect_requests.in_set(ArchitectSystems::Prepare),
                    architect_calls.in_set(ArchitectSystems::Gate),
                ),
            )
            .add_observer(brief_editor)
            .add_observer(editor_spawned)
            .add_observer(editor_call_settled)
            .add_observer(editor_finished)
            .add_observer(stop_editor);
        let invoked = app.world().get_resource::<Invoked>();
        if invoked.is_some_and(|invoked| invoked.flag("architect")) {
            app.add_observer(architect_from_start);
        }
        #[cfg(feature = "tui")]
        look::add(app);
    }
}

/// `--architect`: every agent nothing spawned starts in architect mode.
fn architect_from_start(
    added: On<Add<Agent>>,
    roots: Query<(), Without<SpawnedBy>>,
    mut commands: Commands,
) {
    if roots.contains(added.entity) {
        commands.entity(added.entity).insert(ArchitectMode);
    }
}

/// `/architect on` enters architect mode; `/architect off` leaves it.
fn architect(
    In(args): In<CommandArgs>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    match args.args.trim() {
        "on" | "" => {
            commands.entity(agent).insert(ArchitectMode);
            notices.write(Notice::info(
                agent,
                "Architect mode: plans here, edits by the editor.",
            ));
        }
        "off" => {
            commands.entity(agent).remove::<ArchitectMode>();
            notices.write(Notice::info(agent, "Left architect mode."));
        }
        other => {
            notices.write(Notice::error(
                agent,
                format!("Unknown /architect argument `{other}`: use /architect on or off."),
            ));
        }
    }
}

/// An architect's turn calls go to the architect model, offer only the
/// read-only tools and `implement`, and say how to work; other agents'
/// are not offered `implement`.
fn architect_requests(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    architects: Query<(), With<ArchitectMode>>,
    (settings, models): (Res<ArchitectSettings>, Res<Models>),
    mut notices: MessageWriter<Notice>,
) {
    let reads: Vec<&str> = tools
        .iter()
        .filter(|(_, footprint)| footprint.0.reads_only())
        .map(|(def, _)| def.0.name.as_str())
        .collect();
    for (mut call, &ForAgent(agent)) in &mut calls {
        if !architects.contains(agent) {
            call.request_mut()
                .tools
                .retain(|tool| tool.name.as_str() != IMPLEMENT);
            continue;
        }
        call.request_mut().tools.retain(|tool| {
            let name = tool.name.as_str();
            name == IMPLEMENT || reads.contains(&name)
        });
        if let Some(prompt) = &mut call.prompt {
            prompt
                .sections
                .push(Section::new("architect_mode", ARCHITECT_PROMPT));
        }
        if let Some(model) = &settings.architect {
            match models.connect(model) {
                Ok(connection) => call.connection = Some(connection),
                Err(why) => {
                    notices.write(Notice::warn(
                        agent,
                        format!("The architect model {model} is not reachable ({why}); using the agent's own."),
                    ));
                }
            }
        }
    }
}

/// Refuses an architect's call of a tool that is neither read-only nor
/// `implement`, and `implement` from an agent not in architect mode.
fn architect_calls(
    mut calls: Query<
        (&ToolRequest, &Footprint, &ForAgent, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    architects: Query<(), With<ArchitectMode>>,
) {
    for (request, footprint, &ForAgent(agent), mut decision) in &mut calls {
        let implement = request.call.function.name.as_str() == IMPLEMENT;
        let architect = architects.contains(agent);
        if architect && !implement && !footprint.0.reads_only() {
            decision.decide(Decider::plugin("architect"), Outcome::refused(REFUSED));
        } else if implement && !architect {
            let why = "`implement` works only in architect mode (/architect on)";
            decision.decide(Decider::plugin("architect"), Outcome::refused(why));
        }
    }
}

/// An `implement` call: counted against its turn's rounds, then the
/// listed files are read off the main thread for the editor's brief.
fn implement(
    called: On<ToolCalled<ImplementArgs>>,
    agents: Query<(Option<&ActiveTurn>, Option<&WorkDir>)>,
    mut rounds: Query<&mut Rounds>,
    mut calls: Query<&mut Decision, ToolCalls>,
    (settings, root, workspace, wake): (
        Res<ArchitectSettings>,
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Wake>,
    ),
    mut commands: Commands,
) {
    let (turn, dir) = agents.get(called.agent).unwrap_or((None, None));
    let used = match turn.map(|turn| (turn.turn(), rounds.get_mut(turn.turn()))) {
        Some((_, Ok(mut rounds))) => {
            rounds.0 += 1;
            rounds.0
        }
        Some((turn, Err(_))) => {
            commands.entity(turn).try_insert(Rounds(1));
            1
        }
        None => 1,
    };
    if used > settings.max_rounds {
        let why = format!(
            "round limit reached: `implement` was called {} times this turn, the most allowed; \
             finish the turn and tell the user what is left",
            settings.max_rounds
        );
        if let Ok(mut decision) = calls.get_mut(called.call) {
            decision.decide(Decider::plugin("architect"), Outcome::refused(why));
        }
        return;
    }
    let base = dir.map_or(&root.0, |dir| &dir.0).clone();
    commands.entity(called.call).insert(Implementing {
        base: base.clone(),
        changes: Vec::new(),
        checks: Vec::new(),
    });
    let (files, workspace) = (called.args.files.clone(), workspace.0.clone());
    let read = blocking(move || read_listed(&*workspace, &base, files));
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    commands.spawn((
        Name::new("reading the files for the editor"),
        Briefing {
            architect: called.agent,
            effect: called.effect,
            description: called.args.description.clone(),
        },
        ChildOf(called.call),
        Running::spawn_into::<Briefed, _>(pool, &wake, read),
    ));
}

/// Each listed file's text, `None` for one not there yet; a file that is
/// there but cannot be read as text fails the whole read.
fn read_listed(
    workspace: &dyn Workspace,
    base: &Path,
    files: Vec<String>,
) -> Result<Vec<Listed>, ToolExecutionError> {
    let mut listed: Vec<Listed> = Vec::new();
    for file in files {
        if listed.iter().any(|(known, _)| *known == file) {
            continue;
        }
        let path = workspace.resolve(Some(base), &file);
        let text = match workspace.stat(&path) {
            Ok(Stat::Missing) => None,
            _ => Some(workspace.read_text(&path)?),
        };
        listed.push((file, text));
    }
    Ok(listed)
}

/// Once the listed files are read, asks for the editor: a one-job helper
/// on the editor model with the file tools, recorded under the call.
fn brief_editor(
    read: On<Insert<Briefed>>,
    briefings: Query<(&Briefing, &Briefed, &ChildOf)>,
    agents: Query<&ToolAccess>,
    tools: Query<&ToolDef>,
    settings: Res<ArchitectSettings>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((briefing, Briefed(listed), of)) = briefings.get(read.entity) else {
        return;
    };
    let call = of.parent();
    commands.entity(read.entity).despawn();
    let listed = match listed {
        Ok(listed) => listed,
        Err(why) => {
            if let Ok(mut decision) = calls.get_mut(call) {
                let why = format!("the listed files could not be read: {why}");
                decision.decide(Decider::plugin("architect"), Outcome::refused(why));
            }
            return;
        }
    };
    let access = agents.get(briefing.architect).ok();
    let registered: Vec<&str> = tools.iter().map(|def| def.0.name.as_str()).collect();
    let editor_tools = EDITOR_TOOLS
        .iter()
        .filter(|name| registered.contains(name) && access.is_none_or(|access| access.keeps(name)))
        .map(|name| (*name).to_owned())
        .collect();
    let editor = SpawnSubagent {
        model: settings.editor.clone(),
        tools: Some(editor_tools),
        ..SpawnSubagent::helper("editor")
            .brief(brief(&briefing.description, listed))
            .system_prompt(EDITOR_PROMPT)
            .once()
    };
    commands.spawn((
        editor,
        ForAgent(briefing.architect),
        ChildOf(call),
        Effect::with_parent_id(Some(briefing.effect)),
    ));
}

/// The editor's first message: the change, then each listed file's text
/// as it is.
fn brief(description: &str, listed: &[Listed]) -> String {
    let mut brief = format!("The change to apply:\n\n{description}");
    for (file, text) in listed {
        match text {
            Some(text) => brief.push_str(&format!("\n\n<file path=\"{file}\">\n{text}\n</file>")),
            None => brief.push_str(&format!("\n\n{file} does not exist yet.")),
        }
    }
    brief
}

/// Marks the agent an editor request spawned, or answers the call with
/// why none was.
fn editor_spawned(
    settled: On<Add<Settled>>,
    requests: Query<(&ChildOf, &Outcome, Option<&SpawnedAgent>), With<SpawnSubagent>>,
    implementing: Query<(), With<Implementing>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((of, outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let call = of.parent();
    if !implementing.contains(call) {
        return;
    }
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(editor)), _) => {
            commands.entity(editor).insert(EditorFor(call));
        }
        (None, result) => {
            let why = match result {
                Err(report) => report.message.clone(),
                Ok(_) => "no editor was started".to_owned(),
            };
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin("architect"), Outcome::refused(why));
            }
        }
    }
}

/// Keeps what each of an editor's calls changed, and how each of its
/// commands ended.
fn editor_call_settled(
    settled: On<Add<Settled>>,
    tool_calls: Query<(&ToolRequest, &ForAgent, Option<&Published>)>,
    editors: Query<&EditorFor>,
    mut implementing: Query<&mut Implementing>,
) {
    let Ok((request, &ForAgent(agent), published)) = tool_calls.get(settled.entity) else {
        return;
    };
    let Ok(&EditorFor(call)) = editors.get(agent) else {
        return;
    };
    let (Ok(mut work), Some(Published(published))) = (implementing.get_mut(call), published) else {
        return;
    };
    if let Ok(Some(FilesChanged(changes))) = published.get::<FilesChanged>() {
        work.changes.extend(changes);
    }
    if let Ok(Some(status)) = published.get::<ExitStatus>() {
        let arguments = &request.call.function.arguments;
        let command = arguments.get("command").and_then(|it| it.as_str());
        let ended = match status {
            ExitStatus::Exited(0) => "passed".to_owned(),
            ExitStatus::Exited(code) => format!("failed, exit code {code}"),
            other => format!("failed, {other:?}"),
        };
        let name = command.unwrap_or(request.call.function.name.as_str());
        work.checks.push(format!("`{name}`: {ended}"));
    }
}

/// Answers the `implement` call once its editor's turn ended: the files
/// changed with their diffs, the checks, and the editor's last message.
fn editor_finished(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    editors: Query<&EditorFor>,
    implementing: Query<&Implementing>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Ok((outcome, &ForAgent(editor))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(&EditorFor(call)) = editors.get(editor) else {
        return;
    };
    let (Ok(work), Ok(mut decision)) = (implementing.get(call), calls.get_mut(call)) else {
        return;
    };
    let said = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => format!("(the editor failed: {why})"),
        TurnOutcome::Stopped(stop) => format!("(the editor was stopped: {})", stop.cause()),
    };
    let report = report(work, &said);
    let result = match outcome {
        TurnOutcome::Answered(_) => ToolResult::success(report.into()),
        _ => ToolResult::failed(ToolExecutionError::other(report)),
    };
    decision.decide(Decider::plugin("architect"), result);
}

/// Stops the editor of an `implement` call that ended first, as when the
/// architect's turn was stopped.
fn stop_editor(
    settled: On<Add<Settled>>,
    implementing: Query<(), With<Implementing>>,
    editors: Query<(Entity, &EditorFor), With<ActiveTurn>>,
    mut commands: Commands,
) {
    if !implementing.contains(settled.entity) {
        return;
    }
    for (editor, _) in editors.iter().filter(|(_, of)| of.0 == settled.entity) {
        let why = "the `implement` call it worked for ended";
        commands.trigger(StopTurn::plugin(editor, "architect-editor", why));
    }
}

/// What an `implement` call answers.
fn report(work: &Implementing, said: &str) -> String {
    let mut files: Vec<(String, usize, usize, Vec<&FileChange>)> = Vec::new();
    for change in &work.changes {
        let path = shown(&work.base, &change.path);
        match files.iter_mut().find(|(known, ..)| *known == path) {
            Some((_, added, removed, changes)) => {
                *added += change.added;
                *removed += change.removed;
                changes.push(change);
            }
            None => files.push((path, change.added, change.removed, vec![change])),
        }
    }
    let mut out = String::new();
    if files.is_empty() {
        out.push_str("Files changed: none.");
    } else {
        out.push_str("Files changed:");
        for (path, added, removed, _) in &files {
            out.push_str(&format!("\n- {path}: +{added} -{removed}"));
        }
        for (path, _, _, changes) in &files {
            let diffs: Vec<String> = changes.iter().map(|change| change.unified()).collect();
            let diff: Vec<&str> = diffs.iter().flat_map(|diff| diff.lines()).collect();
            if diff.is_empty() {
                continue;
            }
            out.push_str(&format!("\n\n{path}:\n```diff\n"));
            out.push_str(
                &diff
                    .iter()
                    .take(DIFF_LINES)
                    .copied()
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            if diff.len() > DIFF_LINES {
                out.push_str(&format!("\n… {} more lines", diff.len() - DIFF_LINES));
            }
            out.push_str("\n```");
        }
    }
    if !work.checks.is_empty() {
        out.push_str("\n\nChecks the editor ran:");
        for check in &work.checks {
            out.push_str(&format!("\n- {check}"));
        }
    }
    out.push_str(&format!("\n\nThe editor's final message:\n{said}"));
    out
}

/// `path` relative to `base` when it is under it.
fn shown(base: &Path, path: &str) -> String {
    Path::new(path).strip_prefix(base).map_or_else(
        |_| path.to_owned(),
        |relative| relative.display().to_string(),
    )
}

/// How the terminal view draws an `implement` call: the description, the
/// editor's calls under it, then the result.
#[cfg(feature = "tui")]
mod look {
    use rig_harness::prelude::*;
    use rig_tui::ratatui::text::Line;
    use rig_tui::{LookLine, RenderBatch, RenderSystems, ToolCallView, TranscriptRender};

    use super::IMPLEMENT;

    pub(super) fn add(app: &mut App) {
        app.add_systems(TranscriptRender, draw.in_set(RenderSystems::Render));
    }

    fn draw(mut batch: ResMut<RenderBatch>) {
        for (call, output) in batch.tools_mut(IMPLEMENT) {
            output.lines = look(&call.view());
        }
    }

    fn look(view: &ToolCallView<'_>) -> Vec<LookLine> {
        let description = view.argument("description").unwrap_or_default();
        let mut lines = vec![view.header("Implement", description.to_owned()).into()];
        for child in view.children {
            let indent = "  ".repeat(child.depth + 2);
            let mut said = child.answer.as_deref().unwrap_or_default().lines();
            let first = said.next().unwrap_or_default();
            lines.push(Line::from(format!("{indent}↳ {} {first}", child.name)).into());
            lines.extend(said.map(|line| Line::from(format!("{indent}  {line}")).into()));
        }
        lines.extend(view.result_lines());
        lines
    }
}
