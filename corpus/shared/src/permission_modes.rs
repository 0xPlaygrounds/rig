//! Permission modes: how much each agent may do without asking. `ask`
//! confirms every call that may change something, `edits` lets file edits
//! run and asks for the rest, `auto` asks for nothing, and `read-only`
//! neither offers nor runs a tool that changes anything. `/mode <name>`
//! sets the agent's mode, `/mode` alone and shift+tab cycle it, and the
//! status line shows it. A subagent starts in its parent's mode, or in the
//! configured one when that is stricter. A call that needs approval is
//! held by a question (Allow, Always for this tool, Deny); "Always" is kept
//! per agent. A run nobody answers denies such calls. Modes and "Always"
//! answers are saved with the session.
//!
//! The `[permissions]` settings table:
//!
//! ```toml
//! [permissions]
//! mode = "ask"           # an agent nobody set a mode for
//! subagents = "edits"    # subagents start no looser than this
//! tools = { shell = "exec", edit = "edit", write = "edit", fetch = "network" }
//! ```

use std::collections::HashMap;

use crate::confirm_destructive::ConfirmSystems;
use crate::plan_mode::PlanModeSystems;
use crate::protected_paths::ProtectedPathsSystems;
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent};
use rig_harness::prelude::*;
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, Side, StatusItem, StatusItems, StatusSystems, Tone,
    crossterm::event::KeyCode, run_command,
};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// After the model's reasoning setting; gone early when the line is short.
const MODE_ITEM: StatusItem = StatusItem::new("permission-modes", Side::Left, 45, 9);

/// How much an agent may do without asking. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[reflect(Component, Saved, Default, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// Nothing that changes anything runs, or is offered.
    ReadOnly,
    /// Every call that may change something asks first.
    #[default]
    Ask,
    /// File edits run; every other call that may change something asks.
    Edits,
    /// Everything runs.
    Auto,
}

impl PermissionMode {
    /// Every mode, strictest first; `/mode` cycles through them.
    pub const ALL: [Self; 4] = [Self::ReadOnly, Self::Ask, Self::Edits, Self::Auto];

    /// Its name, as `/mode` takes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Ask => "ask",
            Self::Edits => "edits",
            Self::Auto => "auto",
        }
    }

    /// The mode named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.name() == name)
    }

    /// The mode after this one, around.
    pub fn next(self) -> Self {
        let at = Self::ALL.iter().position(|mode| *mode == self);
        let next = at.map_or(0, |at| (at + 1) % Self::ALL.len());
        Self::ALL.get(next).copied().unwrap_or_default()
    }

    /// The stricter of the two.
    pub fn stricter(self, other: Self) -> Self {
        let rank = |mode: Self| Self::ALL.iter().position(|m| *m == mode);
        if rank(self) <= rank(other) {
            self
        } else {
            other
        }
    }

    /// What a call of a tool of `class` does in this mode.
    fn decide(self, class: ToolClass) -> Verdict {
        match (self, class) {
            (_, ToolClass::Read) | (Self::Auto, _) | (Self::Edits, ToolClass::Edit) => Verdict::Run,
            (Self::ReadOnly, _) => Verdict::Refuse,
            (Self::Ask | Self::Edits, _) => Verdict::Ask,
        }
    }
}

/// What a tool's calls may do, from its footprint or the settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolClass {
    /// Changes nothing.
    Read,
    /// Edits files.
    Edit,
    /// Runs commands.
    Exec,
    /// Reaches the network.
    Network,
    /// Anything else that may change something.
    Other,
}

/// What happens to a call.
enum Verdict {
    Run,
    Ask,
    Refuse,
}

/// The `[permissions]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Permissions {
    /// The mode of an agent nobody set one for.
    pub mode: PermissionMode,
    /// A mode subagents start in when it is stricter than their parent's.
    pub subagents: Option<PermissionMode>,
    /// The class of a tool by name, over what its footprint says.
    pub tools: HashMap<String, ToolClass>,
}

impl Default for Permissions {
    fn default() -> Self {
        let tools = [
            ("shell", ToolClass::Exec),
            ("edit", ToolClass::Edit),
            ("write", ToolClass::Edit),
            ("fetch", ToolClass::Network),
        ];
        Self {
            mode: PermissionMode::Ask,
            subagents: None,
            tools: tools
                .into_iter()
                .map(|(name, class)| (name.to_owned(), class))
                .collect(),
        }
    }
}

impl Permissions {
    /// The class of the tool `name` whose calls have `footprint`.
    fn class(&self, name: &str, footprint: &Footprint) -> ToolClass {
        self.tools.get(name).copied().unwrap_or(match footprint.0 {
            Touches::Writes(_) => ToolClass::Other,
            Touches::Reads(_) | Touches::Independent => ToolClass::Read,
        })
    }

    /// The mode of an agent with `mode`, if any.
    fn mode(&self, mode: Option<&PermissionMode>) -> PermissionMode {
        mode.copied().unwrap_or(self.mode)
    }
}

/// The tools the agent may call without asking, answered "Always". Saved
/// with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone)]
pub struct AlwaysAllowed(pub Vec<String>);

/// On a question asking to approve a call of `0`, the tool.
#[derive(Component, Clone, Debug)]
pub struct Approval(pub String);

/// On a call the user allowed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermissionSystems {
    /// Prepare: takes the tools that change something out of a read-only
    /// agent's requests, after the default prompt went in and plan mode
    /// took its own out, and before compaction measures them.
    Offer,
    /// Gate: refuses, holds and lets go of calls by their agent's mode,
    /// after the protected-paths guard and before confirm-destructive.
    Gate,
}

/// Permission modes for every agent.
#[derive(Default)]
pub struct PermissionModesPlugin;

impl Plugin for PermissionModesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Permissions>("permissions")
            .register_type::<PermissionMode>()
            .register_type::<AlwaysAllowed>()
            .add_command(
                "mode",
                "Set how much the agent may do without asking: read-only, ask, edits or auto",
                set_mode,
            )
            .configure_sets(
                AgentPass,
                (
                    PermissionSystems::Offer
                        .in_set(PrepareSystems::Edit)
                        .after(PlanModeSystems::Prepare),
                    PermissionSystems::Gate
                        .in_set(Stage::Gate)
                        .after(ProtectedPathsSystems::Gate)
                        .before(ConfirmSystems),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    offer.in_set(PermissionSystems::Offer),
                    (answered, ask).chain().in_set(PermissionSystems::Gate),
                ),
            )
            .add_observer(inherit)
            .add_systems(PostUpdate, show_mode.in_set(StatusSystems::Show));
        app.add_key_binding(
            KeyBinding::new(KeyPress::key(KeyCode::BackTab), "Cycle the permission mode"),
            run_command("mode"),
        );
    }
}

/// An agent in `read-only` is offered only the tools that read.
fn offer(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    agents: Query<Option<&PermissionMode>>,
    tools: Query<(&ToolDef, &Footprint)>,
    settings: Res<Permissions>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if settings.mode(agents.get(agent).ok().flatten()) != PermissionMode::ReadOnly {
            continue;
        }
        let reads = |name: &str| {
            tools.iter().any(|(def, footprint)| {
                def.0.name.as_str() == name && settings.class(name, footprint) == ToolClass::Read
            })
        };
        let tools = &mut call.request_mut().tools;
        tools.retain(|tool| reads(tool.name.as_str()));
    }
}

/// Decides each call about to run by its agent's mode: lets it run,
/// refuses it, or holds it with a question, refused when nobody can
/// answer. A call another gate decided is left alone.
fn ask(
    mut calls: Query<
        (Entity, &ToolRequest, &Footprint, &ForAgent, &mut Decision),
        (ReadyToStart, ToolCalls, Without<Approved>),
    >,
    agents: Query<(
        Option<&Name>,
        Option<&PermissionMode>,
        Option<&AlwaysAllowed>,
        Has<SpawnedBy>,
    )>,
    settings: Res<Permissions>,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, request, footprint, &ForAgent(agent), mut decision) in &mut calls {
        let Ok((name, mode, always, sub)) = agents.get(agent) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        let mode = settings.mode(mode);
        let tool = request.call.function.name.as_str();
        let allowed = always.is_some_and(|always| always.0.iter().any(|name| name == tool));
        let why = match mode.decide(settings.class(tool, footprint)) {
            Verdict::Run => continue,
            Verdict::Ask if allowed => continue,
            Verdict::Refuse => format!(
                "Refused: this agent is in read-only mode, so `{tool}` may not run. Nothing was \
                 changed; use the tools that only read."
            ),
            Verdict::Ask if !answerers.can_answer(Some(agent)) => format!(
                "Denied: `{tool}` needs the user's approval in {} mode, and nobody can approve \
                 it in this run. Nothing was changed.",
                mode.name()
            ),
            Verdict::Ask => {
                let who = match name.filter(|_| sub) {
                    Some(name) => format!("Subagent \"{name}\""),
                    None => "The agent".to_owned(),
                };
                let question = Ask::new(format!("{who} wants to run `{tool}`"))
                    .body(request.call.function.raw_arguments())
                    .choice("allow", "Allow")
                    .choice("always", format!("Always allow `{tool}` for this agent"))
                    .choice("deny", "Deny");
                let approval = Approval(tool.to_owned());
                commands.spawn((question, approval, ForAgent(agent), HoldOf(call)));
                continue;
            }
        };
        decision.decide(Decider::plugin("permission-modes"), Outcome::refused(why));
    }
}

/// Lets a call run, or refuses it, once its question is answered, and
/// remembers "Always" for the agent; the question lets go of the call
/// either way. A question closed unanswered denies.
fn answered(
    asks: Query<(Entity, &Approval, &Outcome, &HoldOf, &ForAgent), (Added<Outcome>, PluginEffects)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut always: Query<&mut AlwaysAllowed>,
    mut commands: Commands,
) {
    for (ask, Approval(tool), outcome, &HoldOf(call), &ForAgent(agent)) in &asks {
        let answer = Answer::of(outcome);
        let is = |id: &str| answer.as_ref().is_some_and(|answer| answer.is(id));
        if is("always") {
            match always.get_mut(agent) {
                Ok(mut always) => always.0.push(tool.clone()),
                Err(_) => {
                    commands
                        .entity(agent)
                        .insert(AlwaysAllowed(vec![tool.clone()]));
                }
            }
        }
        if is("allow") || is("always") {
            commands.entity(call).insert(Approved);
        } else {
            let why = format!(
                "Denied: the user did not allow this `{tool}` call. Nothing was changed; ask the \
                 user or do something else."
            );
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin("permission-modes"), Outcome::refused(why));
            }
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}

/// A subagent starts in its parent's mode, or the configured one when that
/// is stricter. A restored subagent keeps its saved mode: it is not
/// spawned through a request again.
fn inherit(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &ForAgent), With<SpawnSubagent>>,
    agents: Query<Option<&PermissionMode>>,
    settings: Res<Permissions>,
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(child), &ForAgent(parent))) = requests.get(spawned.entity) else {
        return;
    };
    let parent = settings.mode(agents.get(parent).ok().flatten());
    let mode = settings
        .subagents
        .map_or(parent, |configured| configured.stricter(parent));
    commands.entity(child).insert_if_new(mode);
}

/// `/mode [name]`: sets the agent's mode, or cycles it to the next.
fn set_mode(
    In(args): In<CommandArgs>,
    agents: Query<Option<&PermissionMode>>,
    settings: Res<Permissions>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mode = match args.args.trim() {
        "" => settings.mode(agents.get(args.agent).ok().flatten()).next(),
        name => {
            let Some(mode) = PermissionMode::parse(name) else {
                let why = format!("No mode `{name}`: read-only, ask, edits or auto.");
                notices.write(Notice::error(args.agent, why));
                return;
            };
            mode
        }
    };
    commands.entity(args.agent).insert(mode);
    notices.write(Notice::info(args.agent, format!("Mode: {}.", mode.name())));
}

/// Shows each agent's mode on the status line.
fn show_mode(
    mut agents: Query<
        (Option<&PermissionMode>, &mut StatusItems),
        (
            With<Agent>,
            Or<(Changed<PermissionMode>, Added<StatusItems>)>,
        ),
    >,
    settings: Res<Permissions>,
) {
    for (mode, mut items) in &mut agents {
        let mode = settings.mode(mode);
        let tone = match mode {
            PermissionMode::ReadOnly => Tone::Cyan,
            PermissionMode::Ask => Tone::Dim,
            PermissionMode::Edits => Tone::Yellow,
            PermissionMode::Auto => Tone::Red,
        };
        items.show(MODE_ITEM.says(mode.name(), tone));
    }
}
