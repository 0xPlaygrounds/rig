//! The agents of this session: `/fork` makes one from the shown agent
//! (the kernel's [`ForkAgent`]), `/agents` lists them and shows one, and,
//! with the `tui` feature, rewind: select an earlier message of yours with
//! Alt+Up and press Enter, and the shown agent forks just before it, the
//! fork is shown and the message is back in the input. Showing an agent
//! makes its top-level agent the session's current one (the harness's
//! `CurrentAgent`), which a resumed session shows again.

use crate::session_picker::{SessionPicker, SessionPickerPlugin, SessionSource};
use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;
use rig_core::conversation::short_id;
use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Notice, Spawned, SpawnedBy};
use rig_ecs::fork::{AgentForked, ForkAgent, ForkedFrom};
use rig_ecs::inbox::Recalled;
use rig_ecs::model::ModelChoice;
use rig_ecs::prelude::MessageId;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::{PickItem, Picker};
use rig_harness::prelude::{
    CurrentAgentPlugin, CurrentAgentSystems, MainAgentQuery, ShownAgent, main_agent,
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

#[cfg(feature = "tui")]
mod rewind;

/// `/fork`, `/agents` and, with the `tui` feature, rewind.
#[derive(Default)]
pub struct AgentsPlugin;

impl Plugin for AgentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionPickerPlugin>() {
            app.add_plugins(SessionPickerPlugin);
        }
        if !app.is_plugin_added::<CurrentAgentPlugin>() {
            app.add_plugins(CurrentAgentPlugin);
        }
        app.init_resource::<Forking>()
            .add_command(
                "fork",
                "Fork the shown agent into a new one and show it: /fork at the end, \
                 /fork at to pick a message, /fork <message> through it",
                fork,
            )
            .add_command(
                "agents",
                "Pick an agent to show; /agents <id> shows it",
                agents,
            )
            .add_observer(forked)
            .add_systems(
                Last,
                (relist.run_if(any_with_component::<AgentList>)).after(CurrentAgentSystems),
            );
        #[cfg(feature = "tui")]
        rewind::add(app);
    }
}

/// A column of an agent's line in `/agents`, such as what it spent or why
/// it waits: spawn one with a [`ColumnOf`] the agent and keep its `text`
/// current. Columns follow the agent's name, model and state, by `tag`;
/// an empty `text`, or Bevy's `Disabled` on the entity, leaves one out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct AgentColumn {
    /// What it is, and what sorts it, such as `spent`.
    pub tag: String,
    /// What the line shows.
    pub text: String,
}

impl AgentColumn {
    /// `text` in an agent's line, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// On an [`AgentColumn`]: the agent whose line it is in. Despawning the
/// agent despawns it; a restored agent is a new entity, so a plugin
/// spawns its columns again.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = AgentColumns)]
pub struct ColumnOf(pub Entity);

/// The [`AgentColumn`]s of this agent.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ColumnOf, linked_spawn)]
pub struct AgentColumns(Vec<Entity>);

/// The forks this plugin asked for, until the kernel made them: the agent,
/// the message, and the text to put back in the input once the fork is
/// shown.
#[derive(Resource, Default)]
struct Forking(Vec<(Entity, Option<MessageId>, Option<String>)>);

impl Forking {
    /// Forks `agent` through `through`, then shows the fork with `recall`
    /// in its input.
    fn ask(
        &mut self,
        commands: &mut Commands,
        (agent, through): (Entity, Option<MessageId>),
        recall: Option<String>,
    ) {
        self.0.push((agent, through, recall));
        commands.trigger(ForkAgent { agent, through });
    }
}

/// Shows `agent` in the front the user uses, and makes it the
/// [`ShownAgent`] in any app.
fn show(agent: Entity, commands: &mut Commands) {
    #[cfg(feature = "tui")]
    commands.trigger(rig_tui::ShowAgent { entity: agent });
    commands.queue(move |world: &mut World| {
        if let Some(mut shown) = world.get_resource_mut::<ShownAgent>() {
            shown.set_if_neq(ShownAgent(Some(agent)));
        }
    });
}

/// `/fork`: forks the agent the command is for at the end, through the
/// message named (`m4`), or, with `at`, through the message picked from
/// its entries.
fn fork(
    In(args): In<CommandArgs>,
    ids: Query<&AgentId>,
    mut forking: ResMut<Forking>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let through = match args.args.trim() {
        "" => None,
        "at" => {
            let agent = ids.get(args.agent).ok().map(|id| id.0.clone());
            commands.spawn((
                Picker::new("Fork through which message?").for_agent(args.agent),
                SessionPicker {
                    source: SessionSource::Entries {
                        session: None,
                        agent,
                    },
                    command: "fork {entry}".to_owned(),
                    ..SessionPicker::default()
                },
            ));
            return;
        }
        named => match named.parse::<MessageId>() {
            Ok(id) => Some(id),
            Err(_) => {
                let why = format!("`{named}` is no message, such as `m4`; /fork at picks one.");
                notices.write(Notice::error(args.agent, why));
                return;
            }
        },
    };
    forking.ask(&mut commands, (args.agent, through), None);
}

/// Shows each fork this plugin asked for, with what it puts back in the
/// input.
fn forked(
    forked: On<AgentForked>,
    mut forking: ResMut<Forking>,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let asked = forking
        .0
        .iter()
        .position(|(agent, through, _)| (*agent, *through) == (forked.from, forked.through));
    let Some((_, _, recall)) = asked.map(|at| forking.0.remove(at)) else {
        return;
    };
    show(forked.fork, &mut commands);
    let said = "Forked into a new agent; /agents shows the other.";
    notices.write(Notice::info(forked.fork, said));
    if let Some(text) = recall {
        recalled.write(Recalled {
            agent: forked.fork,
            text,
            why: None,
        });
    }
}

/// An agent as `/agents` lists it.
#[derive(QueryData)]
struct Listed {
    entity: Entity,
    id: &'static AgentId,
    name: &'static Name,
    model: Option<&'static ModelChoice>,
    busy: Has<ActiveTurn>,
    from: Option<&'static ForkedFrom>,
    columns: Option<&'static AgentColumns>,
}

/// What `/agents` lists: every agent, each under the agent that spawned
/// it, with the main agent and its columns.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    agents: Query<'w, 's, Listed, With<Agent>>,
    main: MainAgentQuery<'w, 's>,
    columns: Query<'w, 's, &'static AgentColumn>,
    spawned: Query<'w, 's, &'static Spawned>,
    parents: Query<'w, 's, &'static SpawnedBy>,
}

impl Agents<'_, '_> {
    /// A line per agent, top-level agents by id with the agents each
    /// spawned below it, and the position of `agent`'s line.
    fn items(&self, agent: Option<Entity>) -> (Vec<PickItem>, usize) {
        let main = main_agent(&self.main);
        let mut roots: Vec<_> = (self.agents.iter())
            .filter(|agent| !self.parents.contains(agent.entity))
            .collect();
        roots.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        let listed = roots.iter().flat_map(|root| {
            let spawned = self.spawned.iter_descendants_depth_first(root.entity);
            std::iter::once(root.entity).chain(spawned)
        });
        let (mut items, mut selected) = (Vec::new(), 0);
        for listed in listed.filter_map(|listed| self.agents.get(listed).ok()) {
            if Some(listed.entity) == agent {
                selected = items.len();
            }
            let depth = self.parents.iter_ancestors(listed.entity).count();
            let from = (listed.from)
                .map(|from| format!(" · fork of {}", short_id(&from.0)))
                .unwrap_or_default();
            let current = if Some(listed.entity) == main {
                " · current"
            } else {
                ""
            };
            let model = listed.model.map_or("no model", |model| model.0.as_str());
            let state = if listed.busy { "working" } else { "idle" };
            let mut own: Vec<&AgentColumn> = (listed.columns.into_iter())
                .flat_map(|of| of.iter())
                .filter_map(|column| self.columns.get(column).ok())
                .filter(|column| !column.text.is_empty())
                .collect();
            own.sort_by(|a, b| a.tag.cmp(&b.tag));
            let own: String = own
                .iter()
                .map(|column| format!(" · {}", column.text))
                .collect();
            items.push(PickItem {
                label: format!(
                    "{}{} {} · {model} · {state}{current}{from}{own}",
                    "  ".repeat(depth),
                    listed.name,
                    listed.id.short(),
                ),
                command: format!("agents {}", listed.id.0),
                detail: None,
                more: None,
                checked: false,
            });
        }
        (items, selected)
    }
}

/// On the picker `/agents` opened, whose lines follow the agents.
#[derive(Component)]
struct AgentList;

/// `/agents`: picks an agent to show, each listed under the agent that
/// spawned it, marked when it is current or a fork, with its
/// [`AgentColumn`]s, or shows the one whose id starts with the argument.
fn agents(
    In(args): In<CommandArgs>,
    agents: Agents,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !args.args.is_empty() {
        match (agents.agents.iter()).find(|agent| agent.id.0.starts_with(&args.args)) {
            Some(agent) => show(agent.entity, &mut commands),
            None => {
                let why = format!("No agent's id starts with `{}`.", args.args);
                notices.write(Notice::error(args.agent, why));
            }
        }
        return;
    }
    let (items, selected) = agents.items(Some(args.agent));
    commands.spawn((
        Picker {
            items,
            selected,
            ..Picker::new("Show an agent").for_agent(args.agent)
        },
        AgentList,
    ));
}

/// Keeps the lines of an open `/agents` picker current, such as an agent's
/// state once its turn ended.
fn relist(mut lists: Query<&mut Picker, With<AgentList>>, agents: Agents) {
    for mut picker in &mut lists {
        let (items, _) = agents.items(picker.agent);
        if picker.items != items {
            picker.items = items;
        }
    }
}

#[cfg(test)]
mod tests;
