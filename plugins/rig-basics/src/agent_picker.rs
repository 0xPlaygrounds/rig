//! `/agents`, the picker of a session's agents: each listed under the
//! agent that spawned it, marked when it is current or a fork, with the
//! [`AgentColumn`]s other plugins add to its line. Picking one, or
//! `/agents <id>`, shows it and makes its top-level agent the session's
//! current one (the harness's `CurrentAgent`), which a resumed session
//! shows again. The plugins that make more than one agent add
//! [`AgentPickerPlugin`]; it is no entry of `/plugins` of its own.
//!
//! ```
//! use rig_basics::agent_picker::{AgentColumn, ColumnOf};
//! use rig_harness::prelude::*;
//!
//! /// Shows on each agent's line that it is on call.
//! fn on_call(added: On<Add<Agent>>, mut commands: Commands) {
//!     commands.spawn((AgentColumn::new("on-call", "on call"), ColumnOf(added.entity)));
//! }
//! ```

use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;
use rig_core::conversation::short_id;
use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Notice, Spawned, SpawnedBy};
use rig_ecs::fork::ForkedFrom;
use rig_ecs::model::ModelChoice;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::{PickItem, Picker};
use rig_harness::prelude::{
    CurrentAgentPlugin, CurrentAgentSystems, MainAgentQuery, ShownAgent, main_agent,
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

/// `/agents`.
#[derive(Default)]
pub struct AgentPickerPlugin;

impl Plugin for AgentPickerPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CurrentAgentPlugin>() {
            app.add_plugins(CurrentAgentPlugin);
        }
        app.add_command(
            "agents",
            "Pick an agent to show; /agents <id> shows it",
            agents,
        );
        let relist = relist
            .run_if(any_with_component::<AgentList>)
            .after(CurrentAgentSystems);
        #[cfg(feature = "tui")]
        let relist = relist.before(rig_tui::ChoiceSystems);
        app.add_systems(Update, relist);
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

/// Shows `agent` in the front the user uses, and makes it the
/// [`ShownAgent`] in any app.
pub(crate) fn show(agent: Entity, commands: &mut Commands) {
    #[cfg(feature = "tui")]
    commands.trigger(rig_tui::ShowAgent { entity: agent });
    commands.queue(move |world: &mut World| {
        if let Some(mut shown) = world.get_resource_mut::<ShownAgent>() {
            shown.set_if_neq(ShownAgent(Some(agent)));
        }
    });
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
            let from = (listed.from.and_then(|from| from.agents().next()))
                .map(|from| format!(" · fork of {}", short_id(from)))
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
            let label = format!(
                "{}{} {} · {model} · {state}{current}{from}{own}",
                "  ".repeat(depth),
                listed.name,
                listed.id.short(),
            );
            items.push(PickItem::new(label, format!("agents {}", listed.id.0)));
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
/// state once its turn ended, before the terminal view finds its
/// selection again.
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
