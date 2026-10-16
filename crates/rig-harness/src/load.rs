//! What each plugin added, as data in the world: one entity per plugin
//! `plugins.toml` lists (its `Name` and [`PluginSource`]), and
//! [`ProvidedBy`] on what that plugin spawned, such as its commands,
//! tools, prompt sections and observers. "What did plugin X add" is a
//! query for `ProvidedBy(x)`; which plugin added a schedule's system is in
//! [`PluginSystems`]; what it keeps with the session is its [`Keeps`].
//!
//! Every named entity, observer and schedule system a plugin adds in its
//! `build`, `finish` and `cleanup` is attributed. What it spawns later, such as a
//! tool an MCP server lists, it attributes itself, with its entity from
//! [`PluginEntity<P>`]:
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct McpPlugin;
//!
//! impl Plugin for McpPlugin {
//!     fn build(&self, _app: &mut App) {}
//! }
//!
//! fn spawn_listed_tool(me: Option<Res<PluginEntity<McpPlugin>>>, mut commands: Commands) {
//!     let tool = commands.spawn(Name::new("listed tool")).id();
//!     if let Some(me) = me {
//!         commands.entity(tool).insert(ProvidedBy(me.entity()));
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use bevy_app::prelude::*;
use bevy_ecs::observer::Observer;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{InternedScheduleLabel, Schedules, SystemKey};
use bevy_log::error;
use bevy_reflect::prelude::*;
use rig_ecs::prelude::Notice;
use rig_ecs::restore::SavedKeys;
use rig_ecs::stage::AgentPass;

/// Where a plugin comes from, on the entity [`load`] spawns for it.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct PluginSource {
    /// Its package, such as `rig-harness`.
    pub package: String,
    /// Where the package is built from, as `plugins.toml` names it, such
    /// as `path /home/me/.rig/plugins/viz` or `version 0.3`.
    pub source: String,
}

/// The plugin that spawned this entity.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Provides)]
pub struct ProvidedBy(pub Entity);

/// What a plugin spawned.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ProvidedBy)]
pub struct Provides(Vec<Entity>);

/// The entity [`load`] spawned for the plugin `P`, for what `P` spawns
/// while the app runs.
#[derive(Resource)]
pub struct PluginEntity<P: Plugin> {
    entity: Entity,
    plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> PluginEntity<P> {
    /// The plugin's entity.
    pub fn entity(&self) -> Entity {
        self.entity
    }
}

/// Which plugin added each schedule system, by the system's name and its
/// schedule's label (such as `AgentPass`), as [`load`] saw it. Two plugins
/// that add a system of the same name to one schedule are told apart by
/// neither; a system a plugin adds later, while the app runs, is no
/// plugin's.
#[derive(Resource, Default, Debug)]
pub struct PluginSystems(HashMap<(String, String), Option<Entity>>);

impl PluginSystems {
    /// The plugin that added the system `system` to the schedule labelled
    /// `schedule`, if one plugin did.
    pub fn plugin(&self, system: &str, schedule: &str) -> Option<Entity> {
        let key = (system.to_owned(), schedule.to_owned());
        self.0.get(&key).copied().flatten()
    }

    /// Records each system added since `before` as `plugin`'s.
    fn record(
        world: &mut World,
        before: &HashSet<(InternedScheduleLabel, SystemKey)>,
        plugin: Entity,
    ) {
        let mut added = Vec::new();
        if let Some(schedules) = world.get_resource::<Schedules>() {
            for (label, schedule) in schedules.iter() {
                for (key, system, _) in schedule.graph().systems.iter() {
                    if !before.contains(&(schedule.label(), key)) {
                        added.push((system.name().to_string(), format!("{label:?}")));
                    }
                }
            }
        }
        let mut systems = world.get_resource_or_init::<PluginSystems>();
        for key in added {
            let by = systems.0.entry(key).or_insert(Some(plugin));
            if *by != Some(plugin) {
                *by = None;
            }
        }
    }
}

/// Each pair of systems of one [`AgentPass`] stage that use the same data,
/// one writing it, with no order between them, so which runs first is
/// chance: one line each, naming their plugins (by type, or `the harness`
/// for the harness's, the kernel's and Bevy's own), the systems and the
/// data. It initializes the agent loop, as the loop's first run would; an
/// app without one has none. `rig plugin check --build` reports these for
/// the plugins `plugins.toml` lists, and a debug build at its start
/// ([`check_stages_at_start`]).
pub fn stage_conflicts(world: &mut World) -> Vec<String> {
    let label = format!("{AgentPass:?}");
    let conflicts = world.try_schedule_scope(AgentPass, |world, pass| {
        // Named before the build, which moves the systems out of the graph
        // into the schedule a run has.
        let built = pass.systems().into_iter().flatten();
        let names: HashMap<SystemKey, String> = (pass.graph().systems.iter())
            .map(|(key, system, _)| (key, system))
            .chain(built)
            .map(|(key, system)| (key, system.name().to_string()))
            .collect();
        let failure = pass.initialize(world).err();
        let graph = pass.graph();
        let name = |key| {
            names
                .get(key)
                .cloned()
                .unwrap_or_else(|| "an unnamed system".to_owned())
        };
        let conflicts: Vec<(String, String, String)> = (graph.conflicting_systems().iter())
            .map(|(first, second, data)| {
                let data = data
                    .iter()
                    .filter_map(|id| world.components().get_name(*id));
                let data = data.map(|name| name.to_string()).collect::<Vec<_>>();
                (name(first), name(second), data.join(", "))
            })
            .collect();
        match failure {
            Some(failure) if conflicts.is_empty() => vec![(
                String::new(),
                String::new(),
                failure.to_string(graph, world),
            )],
            _ => conflicts,
        }
    });
    let systems = world.get_resource::<PluginSystems>();
    let plugin = |system: &str| {
        systems
            .and_then(|systems| systems.plugin(system, &label))
            .and_then(|plugin| world.get::<Name>(plugin))
            .map_or_else(|| "the harness".to_owned(), |name| format!("`{name}`"))
    };
    let conflicts = conflicts.unwrap_or_default().into_iter();
    conflicts
        .map(|(first, second, data)| {
            if first.is_empty() {
                return format!("the agent loop does not build: {data}");
            }
            let on = if data.is_empty() {
                "the whole world".to_owned()
            } else {
                data
            };
            let plugins = match (plugin(&first), plugin(&second)) {
                (one, other) if one == other => one,
                (one, other) => format!("{one} and {other}"),
            };
            format!("{plugins}: {first} and {second}, on {on}")
        })
        .collect()
}

/// Shows each of the app's [`stage_conflicts`] as a warning notice at its
/// start. The generated `main.rs` adds it to a debug build.
pub fn check_stages_at_start(app: &mut App) {
    app.add_systems(Startup, warn_stage_conflicts);
}

fn warn_stage_conflicts(world: &mut World) {
    for conflict in stage_conflicts(world) {
        world.write_message(Notice::warn(
            None,
            format!(
                "Agent-loop systems use the same data with no order between them, so which \
                 runs first is chance: {conflict}"
            ),
        ));
    }
}

/// Every system of every schedule, by its schedule's label.
fn systems(world: &World) -> HashSet<(InternedScheduleLabel, SystemKey)> {
    let Some(schedules) = world.get_resource::<Schedules>() else {
        return HashSet::new();
    };
    let systems = schedules.iter().flat_map(|(_, schedule)| {
        (schedule.graph().systems.iter()).map(|(key, ..)| (schedule.label(), key))
    });
    systems.collect()
}

/// Adds `P`, unless the app has it already (another plugin added it), as
/// the generated `main.rs` adds each plugin `plugins.toml` lists: with an
/// entity of its own, named after it, from `package` at `source`.
pub fn load<P: Plugin + Default>(app: &mut App, package: &str, source: &str) {
    if app.is_plugin_added::<P>() {
        return;
    }
    let plugin = P::default();
    let source = PluginSource {
        package: package.to_owned(),
        source: source.to_owned(),
    };
    let entity = app
        .world_mut()
        .spawn((Name::new(plugin.name().to_owned()), source))
        .id();
    app.insert_resource(PluginEntity::<P> {
        entity,
        plugin: PhantomData,
    });
    let attribute = attribute(app.world_mut(), entity);
    let before = systems(app.world());
    let outer = app.world_mut().remove_resource::<Loading>();
    app.insert_resource(Loading(entity));
    // `Attribute` before and after it brackets its `finish` and `cleanup`
    // (and the plugins it adds), which run in the order plugins were
    // added.
    app.add_plugins(Attribute::Open(entity));
    app.add_plugins(plugin);
    app.add_plugins(Attribute::Close);
    app.world_mut().remove_resource::<Loading>();
    if let Some(outer) = outer {
        app.insert_resource(outer);
    }
    PluginSystems::record(app.world_mut(), &before, entity);
    stop_attributing(app.world_mut(), attribute);
}

/// What a plugin keeps with the session, on the entity [`load`] spawned for
/// it: the saved values it owns, by their saved key (the type path, such as
/// `my_plugin::Counter`, or a stable one such as `rig::model`), and its
/// folders of session files in
/// [`PluginData::session`](crate::prelude::PluginData::session), such as
/// `vim`. A plugin says so in `build`, `finish` or `cleanup`
/// ([`AppKeepsExt`]); nothing infers it. A saved value no plugin keeps is
/// the kernel's or the harness's, or one whose plugin did not say.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Keeps {
    /// The keys of the saved values it keeps.
    pub saved: Vec<String>,
    /// Its folders in the session's plugin files.
    pub files: Vec<String>,
}

impl Keeps {
    /// Whether the saved value of `key` is this plugin's.
    pub fn saved(&self, key: &str) -> bool {
        self.saved.iter().any(|saved| saved == key)
    }

    /// Whether the session file `name`, as `PluginData::session` names it
    /// (`vim/registers.json`), is in one of this plugin's folders.
    pub fn file(&self, name: &str) -> bool {
        self.files.iter().any(|folder| {
            name.strip_prefix(folder.as_str())
                .is_some_and(|rest| rest.starts_with('/'))
        })
    }
}

/// Saying what a plugin keeps with the session, in its `build`, `finish`
/// or `cleanup`, onto its [`Keeps`]. A plugin added without [`load`] has no
/// entity to say it on, and nothing is recorded.
pub trait AppKeepsExt {
    /// This plugin keeps the saved value `T` (`#[reflect(.., Saved)]`, or
    /// `Follows`).
    fn keeps_saved<T: Component + TypePath>(&mut self) -> &mut Self;

    /// This plugin keeps the session files under `folder` in
    /// `PluginData::session`, such as `vim` for `vim/registers.json`.
    fn keeps_files(&mut self, folder: &str) -> &mut Self;

    /// This plugin's saved value `T` (an agent component or a resource) is
    /// logged and restored under `key`, such as `goals.goal`, rather than its
    /// type path, so a rename keeps it and a program without Bevy reads it
    /// by that key (rig-memory's `Session::value`). Keys are namespaced by
    /// the plugin; a key another type has, or a second key for `T`, is
    /// refused with an error in the log. [`SavedAs::listed`] also lists it.
    fn saved_as<T: TypePath>(&mut self, key: &'static str) -> SavedAs<'_>;
}

/// A saved value's key, as [`AppKeepsExt::saved_as`] recorded it.
#[must_use = "a saved key is recorded already; call `listed` to list its values too"]
pub struct SavedAs<'a> {
    app: &'a mut App,
    /// The key, or `None` when it was refused.
    key: Option<&'static str>,
}

impl<'a> SavedAs<'a> {
    /// The value is listed too: copied into the session's listing
    /// (rig-memory's `SessionMeta::listed`) whenever it changes, so a listing
    /// and a session picker show and filter by it without opening a log.
    /// Keep listed values small.
    pub fn listed(self) -> &'a mut App {
        if let Some(key) = self.key {
            self.app
                .world_mut()
                .get_resource_or_init::<SavedKeys>()
                .list(key);
        }
        self.app
    }
}

impl AppKeepsExt for App {
    fn keeps_saved<T: Component + TypePath>(&mut self) -> &mut Self {
        keep(self.world_mut(), |keeps| &mut keeps.saved, T::type_path());
        self
    }

    fn keeps_files(&mut self, folder: &str) -> &mut Self {
        let folder = folder.trim_end_matches('/');
        keep(self.world_mut(), |keeps| &mut keeps.files, folder);
        self
    }

    fn saved_as<T: TypePath>(&mut self, key: &'static str) -> SavedAs<'_> {
        let world = self.world_mut();
        let added = world
            .get_resource_or_init::<SavedKeys>()
            .add(T::type_path(), key);
        let key = match added {
            Ok(()) => {
                keep(world, |keeps| &mut keeps.saved, key);
                Some(key)
            }
            Err(failure) => {
                error!("not saving `{}` under `{key}`: {failure}", T::type_path());
                None
            }
        };
        SavedAs { app: self, key }
    }
}

/// Adds `what` to the list `of` of the [`Keeps`] of the plugin being
/// built, finished or cleaned up, once.
fn keep(world: &mut World, of: fn(&mut Keeps) -> &mut Vec<String>, what: &str) {
    let loading = world.get_resource::<Loading>().map(|loading| loading.0);
    let plugin = loading.or_else(|| Some(world.get_resource::<Attributing>()?.plugin));
    let Some(mut plugin) = plugin.and_then(|plugin| world.get_entity_mut(plugin).ok()) else {
        return;
    };
    if !plugin.contains::<Keeps>() {
        plugin.insert(Keeps::default());
    }
    let Some(mut keeps) = plugin.get_mut::<Keeps>() else {
        return;
    };
    let list = of(&mut keeps);
    if !list.iter().any(|kept| kept == what) {
        list.push(what.to_owned());
    }
}

/// The plugin [`load`] is building.
#[derive(Resource)]
struct Loading(Entity);

/// Marks each named entity and observer spawned from now on as
/// `plugin`'s, until [`stop_attributing`].
fn attribute(world: &mut World, plugin: Entity) -> Entity {
    world
        .add_observer(
            move |added: On<Add<(Name, Observer)>>, mut commands: Commands| {
                commands
                    .entity(added.entity)
                    .try_insert_if_new(ProvidedBy(plugin));
            },
        )
        .id()
}

/// Spawns what was queued while still attributed, then stops.
fn stop_attributing(world: &mut World, attribute: Entity) {
    world.flush();
    world.despawn(attribute);
}

/// Attributes what a plugin spawns in `finish` and `cleanup`, from its
/// [`Attribute::Open`] to the [`Attribute::Close`] after it.
enum Attribute {
    Open(Entity),
    Close,
}

/// The plugin being finished or cleaned up, the observer attributing to
/// it, and the systems before it.
#[derive(Resource)]
struct Attributing {
    plugin: Entity,
    observer: Entity,
    before: HashSet<(InternedScheduleLabel, SystemKey)>,
}

impl Attribute {
    fn bracket(&self, app: &mut App) {
        let world = app.world_mut();
        match self {
            Self::Open(plugin) => {
                let observer = attribute(world, *plugin);
                let before = systems(world);
                world.insert_resource(Attributing {
                    plugin: *plugin,
                    observer,
                    before,
                });
            }
            Self::Close => {
                if let Some(attributing) = world.remove_resource::<Attributing>() {
                    PluginSystems::record(world, &attributing.before, attributing.plugin);
                    stop_attributing(world, attributing.observer);
                }
            }
        }
    }
}

impl Plugin for Attribute {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        self.bracket(app);
    }

    fn cleanup(&self, app: &mut App) {
        self.bracket(app);
    }

    fn is_unique(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
