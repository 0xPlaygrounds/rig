//! `inspect`, an optional plugin of the rig-harness agent: the agent reads
//! its own world through the Bevy Remote Protocol, answered in process by
//! Bevy's `RemotePlugin`, with no transport and no port. Bevy answers each
//! request; rig writes no query handler. [`InspectPlugin`] is listed in
//! `plugins.toml` as `crate = "rig-inspect"`.
//!
//! Only read-only methods are sent: a method is read-only when its system's
//! entity has [`ReadOnlyMethod`]. This plugin marks Bevy's nine read-only
//! methods; a plugin that adds its own method with
//! [`RemoteMethods::insert`] (in `finish`, or listed after this plugin)
//! marks it the same way:
//!
//! ```ignore
//! let id = app.world_mut().register_system(count_calls);
//! app.world_mut().entity_mut(id.entity()).insert(ReadOnlyMethod);
//! app.world_mut().resource_mut::<RemoteMethods>()
//!     .insert("tools.counts", RemoteMethodSystemId::Instant(id));
//! ```
//!
//! A method is checked before it is sent: Bevy stops reading its mailbox
//! for the frame at a method it does not know.

use std::io::Write as _;

use bevy_remote::builtin_methods::{
    BRP_GET_COMPONENTS_METHOD, BRP_GET_RESOURCE_METHOD, BRP_LIST_COMPONENTS_METHOD,
    BRP_LIST_RESOURCES_METHOD, BRP_QUERY_METHOD, BRP_REGISTRY_SCHEMA_METHOD, BRP_SCHEDULE_GRAPH,
    BRP_SCHEDULE_LIST, RPC_DISCOVER_METHOD,
};
use bevy_remote::schemas::SchemaTypesMetadata;
use bevy_remote::{BrpMessage, BrpResult, BrpSender, RemoteMethodSystemId, RemoteMethods};
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::calls::Running;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

pub use bevy_remote::RemotePlugin;

/// The tool's name.
pub const INSPECT_TOOL: &str = "inspect";

/// Bevy's methods that only read.
const READ_ONLY: [&str; 9] = [
    RPC_DISCOVER_METHOD,
    BRP_QUERY_METHOD,
    BRP_GET_COMPONENTS_METHOD,
    BRP_LIST_COMPONENTS_METHOD,
    BRP_GET_RESOURCE_METHOD,
    BRP_LIST_RESOURCES_METHOD,
    BRP_REGISTRY_SCHEMA_METHOD,
    BRP_SCHEDULE_LIST,
    BRP_SCHEDULE_GRAPH,
];

/// The longest answer returned whole, in bytes; a longer one is cut and
/// kept whole in the session's spill directory.
const CAP: usize = 16 * 1024;

const DESCRIPTION: &str = "Read your own running app, a Bevy world, through the Bevy Remote \
    Protocol: your plugins and what each added, agents, commands, tools, prompt sections, \
    settings, saved state, schedules and types. Read-only: `rpc.discover` lists the methods \
    you can send. Type names may be short (`Name`) where only one type has the name. Params: \
    `world.list_components` {entity} (or none: every registered one); `world.list_resources`; \
    `world.query` {data: {components: [..], option: [..] or \"all\", has: [..]}, filter: {with: \
    [..], without: [..]}}; `world.get_components` {entity, components: [..]}; \
    `world.get_resources` {resource}; `registry.schema` {with_crates: [..], type_limit: {with: \
    [..]}}, where `type_limit.with` takes reflect trait names, not type names: `Saved` lists \
    what is saved with the session, `Component` the components; `schedule.list`; \
    `schedule.graph` {schedule_label}. Plugins are the entities with `PluginSource`; what one \
    added has `ProvidedBy` naming it. A long answer is cut and kept whole in a file to read or \
    search.";

/// The system prompt's line, after the section on what the agent is.
const PROMPT: &str = "To learn about yourself (plugins and what each added, agents, settings, \
    saved state, warnings in the `Diagnostics` resource, types), use `inspect` {method, params} \
    before reading rig's source or running commands; `rpc.discover` lists its methods.";

/// A Bevy Remote method `inspect` may send, on the entity of the method's
/// system ([`RemoteMethodSystemId`]).
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct ReadOnlyMethod;

/// Adds Bevy's `RemotePlugin` without a transport, the `inspect` tool, and
/// a system prompt line on when to use it.
#[derive(Default)]
pub struct InspectPlugin;

impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<RemotePlugin>() {
            app.add_plugins(RemotePlugin::default());
        }
        let world = app.world_mut();
        let read_only: Vec<Entity> = world
            .get_resource::<RemoteMethods>()
            .map(|methods| READ_ONLY.iter().filter_map(|name| methods.get(name)))
            .into_iter()
            .flatten()
            .map(system)
            .collect();
        for entity in read_only {
            world.entity_mut(entity).insert(ReadOnlyMethod);
        }
        // `registry.schema` lists `Saved` among a type's `reflectTypes`.
        if let Some(mut types) = world.get_resource_mut::<SchemaTypesMetadata>() {
            types.map_type_data::<ReflectSaved>("Saved");
        }
        let spill = world
            .get_resource::<SessionPaths>()
            .map(SessionPaths::spill);
        app.insert_resource(Answers(spill))
            .add_open_tool(
                INSPECT_TOOL,
                DESCRIPTION,
                ToolOptions {
                    rules: &[],
                    footprint: Footprint::ReadOnly,
                },
                on_inspect,
            )
            .world_mut()
            .spawn((
                Name::new("prompt:inspect"),
                PromptSection::new(PromptSection::ORDER_PROJECT - 99, "inspect", PROMPT),
            ));
    }
}

/// Where answers that are too long are kept whole.
#[derive(Resource)]
struct Answers(Option<Spill>);

/// The arguments of an `inspect` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectArgs {
    /// A read-only Bevy Remote method, such as `world.query`.
    method: String,
    /// The method's params; none, or empty, for a method that takes none.
    params: Option<Map<String, Value>>,
}

/// The entity of a method's system.
fn system(method: &RemoteMethodSystemId) -> Entity {
    match method {
        RemoteMethodSystemId::Instant(id) => id.entity(),
        RemoteMethodSystemId::Watching(id) => id.entity(),
    }
}

/// Sends a read-only call's request to Bevy and answers the call with the
/// reply, off the main thread; refuses any other at once.
fn on_inspect(
    called: On<ToolCalled<InspectArgs>>,
    (methods, read_only): (Option<Res<RemoteMethods>>, Query<(), With<ReadOnlyMethod>>),
    (sender, registry, answers, wake): (
        Option<Res<BrpSender>>,
        Res<AppTypeRegistry>,
        Res<Answers>,
        Res<Wake>,
    ),
    mut commands: Commands,
) {
    let allowed = |name: &str| {
        let method = methods.as_ref().and_then(|methods| methods.get(name));
        method.is_some_and(|method| read_only.contains(system(method)))
    };
    let mut names = methods.as_ref().map(|m| m.methods()).unwrap_or_default();
    names.retain(|name| allowed(name));
    names.sort();
    let method = called.args.method.trim();
    let failed = |why: String| ToolOutput(ToolResult::failed(ToolExecutionError::refused(why)));
    let mut call = commands.entity(called.call);
    if !allowed(method) {
        call.insert(failed(format!(
            "`{method}` is not a read-only method here, so nothing was sent. inspect sends: {}.",
            names.join(", ")
        )));
        return;
    }
    let params = called
        .args
        .params
        .clone()
        .filter(|params| !params.is_empty());
    let mut params = params.map(Value::Object);
    if let Some(params) = &mut params {
        let registry = registry.read();
        rename(params, &|name| {
            let registration = registry.get_with_short_type_path(name)?;
            Some(registration.type_info().type_path().to_owned())
        });
    }
    let (reply, replied) = async_channel::bounded(1);
    let message = BrpMessage {
        method: method.to_owned(),
        params,
        sender: reply,
    };
    if sender.is_none_or(|sender| sender.try_send(message).is_err()) {
        call.insert(failed("Bevy Remote is not taking requests now.".to_owned()));
        return;
    }
    wake.wake();
    let listed = (method == RPC_DISCOVER_METHOD).then_some(names);
    let (registry, spill) = (registry.clone(), answers.0.clone());
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    call.insert(Running::spawn(pool, &wake, async move {
        let reply = replied.recv().await.ok();
        blocking(move || Ok(answer(reply, listed, &registry, spill.as_ref())))
            .await
            .unwrap_or_else(ToolResult::failed)
    }));
}

/// The tool's output for Bevy's `reply`: the methods `listed`, for
/// `rpc.discover`; type paths shortened where `registry` has one type of
/// that short name; cut at [`CAP`], the whole kept in `spill`.
fn answer(
    reply: Option<BrpResult>,
    listed: Option<Vec<String>>,
    registry: &AppTypeRegistry,
    spill: Option<&Spill>,
) -> ToolResult {
    let mut value = match reply {
        Some(Ok(value)) => value,
        Some(Err(error)) => {
            let data = error
                .data
                .map(|data| format!(" {data}"))
                .unwrap_or_default();
            let why = format!("Bevy Remote error {}: {}{data}", error.code, error.message);
            return ToolResult::failed(ToolExecutionError::invalid_args(why));
        }
        None => return ToolResult::failed(ToolExecutionError::other("Bevy did not answer.")),
    };
    if let (Some(listed), Some(Value::Array(methods))) = (listed, value.get_mut("methods")) {
        methods.retain(|method| {
            let name = method.get("name").and_then(Value::as_str);
            name.is_some_and(|name| listed.iter().any(|listed| listed == name))
        });
    }
    let registry = registry.read();
    rename(&mut value, &|path| {
        let short = registry
            .get_with_type_path(path)?
            .type_info()
            .type_path_table();
        let short = short.short_path();
        registry.get_with_short_type_path(short)?;
        Some(short.to_owned())
    });
    let mut text = value.to_string();
    if text.len() <= CAP {
        return ToolResult::success(text.into());
    }
    let kept = spill.and_then(|spill| {
        let (path, mut file) = spill.create(INSPECT_TOOL).ok()?;
        file.write_all(text.as_bytes()).ok().map(|()| path)
    });
    let kept = kept.map_or("ask for less".to_owned(), |path| {
        format!("all of it is in {}: read or search it", path.display())
    });
    let total = text.len();
    text.truncate(text.floor_char_boundary(CAP));
    ToolResult::success(format!("{text}\n[Cut at {} of {total} bytes; {kept}]", text.len()).into())
}

/// Replaces each string and object key of `value` that `to` maps.
fn rename(value: &mut Value, to: &impl Fn(&str) -> Option<String>) {
    match value {
        Value::String(text) => {
            if let Some(renamed) = to(text) {
                *text = renamed;
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|item| rename(item, to)),
        Value::Object(map) => {
            *map = std::mem::take(map)
                .into_iter()
                .map(|(key, mut item)| {
                    rename(&mut item, to);
                    (to(&key).unwrap_or(key), item)
                })
                .collect();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
