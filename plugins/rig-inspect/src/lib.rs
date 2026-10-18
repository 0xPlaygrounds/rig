//! `inspect`, an optional plugin of the rig-harness agent: the agent reads
//! its own world through the Bevy Remote Protocol, answered in process by
//! Bevy's `RemotePlugin`, with no transport and no port. Bevy answers each
//! request; rig writes no query handler.
//!
//! Only read-only methods are sent: a method is read-only when its system's
//! entity has [`ReadOnlyMethod`]. This plugin marks Bevy's nine read-only
//! methods; a plugin that adds its own method with
//! [`RemoteMethods::insert`] (in `finish`, or listed after this plugin)
//! marks it the same way:
//!
//! ```no_run
//! use bevy_remote::{BrpResult, RemoteMethodSystemId, RemoteMethods};
//! use rig_harness::prelude::*;
//! use rig_inspect::ReadOnlyMethod;
//! use serde_json::Value;
//!
//! fn count_calls(In(_): In<Option<Value>>) -> BrpResult {
//!     Ok(Value::Null)
//! }
//!
//! let mut app = App::new();
//! let id = app.world_mut().register_system(count_calls);
//! app.world_mut().entity_mut(id.entity()).insert(ReadOnlyMethod);
//! app.world_mut().resource_mut::<RemoteMethods>()
//!     .insert("tools.counts", RemoteMethodSystemId::Instant(id));
//! ```
//!
//! A method is checked before it is sent: Bevy stops reading its mailbox
//! for the frame at a method it does not know.
//!
//! The same read-only methods are the session's channel method
//! `x/inspect/query` ([`InspectQuery`]), for the session's owner only: a
//! driver or a script reads the agent's world through `rigd`, which routes
//! the call to the session's worker. A child session's agent may not.

use std::io::Write as _;

use bevy_ecs::system::SystemParam;
use bevy_remote::builtin_methods::{
    BRP_GET_COMPONENTS_METHOD, BRP_GET_RESOURCE_METHOD, BRP_LIST_COMPONENTS_METHOD,
    BRP_LIST_RESOURCES_METHOD, BRP_QUERY_METHOD, BRP_REGISTRY_SCHEMA_METHOD, BRP_SCHEDULE_GRAPH,
    BRP_SCHEDULE_LIST, RPC_DISCOVER_METHOD,
};
use bevy_remote::schemas::SchemaTypesMetadata;
use bevy_remote::{BrpMessage, BrpResult, BrpSender, RemoteMethodSystemId, RemoteMethods};
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::channel::{Call, ChannelAppExt};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::tasks::Running;
use rig_harness::server::Principal;
use rig_harness::view::protocol::{Method, RpcError, SessionId, codes};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
/// kept whole in the session's [`Scratch`] spill directory.
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
        app.add_method::<InspectQuery, _, _>(query);
        app.add_open_tool(
            INSPECT_TOOL,
            DESCRIPTION,
            ToolOptions {
                footprint: Some(Touches::Reads(Scope::Anything)),
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

/// The arguments of an `inspect` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectArgs {
    /// A read-only Bevy Remote method, such as `world.query`.
    method: String,
    /// The method's params; none, or empty, for a method that takes none.
    params: Option<Map<String, Value>>,
}

/// `x/inspect/query {sessionId, method, params?}`: a read-only Bevy Remote
/// method's answer, for the session's owner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectQuery {
    /// The session whose agent's world is read.
    pub session_id: SessionId,
    /// A read-only Bevy Remote method, such as `world.query`.
    pub method: String,
    /// The method's params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl Method for InspectQuery {
    const NAME: &'static str = "x/inspect/query";
    type Params = InspectQuery;
    type Result = Value;
}

/// Answers an [`InspectQuery`] at once, running the method's system: only
/// for an owner, and only a read-only method.
fn query(In(call): In<Call<InspectQuery>>, world: &mut World) -> Result<Value, RpcError> {
    if call.caller.principal != Principal::Owner {
        let why = "only the session's owner may inspect its agent";
        return Err(RpcError::new(codes::UNAUTHORIZED, why));
    }
    let method = call.params.method.trim();
    let found = world
        .get_resource::<RemoteMethods>()
        .and_then(|methods| methods.get(method))
        .cloned();
    let read_only = found
        .as_ref()
        .filter(|found| world.get::<ReadOnlyMethod>(system(found)).is_some());
    let Some(RemoteMethodSystemId::Instant(id)) = read_only.cloned() else {
        let why = format!("`{method}` is not a read-only method here");
        return Err(RpcError::new(codes::INVALID_PARAMS, why));
    };
    let mut params = call
        .params
        .params
        .filter(|params| params.as_object().is_none_or(|map| !map.is_empty()));
    if let Some(params) = &mut params {
        let registry = world.resource::<AppTypeRegistry>().clone();
        let registry = registry.read();
        rename(params, &|name| {
            let registration = registry.get_with_short_type_path(name)?;
            Some(registration.type_info().type_path().to_owned())
        });
    }
    match world.run_system_with(id, params) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(RpcError::new(
            codes::INVALID_PARAMS,
            format!("Bevy Remote error {}: {}", error.code, error.message),
        )),
        Err(why) => Err(RpcError::new(codes::INTERNAL, why.to_string())),
    }
}

/// The entity of a method's system.
fn system(method: &RemoteMethodSystemId) -> Entity {
    match method {
        RemoteMethodSystemId::Instant(id) => id.entity(),
        RemoteMethodSystemId::Watching(id) => id.entity(),
    }
}

/// Bevy Remote as an `inspect` call uses it: its methods, the read-only
/// ones, where requests go, the types answers name, and where a long
/// answer is kept.
#[derive(SystemParam)]
struct Remote<'w, 's> {
    methods: Option<Res<'w, RemoteMethods>>,
    read_only: Query<'w, 's, (), With<ReadOnlyMethod>>,
    sender: Option<Res<'w, BrpSender>>,
    registry: Res<'w, AppTypeRegistry>,
    scratch: Option<Res<'w, Scratch>>,
    wake: Res<'w, Wake>,
}

/// Sends a read-only call's request to Bevy and answers the call with the
/// reply, off the main thread; refuses any other at once.
fn on_inspect(
    called: On<ToolCalled<InspectArgs>>,
    remote: Remote,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let methods = remote.methods.as_ref();
    let allowed = |name: &str| {
        let method = methods.and_then(|methods| methods.get(name));
        method.is_some_and(|method| remote.read_only.contains(system(method)))
    };
    let mut names = methods.map(|m| m.methods()).unwrap_or_default();
    names.retain(|name| allowed(name));
    names.sort();
    let method = called.args.method.trim();
    let mut fail = |why: String| {
        if let Ok(mut call) = calls.get_mut(called.call) {
            call.decide(
                Decider::plugin("rig-inspect"),
                ToolResult::failed(ToolExecutionError::refused(why)),
            );
        }
    };
    if !allowed(method) {
        fail(format!(
            "`{method}` is not a read-only method here, so nothing was sent. inspect sends: {}.",
            names.join(", ")
        ));
        return;
    }
    let params = called
        .args
        .params
        .clone()
        .filter(|params| !params.is_empty());
    let mut params = params.map(Value::Object);
    if let Some(params) = &mut params {
        let registry = remote.registry.read();
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
    let sender = remote.sender.as_ref();
    if sender.is_none_or(|sender| sender.try_send(message).is_err()) {
        fail("Bevy Remote is not taking requests now.".to_owned());
        return;
    }
    let wake = &remote.wake;
    wake.wake();
    let listed = (method == RPC_DISCOVER_METHOD).then_some(names);
    let spill = remote.scratch.as_ref().and_then(|scratch| scratch.spill());
    let registry = remote.registry.clone();
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    let running = Running::answering(pool, wake, async move {
        let reply = replied.recv().await.ok();
        blocking(move || Ok(answer(reply, listed, &registry, spill.as_ref())))
            .await
            .unwrap_or_else(ToolResult::failed)
    });
    commands.entity(called.call).insert(running);
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
