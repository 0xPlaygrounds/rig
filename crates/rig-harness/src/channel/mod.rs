//! The channel's method table: each method a registered one-shot system,
//! called with its typed params and its caller, the way Bevy Remote
//! dispatches (`RemoteMethods`, a name → `SystemId` table). It works in
//! `rigd` and in workers; a plugin's method in a worker is routed there by
//! `rigd`.
//!
//! ```
//! use rig_harness::channel::{Call, ChannelAppExt, ChannelMethods, Caller, call};
//! use rig_harness::server::Principal;
//! use rig_harness::App;
//! use rig_protocol::{Method, RpcError};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Ping { n: u32 }
//! impl Method for Ping {
//!     const NAME: &'static str = "x/demo/ping";
//!     type Params = Ping;
//!     type Result = u32;
//! }
//!
//! fn ping(call: bevy_ecs::system::In<Call<Ping>>) -> Result<u32, RpcError> {
//!     Ok(call.params.n + 1)
//! }
//!
//! let mut app = App::new();
//! app.add_method::<Ping, _, _>(ping);
//! let world = app.world_mut();
//! let caller = Caller { client: bevy_ecs::entity::Entity::PLACEHOLDER, principal: Principal::Owner };
//! let reply = call(world, "x/demo/ping", caller, serde_json::json!({"n": 1}));
//! assert_eq!(reply.map(Result::ok), Some(Some(serde_json::json!(2))));
//! assert!(app.world().resource::<ChannelMethods>().contains("x/demo/ping"));
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_protocol::methods::{EventInfo, MethodInfo};
use rig_protocol::{Method, RpcError, codes};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::server::Principal;

/// One call of a method.
pub struct Call<P> {
    pub caller: Caller,
    pub params: P,
}

/// Who called: set by the transport, never by params.
#[derive(Clone, Debug)]
pub struct Caller {
    pub client: Entity,
    pub principal: Principal,
}

pub type Reply = Result<serde_json::Value, RpcError>;

/// How built-ins and plugins extend the channel.
pub trait ChannelAppExt {
    /// Registers `M::NAME` as a one-shot system. A second registration of
    /// the same name replaces the first, with a warning. In `rigd`, a call
    /// that panics answers `-32603` and the method is registered again
    /// from `system`, so it keeps answering.
    fn add_method<M: Method, Sys, Marker>(&mut self, system: Sys) -> &mut Self
    where
        Sys: IntoSystem<In<Call<M::Params>>, Result<M::Result, RpcError>, Marker>
            + Clone
            + Send
            + Sync
            + 'static;

    /// Declares an event kind a plugin publishes (`x/<plugin>/<name>`).
    fn add_channel_event<E: ChannelEvent>(&mut self) -> &mut Self;
}

/// A plugin's event kind.
pub trait ChannelEvent: Serialize + DeserializeOwned + Send + Sync + 'static {
    const KIND: &'static str;
    const VERSION: u32 = 1;
}

/// A type-erased method: parses the params, runs the system, encodes the
/// result.
type Handler = Arc<dyn Fn(&mut World, Caller, serde_json::Value) -> Reply + Send + Sync>;

/// The method table and the event kinds declared, the catalog `rig/hello`
/// lists.
#[derive(Resource, Default)]
pub struct ChannelMethods {
    methods: BTreeMap<String, Handler>,
    events: BTreeMap<String, u32>,
}

impl ChannelMethods {
    /// Whether `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.methods.contains_key(name)
    }

    /// Every registered method, by name.
    pub fn methods(&self) -> Vec<MethodInfo> {
        self.methods
            .keys()
            .map(|name| MethodInfo {
                name: name.clone(),
                version: 1,
            })
            .collect()
    }

    /// Every declared event kind, by kind.
    pub fn events(&self) -> Vec<EventInfo> {
        self.events
            .iter()
            .map(|(kind, version)| EventInfo {
                kind: kind.clone(),
                version: *version,
            })
            .collect()
    }
}

/// Runs the method `name` for `caller`, or `None` when no such method is
/// registered.
pub fn call(
    world: &mut World,
    name: &str,
    caller: Caller,
    params: serde_json::Value,
) -> Option<Reply> {
    let handler = world
        .get_resource::<ChannelMethods>()?
        .methods
        .get(name)?
        .clone();
    Some(handler(world, caller, params))
}

fn internal(e: impl std::fmt::Display) -> RpcError {
    RpcError::new(codes::INTERNAL, e.to_string())
}

impl ChannelAppExt for App {
    fn add_method<M: Method, Sys, Marker>(&mut self, system: Sys) -> &mut Self
    where
        Sys: IntoSystem<In<Call<M::Params>>, Result<M::Result, RpcError>, Marker>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let world = self.world_mut();
        let id = std::sync::Mutex::new(world.register_system(system.clone()));
        let handler: Handler = Arc::new(move |world, caller, params| {
            let params: M::Params = serde_json::from_value(params)
                .map_err(|e| RpcError::new(codes::INVALID_PARAMS, format!("{}: {e}", M::NAME)))?;
            let current = *id.lock().map_err(internal)?;
            let call = Call { caller, params };
            let ran = if world.contains_resource::<crate::server::CatchPanics>() {
                let run = std::panic::AssertUnwindSafe(|| world.run_system_with(current, call));
                match std::panic::catch_unwind(run) {
                    Ok(ran) => ran,
                    Err(_) => {
                        // The panic took the system with it: register it again.
                        let again = world.register_system(system.clone());
                        if let Ok(mut id) = id.lock() {
                            *id = again;
                        }
                        return Err(internal(format!("{} failed inside rigd", M::NAME)));
                    }
                }
            } else {
                world.run_system_with(current, call)
            };
            match ran {
                Ok(Ok(result)) => serde_json::to_value(result).map_err(internal),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(internal(e)),
            }
        });
        let mut table = world.get_resource_or_init::<ChannelMethods>();
        if table.methods.insert(M::NAME.to_owned(), handler).is_some() {
            bevy_log::warn!(
                "the channel method {} is registered twice; the later one answers",
                M::NAME
            );
        }
        self
    }

    fn add_channel_event<E: ChannelEvent>(&mut self) -> &mut Self {
        self.world_mut()
            .get_resource_or_init::<ChannelMethods>()
            .events
            .insert(E::KIND.to_owned(), E::VERSION);
        self
    }
}

#[cfg(test)]
mod tests;
