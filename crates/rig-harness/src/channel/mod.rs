//! The channel's method table: each method a registered one-shot system,
//! called with its typed params and its caller, the way Bevy Remote
//! dispatches. It works in `rigd` and in workers; a plugin's method in a
//! worker is routed there by `rigd`.

use bevy_ecs::prelude::*;
use rig_protocol::{Method, RpcError};
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
    /// Registers `M::NAME` as a one-shot system.
    fn add_method<M: Method, Sys, Marker>(&mut self, system: Sys) -> &mut Self
    where
        Sys: IntoSystem<In<Call<M::Params>>, Result<M::Result, RpcError>, Marker> + 'static;

    /// Declares an event kind a plugin publishes (`x/<plugin>/<name>`).
    fn add_channel_event<E: ChannelEvent>(&mut self) -> &mut Self;
}

/// A plugin's event kind.
pub trait ChannelEvent: Serialize + DeserializeOwned + Send + Sync + 'static {
    const KIND: &'static str;
    const VERSION: u32 = 1;
}
