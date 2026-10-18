//! The view: the terminal front as a client of `rigd`, with a mirror world
//! of the kernel's own components built from a session's events, so the
//! TUI's queries are unchanged.

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use rig_protocol::{AgentId, SessionId};

/// Subscribes to the viewed session and applies its events to the mirror.
pub struct MirrorPlugin;

impl Plugin for MirrorPlugin {
    fn build(&self, _app: &mut App) {}
}

/// The view's connection, the session it shows, and whether it drives it.
#[derive(Resource)]
pub struct ViewClient {
    pub client: rig_client::Client,
    pub session: Option<SessionId>,
    pub driver: bool,
}

/// On a mirror entity: the agent it mirrors.
#[derive(Component)]
pub struct Mirrored {
    pub agent: AgentId,
}

/// What the TUI's input triggers in place of delivering to a local agent.
#[derive(Event)]
pub struct SendCommand(pub rig_protocol::methods::Send);
