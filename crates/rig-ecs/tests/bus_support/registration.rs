//! Shared handler registration for native bus and run suites.

use bevy_app::App;
use bevy_ecs::entity::Entity;
use rig_core::serve::Serve;
use rig_ecs::bus::Handlers;

/// Register `handler` under `key` from outside a system.
pub fn register(app: &mut App, key: &str, handler: impl Serve + 'static) -> Entity {
    Handlers::with(app.world_mut(), |handlers| handlers.register(key, handler))
        .expect("the world has a bus")
        .expect("a fresh key")
}
