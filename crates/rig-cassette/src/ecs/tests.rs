use bevy_app::App;
use rig_ecs::bus::BusPlugin;

use super::ReplayPlugin;

#[test]
#[should_panic]
fn replay_plugin_refuses_to_precede_the_runtime() {
    let mut app = App::new();
    app.add_plugins((ReplayPlugin, BusPlugin::default()));
}
