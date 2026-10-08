//! The agent with the default plugin list, the same program the `rig`
//! launcher generates for the default `plugins.toml`.

fn main() -> rig_code::bevy::app::AppExit {
    rig_code::run(|app| {
        app.add_plugins(<rig_code::BuiltinTools as Default>::default());
        app.add_plugins(<rig_code::BuiltinCommands as Default>::default());
        app.add_plugins(<rig_code::TuiPlugin as Default>::default());
    })
}
