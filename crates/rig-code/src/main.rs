//! The agent with only its built-in plugins, for `cargo run -p rig-code`.

fn main() -> rig_code::bevy_app::AppExit {
    rig_code::app().run()
}
