//! `cargo run -p rig-code`: the agent with the built-in plugins only.

fn main() -> bevy_app::AppExit {
    rig_code::app().run()
}
