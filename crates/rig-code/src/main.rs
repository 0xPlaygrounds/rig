use rig_code::bevy::app::{App, AppExit};

fn main() -> AppExit {
    App::new().add_plugins(rig_code::RigCodePlugins).run()
}
