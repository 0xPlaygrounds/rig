//! Runs rig-code with the built-in plugins.

use rig_code::prelude::*;

fn main() -> AppExit {
    App::new().add_plugins(RigCodePlugins).run()
}
