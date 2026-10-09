//! The `src/lib.rs` that `rig plugin new` writes builds against this
//! crate's prelude and registers its command.

#[path = "../../../src/launcher/plugin/scaffold.rs"]
mod scaffold;

use rig_harness::prelude::*;
use rig_harness::rig_ecs::commands::SlashCommand;

#[test]
fn the_scaffold_plugin_adds_its_command() {
    let mut app = App::new();
    app.add_message::<Notice>()
        .add_plugins(scaffold::ScaffoldPlugin);
    let world = app.world_mut();
    let names: Vec<String> = world
        .query::<&SlashCommand>()
        .iter(world)
        .map(|command| command.name.clone())
        .collect();
    assert_eq!(names, ["__name__"]);
}
