//! The `src/lib.rs` that `rig plugin new` writes builds against this
//! crate's prelude, and loaded as the generated `main.rs` loads it, it is
//! added once and provides its command.

#[path = "../../../src/launcher/plugin/scaffold.rs"]
mod scaffold;

use rig_harness::prelude::*;

#[test]
fn the_scaffold_plugin_is_loaded_once_and_provides_its_command() {
    let mut app = App::new();
    app.add_message::<Notice>();
    for _ in 0..2 {
        rig_harness::load::<scaffold::ScaffoldPlugin>(&mut app, "scaffold", "path plugins/x");
    }
    let world = app.world_mut();
    let plugins: Vec<(Entity, String)> = world
        .query::<(Entity, &PluginSource)>()
        .iter(world)
        .map(|(plugin, source)| (plugin, source.krate.clone()))
        .collect();
    let commands: Vec<(Entity, String)> = world
        .query::<(&ProvidedBy, &Name)>()
        .iter(world)
        .map(|(by, name)| (by.0, name.as_str().to_owned()))
        .collect();
    let plugin = plugins.first().map(|(plugin, _)| *plugin);
    assert_eq!(plugins.len(), 1);
    assert_eq!(
        commands,
        plugin
            .map(|plugin| (plugin, "/__name__".to_owned()))
            .into_iter()
            .collect::<Vec<_>>()
    );
}
