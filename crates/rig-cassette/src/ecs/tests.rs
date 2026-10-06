use bevy_app::App;
use rig_ecs::bus::BusPlugin;

use super::ReplayPlugin;

#[test]
#[should_panic]
fn replay_plugin_refuses_to_precede_the_runtime() {
    let mut app = App::new();
    app.add_plugins((ReplayPlugin, BusPlugin::default()));
}

#[test]
fn provider_options_enter_the_spec_with_the_runs_entries_over_the_agents() {
    use bevy_ecs::prelude::World;
    use rig_core::completion::ProviderOptions as Entries;
    use rig_ecs::agent::{ProviderOptions, RunOf};

    use super::identity::spec_json;

    fn entries(value: serde_json::Value) -> Entries {
        serde_json::from_value(value).expect("provider options")
    }

    let mut world = World::new();
    let bare = world.spawn_empty().id();
    let bare_run = world.spawn(RunOf(bare)).id();
    for subject in [bare, bare_run] {
        assert!(
            spec_json(&mut world, subject)
                .get("provider_options")
                .is_none()
        );
    }

    let agents = entries(serde_json::json!({
        "alpha": {"*": {"top_k": 4}},
        "beta": {"*": {"min_p": 0.1}},
    }));
    let agent = world.spawn(ProviderOptions(agents.clone())).id();
    let plain = world.spawn(RunOf(agent)).id();
    let overlaid = world
        .spawn((
            RunOf(agent),
            ProviderOptions(entries(serde_json::json!({"alpha": {"*": {"top_k": 8}}}))),
        ))
        .id();
    let overlay = entries(serde_json::json!({
        "alpha": {"*": {"top_k": 8}},
        "beta": {"*": {"min_p": 0.1}},
    }));
    for (subject, expected) in [(agent, &agents), (plain, &agents), (overlaid, &overlay)] {
        assert_eq!(
            spec_json(&mut world, subject).get("provider_options"),
            Some(&serde_json::json!(expected))
        );
    }
}
