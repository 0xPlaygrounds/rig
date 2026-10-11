use std::path::PathBuf;

use super::*;

/// The notices written so far.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

fn collect(mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>) {
    seen.0
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// The bindings of `keys` in force.
fn bound(app: &mut App, keys: &str) -> Vec<(Entity, bool)> {
    let keys: Result<Keys, String> = keys.parse();
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding, Has<UserKey>)>();
    bindings
        .iter(world)
        .filter(|(_, binding, _)| Ok(&binding.keys) == keys.as_ref())
        .map(|(entity, _, user)| (entity, user))
        .collect()
}

/// Presses the user's binding of `keys` for `agent`, as the terminal view
/// does, then runs a frame.
fn press(app: &mut App, keys: &str, agent: Entity) {
    let user = bound(app, keys).into_iter().find(|(_, user)| *user);
    assert!(user.is_some(), "{keys} is not bound");
    if let Some((entity, _)) = user {
        app.world_mut().trigger(KeyPressed {
            entity,
            agent: Some(agent),
            released: false,
        });
    }
    app.update();
}

fn model(app: &App, agent: Entity) -> Option<&str> {
    app.world()
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.as_str())
}

#[test]
fn keys_cycle_the_model_and_a_taken_key_needs_override() {
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target")),
        PathBuf::from,
    );
    let dir = target.join(format!("corpus-keybindings-{}", std::process::id()));
    assert!(std::fs::create_dir_all(&dir).is_ok());
    // The catalog has no `deepseek/deepseek-pro`, the spec's second model.
    let keys = r#"
"alt+m" = { cycle_model = ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"] }
"ctrl+k" = { command = "/help" }
"ctrl+t" = { insert = "Think." }
"ctrl+e" = { editor = true, override = true }
"enter" = { send = "hello" }
"#;
    assert!(std::fs::write(dir.join(USER_FILE), keys).is_ok());

    let mut app = App::new();
    app.insert_resource(Invoked::default())
        .insert_resource(RigHome(Home::at(&dir)))
        .insert_resource(Settings::default())
        .init_resource::<Seen>()
        .add_plugins((AgentPlugin, KeyBindingsPlugin))
        .add_systems(Last, collect);
    app.finish();
    app.cleanup();
    app.update();
    std::fs::remove_dir_all(&dir).ok();

    // `enter` sends and `ctrl+k` deletes to the line's end in the view, so
    // both are refused without `override`; `ctrl+e` replaced the line end.
    let seen = &app.world().resource::<Seen>().0;
    let refused = |key: &str| {
        seen.iter()
            .any(|text| text.starts_with(&format!("keys.toml: `{key}` refused")))
    };
    assert!(refused("enter") && refused("ctrl+k"), "{seen:?}");
    assert!(!refused("ctrl+e") && !refused("alt+m"), "{seen:?}");
    assert_eq!(bound(&mut app, "enter").len(), 1);
    assert!(bound(&mut app, "enter").iter().all(|(_, user)| !user));
    assert!(bound(&mut app, "ctrl+e").iter().all(|(_, user)| *user));

    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "no agent");
    let Ok(agent) = agent else {
        return;
    };
    world.trigger(SetModel {
        entity: agent,
        model: "deepseek/deepseek-flash".to_owned(),
    });
    app.update();
    press(&mut app, "alt+m", agent);
    assert_eq!(model(&app, agent), Some("deepseek/deepseek-v4-pro"));
    press(&mut app, "alt+m", agent);
    assert_eq!(model(&app, agent), Some("deepseek/deepseek-flash"));

    press(&mut app, "ctrl+t", agent);
    assert_eq!(app.world().resource::<TuiInput>().text(), "Think.");

    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "keys".to_owned(),
    });
    app.update();
    let seen = &app.world().resource::<Seen>().0;
    let listed = seen.last().map(String::as_str).unwrap_or_default();
    assert!(
        listed
            .lines()
            .any(|line| line.starts_with("alt+m") && line.ends_with("(keys.toml)")),
        "{listed}"
    );
}
