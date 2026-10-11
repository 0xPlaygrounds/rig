use std::path::PathBuf;

use super::*;

/// The notices written so far.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

fn collect(mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>) {
    seen.0
        .extend(notices.read().map(|notice| notice.text.clone()));
}

fn model(app: &App, agent: Entity) -> Option<String> {
    app.world()
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.clone())
}

#[test]
fn alt_m_cycles_the_model_and_a_built_in_key_needs_override() {
    let base = std::env::var_os("CARGO_TARGET_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join(format!("corpus-keybindings-{}", std::process::id()));
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let file = dir.join("keys.toml");
    // The catalog has no `deepseek/deepseek-pro`, the spec's second model.
    let keys = r#"
"alt+m" = { cycle_model = ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"] }
"ctrl+k" = { command = "/help" }
"ctrl+g" = { command = "/help", override = true }
"enter" = { send = "hello" }
"#;
    assert!(std::fs::write(&file, keys).is_ok());

    let mut app = App::new();
    app.insert_resource(KeysFile(file))
        .init_resource::<Seen>()
        .add_plugins((AgentPlugin, KeyBindingsPlugin))
        .add_systems(Last, collect);
    app.update();
    app.update();
    std::fs::remove_dir_all(&dir).ok();

    let seen = &app.world().resource::<Seen>().0;
    assert!(
        seen.iter()
            .any(|text| text.starts_with("Key enter (keys.toml) refused") && text.contains("send")),
        "{seen:?}"
    );
    // `ctrl+k` deletes to the line end in the terminal view, so the spec's
    // own example is refused without `override` too.
    assert!(
        seen.iter().any(|text| text.starts_with("Key ctrl+k")),
        "{seen:?}"
    );
    let bound: Vec<String> = app
        .world()
        .resource::<KeyBindings>()
        .0
        .iter()
        .map(|binding| binding.chord.to_string())
        .collect();
    assert_eq!(bound, ["alt+m", "ctrl+g"]);

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
    assert_eq!(
        model(&app, agent).as_deref(),
        Some("deepseek/deepseek-flash")
    );

    let alt_m = KeyEvent::new(KeyCode::Char('m'), KeyModifiers::ALT);
    app.world_mut().trigger(KeyPressed {
        entity: agent,
        key: alt_m,
    });
    app.update();
    assert_eq!(
        model(&app, agent).as_deref(),
        Some("deepseek/deepseek-v4-pro")
    );
    app.world_mut().trigger(KeyPressed {
        entity: agent,
        key: alt_m,
    });
    app.update();
    assert_eq!(
        model(&app, agent).as_deref(),
        Some("deepseek/deepseek-flash")
    );
}
