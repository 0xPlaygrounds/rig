use rig_harness_test_support::{Collected, collect, first_agent, scratch_home};

use super::*;

/// The bindings of `keys` in force, and whether each is the user's.
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
/// does on a match (there is no public headless view), then runs a frame.
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

fn notices(app: &App) -> Vec<String> {
    let seen = &app.world().resource::<Collected<Notice>>().0;
    seen.iter().map(|notice| notice.text.clone()).collect()
}

#[test]
fn keys_cycle_the_model_and_a_taken_key_needs_override() {
    let home = scratch_home("corpus-keybindings");
    assert!(std::fs::create_dir_all(home.root()).is_ok());
    // The catalog has no `deepseek/deepseek-pro`, the spec's second model.
    let keys = r#"
"alt+m" = { cycle_model = ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"] }
"ctrl+k" = { command = "/help" }
"ctrl+t" = { insert = "Think." }
"ctrl+e" = { editor = true, override = true }
"enter" = { send = "hello" }
"#;
    assert!(std::fs::write(home.root().join(USER_FILE), keys).is_ok());

    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(Settings::default())
        .add_plugins((AgentPlugin, KeyBindingsPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    std::fs::remove_dir_all(home.root()).ok();

    // `enter` sends and `ctrl+k` deletes to the line's end in the view, so
    // both are refused without `override`; `ctrl+e` replaced the line end.
    let seen = notices(&app);
    let refused = |key: &str| {
        seen.iter()
            .any(|text| text.starts_with(&format!("keys.toml: `{key}` refused")))
    };
    assert!(refused("enter") && refused("ctrl+k"), "{seen:?}");
    assert!(!refused("ctrl+e") && !refused("alt+m"), "{seen:?}");
    assert_eq!(bound(&mut app, "enter").len(), 1);
    assert!(bound(&mut app, "enter").iter().all(|(_, user)| !user));
    assert!(bound(&mut app, "ctrl+e").iter().all(|(_, user)| *user));

    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "no agent");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "deepseek/deepseek-flash".to_owned(),
    });
    app.update();
    press(&mut app, "alt+m", agent);
    assert_eq!(model(&app, agent), Some("deepseek/deepseek-v4-pro"));
    press(&mut app, "alt+m", agent);
    assert_eq!(model(&app, agent), Some("deepseek/deepseek-flash"));

    press(&mut app, "ctrl+t", agent);
    let world = app.world_mut();
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let typed = main.single(world).ok().map(|input| input.text().to_owned());
    assert_eq!(typed.as_deref(), Some("Think."));

    app.world_mut()
        .trigger(RunCommand::new(agent, "keys".to_owned()));
    app.update();
    let listed = notices(&app).pop().unwrap_or_default();
    assert!(
        listed
            .lines()
            .any(|line| line.starts_with("alt+m") && line.ends_with("(keys.toml)")),
        "{listed}"
    );
}

#[cfg(unix)]
#[test]
fn a_failed_editor_leaves_the_draft_and_says_why() {
    let home = scratch_home("corpus-keybindings-draft");
    assert!(std::fs::create_dir_all(home.root()).is_ok());
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(Settings::default())
        .add_plugins((AgentPlugin, KeyBindingsPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.cleanup();
    app.update();

    // The editor wrote the draft, then exited with 3.
    use std::os::unix::process::ExitStatusExt;
    let draft = home.root().join("draft.md");
    assert!(std::fs::write(&draft, "edited").is_ok());
    let failed = TerminalExit(Ok(std::process::ExitStatus::from_raw(3 << 8)));
    let agent = first_agent(app.world_mut());
    app.world_mut().spawn((Draft(draft, agent), failed));
    app.update();
    std::fs::remove_dir_all(home.root()).ok();

    let world = app.world_mut();
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let typed = main.single(world).ok().map(|input| input.text().to_owned());
    assert_eq!(typed.as_deref(), Some(""));
    let seen = notices(&app);
    assert!(
        seen.iter()
            .any(|text| text == "The editor failed (exit status 3); the draft is unchanged."),
        "{seen:?}"
    );
}
