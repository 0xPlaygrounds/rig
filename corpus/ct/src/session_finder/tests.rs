use std::fs;
use std::path::Path;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::CommandPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_sessions::SessionsPlugin;

use super::*;

/// The linker error session B's build printed.
const LINKER: &str = "undefined reference to 'png_init'";

/// A `build` tool that fails to link.
fn build_tool(
    called: On<ToolCalled<serde_json::Value>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        let output = format!("ld: error: {LINKER}\ncollect2: error: ld returned 1 exit status");
        call.decide(Decider::plugin("build"), ToolResult::success(output.into()));
    }
}

/// Runs a saved session working in `project` under `home`, whose user
/// says each of `asks` to a model answering with `turns`; its id.
fn saved_session(
    home: &Home,
    project: &Path,
    asks: &[&str],
    turns: Vec<Vec<rig_core::test_utils::MockStreamEvent>>,
) -> SessionId {
    let id = SessionId::generate();
    let (mut app, wakes) = app_in(home, &id);
    fs::write(
        home.session(&id).directory(),
        project.to_string_lossy().as_bytes(),
    )
    .ok();
    let model = MockCompletionModel::from_stream_turns(turns);
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, SessionsPlugin))
        .add_open_tool(
            "build",
            "Build the project.",
            ToolOptions::default(),
            build_tool,
        )
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    for (turn, ask) in asks.iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, *ask, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > turn
        });
    }
    app.update();
    // The next session is newer by its listing's time.
    std::thread::sleep(std::time::Duration::from_millis(5));
    id
}

/// Runs `/find <args>` in `app` and waits until every session was read;
/// the search's picker.
fn find(
    app: &mut App,
    wakes: &std::sync::mpsc::Receiver<()>,
    agent: Entity,
    args: &str,
) -> Option<Entity> {
    app.world_mut()
        .trigger(RunCommand::new(agent, format!("find {args}")));
    run_until(app, wakes, |world| {
        let Some(mut searches) = world.try_query::<&Search>() else {
            return false;
        };
        let query = FindQuery::parse(args);
        searches
            .iter(world)
            .any(|search| search.finished && search.query == query)
    });
    let world = app.world_mut();
    let mut searches = world.query_filtered::<Entity, With<Search>>();
    let found = searches.iter(world).collect::<Vec<_>>();
    assert_eq!(found.len(), 1, "a new /find drops the one before");
    found.first().copied()
}

/// The ids of the sessions the search in `picker` found, in order.
fn found(app: &App, picker: Option<Entity>) -> Vec<SessionId> {
    let search = picker.and_then(|picker| app.world().get::<Search>(picker));
    search
        .map(|search| {
            search
                .found
                .iter()
                .map(|found| found.session.id.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// The messages of the main agent of the saved session `id`.
fn messages(home: &Home, id: &SessionId) -> Vec<Message> {
    let store = StoreOpener::default().open(&home.session(id));
    let session = store.ok().and_then(|store| Session::read(&*store).ok());
    let main = session.as_ref().and_then(Session::main_agent);
    main.map(|agent| {
        agent
            .conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
    .unwrap_or_default()
}

#[test]
fn finds_sessions_by_text_skips_a_broken_one_and_opens_a_hit() {
    let home = scratch_home("ct-session-finder");
    let project = home.root().join("project");
    fs::create_dir_all(&project).ok();
    let elsewhere = home.root().join("elsewhere");
    fs::create_dir_all(&elsewhere).ok();

    let a = saved_session(
        &home,
        &project,
        &["Rename the module."],
        vec![reply("Renamed it.")],
    );
    let b = saved_session(
        &home,
        &project,
        &["Build the project."],
        vec![
            calls("c1", "build", serde_json::json!({ "target": "app" })),
            reply("The link failed; I linked the image library."),
        ],
    );
    let c = saved_session(
        &home,
        &project,
        &["The png icons look blurry."],
        vec![reply("Scale them with nearest-neighbour.")],
    );
    // Another project's session matches only with --all-projects.
    let other = saved_session(
        &home,
        &elsewhere,
        &["Is png_init in libpng?"],
        vec![reply("Yes.")],
    );
    // A session whose log does not read.
    let broken = SessionId::generate();
    let dir = home.session(&broken);
    fs::create_dir_all(dir.path()).ok();
    fs::write(dir.directory(), project.to_string_lossy().as_bytes()).ok();
    let store = StoreOpener::default().open(&dir);
    assert!(store.is_ok_and(|store| {
        store
            .append_log("a1", b"{ not a record\n{ nor this\n")
            .is_ok()
    }));

    // The running session.
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(ProjectRoot(project.clone()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            CommandPlugin,
            PickPlugin,
            SessionFinderPlugin,
        ));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);

    // The linker error is only in a tool result.
    let picker = find(&mut app, &wakes, agent, "png_init");
    assert_eq!(found(&app, picker), []);
    let summary = picker
        .and_then(|picker| app.world().get::<Picker>(picker))
        .map(|picker| picker.body.clone());
    assert!(
        summary
            .as_deref()
            .is_some_and(|body| body.contains("1 skipped")),
        "{summary:?}"
    );
    let picker = find(&mut app, &wakes, agent, "--tools png_init");
    assert_eq!(found(&app, picker), std::slice::from_ref(&b));
    let picker = find(&mut app, &wakes, agent, "png_init --all-projects --tools");
    assert_eq!(found(&app, picker), [other, b.clone()]);

    // Newest first; the snippet marks the match.
    let picker = find(&mut app, &wakes, agent, "png");
    assert_eq!(found(&app, picker), std::slice::from_ref(&c));
    let picker = find(&mut app, &wakes, agent, "--tools png");
    assert_eq!(found(&app, picker), [c.clone(), b.clone()]);
    assert!(!found(&app, picker).contains(&a));
    let shown = picker
        .and_then(|picker| app.world().get::<Picker>(picker))
        .cloned()
        .unwrap_or_default();
    let first = shown.items.first().map(|item| item.label.as_str());
    assert_eq!(first, Some("user: The «png» icons look blurry."));

    // The highlighted hit, and the hit opened, show C's conversation at it.
    let c_before = messages(&home, &c);
    let results = picker.unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().entity_mut(results).insert(Highlighted(0));
    app.world_mut().flush();
    let at = |app: &App, entity: Entity| match app.world().get::<PickPreview>(entity) {
        Some(PickPreview::Conversation { conversation, at }) => at
            .and_then(|at| conversation.index_of(at))
            .and_then(|at| conversation.messages().get(at).map(|m| (**m).clone())),
        _ => None,
    };
    assert_eq!(at(&app, results).as_ref(), c_before.first());
    app.world_mut()
        .trigger(Pick::person(results, Some(Chosen::item(0))));
    app.update();
    let world = app.world_mut();
    let mut opened = world.query_filtered::<Entity, With<Opened>>();
    let opened = opened.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    assert_eq!(at(&app, opened).as_ref(), c_before.first());
    assert_eq!(messages(&home, &c), c_before);
    assert_eq!(c_before.len(), 2);
    fs::remove_dir_all(home.root()).ok();
}
