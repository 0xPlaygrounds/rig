use rig_harness::prelude::*;
use rig_harness_test_support::{MemoryStore, app_on, first_agent};

use super::*;

/// The kernel and the welcome on a fresh session, after its first frame,
/// with its agent.
fn start() -> (App, Option<Entity>) {
    let (mut app, _) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, WelcomePlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    (app, agent)
}

/// The welcome asides of `agent`'s conversation, with their ids.
fn welcomes(app: &App, agent: Option<Entity>) -> Vec<(AsideId, Aside)> {
    let conversation = agent.and_then(|agent| app.world().get::<Conversation>(agent));
    let asides = conversation
        .into_iter()
        .flat_map(|conversation| conversation.asides());
    asides
        .filter(|(_, aside)| aside.kind == WELCOME)
        .map(|(id, aside)| (id, aside.clone()))
        .collect()
}

#[test]
fn the_main_agent_has_one_unsaved_welcome_kept_current() {
    let (mut app, agent) = start();
    let shown = welcomes(&app, agent);
    assert_eq!(shown.len(), 1, "{shown:?}");
    let Some((id, aside)) = shown.first() else {
        return;
    };
    assert!(!aside.saved && aside.after.is_none());
    let says: Option<Welcome> = serde_json::from_value(aside.value.clone()).ok();
    assert_eq!(
        says.as_ref().map(|says| says.version.as_str()),
        Some(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(says.and_then(|says| says.model), None);

    let Some(agent_entity) = agent else {
        return;
    };
    app.world_mut()
        .entity_mut(agent_entity)
        .insert(ModelChoice("acme/fast".to_owned()));
    app.update();
    app.update();
    let shown = welcomes(&app, agent);
    assert_eq!(shown.len(), 1, "{shown:?}");
    let Some((again, aside)) = shown.first() else {
        return;
    };
    // Replaced in its place, not added again.
    assert_eq!(again, id);
    let says: Option<Welcome> = serde_json::from_value(aside.value.clone()).ok();
    assert_eq!(
        says.and_then(|says| says.model).as_deref(),
        Some("acme/fast")
    );
}

#[cfg(feature = "tui")]
#[test]
fn the_welcome_is_the_mark_beside_the_version_model_and_directory() {
    let says = Welcome {
        version: "1.2.3".to_owned(),
        model: Some("acme/fast".to_owned()),
        reasoning: "high".to_owned(),
        directory: "~/code/rig".to_owned(),
    };
    let aside = Aside::new(
        WELCOME,
        Author::plugin(PLUGIN),
        serde_json::to_value(&says).unwrap_or_default(),
    );
    let (mut app, agent) = start();
    let Some(agent) = agent else {
        return;
    };
    let id = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .map(|mut conversation| conversation.add_aside(aside.clone()));
    let Some(id) = id else {
        return;
    };
    let rows: Vec<String> = tui::lines(&rig_tui::AsideView { id, aside: &aside })
        .iter()
        .map(ToString::to_string)
        .collect();
    let row = |at: usize| rows.get(at).map(String::as_str).unwrap_or_default();
    assert!(
        row(0).contains(tui::MARK[0]) && row(0).ends_with("rig v1.2.3"),
        "{rows:?}"
    );
    assert!(row(1).ends_with("acme/fast · reasoning high"), "{rows:?}");
    assert!(row(2).ends_with("~/code/rig"), "{rows:?}");
}
