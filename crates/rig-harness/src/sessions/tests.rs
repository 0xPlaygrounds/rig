use std::collections::BTreeMap;
use std::fs;
use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use rig_core::completion::Message;
use rig_core::conversation::Conversation;
use rig_ecs::AgentPlugin;
use rig_harness_test_support::scratch_home;
use rig_memory::{DirStore, SessionMeta, save};
use serde_json::json;

use super::*;
use crate::pick::{Chosen, Pick, PickPlugin, Picked};

/// What the pickers' entities named when picked.
#[derive(Resource, Default)]
struct Seen(Vec<SessionPicked>);

/// Saves session `n` of `home`: one message, listed as `session n`,
/// changed at `n` ms, with `data` as its listed values.
fn saved(home: &Home, n: u64, data: &[(&str, &str)]) -> SessionId {
    let id: SessionId = format!("{}-{n}", 1_700_000_000 + n)
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let dir = home.session(&id);
    fs::create_dir_all(dir.path()).ok();
    let store = DirStore::open(dir.path());
    assert!(store.is_ok());
    let Ok(store) = store else { return id };
    let mut conversation = Conversation::from_messages([Message::user(format!("hello {n}"))]);
    assert!(save(&store, "agent", &mut conversation).is_ok());
    let meta = SessionMeta {
        title: format!("session {n}"),
        updated: n,
        listed: data
            .iter()
            .map(|(key, value)| ((*key).to_owned(), json!(value)))
            .collect::<BTreeMap<_, _>>(),
        ..SessionMeta::default()
    };
    assert!(meta.write(&store).is_ok());
    assert!(dir.mark_saved().is_ok());
    id
}

/// Updates `app` until `done` holds, for at most ten seconds.
fn until(app: &mut App, done: impl Fn(&mut World) -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        app.update();
        if done(app.world_mut()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// A session picker over a home of more sessions than one task reads fills
/// its picker newest first, without the running session; previews the
/// highlighted session's line; shows and filters on a plugin's listed
/// value, an empty one adding nothing to the label; and names the session
/// picked.
#[test]
fn a_session_picker_fills_previews_filters_on_listed_values_and_names_the_pick() {
    let home = scratch_home("rig-harness-session-picker");
    let goal = [("goals.goal", "ship the parser")];
    let unset = [("goals.goal", "")];
    let ids: Vec<SessionId> = (1..=20)
        .map(|n| match n {
            7 => saved(&home, n, &goal),
            8 => saved(&home, n, &unset),
            _ => saved(&home, n, &[]),
        })
        .collect();
    let (Some(running), Some(newest), Some(goal_id)) = (ids.first(), ids.last(), ids.get(6)) else {
        return;
    };
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, PickPlugin))
        .add_plugins(SessionPickerPlugin)
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(home.session(running)))
        .init_resource::<Seen>()
        .add_observer(
            |picked: On<Picked>, named: Query<&SessionPicked>, mut seen: ResMut<Seen>| {
                seen.0.extend(named.get(picked.entity).cloned());
            },
        );
    let picker = app
        .world_mut()
        .spawn((
            Picker::new("Resume"),
            SessionPicker::default(),
            Highlighted(0),
        ))
        .id();

    let filled = until(&mut app, |world| {
        world
            .get::<Picker>(picker)
            .is_some_and(|it| it.items.len() == 19)
    });
    let labels = |world: &World| -> Vec<String> {
        let items = world.get::<Picker>(picker).map(|it| it.items.clone());
        items
            .unwrap_or_default()
            .into_iter()
            .map(|it| it.label)
            .collect()
    };
    assert!(filled, "{:?}", labels(app.world()));
    let shown = labels(app.world());
    assert_eq!(shown.first().map(String::as_str), Some("session 20"));
    assert!(
        !shown.contains(&"session 1".to_owned()),
        "the running one is left out"
    );
    assert!(
        shown.contains(&"session 8".to_owned()),
        "an empty listed value adds nothing: {shown:?}"
    );
    let previewed = until(&mut app, |world| world.get::<PickPreview>(picker).is_some());
    assert!(previewed);
    let preview = app.world().get::<PickPreview>(picker).cloned();
    let line = match &preview {
        Some(PickPreview::Conversation {
            conversation,
            at: None,
        }) => conversation.messages(),
        _ => &[],
    };
    assert_eq!(
        line,
        [Message::user("hello 20")].map(std::sync::Arc::new),
        "{preview:?}"
    );

    if let Some(mut session) = app.world_mut().get_mut::<SessionPicker>(picker) {
        session.filter.text = "SHIP the".to_owned();
    }
    app.update();
    let shown = labels(app.world());
    assert_eq!(shown, ["session 7  · ship the parser"]);

    app.world_mut()
        .trigger(Pick::person(picker, Some(Chosen::item(0))));
    app.update();
    let seen = &app.world().resource::<Seen>().0;
    assert_eq!(
        seen.first().and_then(|it| it.session.as_ref()),
        Some(goal_id)
    );
    assert_ne!(goal_id, newest);
    fs::remove_dir_all(home.root()).ok();
}
