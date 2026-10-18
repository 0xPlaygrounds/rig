use std::fs;

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Notice;
use rig_ecs::session_log::{SessionLogPlugin, SessionStore};
use rig_harness_protocol::SessionId;
use rig_harness_test_support::scratch_home;
use rig_memory::DirStore;

use super::{PluginData, project_dir};
use crate::host::session::{SessionPaths, SessionPlugin, StoreOpener};
use crate::services::{ProjectRoot, RigHome};

/// A plugin's file of the session is kept in the session's store, so
/// whoever opens the session reads it; its project's and user's files are
/// kept under the home.
#[test]
fn a_session_file_is_kept_in_its_store_and_the_other_places_are_in_the_home() {
    let home = scratch_home("rig-plugin-data");
    let Ok(id) = "1700000000-7".parse::<SessionId>() else {
        return;
    };
    let session = home.session(&id);
    fs::create_dir_all(session.path()).ok();
    let store = DirStore::open(session.path());
    assert!(store.is_ok());
    let Ok(store) = store else { return };
    let mut app = App::new();
    app.add_message::<Notice>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .insert_resource(SessionStore::new(store))
        .insert_resource(ProjectRoot(home.root().to_owned()))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
        .add_plugins(SessionPlugin);
    drop(std::panic::take_hook());
    app.finish();
    let data = app.world().resource::<PluginData>().clone();
    let written = [
        data.session.write_file("notes/today.md", b"session"),
        data.project.write_file("notes/today.md", b"project"),
        data.user.write_file("notes/today.md", b"user"),
    ];
    assert!(written.iter().all(Result::is_ok), "{written:?}");
    assert_eq!(
        data.session.files().ok(),
        Some(vec!["notes/today.md".to_owned()])
    );
    let read = |dir: &std::path::Path| fs::read_to_string(dir.join("notes/today.md")).ok();
    let opened = PluginData::of(&StoreOpener::default(), &session).ok();
    let in_session = opened.and_then(|data| data.read_file("notes/today.md").ok());
    let project = read(&project_dir(&home, home.root(), None));
    let user = read(&home.data());
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(in_session.as_deref(), Some(&b"session"[..]));
    assert_eq!(project.as_deref(), Some("project"));
    assert_eq!(user.as_deref(), Some("user"));
}

/// Two spellings of one project, such as through a symbolic link, share
/// its directory; another project has its own.
#[cfg(unix)]
#[test]
fn every_spelling_of_a_project_shares_its_directory() {
    let home = scratch_home("rig-plugin-data-project");
    let (project, other) = (home.root().join("work/app"), home.root().join("work/lib"));
    let link = home.root().join("app-link");
    let made = [fs::create_dir_all(&project), fs::create_dir_all(&other)];
    let linked = std::os::unix::fs::symlink(&project, &link);
    let dirs = [
        project_dir(&home, &project, None),
        project_dir(&home, &link, None),
        project_dir(&home, &home.root().join("work/lib/../app"), None),
        project_dir(&home, &other, None),
    ];
    fs::remove_dir_all(home.root()).ok();
    assert!(made.iter().all(Result::is_ok) && linked.is_ok());
    let [app, by_link, dotted, lib] = dirs;
    assert_eq!((&by_link, &dotted), (&app, &app));
    assert_ne!(app, lib);
    assert!(
        app.starts_with(rig_harness_protocol::Home::projects(&home)),
        "{app:?}"
    );
    let name = app
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    assert!(name.is_some_and(|name| name.starts_with("app-") && name.len() == 12));
}
