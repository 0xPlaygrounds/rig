use rig_core::catalog::ModelStatus;
use rig_core::providers::registry::ProviderId;
use rig_ecs::model::{ChoiceKind, ModelChoice, SetModel};
use rig_ecs::prelude::*;
use rig_harness::commands::RunCommand;
use rig_harness::pick::{Chosen, Pick, PickPlugin, Picker};
use rig_harness::services::RigHome;
use rig_harness_test_support::{Collected, MemoryStore, app_on, collect, scratch_home};

use super::super::LoginPlugin;
use super::{Keys, NOTHING_STORED, key_vendors};
use crate::models::FirstPick;

/// A provider a key alone reaches whose variables are all unset here, so
/// only a saved key connects its models.
fn unset_vendor(models: &Models) -> Option<&'static str> {
    let unset = |env: &&str| std::env::var_os(env).is_none_or(|value| value.is_empty());
    key_vendors(&models.0).into_iter().find(|vendor| {
        ProviderId::catalog(vendor).is_some_and(|id| id.api_key_envs().iter().all(unset))
    })
}

/// Whether the first of `vendor`'s models that call tools connects now.
fn status(app: &App, vendor: &str) -> Option<ModelStatus> {
    let models = app.world().resource::<Models>();
    let mut specs = models.0.catalog().iter();
    let spec = specs.find(|spec| spec.tools && spec.provider.vendor() == vendor)?;
    Some(models.0.status(spec))
}

/// Runs `/line` for `agent`, then a frame.
fn run(app: &mut App, agent: Entity, line: String) {
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, line));
    world.flush();
    app.update();
}

/// A key typed in `/login <provider>`'s hidden picker is saved readable by
/// the owner alone, connects the provider's models and appears in no
/// notice; `/logout <provider>` removes it, leaving the environment, and
/// with nothing stored `/logout` says so.
#[test]
fn a_typed_key_is_saved_privately_and_connects_until_logout() {
    let home = scratch_home("rig-login-keys");
    let (mut app, _) = app_on(&MemoryStore::default());
    app.insert_resource(RigHome(home.clone()))
        .add_plugins((AgentPlugin, PickPlugin, LoginPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let models = app.world().resource::<Models>();
    let vendors = key_vendors(&models.0);
    assert!(vendors.contains(&"anthropic"), "{vendors:?}");
    assert!(
        !vendors.contains(&"chatgpt") && !vendors.contains(&"copilot"),
        "{vendors:?}"
    );
    let Some(vendor) = unset_vendor(models) else {
        return;
    };
    assert!(matches!(status(&app, vendor), Some(ModelStatus::NoKey(_))));
    let agent = spawn_root(app.world_mut()).id();
    run(&mut app, agent, format!("login {vendor}"));
    let world = app.world_mut();
    let mut pickers = world.query::<(Entity, &Picker)>();
    let picker = pickers.iter(world).find(|(_, picker)| picker.secret);
    let picker = picker.map(|(picker, _)| picker);
    assert!(picker.is_some(), "/login {vendor} asks for the key");
    let Some(picker) = picker else {
        return;
    };
    let typed = Chosen::text(" sk-rig-test ");
    world.trigger(Pick::person(picker, Some(typed)));
    world.flush();
    app.update();

    let keys = Keys::at(&home);
    assert_eq!(
        keys.load(vendor).ok().flatten().as_deref(),
        Some("sk-rig-test")
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(keys.path(vendor)).map(|meta| meta.permissions().mode());
        assert_eq!(mode.ok().map(|mode| mode & 0o777), Some(0o600));
    }
    assert_eq!(status(&app, vendor), Some(ModelStatus::Usable));
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .all(|notice| !notice.text.contains("sk-rig-test"))
    );
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.starts_with("Saved the API key"))
    );

    run(&mut app, agent, format!("logout {vendor}"));
    assert!(!keys.path(vendor).exists());
    assert!(matches!(status(&app, vendor), Some(ModelStatus::NoKey(_))));
    run(&mut app, agent, "logout".to_owned());
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let last = notices.last().map(|notice| notice.text.as_str());
    assert_eq!(last, Some(NOTHING_STORED));
    std::fs::remove_dir_all(home.root()).ok();
}

/// An app with the login plugin on `home`, its first frame run.
fn login_app(home: &rig_harness::harness_protocol::Home) -> App {
    let (mut app, _) = app_on(&MemoryStore::default());
    app.insert_resource(RigHome(home.clone()))
        .add_plugins((AgentPlugin, PickPlugin, LoginPlugin));
    app.finish();
    app.update();
    app
}

/// A saved key file that exists but does not parse refuses the provider's
/// models, naming the file but not what it holds, instead of connecting
/// with the environment.
#[test]
fn a_saved_key_that_does_not_parse_refuses_the_connect() {
    let home = scratch_home("rig-login-keys-corrupt");
    let Some(vendor) = unset_vendor(&Models::default()) else {
        return;
    };
    let path = Keys::at(&home).path(vendor);
    let written = path.parent().map(|dir| {
        std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, "\"sk-private\""))
    });
    assert!(matches!(written, Some(Ok(()))));
    let app = login_app(&home);
    let status = status(&app, vendor);
    let shown = path.display().to_string();
    assert!(
        matches!(&status, Some(ModelStatus::Unavailable(why))
            if why.contains(&shown) && !why.contains("sk-private")),
        "{status:?}"
    );
    std::fs::remove_dir_all(home.root()).ok();
}

/// A saved key behind a symlink refuses the provider's models, naming the
/// file, instead of connecting with what it points at; one readable by
/// others connects and is the owner's alone afterwards.
#[cfg(unix)]
#[test]
fn a_saved_key_behind_a_symlink_refuses_and_a_shared_one_is_narrowed() {
    use std::os::unix::fs::PermissionsExt;
    let home = scratch_home("rig-login-keys-symlink");
    let Some(vendor) = unset_vendor(&Models::default()) else {
        return;
    };
    let keys = Keys::at(&home);
    assert!(keys.save(vendor, "sk-rig-test").is_ok());
    let path = keys.path(vendor);
    let elsewhere = home.root().join("elsewhere.json");
    assert!(std::fs::rename(&path, &elsewhere).is_ok());
    assert!(std::os::unix::fs::symlink(&elsewhere, &path).is_ok());
    let linked = status(&login_app(&home), vendor);
    let shown = path.display().to_string();
    assert!(
        matches!(&linked, Some(ModelStatus::Unavailable(why)) if why.contains(&shown)),
        "{linked:?}"
    );

    assert!(std::fs::remove_file(&path).is_ok());
    assert!(std::fs::rename(&elsewhere, &path).is_ok());
    let widened = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644));
    assert!(widened.is_ok());
    assert_eq!(status(&login_app(&home), vendor), Some(ModelStatus::Usable));
    let mode = std::fs::metadata(&path).map(|meta| meta.permissions().mode() & 0o777);
    std::fs::remove_dir_all(home.root()).ok();
    assert_eq!(mode.ok(), Some(0o600));
}

/// A key directory that cannot be listed refuses the providers' models,
/// naming the key file, instead of reading as no saved keys.
#[cfg(unix)]
#[test]
fn an_unlistable_key_directory_refuses_the_connect() {
    use std::os::unix::fs::PermissionsExt;
    let home = scratch_home("rig-login-keys-unlistable");
    let Some(vendor) = unset_vendor(&Models::default()) else {
        return;
    };
    let keys = Keys::at(&home);
    assert!(keys.save(vendor, "sk-rig-test").is_ok());
    let path = keys.path(vendor);
    let Some(dir) = path.parent() else {
        return;
    };
    let mode = |mode| std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode));
    assert!(mode(0o000).is_ok());
    // Permissions bind no one running as root.
    let status = std::fs::read_dir(dir)
        .is_err()
        .then(|| status(&login_app(&home), vendor));
    assert!(mode(0o700).is_ok());
    std::fs::remove_dir_all(home.root()).ok();
    let shown = path.display().to_string();
    assert!(
        status.as_ref().is_none_or(|status| {
            matches!(status, Some(ModelStatus::Unavailable(why)) if why.contains(&shown))
        }),
        "{status:?}"
    );
}

/// Writes something that is not a saved key as `vendor`'s, returning the
/// file's path as errors show it.
fn write_broken_key(home: &rig_harness::harness_protocol::Home, vendor: &str) -> String {
    let path = Keys::at(home).path(vendor);
    let written = path
        .parent()
        .map(|dir| std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, "\"x\"")));
    assert!(matches!(written, Some(Ok(()))));
    path.display().to_string()
}

/// With nothing else to reach, the first pick's reason names the saved key
/// file that cannot be read.
#[test]
fn a_first_pick_with_only_an_unreadable_saved_key_names_the_file() {
    let home = scratch_home("rig-login-keys-first-pick");
    let Some(vendor) = unset_vendor(&Models::default()) else {
        return;
    };
    let shown = write_broken_key(&home, vendor);
    let app = login_app(&home);
    let pick = FirstPick::of(app.world().resource::<Models>());
    std::fs::remove_dir_all(home.root()).ok();
    // A key in this machine's environment reaches a model after all.
    if matches!(pick, FirstPick::Model(_)) {
        return;
    }
    assert!(
        matches!(&pick, FirstPick::Nothing(Some(why)) if why.contains(&shown)),
        "{pick:?} does not name {shown}"
    );
}

/// A first pick that lands on another provider because a saved key cannot
/// be read says so once, naming the file and the model picked.
#[test]
fn a_first_pick_past_an_unreadable_saved_key_says_so_once() {
    let home = scratch_home("rig-login-keys-passed-over");
    let empty = login_app(&home);
    let models = empty.world().resource::<Models>();
    let vendors = key_vendors(&models.0);
    let tooled = models.0.catalog().iter().filter(|spec| spec.tools);
    let mut ordered = tooled
        .map(|spec| spec.provider.vendor())
        .filter(|vendor| vendors.contains(vendor));
    let Some(broken) = ordered.next() else {
        return;
    };
    let Some(usable) = ordered.find(|vendor| *vendor != broken) else {
        return;
    };
    let shown = write_broken_key(&home, broken);
    assert!(Keys::at(&home).save(usable, "sk-rig-test").is_ok());
    let (mut app, _) = app_on(&MemoryStore::default());
    app.insert_resource(RigHome(home.clone()))
        .add_plugins((AgentPlugin, PickPlugin, LoginPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let pick = FirstPick::of(app.world().resource::<Models>());
    let model = match pick {
        FirstPick::Model(model) if model.starts_with(&format!("{usable}/")) => model,
        // A key in this machine's environment reaches an earlier model.
        _ => return,
    };
    // An agent already on a model was not moved there by the first pick.
    let on_it = spawn_root(app.world_mut())
        .insert(ModelChoice(model.clone()))
        .id();
    let world = app.world_mut();
    world.trigger(SetModel {
        entity: on_it,
        model: model.clone(),
        kind: ChoiceKind::ForThisRun,
    });
    world.flush();
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices.iter().all(|notice| !notice.text.contains(&shown)),
        "an explicit choice was told about {shown}: {notices:?}"
    );
    for _ in 0..2 {
        let entity = spawn_root(app.world_mut()).id();
        let world = app.world_mut();
        world.trigger(SetModel {
            entity,
            model: model.clone(),
            kind: ChoiceKind::ForThisRun,
        });
        world.flush();
        app.update();
    }
    std::fs::remove_dir_all(home.root()).ok();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said = notices
        .iter()
        .filter(|notice| notice.text.contains(&shown) && notice.text.contains(&model))
        .count();
    assert_eq!(said, 1, "the file {shown} and {model} in {notices:?}");
}
