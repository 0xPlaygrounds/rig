use rig_core::catalog::ModelStatus;
use rig_core::providers::registry::ProviderId;
use rig_ecs::prelude::*;
use rig_harness::commands::RunCommand;
use rig_harness::pick::{Chosen, Pick, PickPlugin, Picker};
use rig_harness::services::RigHome;
use rig_harness_test_support::{Collected, MemoryStore, app_on, collect, scratch_home};

use super::super::LoginPlugin;
use super::{Keys, NOTHING_STORED, key_vendors};

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
    let agent = app.world_mut().spawn(Agent).id();
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
    assert_eq!(keys.load(vendor).as_deref(), Some("sk-rig-test"));
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
