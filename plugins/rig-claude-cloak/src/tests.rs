use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

use rig_harness::harness_protocol::Home;
use rig_harness::rig_core::catalog::ModelStatus;
use rig_harness::services::RigHome;

use super::*;

#[test]
fn plugin_registers_separate_vendor_without_changing_anthropic() {
    let mut app = App::new();
    app.init_resource::<Models>();
    let original: Vec<_> = app
        .world()
        .resource::<Models>()
        .0
        .catalog()
        .iter()
        .filter(|spec| spec.provider.vendor() == "anthropic")
        .map(|spec| spec.reference())
        .collect();
    app.add_plugins(ClaudeCloakPlugin);
    let models = app.world().resource::<Models>();
    let after: Vec<_> = models
        .0
        .catalog()
        .iter()
        .filter(|spec| spec.provider.vendor() == "anthropic")
        .map(|spec| spec.reference())
        .collect();
    assert!(!original.is_empty());
    assert_eq!(original, after);
    for reference in original {
        let separate = reference.replacen("anthropic/", "claude-cloak/", 1);
        assert!(models.0.catalog().resolve(&separate).is_ok());
        let retired = reference.replacen("anthropic/", "anthropic-oauth/", 1);
        assert!(models.0.catalog().resolve(&retired).is_err());
    }
}

#[test]
fn the_models_are_usable_only_once_logged_in() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.path())))
        .init_resource::<Models>()
        .add_plugins(ClaudeCloakPlugin);
    let usable = |app: &App| {
        let models = &app.world().resource::<Models>().0;
        let statuses = models.statuses().into_iter();
        let cloaked = statuses.filter(|(spec, _)| spec.provider.vendor() == VENDOR);
        cloaked
            .map(|(_, status)| status == ModelStatus::Usable)
            .collect::<Vec<_>>()
    };
    let before = usable(&app);
    anyhow::ensure!(!before.is_empty() && before.iter().all(|usable| !usable));
    let stored = root.path().join("anthropic-oauth");
    std::fs::DirBuilder::new().mode(0o700).create(&stored)?;
    let credentials = stored.join("credentials.json");
    std::fs::write(&credentials, "{}")?;
    std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o644))?;
    // A file others may read is not one a request reads.
    anyhow::ensure!(usable(&app).iter().all(|usable| !usable));
    std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600))?;
    let after = usable(&app);
    anyhow::ensure!(after.iter().any(|usable| *usable), "{after:?}");
    Ok(())
}
