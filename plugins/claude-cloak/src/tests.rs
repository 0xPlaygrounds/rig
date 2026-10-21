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
