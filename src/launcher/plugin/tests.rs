use super::*;

#[test]
fn type_names_come_from_the_crate_name() {
    assert_eq!(type_name("agent-viz"), "AgentVizPlugin");
    assert_eq!(type_name("hello"), "HelloPlugin");
    assert_eq!(type_name("my_plugin"), "MyPlugin");
}

#[test]
fn names_are_lowercase_crate_names_not_the_agents_own() {
    assert!(validate_name("agent-viz").is_ok());
    assert!(validate_name("Viz").is_err());
    assert!(validate_name("1viz").is_err());
    assert!(validate_name("viz/../x").is_err());
    assert!(validate_name("rig-harness").is_err());
    assert!(validate_name(PACKAGE).is_err());
    assert!(validate_name("bevy_ui").is_err());
}

#[test]
fn manifest_strings_are_read_from_their_table() {
    let manifest = "[package]\nname = \"viz\" # the name\n\n[lib]\nname = \"viz_lib\"\n";
    assert_eq!(
        manifest_string(manifest, "package", "name").as_deref(),
        Some("viz")
    );
    assert_eq!(
        manifest_string(manifest, "lib", "name").as_deref(),
        Some("viz_lib")
    );
    assert_eq!(manifest_string(manifest, "lib", "path"), None);
}

#[test]
fn the_scaffold_depends_on_the_rig_version_and_patches_a_checkout() {
    let checkout = Path::new("/rig");
    let local = manifest("viz", "0.44.0", &RigSource::Local(checkout.to_path_buf()));
    assert!(local.contains("rig-harness = \"0.44.0\"\n"));
    assert!(local.contains(&format!(
        "bevy_ecs = {{ version = \"={}\", default-features = false }}\n",
        super::super::BEVY_VERSION
    )));
    assert!(local.contains("[patch.crates-io]\n"));
    assert!(local.contains("rig-harness = { path = "));
    let registry = manifest("viz", "0.44.0", &RigSource::Registry);
    assert!(!registry.contains("[patch"));
    assert_eq!(
        manifest_string(&registry, "package", "name").as_deref(),
        Some("viz")
    );
}

#[test]
fn remove_deletes_only_a_crate_directly_in_rig_home_plugins() {
    let root = std::env::temp_dir().join(format!("rig-plugin-delete-{}", std::process::id()));
    let plugins = root.join("plugins");
    let stats = plugins.join("stats");
    let nested = stats.join("inner");
    let outside = root.join("elsewhere");
    for crate_dir in [&stats, &nested, &outside] {
        assert!(fs::create_dir_all(crate_dir).is_ok());
        assert!(fs::write(crate_dir.join("Cargo.toml"), "[package]\n").is_ok());
    }
    assert!(fs::create_dir_all(plugins.join("empty")).is_ok());
    assert_eq!(
        deletable(&plugins, &nested.join("../..").join("stats")),
        fs::canonicalize(&stats).ok()
    );
    assert_eq!(deletable(&plugins, &nested), None);
    assert_eq!(deletable(&plugins, &outside), None);
    assert_eq!(deletable(&plugins, &plugins), None);
    assert_eq!(deletable(&plugins, &plugins.join("empty")), None);
    assert_eq!(deletable(&plugins, &plugins.join("missing")), None);
    assert!(fs::remove_dir_all(&root).is_ok());
}
