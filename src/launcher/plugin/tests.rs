use super::*;

#[test]
fn the_scaffold_names_the_plugin_and_its_command() {
    assert_eq!(type_name("agent-viz"), "AgentVizPlugin");
    assert_eq!(type_name("my_plugin"), "MyPlugin");
    let lib = lib_rs("word-count", &type_name("word-count"));
    assert!(lib.contains("pub struct WordCountPlugin;"));
    assert!(lib.contains("impl Plugin for WordCountPlugin {"));
    assert!(lib.contains("\"word-count\",\n"));
    assert!(!lib.contains("__name__"));
    assert!(!lib.contains("Scaffold"));
}

#[test]
fn names_are_lowercase_crate_names_not_the_agents_own() {
    let registry = &RigSource::Registry;
    assert!(validate_name("agent-viz", registry).is_ok());
    assert!(validate_name("Viz", registry).is_err());
    assert!(validate_name("1viz", registry).is_err());
    assert!(validate_name("viz/../x", registry).is_err());
    assert!(validate_name("rig-harness", registry).is_err());
    assert!(validate_name(PACKAGE, registry).is_err());
    assert!(validate_name("bevy_ui", registry).is_err());
    // A plugin crate of the checkout this launcher is built in.
    let checkout = RigSource::Local(Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf());
    assert!(validate_name("rig-tui", &checkout).is_err());
    assert!(validate_name("rig-tui", registry).is_ok());
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
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest("viz", "0.44.0", &RigSource::Local(checkout.to_path_buf()));
    assert!(local.contains("rig-harness = \"0.44.0\"\n"));
    // The scaffold's saved component derives from both Bevy crates, which
    // its derives name as the manifest lists them.
    let lib = lib_rs("viz", "VizPlugin");
    assert!(lib.contains("#[derive(Component, Reflect, Default)]\n#[reflect(Component, Saved)]"));
    for krate in ["bevy_ecs", "bevy_reflect"] {
        assert!(local.contains(&format!(
            "\n{krate} = {{ version = \"={}\", default-features = false }}\n",
            super::super::BEVY_VERSION
        )));
    }
    // The rig crates it uses through rig-harness, and nothing else.
    let patched: Vec<&str> = local
        .split("[patch.crates-io]\n")
        .nth(1)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.strip_suffix(" }")?.split_once(" = { path = "))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(patched, CORE_CRATES);
    let registry = manifest("viz", "0.44.0", &RigSource::Registry);
    assert!(!registry.contains("[patch"));
    assert_eq!(
        manifest_string(&registry, "package", "name").as_deref(),
        Some("viz")
    );
}

#[test]
fn check_names_the_new_home_of_a_moved_plugin() {
    let root = std::env::temp_dir().join(format!("rig-moved-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    fs::create_dir_all(&root).ok();
    // A merged crate, an old path, a moved harness plugin, a current entry,
    // and a copy of a merged crate kept elsewhere, which still builds.
    let list = "[[plugin]]\ncrate = \"rig-compaction\"\nplugin = \"rig_compaction::CompactionPlugin\"\n\n\
                [[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_coding_tools::ReadTool\"\n\n\
                [[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_harness::prelude::ApprovalPlugin\"\n\n\
                [[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::prompt::DefaultPromptPlugin\"\n\n\
                [[plugin]]\ncrate = \"rig-print\"\ngit = \"https://example.com/print\"\nplugin = \"rig_print::PrintPlugin\"\n";
    fs::write(home.plugins_toml(), list).ok();
    let failure = check(&home).err().map(|failure| failure.to_string());
    fs::remove_dir_all(&root).ok();
    let failure = failure.unwrap_or_default();
    let problems: Vec<&str> = failure.lines().skip(1).map(str::trim).collect();
    assert_eq!(
        problems,
        [
            "`rig-compaction` is now part of `rig-basics`: for `rig_compaction::CompactionPlugin`, \
             write `crate = \"rig-basics\"` and `plugin = \"rig_basics::compaction::CompactionPlugin\"`",
            "`rig_coding_tools::ReadTool` is now part of `rig-basics`: write `crate = \"rig-basics\"` \
             and `plugin = \"rig_basics::coding_tools::ReadTool\"`",
            "`rig_harness::prelude::ApprovalPlugin` is now part of `rig-basics`: write \
             `crate = \"rig-basics\"` and `plugin = \"rig_basics::approval::ApprovalPlugin\"`",
        ],
        "{failure}"
    );
}
