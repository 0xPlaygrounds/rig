use super::*;

const MCP: &str = "rig_mcp_client::McpPlugin";

fn old_home(test: &str) -> Home {
    Home::at(std::env::temp_dir().join(format!("rig-plugin-{test}-{}", std::process::id())))
}

#[test]
fn enabling_a_shipped_plugin_missing_from_an_old_home_adds_its_description() {
    let home = old_home("enable");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.plugins_toml(), "# An older home.\n")?;
        run(&home, &["enable", MCP])?;
        let parsed = PluginList::load(&home.plugins_toml())?;
        let text = fs::read_to_string(home.plugins_toml())?;
        run(&home, &["enable", MCP])?;
        Ok((
            parsed.plugins.iter().any(|plugin| plugin.type_path == MCP),
            text.contains("# MCP servers:"),
            text == fs::read_to_string(home.plugins_toml())?,
        ))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(matches!(result, Ok((true, true, true))), "{result:?}");
}

#[test]
fn adding_a_shipped_plugin_without_a_source_keeps_its_template_description() {
    let home = old_home("add");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.plugins_toml(), "# An older home.\n")?;
        run(&home, &["add", MCP])?;
        let text = fs::read_to_string(home.plugins_toml())?;
        Ok(text.contains("# MCP servers:") && !text.contains("# Added by"))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(matches!(result, Ok(true)), "{result:?}");
}

#[test]
fn plugin_help_anywhere_succeeds_without_creating_a_home() {
    let home = old_home("help");
    fs::remove_dir_all(home.root()).ok();
    let cases: &[&[&str]] = &[
        &["--help"],
        &["add", "--help"],
        &["enable", "-h"],
        &["add", MCP, "--path", "--help"],
        &["-h", "new", "example"],
        &["unknown", "--help"],
    ];
    let results: Vec<_> = cases.iter().map(|args| run(&home, args)).collect();
    let created = home.root().exists();
    fs::remove_dir_all(home.root()).ok();
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Ok(code) if *code == ExitCode::SUCCESS)),
        "{results:?}"
    );
    assert!(!created);
}

#[test]
fn missing_or_unknown_plugin_commands_are_refused_without_creating_a_home() {
    let home = old_home("refused");
    fs::remove_dir_all(home.root()).ok();
    for args in [vec![], vec!["unknown"], vec!["enable"]] {
        let result = run(&home, &args);
        assert!(
            matches!(result, Ok(code) if code == ExitCode::from(REFUSED_EXIT_CODE)),
            "{args:?}: {result:?}"
        );
    }
    assert!(!home.root().exists());
}

#[test]
fn an_explicit_source_overrides_the_shipped_entry() {
    let entry = added_entry(
        &old_home("source"),
        MCP,
        &["--crate", "rig-mcp-client", "--version", "1"],
    );
    let parsed = entry
        .and_then(|entry| plugin_list::parse(&entry, Path::new("/home")))
        .ok();
    assert!(parsed.is_some_and(|list| list.plugins.iter().any(|entry| {
        entry.type_path == MCP && entry.package.source == Source::Version("1".to_owned())
    })));
}

#[test]
fn enabling_an_unknown_type_does_not_change_the_file() {
    let home = old_home("unknown");
    let original = "# An older home.\n";
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.plugins_toml(), original)?;
        let failure = run(&home, &["enable", "rig_mcp_client::UnknownPlugin"])
            .err()
            .ok_or("unknown type was accepted")?;
        Ok((
            failure.to_string(),
            fs::read_to_string(home.plugins_toml())?,
        ))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(
        matches!(&result, Ok((failure, text)) if failure.contains("rig plugin add") && text == original),
        "{result:?}"
    );
}

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
fn a_type_of_a_shipped_plugin_crate_names_its_crate() {
    assert_eq!(
        rig_plugin_crate("rig_traffic::TrafficPlugin").as_deref(),
        Some("rig-traffic")
    );
    assert_eq!(
        rig_plugin_crate("rig_claude_cloak::ClaudeCloakPlugin").as_deref(),
        Some("rig-claude-cloak")
    );
    assert_eq!(
        rig_plugin_crate("rig_basics::reload::ReloadPlugin").as_deref(),
        Some("rig-basics")
    );
    // Another crate needs its source.
    assert_eq!(rig_plugin_crate("viz::VizPlugin"), None);
    assert_eq!(rig_plugin_crate("VizPlugin"), None);
}
