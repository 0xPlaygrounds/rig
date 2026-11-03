use super::*;

const MCP: &str = "rig_mcp_client::McpPlugin";
const SKILLS: &str = "rig_basics::skills::SkillsPlugin";

fn old_home(test: &str) -> Home {
    Home::at(std::env::temp_dir().join(format!("rig-plugin-{test}-{}", std::process::id())))
}

/// A user's own `settings.toml`, with comments.
const COMMENTED: &str = "# My rig.\n[tui] # the view\n# keep this\n\n# Plugins I tried:\n";

#[test]
fn enable_and_disable_keep_the_users_comments_and_undo_each_other() {
    let home = old_home("comments");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["enable", MCP])?;
        let enabled = fs::read_to_string(home.settings())?;
        run(&home, &["disable", SKILLS])?;
        run(&home, &["enable", SKILLS])?;
        run(&home, &["disable", MCP])?;
        Ok((enabled, fs::read_to_string(home.settings())?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (enabled, undone) = result.unwrap_or_default();
    assert!(enabled.starts_with(COMMENTED), "{enabled}");
    assert!(
        enabled.ends_with(&format!("[plugins]\nenable = [\"{MCP}\"]\n")),
        "{enabled}"
    );
    assert!(undone.starts_with(COMMENTED) && !undone.contains(MCP) && !undone.contains(SKILLS));
}

#[test]
fn a_plugin_that_does_not_build_here_is_refused() {
    let home = old_home("target");
    let elsewhere = BUILT_IN.iter().find(|built_in| !built_in.available());
    if let Some(elsewhere) = elsewhere {
        let refused = run(&home, &["enable", elsewhere.plugin]).err();
        assert!(refused.is_some_and(|why| why.to_string().contains(std::env::consts::OS)));
    }
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn a_new_plugin_crate_is_an_extra_of_the_home() {
    let home = old_home("new");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["new", "word-count"])?;
        let text = fs::read_to_string(home.settings())?;
        let extras = plugin_list::of_home(&home)?.extra;
        Ok((text, extras))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (text, extras) = result.unwrap_or_default();
    assert!(
        text.starts_with(COMMENTED) && text.contains("# Made by `rig plugin new word-count`."),
        "{text}"
    );
    assert_eq!(
        extras
            .iter()
            .map(|extra| extra.plugin.as_str())
            .collect::<Vec<_>>(),
        ["word_count::WordCountPlugin"]
    );
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
fn enabling_an_unknown_type_does_not_change_the_file() {
    let home = old_home("unknown");
    let original = "# An older home.\n";
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), original)?;
        let failure = run(&home, &["enable", "rig_mcp_client::UnknownPlugin"])
            .err()
            .ok_or("unknown type was accepted")?;
        Ok((failure.to_string(), fs::read_to_string(home.settings())?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(
        matches!(&result, Ok((failure, text)) if failure.contains("rig plugin list") && text == original),
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
    let checkout =
        RigSource::Local(std::path::absolute(super::super::project::CHECKOUT).unwrap_or_default());
    assert!(validate_name("rig-tui", &checkout).is_err());
    assert!(validate_name("rig-tui", registry).is_ok());
}

#[test]
fn the_scaffold_depends_on_the_rig_version_and_patches_a_checkout() {
    let checkout = std::path::absolute(super::super::project::CHECKOUT).unwrap_or_default();
    let local = manifest("viz", "0.44.0", &RigSource::Local(checkout));
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
    let parsed = registry.parse::<config::toml_edit::DocumentMut>().ok();
    let name = parsed
        .as_ref()
        .and_then(|manifest| manifest.get("package")?.get("name")?.as_str());
    assert_eq!(name, Some("viz"));
}
