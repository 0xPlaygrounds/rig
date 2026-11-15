#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures write files of their own"
)]

use super::*;

/// A home in the temporary directory, named `name`, with `files` in it.
fn home(name: &str, files: &[(&str, &str)]) -> Home {
    let root = std::env::temp_dir().join(format!("rig-legacy-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    fs::create_dir_all(&root).ok();
    for (file, text) in files {
        fs::write(root.join(file), text).ok();
    }
    Home::at(&root)
}

/// A 0.46 list: one of rig's plugins turned off, two turned on, one at its
/// default, one gone from rig, and two crates of the user's, one off.
const PLUGINS: &str = r#"# rig's own system prompt.
[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::prompt::DefaultPromptPlugin"

# The welcome.
# [[plugin]]
# crate = "rig-basics"
# plugin = "rig_basics::welcome::WelcomePlugin"

# MCP servers: with plugin = "x" in this comment.
[[plugin]]
crate = "rig-mcp-client"
plugin = "rig_mcp_client::McpPlugin"
# The effect log.
[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::telemetry::effect_log::EffectLogPlugin"

[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::checkpoint::CheckpointPlugin"

# Mine.
[[plugin]]
crate = "hello"
plugin = "hello::HelloPlugin"
path = "plugins/hello"

# [[plugin]]
# crate = "viz"
# plugin = "viz::VizPlugin"
# git = "https://example.com/viz"
# rev = "abc"
"#;

#[test]
fn a_0_46_plugin_list_and_model_are_warned_of_with_what_moves_them() {
    let defaults = "{\n  \"model\": \"gcp.gemini/gemini-3.7-flash\",\n  \"effort\": null\n}";
    let home = home(
        "moves",
        &[("plugins.toml", PLUGINS), ("defaults.json", defaults)],
    );
    let found = warnings(&home);
    let settings = home.settings().display().to_string();
    let (plugins, model) = (
        home.root().join("plugins.toml"),
        home.root().join("defaults.json"),
    );
    let expected = [
        format!(
            "rig no longer reads {}. Move its plugin list to [plugins] in {settings} (see \
             MIGRATING.md):\n  Add to {settings}, before the commands below:\n  \
             [[plugins.extra]]\n  crate = \"hello\"\n  plugin = \"hello::HelloPlugin\"\n  \
             path = \"plugins/hello\"\n  [[plugins.extra]]\n  crate = \"viz\"\n  \
             plugin = \"viz::VizPlugin\"\n  git = \"https://example.com/viz\"\n  \
             rev = \"abc\"\n  rig plugin enable rig_mcp_client::McpPlugin\n  \
             rig plugin enable rig_mcp_client::approval::McpApprovalPlugin\n  \
             rig plugin enable rig_basics::telemetry::effect_log::EffectLogPlugin\n  \
             rig plugin disable rig_basics::welcome::WelcomePlugin\n  \
             rig plugin disable viz::VizPlugin\n  \
             rig_basics::checkpoint::CheckpointPlugin is no longer one of rig's plugins.\n\
             The file was not changed.",
            plugins.display()
        ),
        format!(
            "rig no longer reads {}. Move its model to [model] in {settings} (see \
             MIGRATING.md):\n  Add to {settings}:\n  [model]\n  \
             default = \"gcp.gemini/gemini-3.7-flash\"\nThe file was not changed.",
            model.display()
        ),
    ];
    let left = (
        fs::read_to_string(&plugins).ok(),
        fs::read_to_string(&model).ok(),
    );
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(found, expected);
    assert_eq!(left, (Some(PLUGINS.to_owned()), Some(defaults.to_owned())));
}

#[test]
fn a_0_46_list_of_the_defaults_and_no_model_say_to_add_the_empty_table() {
    let plugins = "[[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::prompt::DefaultPromptPlugin\"\n";
    let home = home(
        "defaults",
        &[
            ("plugins.toml", plugins),
            ("defaults.json", "{ \"model\": null, \"effort\": null }"),
        ],
    );
    let found = warnings(&home);
    fs::remove_dir_all(home.root()).ok();
    let settings = home.settings().display().to_string();
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(
        found
            .iter()
            .all(|warning| warning.contains("Add an empty [") && warning.contains(&settings))
    );
}

/// v0.46.0 read a live `[[plugin]]` table as TOML: a blank line or comment
/// inside it, or a comment after a value, keeps its keys. MCP, on, also gets
/// the approval plugin, which v0.46.0's MCP plugin held.
#[test]
fn a_live_entry_keeps_its_keys_across_blank_lines_and_comments() {
    let plugins = "[[plugin]]\n\ncrate = \"hello\" # mine\n# the type\nplugin = \"hello::HelloPlugin\"\n\n\
                   path = \"plugins/hello\"\n\n[[plugin]]\n# MCP\ncrate = \"rig-mcp-client\"\n\n\
                   plugin = \"rig_mcp_client::McpPlugin\"\n\n# [[plugin]]\n# crate = \"viz\" # off\n\
                   # plugin = \"viz::VizPlugin\"\n# path = \"plugins/viz\"\n";
    let moves = plugin_moves(plugins, Path::new("/h/settings.toml"));
    assert_eq!(
        moves,
        [
            "  Add to /h/settings.toml, before the commands below:",
            "  [[plugins.extra]]",
            "  crate = \"hello\"",
            "  plugin = \"hello::HelloPlugin\"",
            "  path = \"plugins/hello\"",
            "  [[plugins.extra]]",
            "  crate = \"viz\"",
            "  plugin = \"viz::VizPlugin\"",
            "  path = \"plugins/viz\"",
            "  rig plugin enable rig_mcp_client::McpPlugin",
            "  rig plugin enable rig_mcp_client::approval::McpApprovalPlugin",
            "  rig plugin disable viz::VizPlugin",
        ]
    );
}

/// A list whose only change is a plugin rig no longer has still says to
/// add the table that ends the warning.
#[test]
fn a_list_with_only_a_gone_plugin_says_to_add_the_empty_table() {
    let plugins = "[[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::checkpoint::CheckpointPlugin\"\n";
    let moves = plugin_moves(plugins, Path::new("/h/settings.toml"));
    assert_eq!(
        moves,
        [
            "  Add an empty [plugins] table to /h/settings.toml: the list was rig's defaults.",
            "  rig_basics::checkpoint::CheckpointPlugin is no longer one of rig's plugins.",
        ]
    );
}

/// A list that does not read as TOML says so, and to add the table.
#[test]
fn a_list_that_is_not_toml_is_named() {
    let moves = plugin_moves("[[plugin]\n", Path::new("/h/settings.toml"));
    assert_eq!(moves.len(), 1, "{moves:?}");
    assert!(
        moves.iter().all(|line| line.contains("not TOML")),
        "{moves:?}"
    );
}
