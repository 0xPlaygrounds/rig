use super::*;

const LIST: &str = "# The plugins.\n\n# The tools.\n[[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::coding_tools::ReadTool\"\n\n# A crate.\n[[plugin]]\ncrate = \"viz\"\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n\n# The view.\n[[plugin]]\ncrate = \"rig-tui\"\nplugin = \"rig_tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n";

#[test]
fn removing_a_table_takes_its_comments_and_keeps_the_rest() {
    let base = Path::new("/home");
    assert_eq!(
        without_table(LIST, "viz::VizPlugin", base).ok().as_deref(),
        Some(
            "# The plugins.\n\n# The tools.\n[[plugin]]\ncrate = \"rig-basics\"\nplugin = \"rig_basics::coding_tools::ReadTool\"\n\n# The view.\n[[plugin]]\ncrate = \"rig-tui\"\nplugin = \"rig_tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n"
        )
    );
    let last = without_table(LIST, "rig_tui::TuiPlugin", base)
        .ok()
        .unwrap_or_default();
    assert!(last.ends_with("plugin = \"viz::VizPlugin\"\n\n# An example.\n# [[plugin]]\n"));
    assert_eq!(
        parse(&last, base).ok().map(|list| list.plugins.len()),
        Some(2)
    );
}

#[test]
fn removing_an_unlisted_type_fails() {
    assert!(without_table(LIST, "viz::Other", Path::new("/home")).is_err());
}

#[test]
fn the_template_lists_every_shipped_plugin_the_optional_ones_disabled() {
    let base = Path::new("/home");
    let listed = entries(TEMPLATE, base).unwrap_or_default();
    assert!(
        listed
            .iter()
            .all(|listed| listed.entry.package.source == Source::Rig)
    );
    assert_eq!(listed.iter().filter(|listed| listed.enabled).count(), 20);
    let disabled: Vec<&str> = listed
        .iter()
        .filter(|listed| !listed.enabled)
        .map(|listed| listed.entry.type_path.as_str())
        .collect();
    assert_eq!(
        disabled,
        [
            "rig_basics::approval::ApprovalPlugin",
            "rig_basics::preview::PreviewPlugin",
            "rig_basics::checkpoint::CheckpointPlugin",
            "rig_inspect::InspectPlugin",
            "rig_basics::subagents::SubagentsPlugin",
            "rig_steel::SteelPlugin",
            "rig_basics::telemetry::effect_log::EffectLogPlugin",
            "rig_basics::telemetry::diagnostics::DiagnosticsPlugin",
        ]
    );
    assert!(listed.iter().all(|listed| {
        !listed.description.is_empty()
            || listed
                .entry
                .type_path
                .starts_with("rig_basics::coding_tools::")
    }));
}

#[test]
fn disabling_then_enabling_gives_back_the_same_file() {
    let base = Path::new("/home");
    let text = format!(
        "{TEMPLATE}\n# Mine.\n[[plugin]]\ncrate = \"viz\" # by path\r\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n"
    );
    for type_path in [
        "rig_tui::TuiPlugin",
        "rig_basics::reload::ReloadPlugin",
        "viz::VizPlugin",
    ] {
        let disabled = toggled(&text, type_path, false, base).ok().flatten();
        let listed = disabled
            .as_deref()
            .and_then(|disabled| entries(disabled, base).ok())
            .unwrap_or_default();
        assert!(
            listed
                .iter()
                .any(|listed| listed.entry.type_path == type_path && !listed.enabled),
            "{type_path} is listed disabled"
        );
        let enabled = disabled
            .as_deref()
            .and_then(|disabled| toggled(disabled, type_path, true, base).ok().flatten());
        assert_eq!(enabled.as_deref(), Some(text.as_str()), "{type_path}");
    }
    // Enabling a disabled one of the template, then disabling it again.
    let enabled = toggled(
        TEMPLATE,
        "rig_basics::subagents::SubagentsPlugin",
        true,
        base,
    )
    .ok()
    .flatten()
    .unwrap_or_default();
    assert_eq!(
        parse(&enabled, base).map(|list| list.plugins.len()).ok(),
        Some(21)
    );
    assert_eq!(
        toggled(
            &enabled,
            "rig_basics::subagents::SubagentsPlugin",
            false,
            base
        )
        .ok()
        .flatten()
        .as_deref(),
        Some(TEMPLATE)
    );
    // Already so, and not listed.
    assert!(matches!(
        toggled(TEMPLATE, "rig_tui::TuiPlugin", true, base),
        Ok(None)
    ));
    assert!(toggled(TEMPLATE, "viz::Other", true, base).is_err());
}

#[test]
fn a_toggle_that_would_change_another_entry_is_refused() {
    // A key after a blank line would stay active and land in the entry above.
    let text = "[[plugin]]\ncrate = \"a\"\nplugin = \"a::A\"\n\n[[plugin]]\nplugin = \"b::B\"\n\ncrate = \"b\"\n";
    assert!(toggled(text, "b::B", false, Path::new("/home")).is_err());
}

#[test]
fn every_entry_names_its_crate_and_at_most_one_source() {
    let entry = |keys: &str| {
        parse(
            &format!("[[plugin]]\n{keys}plugin = \"viz::VizPlugin\"\n"),
            Path::new("/home"),
        )
    };
    assert!(entry("").is_err());
    assert!(entry("crate = \"viz\"\npath = \"v\"\nversion = \"1\"\n").is_err());
    assert!(entry("crate = \"rig-harness\"\n").is_err());
    let source = |keys: &str| {
        entry(keys)
            .ok()
            .and_then(|list| list.plugins.into_iter().next())
            .map(|plugin| plugin.package.source)
    };
    assert_eq!(source("crate = \"viz\"\n"), Some(Source::Rig));
    assert_eq!(
        source("crate = \"viz\"\nversion = \"1\"\n"),
        Some(Source::Version("1".to_owned()))
    );
}

#[test]
fn quoted_strings_read_back_with_their_escapes() {
    let text = "a\u{1}b\"c\\d\te\u{7f}";
    assert_eq!(
        parse_value(&crate::launcher::project::quoted(text)).as_deref(),
        Some(text)
    );
    assert_eq!(
        parse_value("\"\\U0001F600 \\u00e9\"").as_deref(),
        Some("\u{1F600} \u{e9}")
    );
    assert_eq!(parse_value("\"\\uD800\""), None);
    assert_eq!(parse_value("\"\\u12\""), None);
}

#[test]
fn a_moved_harness_plugin_listed_from_the_harness_names_its_new_home() {
    let entry = |plugin: &str| {
        parse(
            &format!("[[plugin]]\ncrate = \"rig-harness\"\nplugin = \"{plugin}\"\n"),
            Path::new("/home"),
        )
        .err()
        .map(|failure| failure.to_string())
        .unwrap_or_default()
    };
    let picker = entry("rig_harness::sessions::SessionPickerPlugin");
    assert!(
        picker.ends_with(
            "write `crate = \"rig-basics\"` and \
             `plugin = \"rig_basics::session_picker::SessionPickerPlugin\"`"
        ),
        "{picker}"
    );
    let other = entry("rig_harness::RigHarnessPlugins");
    assert!(
        other.ends_with("`rig-harness` is part of the agent itself"),
        "{other}"
    );
}
