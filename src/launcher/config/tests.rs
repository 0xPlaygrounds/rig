use super::*;

const LIST: &str = "# The plugins.\n\n# The tools.\n[[plugin]]\ncrate = \"rig-coding-tools\"\nplugin = \"rig_coding_tools::ReadTool\"\n\n# A crate.\n[[plugin]]\ncrate = \"viz\"\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n\n# The view.\n[[plugin]]\ncrate = \"rig-tui\"\nplugin = \"rig_tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n";

#[test]
fn removing_a_table_takes_its_comments_and_keeps_the_rest() {
    let base = Path::new("/home");
    assert_eq!(
        without_table(LIST, "viz::VizPlugin", base).ok().as_deref(),
        Some(
            "# The plugins.\n\n# The tools.\n[[plugin]]\ncrate = \"rig-coding-tools\"\nplugin = \"rig_coding_tools::ReadTool\"\n\n# The view.\n[[plugin]]\ncrate = \"rig-tui\"\nplugin = \"rig_tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n"
        )
    );
    let last = without_table(LIST, "rig_tui::TuiPlugin", base)
        .ok()
        .unwrap_or_default();
    assert!(last.ends_with("plugin = \"viz::VizPlugin\"\n\n# An example.\n# [[plugin]]\n"));
    assert_eq!(
        parse(&last, base).ok().map(|config| config.plugins.len()),
        Some(2)
    );
}

#[test]
fn removing_an_unlisted_type_fails() {
    assert!(without_table(LIST, "viz::Other", Path::new("/home")).is_err());
}

#[test]
fn the_template_lists_rig_plugin_crates_and_inspect_uncommented() {
    let plugins = parse(TEMPLATE, Path::new("/home")).map(|config| config.plugins);
    assert!(plugins.is_ok_and(|plugins| {
        plugins.len() == 21
            && plugins
                .iter()
                .all(|plugin| plugin.package.source == Source::Rig)
    }));
    // The commented-out rig-inspect entry, uncommented as a user would.
    let uncommented = TEMPLATE
        .replace(
            "# [[plugin]]\n# crate = \"rig-inspect\"",
            "[[plugin]]\ncrate = \"rig-inspect\"",
        )
        .replace("# plugin = \"rig_inspect", "plugin = \"rig_inspect");
    let config = parse(&uncommented, Path::new("/home"));
    let inspect = config
        .ok()
        .and_then(|config| config.plugins.into_iter().nth(19));
    let package = inspect.as_ref().map(|plugin| &plugin.package);
    assert_eq!(
        package.map(|package| (package.name.as_str(), &package.source)),
        Some(("rig-inspect", &Source::Rig))
    );
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
            .and_then(|config| config.plugins.into_iter().next())
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
