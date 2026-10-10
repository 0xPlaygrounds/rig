use super::*;

const LIST: &str = "# The plugins.\n\n# The tools.\n[[plugin]]\nplugin = \"rig_harness::plugins::tools::BuiltinToolsPlugin\"\n\n# A crate.\n[[plugin]]\ncrate = \"viz\"\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n\n# The view.\n[[plugin]]\nplugin = \"rig_harness::tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n";

#[test]
fn removing_a_table_takes_its_comments_and_keeps_the_rest() {
    let base = Path::new("/home");
    assert_eq!(
        without_table(LIST, "viz::VizPlugin", base).ok().as_deref(),
        Some(
            "# The plugins.\n\n# The tools.\n[[plugin]]\nplugin = \"rig_harness::plugins::tools::BuiltinToolsPlugin\"\n\n# The view.\n[[plugin]]\nplugin = \"rig_harness::tui::TuiPlugin\"\n\n# An example.\n# [[plugin]]\n"
        )
    );
    let last = without_table(LIST, "rig_harness::tui::TuiPlugin", base)
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
fn the_template_lists_rig_harness_plugins_only() {
    let plugins = parse(TEMPLATE, Path::new("/home")).map(|config| config.plugins);
    assert!(plugins.is_ok_and(
        |plugins| plugins.len() == 20 && plugins.iter().all(|plugin| plugin.package.is_none())
    ));
}
