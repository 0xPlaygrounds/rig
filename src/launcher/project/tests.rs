use super::super::config::{Package, Plugin, TEMPLATE, parse};
use super::*;

fn checkout() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn the_core_crates_are_rig_harness_and_its_rig_dependencies() {
    let mut core: BTreeSet<String> = dependencies(&checkout().join("crates/rig-harness"))
        .into_iter()
        .filter(|name| name == "rig" || name.starts_with("rig-"))
        .collect();
    core.insert("rig-harness".to_owned());
    assert_eq!(core, CORE_CRATES.map(str::to_owned).into());
}

#[test]
fn the_default_plugins_come_from_the_checkout_and_only_what_they_use_is_patched() {
    let checkout = checkout();
    let source = RigSource::Local(checkout.clone());
    let used = parse(TEMPLATE, Path::new("/home"))
        .map(|config| rig_crates(&config, &source))
        .unwrap_or_default();
    // Every plugin crate but the optional ones, and what they depend on.
    for name in [
        "rig-tui",
        "rig-activity",
        "rig-compaction",
        "rig-memory",
        "rig-reqwest",
    ] {
        assert!(used.contains(name), "{name} is used");
    }
    for name in ["rig-inspect", "rig-steel", "rig-agent"] {
        assert!(!used.contains(name), "{name} is not used");
    }
    let patch = rig_patch(&checkout, &used);
    assert_eq!(patch.lines().count(), used.len() + 1);
    let tui = quoted_path(&checkout.join("plugins/rig-tui"));
    assert!(patch.contains(&format!("\nrig-tui = {{ path = {tui} }}\n")));
    assert_eq!(
        rig_source(&source, "rig-tui").ok(),
        Some(format!("path = {tui}"))
    );
    assert!(rig_source(&source, "rig-nothing").is_err());
    assert_eq!(
        rig_source(&RigSource::Registry, "rig-tui").ok(),
        Some(format!("version = \"={VERSION}\""))
    );
}

#[test]
fn a_path_plugin_gets_the_rig_crates_it_depends_on() {
    let checkout = checkout();
    // A crate outside rig that depends on what rig-tui depends on.
    let viz = Plugin {
        type_path: "viz::VizPlugin".to_owned(),
        package: Package {
            name: "viz".to_owned(),
            source: Source::Path(checkout.join("plugins/rig-tui")),
        },
    };
    let used = rig_crates(&Config { plugins: vec![viz] }, &RigSource::Local(checkout));
    assert!(used.contains("rig-activity") && used.contains("rig-telemetry"));
    assert!(!used.contains("rig-tui") && !used.contains("viz"));
}
