use super::super::plugin_list::{Package, PluginEntry, TEMPLATE, parse};
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
        .map(|list| rig_crates(&list, &source))
        .unwrap_or_default();
    // Every plugin crate but the optional ones, and what they depend on.
    for name in [
        "rig-tui",
        "rig-basics",
        "rig-traffic",
        "rig-memory",
        "rig-reqwest",
    ] {
        assert!(used.contains(name), "{name} is used");
    }
    for name in ["rig-inspect", "rig-steel", "rig-claude-cloak", "rig-agent"] {
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
    let viz = PluginEntry {
        type_path: "viz::VizPlugin".to_owned(),
        package: Package {
            name: "viz".to_owned(),
            source: Source::Path(checkout.join("plugins/rig-tui")),
        },
    };
    let list = PluginList { plugins: vec![viz] };
    let source = RigSource::Local(checkout);
    let used = rig_crates(&list, &source);
    assert!(used.contains("rig-harness") && used.contains("rig-ecs"));
    assert!(!used.contains("rig-tui") && !used.contains("viz"));
    // What /reload rebuilds stays at opt-level 1; the rest builds at 3.
    let profile = profile(&list, &source);
    assert!(profile.contains("[profile.dev.package.\"*\"]\nopt-level = 3\n"));
    for name in ["viz", "rig-harness", "rig-ecs"] {
        assert!(profile.contains(&format!("\n[profile.dev.package.{name}]\nopt-level = 1\n")));
    }
}

#[test]
fn a_crate_of_another_rig_checkout_by_path_is_refused() {
    let checkout = checkout();
    let cloak = Package {
        name: "rig-claude-cloak".to_owned(),
        source: Source::Path(checkout.join("plugins/rig-claude-cloak")),
    };
    assert_eq!(
        foreign_rig_crate(&cloak, &RigSource::Local(checkout.clone())),
        None
    );
    let problem = foreign_rig_crate(&cloak, &RigSource::Registry).unwrap_or_default();
    assert!(
        problem.contains("`crate = \"rig-claude-cloak\"` with no `path`"),
        "{problem}"
    );
    // A crate rig does not ship has no plain entry to point at.
    let renamed = Package {
        name: "claude-cloak".to_owned(),
        source: Source::Path(checkout.join("plugins/rig-claude-cloak")),
    };
    let problem = foreign_rig_crate(&renamed, &RigSource::Registry).unwrap_or_default();
    assert!(
        problem.contains("rig ships no crate of that name") && !problem.contains("crate = "),
        "{problem}"
    );
    let list = PluginList {
        plugins: vec![PluginEntry {
            type_path: "rig_claude_cloak::ClaudeCloakPlugin".to_owned(),
            package: cloak,
        }],
    };
    let home = Home::at("/home/rig");
    assert!(manifest(&home, &list, &RigSource::Registry).is_err());
    // An alias of the same directory is the same crate.
    let alias = Package {
        name: "rig-claude-cloak".to_owned(),
        source: Source::Path(checkout.join("plugins/rig-claude-cloak/src/..")),
    };
    assert!(foreign_rig_crate(&alias, &RigSource::Registry).is_some());
    assert_eq!(
        foreign_rig_crate(&alias, &RigSource::Local(checkout.join("crates/.."))),
        None
    );
    // A crate outside every rig checkout is the user's own.
    let viz = Package {
        name: "viz".to_owned(),
        source: Source::Path(PathBuf::from("/home/viz")),
    };
    assert_eq!(foreign_rig_crate(&viz, &RigSource::Registry), None);
    // Nor is a directory that is not there.
    let gone = Package {
        name: "claude-cloak".to_owned(),
        source: Source::Path(checkout.join("plugins/claude-cloak")),
    };
    assert_eq!(foreign_rig_crate(&gone, &RigSource::Registry), None);
}
