use super::*;

fn checkout() -> PathBuf {
    std::path::absolute(CHECKOUT).unwrap_or_default()
}

fn extra(plugin: &str, krate: &str, directory: PathBuf) -> Extra {
    Extra {
        plugin: plugin.to_owned(),
        krate: krate.to_owned(),
        source: Source::Path(directory),
    }
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
fn every_rig_plugin_is_in_the_one_build_and_added_when_on() {
    let checkout = checkout();
    let source = RigSource::Local(checkout.clone());
    let used = rig_crates(&[], &source);
    // Every plugin crate of rig's that builds here, and what they depend on.
    for built_in in BUILT_IN.iter().filter(|built_in| built_in.available()) {
        assert!(used.contains(built_in.krate), "{} is built", built_in.krate);
    }
    for name in ["rig-memory", "rig-reqwest"] {
        assert!(used.contains(name), "{name} is used");
    }
    assert!(!used.contains("rig-agent"));
    let home = Home::at("/home/rig");
    let main = main_rs(&home, &[], &source);
    for built_in in BUILT_IN.iter().filter(|built_in| built_in.available()) {
        let added = format!("rig_harness::Compiled::new::<{0}>({0:?}, ", built_in.plugin);
        assert!(main.contains(&added), "{main}");
    }
    assert!(main.contains("rig_harness::load_chosen("));
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
    let viz = [extra(
        "viz::VizPlugin",
        "viz",
        checkout.join("plugins/rig-tui"),
    )];
    let source = RigSource::Local(checkout);
    let used = rig_crates(&viz, &source);
    assert!(used.contains("rig-harness") && used.contains("rig-ecs"));
    assert!(!used.contains("viz"));
    // What /reload rebuilds stays at opt-level 1; the rest builds at 3.
    let profile = profile(&viz, &source);
    assert!(profile.contains("[profile.dev.package.\"*\"]\nopt-level = 3\n"));
    for name in ["viz", "rig-harness", "rig-ecs"] {
        assert!(profile.contains(&format!("\n[profile.dev.package.{name}]\nopt-level = 1\n")));
    }
}

#[test]
fn a_crate_of_another_rig_checkout_by_path_is_refused() {
    let checkout = checkout();
    let cloak = extra(
        "rig_claude_cloak::ClaudeCloakPlugin",
        "rig-claude-cloak",
        checkout.join("plugins/rig-claude-cloak"),
    );
    assert_eq!(
        foreign_rig_crate(&cloak, &RigSource::Local(checkout.clone())),
        None
    );
    let problem = foreign_rig_crate(&cloak, &RigSource::Registry).unwrap_or_default();
    assert!(
        problem.contains("name them in `[plugins] enable` instead"),
        "{problem}"
    );
    // A crate rig does not ship has no plain entry to point at.
    let renamed = extra(
        "x::X",
        "claude-cloak",
        checkout.join("plugins/rig-claude-cloak"),
    );
    let problem = foreign_rig_crate(&renamed, &RigSource::Registry).unwrap_or_default();
    assert!(
        problem.contains("rig ships no crate of that name") && !problem.contains("enable"),
        "{problem}"
    );
    let home = Home::at("/home/rig");
    assert!(manifest(&home, &[cloak], &RigSource::Registry).is_err());
    // An alias of the same directory is the same crate.
    let alias = extra(
        "x::X",
        "rig-claude-cloak",
        checkout.join("plugins/rig-claude-cloak/src/.."),
    );
    assert!(foreign_rig_crate(&alias, &RigSource::Registry).is_some());
    assert_eq!(
        foreign_rig_crate(&alias, &RigSource::Local(checkout.join("crates/.."))),
        None
    );
    // A crate outside every rig checkout is the user's own.
    let viz = extra("viz::VizPlugin", "viz", PathBuf::from("/home/viz"));
    assert_eq!(foreign_rig_crate(&viz, &RigSource::Registry), None);
    // Nor is a directory that is not there.
    let gone = extra(
        "x::X",
        "claude-cloak",
        checkout.join("plugins/claude-cloak"),
    );
    assert_eq!(foreign_rig_crate(&gone, &RigSource::Registry), None);
}
