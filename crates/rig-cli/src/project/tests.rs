use crate::LONG_VERSION;
use rig_harness_protocol::config::SettingsFile;

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

/// `extras`, each on, and none of rig's plugins.
fn only(extras: &[Extra]) -> Plugins {
    Plugins {
        on: extras.iter().map(|extra| extra.plugin.clone()).collect(),
        extra: extras.to_vec(),
    }
}

/// The plugins of a home whose `settings.toml` is `text`.
fn home_with(text: &str) -> std::result::Result<Plugins, rig_harness_protocol::config::Error> {
    let file = SettingsFile::parse(PathBuf::from("/home/rig/settings.toml"), text)?;
    rig_harness_protocol::config::plugins(&file)
}

/// How the generated program adds `plugin`.
fn added(plugin: &str) -> String {
    format!(".add(rig_harness::Loaded::<{plugin}>::new(")
}

#[test]
fn the_core_crates_are_rig_harness_and_its_rig_dependencies() {
    let mut core: BTreeSet<String> = dependencies(&checkout().join("crates/rig-harness"))
        .unwrap_or_default()
        .into_iter()
        .filter(|name| name == "rig" || name.starts_with("rig-"))
        .collect();
    core.insert("rig-harness".to_owned());
    assert_eq!(core, CORE_CRATES.map(str::to_owned).into());
}

#[test]
fn a_home_without_plugins_builds_exactly_rigs_defaults() {
    let checkout = checkout();
    let source = RigSource::Local(checkout.clone());
    let plugins = home_with("[tui]\n").unwrap_or_default();
    let home = Home::at("/home/rig");
    let main = main_rs(&home, &plugins, &source);
    for built_in in BUILT_IN {
        assert_eq!(
            main.contains(&added(built_in.plugin)),
            built_in.on,
            "{}: {main}",
            built_in.plugin
        );
    }
    assert!(main.contains("rig_harness::PluginGroupBuilder::start::<Self>()"));
    let manifest = manifest(&home, &plugins, &source).unwrap_or_default();
    assert!(manifest.contains("\nrig-tui = "), "{manifest}");
    assert!(!manifest.contains("\nrig-steel = "), "off: {manifest}");
    assert!(
        manifest.contains("default-features = false, features = [\"tui\"] }\n"),
        "{manifest}"
    );
    let used = rig_crates(&plugins, &source).unwrap_or_default();
    for name in ["rig-memory", "rig-reqwest", "rig-tui", "rig-basics"] {
        assert!(used.contains(name), "{name} is used");
    }
    assert!(!used.contains("rig-steel") && !used.contains("rig-agent"));
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
fn enable_and_disable_change_the_programs_plugin_group() {
    const STEEL: &str = "rig_steel::SteelPlugin";
    const SKILLS: &str = "rig_basics::skills::SkillsPlugin";
    let source = RigSource::Registry;
    let home = Home::at("/home/rig");
    let plugins = home_with(&format!(
        "[plugins]\nenable = [\"{STEEL}\"]\ndisable = [\"{SKILLS}\"]\n"
    ));
    assert!(plugins.is_ok(), "{plugins:?}");
    let plugins = plugins.unwrap_or_default();
    let main = main_rs(&home, &plugins, &source);
    assert!(main.contains(&added(STEEL)), "{main}");
    assert!(!main.contains(&added(SKILLS)), "{main}");
    let manifest = manifest(&home, &plugins, &source).unwrap_or_default();
    assert!(manifest.contains("\nrig-steel = "), "{manifest}");
    // Without the terminal view, nothing builds rig-tui.
    let headless = home_with(&format!("[plugins]\ndisable = [\"{TUI}\"]\n")).unwrap_or_default();
    let manifest = super::manifest(&home, &headless, &source).unwrap_or_default();
    assert!(
        !manifest.contains("rig-tui") && !manifest.contains("\"tui\""),
        "{manifest}"
    );
    assert!(
        manifest.contains("default-features = false }\n"),
        "{manifest}"
    );
    let refused = home_with("[plugins]\nenable = [\"rig_basics::nope::NopePlugin\"]\n")
        .err()
        .map(|why| why.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains("rig_basics::nope::NopePlugin"),
        "{refused}"
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
    let used = rig_crates(&only(&viz), &source).unwrap_or_default();
    assert!(used.contains("rig-harness") && used.contains("rig-ecs"));
    assert!(!used.contains("viz"));
    // What /reload rebuilds stays at opt-level 1; the rest builds at 3.
    let profile = profile(&only(&viz), &source).unwrap_or_default();
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
    assert!(
        manifest(
            &home,
            &only(std::slice::from_ref(&cloak)),
            &RigSource::Registry
        )
        .is_err()
    );
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

/// The project's `Cargo.lock` starts from the checkout's, and again
/// whenever the checkout's changes, such as after a pull or a switch to
/// another checkout, or a start from crates.io came between; otherwise it
/// keeps what cargo resolved.
#[test]
fn the_lock_is_seeded_again_when_the_checkouts_changes() {
    let root = std::env::temp_dir().join(format!("rig-lock-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let checkout = root.join("checkout");
    fs::create_dir_all(checkout.join("crates/rig-harness")).ok();
    fs::write(checkout.join("crates/rig-harness/Cargo.toml"), "").ok();
    let home = Home::at(root.join("home"));
    let source = RigSource::Local(checkout.clone());
    let lock = home.project().join("Cargo.lock");
    let generated = |checkouts: &str| {
        fs::write(checkout.join("Cargo.lock"), checkouts).ok();
        generate(&home, &only(&[]), &source).map_err(|why| why.to_string())?;
        Ok::<_, String>(fs::read_to_string(&lock).unwrap_or_default())
    };
    let first = generated("first");
    fs::write(&lock, "first, resolved").ok();
    let kept = generated("first");
    let pulled = generated("pulled");
    // A start from crates.io in between resolves the lock to its graph.
    let registry = generate(&home, &only(&[]), &RigSource::Registry).map_err(|why| why.to_string());
    fs::write(&lock, "crates.io's").ok();
    let back = generated("pulled");
    fs::remove_dir_all(&root).ok();
    assert_eq!(first.as_deref(), Ok("first"));
    assert_eq!(kept.as_deref(), Ok("first, resolved"));
    assert_eq!(pulled.as_deref(), Ok("pulled"));
    assert_eq!(registry, Ok(()));
    assert_eq!(back.as_deref(), Ok("pulled"));
}

/// A launcher built from a commit of a checkout that is gone builds no
/// agent from crates.io, whose release is other code; it says what to do.
#[test]
fn a_launcher_from_a_commit_does_not_build_from_crates_io() {
    // No commit: a crates.io package, whatever its version.
    assert!(RigSource::Registry.check("").is_ok());
    assert!(RigSource::Local(checkout()).check("b1321662b").is_ok());
    let refused = RigSource::Registry.check("b1321662b").err();
    let refused = refused.map(|why| why.to_string()).unwrap_or_default();
    assert!(
        refused.contains(LONG_VERSION) && refused.contains("cargo install --path"),
        "{refused}"
    );
    assert_eq!(refused.lines().count(), 1, "{refused}");
}

/// Built in a rig checkout, the launcher's version names the checkout's
/// commit, the one checked out now; built elsewhere, it is the version.
#[test]
fn the_version_names_the_commit_of_the_checkout_it_was_built_in() {
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(checkout())
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let root = git(&["rev-parse", "--show-toplevel"]).map(PathBuf::from);
    let in_checkout =
        root.and_then(|root| root.canonicalize().ok()) == checkout().canonicalize().ok();
    let expected = match git(&["rev-parse", "--short=9", "HEAD"]) {
        Some(head) if in_checkout => format!("{VERSION} ({head})"),
        _ => VERSION.to_owned(),
    };
    assert_eq!(LONG_VERSION, expected);
}
