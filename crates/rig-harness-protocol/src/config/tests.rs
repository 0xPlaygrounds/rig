use super::*;

const BUDGET: &str = "rig_basics::budget::BudgetPlugin";
const SKILLS: &str = "rig_basics::skills::SkillsPlugin";

fn file(path: &str, text: &str) -> SettingsFile {
    let parsed = SettingsFile::parse(PathBuf::from(path), text);
    assert!(parsed.is_ok(), "{parsed:?}");
    parsed.unwrap_or_else(|_| SettingsFile {
        path: PathBuf::from(path),
        document: DocumentMut::new(),
    })
}

fn defaults() -> Vec<String> {
    (BUILT_IN.iter())
        .filter(|built_in| built_in.on)
        .map(|built_in| built_in.plugin.to_owned())
        .collect()
}

#[test]
fn a_home_without_plugins_runs_the_built_in_defaults() {
    let chosen = plugins(&file("/home/.rig/settings.toml", "[tui]\n"));
    assert_eq!(chosen.map(|chosen| chosen.on), Ok(defaults()));
}

#[test]
fn enable_and_disable_change_the_defaults_in_their_order() {
    let home = file(
        "/home/.rig/settings.toml",
        &format!("[plugins]\nenable = [\"{BUDGET}\"]\ndisable = [\"{SKILLS}\"]\n"),
    );
    let on = plugins(&home).map(|chosen| chosen.on).unwrap_or_default();
    assert!(on.iter().any(|plugin| plugin == BUDGET));
    assert!(!on.iter().any(|plugin| plugin == SKILLS));
    let order: Vec<&str> = BUILT_IN.iter().map(|built_in| built_in.plugin).collect();
    let mut sorted = on.clone();
    sorted.sort_by_key(|plugin| order.iter().position(|it| it == plugin));
    assert_eq!(on, sorted, "the built-in order");
}

#[test]
fn an_unknown_plugin_is_refused_naming_it_and_the_file() {
    let home = file(
        "/home/.rig/settings.toml",
        "[plugins]\nenable = [\"rig_basics::nope::NopePlugin\"]\n",
    );
    let refused = plugins(&home)
        .err()
        .map(|why| why.to_string())
        .unwrap_or_default();
    assert!(refused.contains("/home/.rig/settings.toml"), "{refused}");
    assert!(
        refused.contains("rig_basics::nope::NopePlugin"),
        "{refused}"
    );
}

#[test]
fn the_homes_extras_are_on_after_rig_plugins_unless_disabled() {
    let home = file(
        "/home/.rig/settings.toml",
        "[[plugins.extra]]\ncrate = \"viz\"\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n\
         [[plugins.extra]]\ncrate = \"off\"\nversion = \"1\"\nplugin = \"off::OffPlugin\"\n\
         [plugins]\n",
    );
    let chosen = plugins(&home).unwrap_or_default();
    assert_eq!(chosen.on.last().map(String::as_str), Some("off::OffPlugin"));
    assert_eq!(
        chosen.extra.first().map(|extra| &extra.source),
        Some(&Source::Path(PathBuf::from("/home/.rig/plugins/viz")))
    );
    let disabled = file(
        "/home/.rig/settings.toml",
        "[plugins]\ndisable = [\"viz::VizPlugin\"]\n\
         [[plugins.extra]]\ncrate = \"viz\"\npath = \"plugins/viz\"\nplugin = \"viz::VizPlugin\"\n",
    );
    let chosen = plugins(&disabled).unwrap_or_default();
    assert_eq!(chosen.extra.len(), 1, "listed, though off");
    assert!(!chosen.on.iter().any(|plugin| plugin == "viz::VizPlugin"));
    assert_eq!(chosen.extras_on().count(), 0, "not built");
}

#[test]
fn bad_tables_are_refused() {
    for text in [
        "[plugins]\nenable = \"x\"\n",
        "[plugins]\nturn_on = []\n",
        &format!("[plugins]\nenable = [\"{BUDGET}\"]\ndisable = [\"{BUDGET}\"]\n"),
        "[[plugins.extra]]\ncrate = \"x\"\nplugin = \"x::X\"\n",
        "[[plugins.extra]]\ncrate = \"x\"\npath = \"x\"\nbranch = \"main\"\nplugin = \"x::X\"\n",
        "[[plugins.extra]]\ncrate = \"rig-harness\"\npath = \"x\"\nplugin = \"x::X\"\n",
        &format!(
            "[[plugins.extra]]\ncrate = \"rig-basics\"\npath = \"x\"\nplugin = \"{BUDGET}\"\n"
        ),
        "[[plugins.extra]]\ncrate = \"x\"\npath = \"x\"\nplugin = \"x::X\"\n\
         [[plugins.extra]]\ncrate = \"x\"\nversion = \"1\"\nplugin = \"x::Y\"\n",
    ] {
        assert!(plugins(&file("/h/settings.toml", text)).is_err(), "{text}");
    }
}

#[test]
fn a_plugin_built_for_other_targets_only_is_unavailable() {
    let elsewhere = BUILT_IN.first().map(|first| BuiltIn {
        targets: &["nowhere"],
        ..*first
    });
    assert!(elsewhere.is_some_and(|elsewhere| !elsewhere.available()));
    assert!(
        BUILT_IN
            .iter()
            .filter(|built_in| built_in.on)
            .all(BuiltIn::available)
    );
}

#[test]
fn turning_a_plugin_on_and_back_keeps_every_comment() {
    let text = "# my settings\n[plugins] # changes\n# on as well:\nenable = [\"rig_basics::helpers::HelpersPlugin\"] # helpers\n\n[tui]\n# keep\nx = 1\n";
    let mut document: DocumentMut = text.parse().unwrap_or_default();
    assert!(set_on(&mut document, BUDGET, true, false));
    let enabled = document.to_string();
    assert!(
        enabled.contains("# my settings")
            && enabled.contains("# helpers")
            && enabled.contains("# keep"),
        "{enabled}"
    );
    assert!(enabled.contains(BUDGET), "{enabled}");
    assert!(!set_on(&mut document, BUDGET, true, false), "already on");
    assert!(set_on(&mut document, BUDGET, false, false));
    assert_eq!(document.to_string(), text);
}

#[test]
fn emptying_a_list_keeps_the_users_comment_on_it() {
    let text = format!("[plugins]\nenable = [\"{BUDGET}\"] # spending guard\n");
    let mut document: DocumentMut = text.parse().unwrap_or_default();
    assert!(set_on(&mut document, BUDGET, false, false));
    let kept = document.to_string();
    assert!(
        kept.contains("# spending guard") && !kept.contains(BUDGET),
        "{kept}"
    );
}

#[test]
fn a_comment_above_a_removed_name_stays() {
    let text = format!(
        "[plugins]\nenable = [\n  # spending guard\n  \"{BUDGET}\",\n  # and helpers\n  \"rig_basics::helpers::HelpersPlugin\",\n]\n"
    );
    let mut document: DocumentMut = text.parse().unwrap_or_default();
    assert!(set_on(&mut document, BUDGET, false, false));
    assert!(set_on(
        &mut document,
        "rig_basics::helpers::HelpersPlugin",
        false,
        false
    ));
    let kept = document.to_string();
    assert!(
        kept.contains("# spending guard") && kept.contains("# and helpers"),
        "{kept}"
    );
    assert!(
        !kept.contains(BUDGET) && !kept.contains("HelpersPlugin"),
        "{kept}"
    );
    assert!(kept.parse::<DocumentMut>().is_ok(), "{kept}");
}

#[test]
fn turning_a_default_off_writes_only_that() {
    let mut document = DocumentMut::new();
    assert!(set_on(&mut document, SKILLS, false, true));
    assert_eq!(
        document.to_string(),
        format!("[plugins]\ndisable = [\"{SKILLS}\"]\n")
    );
    assert!(set_on(&mut document, SKILLS, true, true));
    assert!(!document.to_string().contains(SKILLS));
}

#[test]
fn an_edit_replaces_a_symlinked_file_where_it_points() {
    let root = std::env::temp_dir().join(format!("rig-config-edit-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let dotfiles = root.join("dotfiles");
    let home = root.join("home");
    assert!(std::fs::create_dir_all(&dotfiles).is_ok() && std::fs::create_dir_all(&home).is_ok());
    let real = dotfiles.join("settings.toml");
    assert!(std::fs::write(&real, "# mine\n").is_ok());
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&real, home.join("settings.toml")).is_ok();
    #[cfg(not(unix))]
    let linked = std::fs::copy(&real, home.join("settings.toml")).is_ok();
    assert!(linked);
    let edited = edit(&home.join("settings.toml"), |file| {
        Ok::<_, Error>((set_on(&mut file.document, BUDGET, true, false), ()))
    });
    assert_eq!(edited, Ok((true, ())));
    let written = std::fs::read_to_string(home.join("settings.toml")).unwrap_or_default();
    assert!(
        written.starts_with("# mine\n") && written.contains(BUDGET),
        "{written}"
    );
    #[cfg(unix)]
    assert!(
        std::fs::read_link(home.join("settings.toml")).is_ok(),
        "still a link"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_file_that_is_not_toml_is_refused_at_its_line() {
    let refused = SettingsFile::parse(PathBuf::from("/h/settings.toml"), "[tui]\nx = \n").err();
    let refused = refused.map(|why| why.to_string()).unwrap_or_default();
    assert!(
        refused.starts_with("/h/settings.toml: not TOML at line 2"),
        "{refused}"
    );
}
