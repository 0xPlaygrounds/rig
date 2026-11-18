#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures write and read files of their own"
)]

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
        // A crate, not a type in it: the agent would not compile.
        "[[plugins.extra]]\ncrate = \"x\"\nversion = \"1\"\nplugin = \"x\"\n",
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

/// A private file's staged copy is never readable by others, watched while
/// edits run; a link planted at the staged name is not followed; a new file
/// is private too.
#[cfg(unix)]
#[test]
fn a_private_settings_file_is_never_staged_readable() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicBool, Ordering};

    let mode =
        |path: &Path| std::fs::symlink_metadata(path).map(|meta| meta.permissions().mode() & 0o777);
    let set_key = |path: &Path, value: i64| {
        edit::<_, Error>(path, |file| {
            file.document.insert("key", toml_edit::value(value));
            Ok((true, ()))
        })
    };
    let dir = std::env::temp_dir().join(format!("rig-config-staged-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let path = dir.join("settings.toml");
    // Long enough that writing it outlasts one look at the directory.
    let padded = format!("# {}\nkey = 0\n", "x".repeat(1 << 20));
    assert!(std::fs::write(&path, padded).is_ok());
    assert!(std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).is_ok());

    let done = AtomicBool::new(false);
    let (open, edits) = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if !name.starts_with(".settings.toml.") || name.ends_with(".lock") {
                        continue;
                    }
                    let seen = entry
                        .metadata()
                        .map(|meta| meta.permissions().mode() & 0o777);
                    if let Some(open) = seen.ok().filter(|seen| seen & 0o077 != 0) {
                        return Some(open);
                    }
                }
            }
            None
        });
        let edits: Vec<_> = (0..40)
            .take_while(|_| !watcher.is_finished())
            .map(|at| set_key(&path, at))
            .collect();
        done.store(true, Ordering::Relaxed);
        (watcher.join().ok().flatten(), edits)
    });
    assert!(
        edits.iter().all(|edit| *edit == Ok((true, ()))),
        "{edits:?}"
    );
    assert_eq!(open, None, "staged readable by others");
    assert_eq!(mode(&path).ok(), Some(0o600));

    // Links at every staged name the next edit may take, as each call has
    // its own.
    let staged: Vec<_> = (0..1024)
        .map(|call| dir.join(format!(".settings.toml.{}.{call}", std::process::id())))
        .collect();
    let outside = dir.join("outside");
    assert!(std::fs::write(&outside, "").is_ok());
    assert!(
        staged
            .iter()
            .all(|staged| symlink(&outside, staged).is_ok())
    );
    assert_eq!(set_key(&path, -1), Ok((true, ())));
    assert_eq!(std::fs::read_to_string(&outside).ok().as_deref(), Some(""));
    assert_eq!(mode(&path).ok(), Some(0o600));
    let left = staged
        .iter()
        .filter(|staged| std::fs::symlink_metadata(staged).is_ok())
        .count();
    assert_eq!(
        left,
        staged.len() - 1,
        "the edit's own staged name is cleared"
    );

    let fresh = dir.join("fresh.toml");
    assert_eq!(set_key(&fresh, 1), Ok((true, ())));
    assert_eq!(mode(&fresh).ok(), Some(0o600));
    std::fs::remove_dir_all(&dir).ok();
}

/// An edit keeps the file's group, which a new file in its folder may not
/// get.
#[cfg(unix)]
#[test]
fn an_edit_keeps_the_files_group() {
    use std::os::unix::fs::MetadataExt;

    let dir = std::env::temp_dir().join(format!("rig-config-group-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let path = dir.join("settings.toml");
    assert!(std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&path, "key = 0\n").is_ok());
    let own = std::fs::metadata(&path).map(|meta| meta.gid()).ok();
    // Another group the user is in, if any.
    let groups = std::process::Command::new("id").arg("-G").output();
    let groups = groups.map(|out| String::from_utf8_lossy(&out.stdout).into_owned());
    let other = groups
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|gid| gid.parse().ok())
        .find(|gid: &u32| Some(*gid) != own);
    let Some(other) = other else {
        std::fs::remove_dir_all(&dir).ok();
        return;
    };
    assert!(std::os::unix::fs::chown(&path, None, Some(other)).is_ok());
    let edited = edit::<_, Error>(&path, |file| {
        file.document.insert("key", toml_edit::value(1));
        Ok((true, ()))
    });
    let gid = std::fs::metadata(&path).map(|meta| meta.gid()).ok();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(edited, Ok((true, ())));
    assert_eq!(gid, Some(other));
}

/// A link whose target does not exist yet is written through, absolute,
/// relative or chained: the target is made and the link stays.
#[cfg(unix)]
#[test]
fn an_edit_makes_a_dangling_link_target_and_keeps_the_link() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!("rig-config-dangling-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = root.join("home");
    let dotfiles = root.join("dotfiles");
    assert!(std::fs::create_dir_all(&home).is_ok() && std::fs::create_dir_all(&dotfiles).is_ok());
    let path = home.join("settings.toml");
    let cases: [(&str, &dyn Fn(&Path) -> std::io::Result<()>); 3] = [
        ("absolute", &|target| symlink(target, &path)),
        ("relative", &|target| {
            symlink(
                Path::new("../dotfiles").join(target.file_name().unwrap_or_default()),
                &path,
            )
        }),
        ("chained", &|target| {
            std::fs::remove_file(home.join("intermediate")).ok();
            symlink(target, home.join("intermediate"))?;
            symlink("intermediate", &path)
        }),
    ];
    for (kind, link) in cases {
        let target = dotfiles.join(format!("{kind}.toml"));
        std::fs::remove_file(&path).ok();
        assert!(link(&target).is_ok(), "{kind}");
        let edited = edit(&path, |file| {
            Ok::<_, Error>((set_on(&mut file.document, BUDGET, true, false), ()))
        });
        assert_eq!(edited, Ok((true, ())), "{kind}");
        assert!(std::fs::read_link(&path).is_ok(), "{kind}: still a link");
        let written = std::fs::read_to_string(&target).unwrap_or_default();
        assert!(written.contains(BUDGET), "{kind}: {written}");
    }
    std::fs::remove_dir_all(&root).ok();
}

/// A link into a missing folder, or a loop of links, is refused naming
/// where it fails, and the link stays.
#[cfg(unix)]
#[test]
fn an_edit_refuses_a_link_it_cannot_write_through() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!("rig-config-unwritable-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    assert!(std::fs::create_dir_all(&root).is_ok());
    let path = root.join("settings.toml");
    let missing = root.join("no-folder").join("settings.toml");
    let other = root.join("other.toml");
    assert!(symlink(&missing, &path).is_ok());
    let set = || {
        edit(&path, |file| {
            Ok::<_, Error>((set_on(&mut file.document, BUDGET, true, false), ()))
        })
        .err()
        .map(|why| why.to_string())
        .unwrap_or_default()
    };
    // The step that failed names its own file: the lock beside the target.
    let refused = set();
    let folder = root.join("no-folder");
    assert!(
        refused.starts_with(&format!("open {}/", folder.display())),
        "{refused}"
    );
    assert!(!root.join("no-folder").exists(), "no folder made");
    assert!(std::fs::read_link(&path).is_ok(), "still a link");

    assert!(std::fs::remove_file(&path).is_ok());
    assert!(symlink(&other, &path).is_ok() && symlink(&path, &other).is_ok());
    let refused = set();
    assert!(
        refused.starts_with(&format!("{}: ", path.display())) && refused.contains("links"),
        "{refused}"
    );
    assert!(std::fs::read_link(&path).is_ok(), "still a link");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn turning_several_plugins_writes_all_of_them_or_nothing() {
    let root = std::env::temp_dir().join(format!("rig-config-turn-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    assert!(std::fs::create_dir_all(&root).is_ok());
    let path = root.join("settings.toml");
    let before = "# mine\n[plugins]\ndisable = [\"rig_basics::welcome::WelcomePlugin\"]\n";
    assert!(std::fs::write(&path, before).is_ok());
    let unavailable = BUILT_IN.iter().find(|built_in| !built_in.available());
    let refused = ["nope::NopePlugin"]
        .into_iter()
        .chain(unavailable.map(|built_in| built_in.plugin));
    for plugin in refused {
        let turned = edit(&path, |file| {
            turn(file, &[(BUDGET, true), (plugin, true)]).map(|changed| (changed, ()))
        });
        assert!(turned.is_err(), "{plugin}");
        let after = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(after, before, "{plugin}");
    }
    let turned = edit(&path, |file| {
        let changes = [(BUDGET, true), ("rig_basics::welcome::WelcomePlugin", true)];
        turn(file, &changes).map(|changed| (changed, ()))
    });
    assert_eq!(turned, Ok((true, ())));
    let chosen = SettingsFile::read(&path).and_then(|file| plugins(&file));
    let on = chosen.map(|chosen| chosen.on).unwrap_or_default();
    assert!(
        on.iter().any(|plugin| plugin == BUDGET)
            && on.iter().any(|plugin| plugin.ends_with("WelcomePlugin")),
        "{on:?}"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// The refusal of an unknown name in `enable` names the commands that fix
/// it, with the name in them.
#[test]
fn an_unknown_enabled_plugin_names_its_fixes() {
    let refusal = plugins(&file(
        "/h/settings.toml",
        "[plugins]\nenable = [\"x::XPlugin\"]\n",
    ))
    .err()
    .map(|error| error.to_string());
    assert_eq!(
        refusal.as_deref(),
        Some(
            "/h/settings.toml [plugins]: unknown plugin `x::XPlugin` in `enable`: \
             `rig plugin disable x::XPlugin` takes it out, or `rig plugin add x::XPlugin` \
             with its crate's source adds it"
        )
    );
}

const SESSIONS: &str = "rig_basics::sessions::SessionsPlugin";
const FORKS: &str = "rig_basics::sessions::agents::AgentsPlugin";
const TUI: &str = "rig_tui::TuiPlugin";

/// `text` turned by `changes` with [`turn`], and its text after.
fn turned(text: &str, changes: &[(&str, bool)]) -> (Result<bool, Error>, String) {
    let mut home = file("/home/.rig/settings.toml", text);
    let changed = turn(&mut home, changes);
    (changed, home.document.to_string())
}

#[test]
fn a_plugins_area_is_the_module_under_its_crate() {
    assert_eq!(area(FORKS), Some("sessions"));
    assert_eq!(area(SESSIONS), Some("sessions"));
    assert_eq!(area("rig_basics::BasicCommandsPlugin"), None);
    assert_eq!(area(TUI), None);
}

#[test]
fn a_branch_off_then_on_brings_back_each_plugins_own_choice() {
    let mine = format!("# mine\n[plugins]\nenable = [\"{BUDGET}\"]\ndisable = [\"{SKILLS}\"]\n");
    let before = plugins(&file("/home/.rig/settings.toml", &mine)).unwrap_or_default();
    let (changed, off) = turned(&mine, &[("rig-basics", false)]);
    assert_eq!(changed, Ok(true));
    let chosen = plugins(&file("/home/.rig/settings.toml", &off)).unwrap_or_default();
    assert_eq!(chosen.off, ["rig-basics"], "{off}");
    assert!(
        !chosen
            .on
            .iter()
            .any(|plugin| plugin.starts_with("rig_basics::")),
        "{:?}",
        chosen.on
    );
    assert!(
        chosen.on.iter().any(|plugin| plugin == TUI),
        "other crates stay"
    );
    assert_eq!(chosen.off_with(BUDGET), Some("rig-basics"));
    assert_eq!(chosen.own, before.own, "each plugin's own choice stays");
    let (changed, back) = turned(&off, &[("rig-basics", true)]);
    assert_eq!(changed, Ok(true));
    assert_eq!(back, mine, "on again, the file is as it was");
}

#[test]
fn plugins_of_an_off_branch_stay_off_whatever_their_default() {
    // Every plugin of the area, its defaults on or off and any added later.
    let text = "[plugins]\noff = [\"rig-basics/sessions\", \"mine\"]\n\
                [[plugins.extra]]\ncrate = \"mine\"\nversion = \"1\"\nplugin = \"mine::MinePlugin\"\n";
    let first = plugins(&file("/h/settings.toml", text)).unwrap_or_default();
    let added = format!(
        "{text}[[plugins.extra]]\ncrate = \"mine\"\nversion = \"1\"\nplugin = \"mine::NewPlugin\"\n"
    );
    let later = plugins(&file("/h/settings.toml", &added));
    assert!(later.is_ok(), "{later:?}");
    let later = later.unwrap_or_default();
    for chosen in [&first, &later] {
        assert!(
            !chosen.on.iter().any(|plugin| plugin.starts_with("mine::")
                || plugin.starts_with("rig_basics::sessions::")),
            "{:?}",
            chosen.on
        );
    }
    assert!(later.own.iter().any(|plugin| plugin == "mine::NewPlugin"));
    assert_eq!(later.off_with("mine::NewPlugin"), Some("mine"));
    assert_eq!(later.off_with(SKILLS), None);
}

#[test]
fn nested_branches_each_keep_their_plugins_off() {
    let text = "[plugins]\noff = [\"rig-basics\", \"rig-basics/sessions\"]\n";
    let (changed, crate_on) = turned(text, &[("rig-basics", true)]);
    assert_eq!(changed, Ok(true));
    let chosen = plugins(&file("/h/settings.toml", &crate_on)).unwrap_or_default();
    assert_eq!(chosen.off, ["rig-basics/sessions"], "{crate_on}");
    assert!(chosen.on.iter().any(|plugin| plugin == SKILLS));
    assert!(
        !chosen
            .on
            .iter()
            .any(|plugin| plugin == SESSIONS || plugin == FORKS)
    );
    let (_, area_on) = turned(text, &[("rig-basics/sessions", true)]);
    let chosen = plugins(&file("/h/settings.toml", &area_on)).unwrap_or_default();
    assert_eq!(chosen.off_with(SESSIONS), Some("rig-basics"));
    assert!(
        !chosen
            .on
            .iter()
            .any(|plugin| plugin.starts_with("rig_basics::"))
    );
}

#[test]
fn an_unknown_branch_is_refused_naming_it_and_its_fix() {
    for name in ["rig-basics/nope", "nope", "rig-basics/BasicCommandsPlugin"] {
        let text = format!("[plugins]\noff = [\"{name}\"]\n");
        let refused = (plugins(&file("/home/.rig/settings.toml", &text)).err())
            .map(|why| why.to_string())
            .unwrap_or_default();
        assert!(
            refused.contains("/home/.rig/settings.toml")
                && refused.contains(&format!("`{name}`"))
                && refused.contains(&format!("rig plugin disable {name}")),
            "{refused}"
        );
        let (turned_off, _) = turned("", &[(name, false)]);
        assert!(turned_off.is_err(), "{name}");
    }
}

#[test]
fn a_needed_plugin_under_an_off_branch_stays_on() {
    const SUBAGENTS: &str = "rig_basics::subagents::SubagentsPlugin";
    const APPROVALS: &str = "rig_basics::approval::ApprovalPlugin";
    let text = format!("[plugins]\nenable = [\"{SUBAGENTS}\"]\noff = [\"rig-basics/approval\"]\n");
    let chosen = plugins(&file("/h/settings.toml", &text)).unwrap_or_default();
    assert!(
        chosen.on.iter().any(|plugin| plugin == APPROVALS),
        "{:?}",
        chosen.on
    );
    assert_eq!(chosen.needed_by(APPROVALS), Some(SUBAGENTS));
    let order = |plugin: &str| chosen.on.iter().position(|it| it == plugin);
    assert!(order(APPROVALS) < order(SUBAGENTS), "in the table's order");
    let alone = plugins(&file(
        "/h/settings.toml",
        "[plugins]\noff = [\"rig-basics/approval\"]\n",
    ))
    .unwrap_or_default();
    assert!(!alone.on.iter().any(|plugin| plugin == APPROVALS));
    assert_eq!(alone.needed_by(APPROVALS), None);
}

#[test]
fn every_link_of_the_table_names_one_of_its_rows() {
    for row in BUILT_IN {
        for linked in row.adds.iter().chain(row.only_with) {
            assert!(built_in(linked).is_some(), "{} links {linked}", row.plugin);
        }
    }
}
