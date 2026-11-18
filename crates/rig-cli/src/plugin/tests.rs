use super::*;

const MCP: &str = "rig_mcp_client::McpPlugin";
const SKILLS: &str = "rig_basics::skills::SkillsPlugin";

fn old_home(test: &str) -> Home {
    Home::at(std::env::temp_dir().join(format!("rig-plugin-{test}-{}", std::process::id())))
}

/// `rig plugin` with `args` on `home`.
fn run(home: &Home, args: &[&str]) -> Result<ExitCode> {
    super::run(
        &crate::test_launch(home, Path::new("/opt/rig/bin/rig"))?,
        args,
    )
}

/// A user's own `settings.toml`, with comments.
const COMMENTED: &str = "# My rig.\n[tui] # the view\n# keep this\n\n# Plugins I tried:\n";

#[test]
fn enable_and_disable_keep_the_users_comments_and_undo_each_other() {
    let home = old_home("comments");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["enable", MCP])?;
        let enabled = fs::read_to_string(home.settings())?;
        run(&home, &["disable", SKILLS])?;
        run(&home, &["enable", SKILLS])?;
        run(&home, &["disable", MCP])?;
        Ok((enabled, fs::read_to_string(home.settings())?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (enabled, undone) = result.unwrap_or_default();
    assert!(enabled.starts_with(COMMENTED), "{enabled}");
    assert!(
        enabled.ends_with(&format!("[plugins]\nenable = [\"{MCP}\"]\n")),
        "{enabled}"
    );
    assert!(undone.starts_with(COMMENTED) && !undone.contains(MCP) && !undone.contains(SKILLS));
}

#[test]
fn a_plugin_that_does_not_build_here_is_refused() {
    let home = old_home("target");
    let elsewhere = BUILT_IN.iter().find(|built_in| !built_in.available());
    if let Some(elsewhere) = elsewhere {
        let refused = run(&home, &["enable", elsewhere.plugin]).err();
        assert!(refused.is_some_and(|why| why.to_string().contains(std::env::consts::OS)));
    }
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn disabling_a_plugin_that_does_not_build_here_fixes_the_file() {
    let home = old_home("unavailable-off");
    let elsewhere = BUILT_IN.iter().find(|built_in| !built_in.available());
    if let Some(elsewhere) = elsewhere {
        let result = (|| -> Result<_> {
            fs::create_dir_all(home.root())?;
            let on = format!("[plugins]\nenable = [\"{}\"]\n", elsewhere.plugin);
            fs::write(home.settings(), on)?;
            run(&home, &["disable", elsewhere.plugin])?;
            Ok(config::chosen(&home)?)
        })();
        assert!(result.is_ok(), "{result:?}");
    }
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn a_new_plugin_crate_is_an_extra_of_the_home() {
    let home = old_home("new");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["new", "word-count"])?;
        let text = fs::read_to_string(home.settings())?;
        let extras = config::chosen(&home)?.extra;
        Ok((text, extras))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (text, extras) = result.unwrap_or_default();
    assert!(
        text.starts_with(COMMENTED) && text.contains("# Made by `rig plugin new word-count`."),
        "{text}"
    );
    assert_eq!(
        extras
            .iter()
            .map(|extra| extra.plugin.as_str())
            .collect::<Vec<_>>(),
        ["word_count::WordCountPlugin"]
    );
}

#[test]
fn plugin_help_anywhere_succeeds_without_creating_a_home() {
    let home = old_home("help");
    fs::remove_dir_all(home.root()).ok();
    let cases: &[&[&str]] = &[
        &["--help"],
        &["add", "--help"],
        &["enable", "-h"],
        &["add", MCP, "--path", "--help"],
        &["-h", "new", "example"],
        &["unknown", "--help"],
    ];
    let results: Vec<_> = cases.iter().map(|args| run(&home, args)).collect();
    let created = home.root().exists();
    fs::remove_dir_all(home.root()).ok();
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Ok(code) if *code == ExitCode::SUCCESS)),
        "{results:?}"
    );
    assert!(!created);
}

#[test]
fn missing_or_unknown_plugin_commands_are_refused_without_creating_a_home() {
    let home = old_home("refused");
    fs::remove_dir_all(home.root()).ok();
    for args in [
        vec![],
        vec!["unknown"],
        vec!["enable"],
        vec!["add"],
        vec!["remove"],
    ] {
        let result = run(&home, &args);
        assert!(
            matches!(result, Ok(code) if code == ExitCode::from(REFUSED_EXIT_CODE)),
            "{args:?}: {result:?}"
        );
    }
    assert!(!home.root().exists());
}

#[test]
fn enabling_an_unknown_type_does_not_change_the_file() {
    let home = old_home("unknown");
    let original = "# An older home.\n";
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), original)?;
        let failure = run(&home, &["enable", "rig_mcp_client::UnknownPlugin"])
            .err()
            .ok_or("unknown type was accepted")?;
        Ok((failure.to_string(), fs::read_to_string(home.settings())?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(
        matches!(&result, Ok((failure, text)) if failure.contains("rig plugin list") && text == original),
        "{result:?}"
    );
}

#[test]
fn the_scaffold_names_the_plugin_and_its_command() {
    assert_eq!(type_name("agent-viz"), "AgentVizPlugin");
    assert_eq!(type_name("my_plugin"), "MyPlugin");
    let lib = lib_rs("word-count", &type_name("word-count"));
    assert!(lib.contains("pub struct WordCountPlugin;"));
    assert!(lib.contains("impl Plugin for WordCountPlugin {"));
    assert!(lib.contains("\"word-count\",\n"));
    assert!(!lib.contains("__name__"));
    assert!(!lib.contains("Scaffold"));
}

#[test]
fn names_are_lowercase_crate_names_not_the_agents_own() {
    let registry = &RigSource::Registry;
    assert!(validate_name("agent-viz", registry).is_ok());
    assert!(validate_name("Viz", registry).is_err());
    assert!(validate_name("1viz", registry).is_err());
    assert!(validate_name("viz/../x", registry).is_err());
    assert!(validate_name("rig-harness", registry).is_err());
    assert!(validate_name(PACKAGE, registry).is_err());
    assert!(validate_name("bevy_ui", registry).is_err());
    // A plugin crate of the checkout this launcher is built in.
    let checkout =
        RigSource::Local(std::path::absolute(super::super::project::CHECKOUT).unwrap_or_default());
    assert!(validate_name("rig-tui", &checkout).is_err());
    assert!(validate_name("rig-tui", registry).is_ok());
}

#[test]
fn the_scaffold_depends_on_the_rig_version_and_patches_a_checkout() {
    let checkout = std::path::absolute(super::super::project::CHECKOUT).unwrap_or_default();
    let local = manifest("viz", "0.44.0", &RigSource::Local(checkout));
    assert!(local.contains("rig-harness = \"0.44.0\"\n"));
    // The scaffold's saved component derives from both Bevy crates, which
    // its derives name as the manifest lists them.
    let lib = lib_rs("viz", "VizPlugin");
    assert!(lib.contains("#[derive(Component, Reflect, Default)]\n#[reflect(Component, Saved)]"));
    // Bevy's major and minor, which cargo unifies with the agent's exact
    // Bevy, so a Bevy patch release does not break the plugin.
    for krate in ["bevy_ecs", "bevy_reflect"] {
        assert!(local.contains(&format!(
            "\n{krate} = {{ version = \"0.20\", default-features = false }}\n"
        )));
    }
    assert!(!local.contains("exactly"));
    // The rig crates it uses through rig-harness, and nothing else.
    let patched: Vec<&str> = local
        .split("[patch.crates-io]\n")
        .nth(1)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.strip_suffix(" }")?.split_once(" = { path = "))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(patched, CORE_CRATES);
    let registry = manifest("viz", "0.44.0", &RigSource::Registry);
    assert!(!registry.contains("[patch"));
    let parsed = registry.parse::<config::toml_edit::DocumentMut>().ok();
    let name = parsed
        .as_ref()
        .and_then(|manifest| manifest.get("package")?.get("name")?.as_str());
    assert_eq!(name, Some("viz"));
}

/// A plugin crate whose `Cargo.toml` is missing, unreadable or not TOML is
/// refused with why: the operating system's error, or the TOML error.
#[test]
fn a_plugin_whose_manifest_cannot_be_read_is_refused_with_why() {
    let directory =
        std::env::temp_dir().join(format!("rig-plugin-manifest-{}", std::process::id()));
    let manifest = directory.join("Cargo.toml");
    let extra = Extra {
        plugin: "my_thing::MyThingPlugin".to_owned(),
        krate: "my-thing".to_owned(),
        source: config::Source::Path(directory.clone()),
    };
    let malformed = "[package\nname = \"my-thing\"\n";
    let toml_error =
        (malformed.parse::<config::toml_edit::DocumentMut>().err()).map(|error| error.to_string());
    let mut refusals = Vec::new();
    for case in ["missing", "unreadable", "malformed"] {
        fs::remove_dir_all(&directory).ok();
        fs::create_dir_all(&directory).ok();
        if case != "missing" {
            fs::write(&manifest, malformed).ok();
        }
        #[cfg(unix)]
        if case == "unreadable" {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&manifest, std::fs::Permissions::from_mode(0o000)).ok();
        }
        // The operating system's own words, unless the file reads (as root).
        let why = match fs::read_to_string(&manifest) {
            Err(error) => Some(error.to_string()),
            Ok(_) => toml_error.clone(),
        };
        let refusal = check_plugin(&extra, &RigSource::Registry).unwrap_or_default();
        refusals.push((case, why, refusal));
    }
    fs::remove_dir_all(&directory).ok();
    for (case, why, refusal) in refusals {
        assert!(
            why.is_some_and(|why| refusal.contains(&why))
                && refusal.contains(&manifest.display().to_string()),
            "{case}: {refusal}"
        );
    }
}

#[test]
fn the_list_is_a_tree_of_crates_areas_and_plugins_rigs_before_the_users() {
    let home = old_home("list");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        let without = listing(&home)?;
        run(&home, &["new", "hello"])?;
        run(&home, &["disable", "rig-basics/sessions"])?;
        Ok((without, listing(&home)?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (without, with) = result.unwrap_or_default();
    assert!(
        without.ends_with(
            "\nyour plugins (extras): none; `rig plugin new <name>` makes one, \
             `rig plugin add` adds another crate's.\n"
        ),
        "{without}"
    );
    let lines: Vec<&str> = with.lines().collect();
    let at = |what: &str| lines.iter().position(|line| line.contains(what));
    let line = |at: Option<usize>| at.and_then(|at| lines.get(at)).copied().unwrap_or("");
    let (rigs, yours) = (at("rig's plugins:"), at("your plugins (extras):"));
    let basics = lines.iter().position(|line| *line == "rig-basics");
    let sessions = at("rig-basics/sessions");
    let (skills, forks, hello) = (
        at(SKILLS),
        at("rig_basics::sessions::agents::AgentsPlugin"),
        at("hello::HelloPlugin"),
    );
    let tui = lines.iter().position(|line| *line == "rig-tui");
    let order = [rigs, basics, skills, sessions, forks, tui, yours, hello];
    assert!(
        order.iter().all(Option::is_some) && order.is_sorted(),
        "{order:?}\n{with}"
    );
    // A plugin under its crate, one more step in under an area.
    let skills = line(skills);
    assert!(
        skills.starts_with("  on ") && skills.ends_with("(default on)"),
        "{skills}"
    );
    assert!(line(sessions).starts_with("  rig-basics/sessions") && line(sessions).contains("off"));
    let forks = line(forks);
    assert!(
        forks.starts_with("    off ") && forks.contains("off with its branch rig-basics/sessions"),
        "{forks}"
    );
    let hello = line(hello);
    let path = home.plugins().join("hello");
    assert!(
        hello.starts_with("  on ") && hello.ends_with(&format!("path {}", path.display())),
        "{hello}"
    );
}

/// A plugin crate `name` with the library `library`, in `directory`.
fn plugin_crate(directory: &Path, name: &str) -> Result<()> {
    fs::create_dir_all(directory.join("src"))?;
    fs::write(
        directory.join("Cargo.toml"),
        format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
    )?;
    fs::write(directory.join("src/lib.rs"), "")?;
    Ok(())
}

const SAVE: &str = "reactor_save::SavePlugin";

/// `path` relative to the working directory, which a relative `--path` is
/// read against; absolute where paths have a prefix.
fn relative_to_cwd(path: &Path) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    match path.strip_prefix("/") {
        Ok(rest) if cfg!(unix) => {
            // Built from components: a `..` literal here would read to the
            // verification planner as a path this crate's tests read.
            let mut relative = std::path::PathBuf::new();
            for _ in 1..cwd.components().count() {
                relative.push(std::path::Component::ParentDir);
            }
            relative.join(rest).display().to_string()
        }
        _ => path.display().to_string(),
    }
}

/// `rig plugin add` with each source writes the table the settings read
/// back as that source, keeps the file's comments, and `rig plugin list`
/// says where the plugin comes from.
#[test]
fn add_writes_the_table_of_each_source_and_keeps_the_comments() {
    let home = old_home("add");
    let save = home.root().join("save");
    let git = |branch: Option<&str>, rev: Option<&str>| Source::Git {
        url: "file:///srv/save.git".to_owned(),
        branch: branch.map(str::to_owned),
        rev: rev.map(str::to_owned),
    };
    let made = plugin_crate(&save, "reactor-save");
    // `add` writes the directory's resolved path, symlinks followed.
    let resolved = fs::canonicalize(&save).unwrap_or_else(|_| save.clone());
    let relative = relative_to_cwd(&save);
    let cases: [(&[&str], &str, Source); 5] = [
        (
            &["--path", &relative],
            "reactor-save",
            Source::Path(resolved.clone()),
        ),
        (
            &["--git", "file:///srv/save.git"],
            "reactor_save",
            git(None, None),
        ),
        (
            &["--git", "file:///srv/save.git", "--branch", "next"],
            "reactor_save",
            git(Some("next"), None),
        ),
        (
            &[
                "--rev",
                "abc",
                "--git",
                "file:///srv/save.git",
                "--crate",
                "reactor-save",
            ],
            "reactor-save",
            git(None, Some("abc")),
        ),
        (
            &["--version", "0.2"],
            "reactor_save",
            Source::Version("0.2".to_owned()),
        ),
    ];
    let result = (|| -> Result<_> {
        made?;
        let mut found = Vec::new();
        for (flags, _, _) in &cases {
            fs::write(home.settings(), COMMENTED)?;
            run(&home, &[&["add", SAVE], *flags].concat())?;
            let text = fs::read_to_string(home.settings())?;
            found.push((text, config::chosen(&home)?, listing(&home)?));
        }
        Ok(found)
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    for ((text, chosen, list), (flags, krate, source)) in
        result.unwrap_or_default().iter().zip(&cases)
    {
        let expected = Extra {
            plugin: SAVE.to_owned(),
            krate: (*krate).to_owned(),
            source: source.clone(),
        };
        assert!(text.starts_with(COMMENTED), "{flags:?}: {text}");
        assert!(
            text.contains(&format!("# Added by `rig plugin add {SAVE}`.")),
            "{flags:?}: {text}"
        );
        assert_eq!(chosen.extra, [expected], "{flags:?}");
        assert!(chosen.on.iter().any(|on| on == SAVE), "{flags:?}");
        let line = list.lines().find(|line| line.contains(SAVE)).unwrap_or("");
        assert!(
            line.trim_start().starts_with("on ") && line.ends_with(&source.to_string()),
            "{flags:?}: {line}"
        );
    }
}

/// `rig plugin remove` takes out exactly the table `rig plugin add` wrote:
/// a file that had `[plugins]` comes back byte for byte, and one that had
/// none keeps the empty `[plugins]` the add made.
#[test]
fn add_then_remove_leaves_the_file_as_it_was() {
    let home = old_home("round-trip");
    let other = "[plugins] # mine\ndisable = [\"viz::VizPlugin\"] # not now\nenable = []\n\n\
                 # Charts, from crates.io.\n[[plugins.extra]]\ncrate = \"viz\"\n\
                 version = \"1\" # pinned\nplugin = \"viz::VizPlugin\"\n\n# The end.\n";
    let with_plugins = format!("{COMMENTED}{other}");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        let mut after = Vec::new();
        for original in [with_plugins.as_str(), COMMENTED] {
            fs::write(home.settings(), original)?;
            run(&home, &["add", SAVE, "--version", "0.2"])?;
            run(&home, &["remove", SAVE])?;
            after.push(fs::read_to_string(home.settings())?);
        }
        Ok(after)
    })();
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(
        result.ok(),
        Some(vec![
            with_plugins.clone(),
            format!("{COMMENTED}[plugins]\n")
        ])
    );
}

/// `remove` keeps every other table and comment, takes the type out of
/// `[plugins] enable` and `disable` too (a name nothing provides would
/// make the file refused), and leaves a crate `rig plugin new` made on
/// disk.
#[test]
fn remove_takes_out_one_table_and_its_name_and_keeps_the_crate() {
    let home = old_home("remove");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["new", "hello"])?;
        run(&home, &["add", SAVE, "--version", "0.2"])?;
        run(&home, &["disable", "hello::HelloPlugin"])?;
        run(&home, &["remove", "hello::HelloPlugin"])?;
        let text = fs::read_to_string(home.settings())?;
        let kept = home.plugins().join("hello/Cargo.toml").is_file();
        Ok((text, config::chosen(&home)?.extra, kept))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (text, extra, kept) = result.unwrap_or_default();
    assert!(
        text.starts_with(COMMENTED) && !text.contains("hello"),
        "{text}"
    );
    assert!(
        text.contains(&format!("# Added by `rig plugin add {SAVE}`.")),
        "{text}"
    );
    assert_eq!(
        extra
            .iter()
            .map(|extra| extra.plugin.as_str())
            .collect::<Vec<_>>(),
        [SAVE]
    );
    assert!(kept);
}

/// Each refusal says why and leaves the file as it was.
#[test]
fn add_and_remove_refuse_out_loud_without_changing_the_file() {
    let home = old_home("refusals");
    let original = format!("{COMMENTED}[plugins]\n");
    let cases: [(&[&str], &str); 14] = [
        (&["add", SAVE, "--version", "0.2"], "already"),
        (&["add", SAVE], "--path"),
        (&["add", SAVE, "--path", "a", "--version", "1"], "one of"),
        (&["add", SAVE, "--version", "1", "--version", "2"], "once"),
        (&["add", SAVE, "--rev", "abc", "--version", "1"], "--git"),
        (
            &["add", SAVE, "--git", "u", "--branch", "b", "--rev", "r"],
            "or",
        ),
        (
            &["add", "nowhere::NowherePlugin", "--path", "/nowhere/save"],
            "/nowhere/save",
        ),
        (&["add", SAVE, "--version", "1", "--force"], "--force"),
        (
            &["add", "rig_basics::skills::SkillsPlugin", "--version", "1"],
            "rig's own",
        ),
        (&["remove", "viz::VizPlugin"], "viz::VizPlugin"),
        (
            &["add", "x::XPlugin", "--version", "--crate", "x"],
            "--version needs a value",
        ),
        (&["add", "x::XPlugin", "--git", ""], "--git needs a value"),
        (
            &["add", "x::XPlugin", "--version", "--crate"],
            "--version needs a value, not `--crate`",
        ),
        (&["add", "my_save", "--version", "1"], "crate-qualified"),
    ];
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), &original)?;
        run(&home, &["add", SAVE, "--version", "0.2"])?;
        let added = fs::read_to_string(home.settings())?;
        let mut refusals = Vec::new();
        for (args, _) in &cases {
            let refusal = run(&home, args).err().map(|why| why.to_string());
            refusals.push((refusal, fs::read_to_string(home.settings())?));
        }
        Ok((added, refusals))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (added, refusals) = result.unwrap_or_default();
    for ((refusal, text), (args, why)) in refusals.iter().zip(&cases) {
        assert!(
            refusal
                .as_ref()
                .is_some_and(|refusal| refusal.contains(why)),
            "{args:?}: {refusal:?}"
        );
        assert_eq!(text, &added, "{args:?}");
    }
}

/// `remove` takes only the comment block touching the table's header with
/// it. Comments further up, past a blank line, stay: the end of the table
/// before, a commented-out table, or the file's own header comment.
#[test]
fn remove_keeps_the_comments_past_a_blank_line() {
    let home = old_home("remove-comments");
    let kept = "[plugins]\ndisable = [\"rig_basics::welcome::WelcomePlugin\"]\n\
                # welcome is off because it clutters CI logs; keep it off.\n\n\
                # [[plugins.extra]]\n# crate = \"old\"\n# plugin = \"old::OldPlugin\"\n\n";
    let save = format!(
        "# Added by `rig plugin add {SAVE}`.\n[[plugins.extra]]\ncrate = \"reactor-save\"\n\
         plugin = \"{SAVE}\"\nversion = \"0.2\"\n"
    );
    let table = save.split_once('\n').map_or("", |(_, table)| table);
    let first = format!("# My settings.\n\n# The save plugin.\n{table}\n[tui]\n");
    let cases = [
        (format!("{kept}{save}"), kept.to_owned()),
        (first, "# My settings.\n\n[tui]\n".to_owned()),
    ];
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        let mut after = Vec::new();
        for (original, _) in &cases {
            fs::write(home.settings(), original)?;
            run(&home, &["remove", SAVE])?;
            after.push(fs::read_to_string(home.settings())?);
        }
        Ok(after)
    })();
    fs::remove_dir_all(home.root()).ok();
    let expected: Vec<String> = cases.into_iter().map(|(_, after)| after).collect();
    assert_eq!(result.ok(), Some(expected));
}

/// An inline `plugins = { .. }` table cannot hold `[[plugins.extra]]`, so
/// `add` makes it a table, keeping its keys and comment, and the plugin is
/// in the file it wrote.
#[test]
fn add_to_an_inline_plugins_table_keeps_the_plugin() {
    let home = old_home("inline");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(
            home.settings(),
            "# Mine.\nplugins = { disable = [\"rig_basics::welcome::WelcomePlugin\"] } # quiet\n",
        )?;
        run(&home, &["add", SAVE, "--version", "0.2"])?;
        let text = fs::read_to_string(home.settings())?;
        Ok((text, config::chosen(&home)?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (text, chosen) = result.unwrap_or_default();
    assert!(
        text.starts_with("# Mine.\n") && text.contains("# quiet"),
        "{text}"
    );
    assert!(
        (chosen.extra.iter()).any(|extra| extra.plugin == SAVE)
            && !chosen
                .on
                .iter()
                .any(|on| on == "rig_basics::welcome::WelcomePlugin"),
        "{text}"
    );
}

/// A comment inside a multi-line inline `plugins = { .. }` has no place in
/// a `[plugins]` table that keeps it as written, so `add` refuses the file
/// with the fix and changes nothing.
#[test]
fn add_refuses_an_inline_plugins_table_with_comments_inside() {
    let home = old_home("inline-comments");
    let cases = [
        "plugins = {\n  # quiet in CI\n  disable = [\"rig_basics::welcome::WelcomePlugin\"],\n}\n",
        "plugins = {\n  disable = [\"rig_basics::welcome::WelcomePlugin\"],\n  # the end\n}\n",
    ];
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        let mut found = Vec::new();
        for original in cases {
            fs::write(home.settings(), original)?;
            let refusal = run(&home, &["add", SAVE, "--version", "0.2"]).err();
            let text = fs::read_to_string(home.settings())?;
            found.push((refusal.map(|why| why.to_string()), text == original));
        }
        Ok(found)
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    for (refusal, unchanged) in result.unwrap_or_default() {
        assert!(
            refusal
                .as_ref()
                .is_some_and(|why| why.contains("write it as a [plugins] table"))
                && unchanged,
            "{refusal:?}"
        );
    }
}

#[test]
fn a_branch_turns_off_in_one_line_and_an_unknown_one_is_taken_out() {
    let home = old_home("branch");
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        run(&home, &["disable", "rig-basics/sessions"])?;
        let off = fs::read_to_string(home.settings())?;
        run(&home, &["enable", "rig-basics/sessions"])?;
        let on = fs::read_to_string(home.settings())?;
        fs::write(home.settings(), "[plugins]\noff = [\"rig-basics/gone\"]\n")?;
        let refused = config::chosen(&home).err().map(|why| why.to_string());
        run(&home, &["disable", "rig-basics/gone"])?;
        Ok((off, on, refused, config::chosen(&home)?))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let Ok((off, on, refused, fixed)) = result else {
        return;
    };
    assert!(
        off.ends_with("[plugins]\noff = [\"rig-basics/sessions\"]\n") && off.starts_with(COMMENTED),
        "{off}"
    );
    assert!(
        on.starts_with(COMMENTED) && !on.contains("sessions"),
        "{on}"
    );
    assert!(
        refused.is_some_and(|why| why.contains("rig plugin disable rig-basics/gone")),
        "the refusal names its fix"
    );
    assert!(fixed.off.is_empty());
}

/// Removing a crate's last plugin takes the crate's branches out of
/// `[plugins] off` too, which would name nothing; a branch still holding a
/// plugin stays.
#[test]
fn removing_a_crates_last_plugin_takes_its_branches_out_of_off() {
    let home = old_home("remove-off");
    let other = "reactor_save::store::StorePlugin";
    let result = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.settings(), COMMENTED)?;
        for plugin in [SAVE, other] {
            run(
                &home,
                &["add", plugin, "--version", "0.2", "--crate", "reactor-save"],
            )?;
        }
        run(&home, &["disable", "reactor-save"])?;
        run(&home, &["disable", "reactor-save/store"])?;
        run(&home, &["remove", other])?;
        let one = config::chosen(&home)?.off;
        run(&home, &["remove", SAVE])?;
        Ok((one, config::chosen(&home)?.off))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(
        result.ok(),
        Some((vec!["reactor-save".to_owned()], Vec::new()))
    );
}
