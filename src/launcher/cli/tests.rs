use super::*;

/// A build is installed once at `bin/builds/<id>/rig-agent`, its id read
/// from its stamp, and a changed binary gets another id and place.
#[test]
fn a_build_is_installed_once_under_its_id() {
    let root = std::env::temp_dir().join(format!("rig-install-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let found = (|| -> Result<_> {
        fs::create_dir_all(home.bin())?;
        let good = home.good();
        fs::write(&good, b"one")?;
        let (id, installed) = install(&home, &good)?;
        let again = install(&home, &good)?;
        let parent = installed
            .parent()
            .and_then(Path::file_name)
            .map(|name| name.to_owned());
        fs::write(&good, b"a longer one")?;
        let (other, _) = install(&home, &good)?;
        Ok((
            again == (id.clone(), installed.clone()),
            fs::read(&installed)?,
            parent.is_some_and(|name| name == id.as_str()),
            other != id,
        ))
    })();
    fs::remove_dir_all(&root).ok();
    assert_eq!(found.ok(), Some((true, b"one".to_vec(), true, true)));
}

#[test]
fn the_verbs_and_print_run_in_the_cli_role() {
    let words = |words: &[&str]| {
        words
            .iter()
            .map(|word| (*word).to_owned())
            .collect::<Vec<_>>()
    };
    for args in [
        &["start", "x"][..],
        &["wait", "s-1"],
        &["-p", "hi"],
        &["-r", "s", "--print"],
    ] {
        assert!(is_cli(&words(args)), "{args:?}");
    }
    // -p counts only among the options before the first other word: a
    // plugin command's own -p, or one after `--`, is not rig -p.
    for args in [&["-m", "a/b", "-p", "hi"][..], &["--yes", "--print"]] {
        assert!(is_cli(&words(args)), "{args:?}");
    }
    for args in [
        &["export", "-p"][..],
        &["--", "-p"],
        &["-r", "s", "stats", "--print"],
    ] {
        assert!(!is_cli(&words(args)), "{args:?}");
    }
    for args in [
        &["build"][..],
        &["plugin", "list"],
        &["-r", "s"],
        &["help-agents"],
        &["help"],
    ] {
        assert!(!is_cli(&words(args)), "{args:?}");
    }
}

/// Builds are immutable: verbs started at once on a fresh home all install
/// the same build, and none replaces the copy another already runs from,
/// whose process would then find its own executable deleted.
#[cfg(unix)]
#[test]
fn parallel_installs_never_replace_a_build() {
    use std::os::unix::fs::MetadataExt;
    let root = std::env::temp_dir().join(format!("rig-install-race-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    fs::create_dir_all(home.bin()).ok();
    // Large enough that copying it takes a while.
    fs::write(home.good(), vec![7_u8; 32 << 20]).ok();
    let barrier = std::sync::Barrier::new(8);
    let inodes: Vec<Option<u64>> = std::thread::scope(|scope| {
        let installs: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    let (_, path) = install(&home, &home.good()).ok()?;
                    // The file this verb would exec, as it found it.
                    fs::metadata(path).ok().map(|metadata| metadata.ino())
                })
            })
            .collect();
        installs
            .into_iter()
            .map(|install| install.join().ok().flatten())
            .collect()
    });
    let last = build_id(&home.good())
        .ok()
        .and_then(|id| fs::metadata(home.build(&id)).ok())
        .map(|metadata| metadata.ino());
    fs::remove_dir_all(&root).ok();
    assert!(last.is_some());
    assert!(
        inodes.iter().all(|inode| *inode == last),
        "{inodes:?} vs {last:?}"
    );
}

/// F-015: `rig -r ID` opens the terminal view; without a terminal it is
/// refused with the commands that work headless.
#[test]
fn resume_without_a_terminal_points_at_the_verbs() {
    let refused = resume_refusal("s-1", false).unwrap_or_default();
    assert!(refused.contains("next: rig -p -r s-1"), "{refused}");
    assert!(refused.contains("rig send s-1"), "{refused}");
    assert_eq!(resume_refusal("s-1", true), None);
}

/// F-016: the verbs see that plugins.toml changed since the default build
/// was made, and so rebuild instead of running the old one.
#[test]
fn an_edited_plugin_list_makes_the_build_stale() {
    let root = std::env::temp_dir().join(format!("rig-stale-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let states = (|| -> Result<_> {
        fs::create_dir_all(home.project())?;
        fs::write(home.plugins_toml(), "# one\n")?;
        let unbuilt = stale(&home);
        build::record_built(&home)?;
        let built = stale(&home);
        fs::write(home.plugins_toml(), "# two\n")?;
        Ok((unbuilt, built, stale(&home)))
    })();
    fs::remove_dir_all(&root).ok();
    assert_eq!(states.ok(), Some((true, false, true)));
}
