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
