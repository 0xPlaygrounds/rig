use super::*;

#[test]
fn binary_ids_follow_nextest() {
    let cases = [
        ("lib", "rig-core", "rig_core", vec!["lib"], Some("rig-core")),
        (
            "proc macro",
            "rig-derive",
            "rig_derive",
            vec!["proc-macro"],
            Some("rig-derive"),
        ),
        ("test", "rig", "core", vec!["test"], Some("rig::core")),
        (
            "bin",
            "xtask",
            "xtask",
            vec!["bin"],
            Some("xtask::bin/xtask"),
        ),
        (
            "build script",
            "rig",
            "build-script-build",
            vec!["custom-build"],
            None,
        ),
    ];
    for (name, package, target, kinds, want) in cases {
        assert_eq!(
            binary_id(package, target, &kinds).as_deref(),
            want,
            "{name}"
        );
    }
}

#[test]
fn a_test_is_placed_through_inline_file_and_path_modules() {
    let dir = std::env::temp_dir().join(format!("xtask-locate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let write = |relative: &str, source: &str| {
        let path = dir.join(relative);
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("the directory");
        std::fs::write(path, source).expect("the file");
    };
    write(
        "src/lib.rs",
        "mod a;\nmod b { #[cfg(test)] mod tests; }\n#[cfg(target_family = \"wasm\")]\nmod w;\n#[path = \"elsewhere/p.rs\"]\nmod p;\n",
    );
    write("src/a.rs", "#[cfg(test)]\nmod tests;\n");
    write(
        "src/a/tests.rs",
        "#[test]\nfn one() { assert!(ok()); }\nfn helper() {}\n",
    );
    write(
        "src/b/tests.rs",
        "#[tokio::test]\nasync fn two() { assert!(ok()); }\n",
    );
    write("src/w.rs", "#[test]\nfn three() {}\n");
    write("src/elsewhere/p.rs", "mod q;\n");
    write(
        "src/elsewhere/q.rs",
        "#[test]\nfn four() { assert!(ok()); }\n",
    );
    let mut locator = Locator::new(&dir);
    let start = Path::new("src/lib.rs");
    let place = |locator: &mut Locator<'_>, test: &str| {
        locator
            .locate("x", start, test)
            .map(|found| (found.file, found.module, found.facts.contract.is_some()))
    };
    assert_eq!(
        place(&mut locator, "a::tests::one"),
        Some(("src/a/tests.rs".into(), "a::tests".into(), false))
    );
    assert_eq!(
        place(&mut locator, "b::tests::two"),
        Some(("src/b/tests.rs".into(), "b::tests".into(), false))
    );
    assert_eq!(
        place(&mut locator, "w::three"),
        Some(("src/w.rs".into(), "w".into(), true)),
        "a wasm-only module makes a wasm contract"
    );
    assert_eq!(
        place(&mut locator, "p::q::four"),
        Some(("src/elsewhere/q.rs".into(), "p::q".into(), false))
    );
    assert_eq!(place(&mut locator, "a::tests::helper"), None, "not a test");
    assert_eq!(place(&mut locator, "a::tests::missing"), None);
    let _ = std::fs::remove_dir_all(&dir);
}
