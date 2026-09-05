use super::*;

#[test]
fn only_declared_patch_paths_can_change_and_stale_input_is_refused() {
    let mut project = Project::new().expect("project");
    let image = project.image().expect("image");
    let before = digest(&image).expect("digest");
    for path in [
        "../outside",
        "/tmp/outside",
        "src/../Cargo.toml",
        "Cargo.toml",
        "tests/pagination.rs",
    ] {
        assert!(project.apply(path, "changed\n", &before).is_err(), "{path}");
    }
    assert_eq!(project.image().expect("image"), image);
    assert_eq!(project.writes(), 0);
    let after = project
        .apply(
            "tests/regression.rs",
            "#[test]\nfn regression() { assert!(false); }\n",
            &before,
        )
        .expect("permitted test patch");
    assert_ne!(after, before);
    assert_eq!(project.writes(), 1);
    assert!(project.apply("src/lib.rs", "changed\n", &before).is_err());
    assert_eq!(
        project.read("src/lib.rs").expect("source"),
        image["src/lib.rs"]
    );
}

#[test]
fn restored_images_keep_counts_and_refuse_configuration_changes() {
    let mut project = Project::new().expect("project");
    let before = digest(&project.image().expect("image")).expect("digest");
    project
        .apply(
            "tests/regression.rs",
            "#[test]\nfn regression() {}\n",
            &before,
        )
        .expect("patch");
    let image: Image = serde_json::from_str(
        &serde_json::to_string(&project.image().expect("image")).expect("JSON"),
    )
    .expect("image roundtrip");
    let restored = Project::restore(&image, project.writes()).expect("restore");
    assert_eq!(restored.image().expect("image"), image);
    assert_eq!(restored.writes(), 1);
    let mut invalid = image;
    invalid.insert("Cargo.toml".into(), "[package]\nname=\"replaced\"\n".into());
    assert!(Project::restore(&invalid, 1).is_err());
    invalid = initial();
    invalid.insert("../outside".into(), "invalid".into());
    assert!(Project::restore(&invalid, 0).is_err());
}

#[cfg(unix)]
#[test]
fn symlinks_and_undeclared_build_scripts_are_refused_before_validation() {
    let mut project = Project::new().expect("project");
    let outside = assert_fs::TempDir::new().expect("outside");
    let canary = outside.path().join("canary.rs");
    std::fs::write(&canary, "do not change\n").expect("canary");
    let before = digest(&project.image().expect("image")).expect("digest");
    std::fs::remove_file(project.root().join("src/lib.rs")).expect("remove source");
    std::os::unix::fs::symlink(&canary, project.root().join("src/lib.rs")).expect("symlink");
    assert!(project.read("src/lib.rs").is_err());
    assert!(project.image().is_err());
    assert!(project.apply("src/lib.rs", "changed\n", &before).is_err());
    assert_eq!(
        std::fs::read_to_string(&canary).expect("canary"),
        "do not change\n"
    );
    let project = Project::new().expect("project");
    std::fs::write(project.root().join("build.rs"), "fn main() {}\n")
        .expect("unexpected build script");
    assert!(project.image().is_err());
}
