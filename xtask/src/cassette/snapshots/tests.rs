use super::*;

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn arguments_select_the_mode_and_the_targets() {
    assert_eq!(parse(&[]), Ok(Options::default()));
    assert_eq!(
        parse(&args(&["--check", "--test", "openai", "--test", "xai"])),
        Ok(Options {
            check: true,
            targets: vec!["openai".into(), "xai".into()],
        })
    );
    assert!(parse(&args(&["--test"])).is_err());
    assert!(parse(&args(&["--write"])).is_err());
}

#[test]
fn only_request_snapshots_are_listed() {
    let dir = std::env::temp_dir().join(format!("xtask-snapshots-{}", std::process::id()));
    let nested = dir.join("openai/matrix");
    std::fs::create_dir_all(&nested).unwrap();
    for name in [
        "a.yaml",
        "a.requests.json",
        "a.clock.json",
        "b.requests.json",
    ] {
        std::fs::write(nested.join(name), "{}").unwrap();
    }
    let found = snapshot_files(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        found,
        [
            nested.join("a.requests.json"),
            nested.join("b.requests.json")
        ]
    );
    assert_eq!(snapshot_files(&dir), Ok(Vec::new()));
}
