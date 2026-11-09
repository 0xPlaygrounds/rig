use super::*;

/// The path or URL in backticks on the skill's line about `what`.
fn named(text: &str, what: &str) -> String {
    let line = text
        .lines()
        .find(|line| line.contains(what))
        .unwrap_or_default();
    line.split('`').nth(1).unwrap_or_default().to_owned()
}

#[test]
fn the_skill_names_this_checkouts_guide_and_corpus() {
    let checkout = std::path::absolute(super::super::super::project::CHECKOUT).unwrap_or_default();
    let text = text(&RigSource::Local(checkout), None);
    let guide = PathBuf::from(named(&text, "plugin guide"));
    assert!(
        guide.ends_with("crates/rig-harness/PLUGINS.md") && guide.is_file(),
        "{text}"
    );
    assert!(
        PathBuf::from(named(&text, "working plugins"))
            .join("base")
            .is_dir(),
        "{text}"
    );
    assert!(text.starts_with("---\nname: writing-rig-plugins\n") && !text.contains("__"));
}

#[test]
fn a_crates_io_build_names_the_guide_cargo_fetched() {
    let cargo = std::env::temp_dir().join(format!("rig-skill-cargo-{}", std::process::id()));
    let fetched = cargo.join(format!(
        "registry/src/index.example-1/rig-harness-{VERSION}/PLUGINS.md"
    ));
    let before = text(&RigSource::Registry, Some(&cargo));
    let after = (|| -> Result<_> {
        fs::create_dir_all(fetched.parent().ok_or("no parent")?)?;
        fs::write(&fetched, "# Plugins\n")?;
        Ok(text(&RigSource::Registry, Some(&cargo)))
    })();
    let fetched = fs::canonicalize(&fetched).unwrap_or_default();
    fs::remove_dir_all(&cargo).ok();
    assert_eq!(
        named(&before, "plugin guide"),
        format!("https://docs.rs/rig-harness/{VERSION}/rig_harness/plugin_guide/")
    );
    let after = after.unwrap_or_default();
    assert_eq!(
        PathBuf::from(named(&after, "plugin guide")),
        fetched,
        "{after}"
    );
}
