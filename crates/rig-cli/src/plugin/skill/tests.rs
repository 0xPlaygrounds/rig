use super::*;

#[test]
fn a_new_home_gets_the_skill_once_and_a_deleted_one_stays_deleted() {
    let home = Home::at(std::env::temp_dir().join(format!("rig-skill-{}", std::process::id())));
    let file = file(&home);
    let project = || fs::create_dir_all(home.project()).map_err(Into::into);
    let result = (|| -> Result<_> {
        // A first build that fails before it writes the project.
        let failed = set_up(&home, || Err("bad settings".into())).is_err() && !file.exists();
        let first = set_up(&home, project)?;
        let written = fs::read_to_string(&file)?;
        fs::remove_file(&file)?;
        let again = set_up(&home, project)?;
        Ok((failed, first, written, again, file.exists()))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (failed, first, written, again, back) = result.unwrap_or_default();
    assert_eq!((failed, first, again, back), (true, None, None, false));
    assert!(
        written.starts_with("---\nname: writing-rig-plugins\n"),
        "{written}"
    );
}
