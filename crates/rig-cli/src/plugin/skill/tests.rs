use super::*;

#[test]
fn a_new_home_gets_the_skill_once_and_a_deleted_one_stays_deleted() {
    let home = Home::at(std::env::temp_dir().join(format!("rig-skill-{}", std::process::id())));
    let file = file(&home);
    let result = (|| -> std::io::Result<_> {
        let new = set_up(&home);
        let written = fs::read_to_string(&file)?;
        // The home's first build.
        fs::create_dir_all(home.project())?;
        fs::remove_file(&file)?;
        let again = set_up(&home);
        Ok((new, written, again, file.exists()))
    })();
    fs::remove_dir_all(home.root()).ok();
    assert!(result.is_ok(), "{result:?}");
    let (new, written, again, back) = result.unwrap_or_default();
    assert_eq!((new, again, back), (None, None, false));
    assert!(
        written.starts_with("---\nname: writing-rig-plugins\n"),
        "{written}"
    );
}
