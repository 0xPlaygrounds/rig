use super::{Home, Invocation};

#[test]
fn plugin_options_pass_through_and_read_back() {
    let args = [
        "--json",
        "-m",
        "deepseek/deepseek-flash",
        "--serve=127.0.0.1:8080",
        "-p",
        "hello",
        "world",
    ];
    let invocation = Invocation::parse(&args);
    let expected = Invocation {
        print: Some("hello world".to_owned()),
        model: Some("deepseek/deepseek-flash".to_owned()),
        flags: vec![
            ("json".to_owned(), None),
            ("serve".to_owned(), Some("127.0.0.1:8080".to_owned())),
        ],
    };
    assert_eq!(invocation.as_ref(), Ok(&expected));
    let again: Vec<String> = expected
        .to_args()
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(Invocation::parse(&again), Ok(expected));
    assert_eq!(
        Invocation::parse(&["--model=openai/gpt-5"]).map(|args| args.model),
        Ok(Some("openai/gpt-5".to_owned()))
    );
}

#[test]
fn malformed_options_and_stray_words_are_refused() {
    for args in [
        &["-x"][..],
        &["--Json"],
        &["--=1"],
        &["--9lives"],
        &["word"],
    ] {
        assert!(Invocation::parse(args).is_err(), "{args:?}");
    }
}

#[test]
fn a_home_at_a_root_lays_out_its_files_under_it() {
    let home = Home::at("some/root");
    assert!(home.root().is_absolute());
    assert!(home.root().ends_with("some/root"));
    assert_eq!(home.settings(), home.root().join("settings.toml"));
    assert_eq!(home.flags(), home.root().join("bin/flags"));
}

#[test]
fn a_session_directory_knows_its_id() {
    let id = "1700000000-42".parse::<super::SessionId>().ok();
    assert!(id.is_some());
    let read = id.as_ref().and_then(|id| Home::at("root").session(id).id());
    assert_eq!(read, id);
}

#[test]
fn new_sessions_of_one_second_take_the_next_free_id() {
    let root = std::env::temp_dir().join(format!("rig-new-session-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let made: Vec<_> = (0..2).filter_map(|_| home.new_session().ok()).collect();
    let created = made
        .iter()
        .all(|(id, dir)| dir.id().as_ref() == Some(id) && dir.path().is_dir());
    std::fs::remove_dir_all(&root).ok();
    assert!(created && made.len() == 2, "{made:?}");
    let ids: Vec<_> = made.into_iter().map(|(id, _)| id).collect();
    assert_ne!(ids.first(), ids.last());
}
