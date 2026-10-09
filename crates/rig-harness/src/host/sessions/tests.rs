use super::{delivered, first_typed_in_lines};

fn lines(text: &str) -> impl Iterator<Item = String> + '_ {
    text.lines().map(str::to_owned)
}

#[test]
fn the_title_skips_text_an_older_version_logged_without_its_origin() {
    let log = r#"{"seq":0,"t":1,"type":"header","v":1,"agent":"a","cwd":"/x","created":1}
{"seq":1,"t":1,"type":"settings","model":null,"effort":null}
{"seq":2,"t":1,"type":"message","message":{"role":"user","content":[{"type":"text","text":"[Output of plugin build, not the user's words]\nBuilding the agent failed"}]}}
{"seq":3,"t":2,"type":"message","message":{"role":"user","content":[{"type":"text","text":"A note"}]},"origin":{"kind":"plugin","name":"x"}}
{"seq":4,"t":3,"type":"message","message":{"role":"user","content":[{"type":"text","text":"Refactor the stats module"}]}}"#;
    assert_eq!(
        first_typed_in_lines(lines(log)).as_deref(),
        Some("Refactor the stats module")
    );
}

#[test]
fn a_spawned_agent_log_gives_no_title() {
    let log = r#"{"seq":0,"t":1,"type":"header","v":1,"agent":"b","parent":"a","created":1}
{"seq":1,"t":1,"type":"message","message":{"role":"user","content":[{"type":"text","text":"Do the task"}]}}"#;
    assert_eq!(first_typed_in_lines(lines(log)), None);
}

#[test]
fn a_stale_title_is_told_apart() {
    assert!(delivered(
        "[Output of plugin build, not the user's words] Building the …"
    ));
    assert!(!delivered("Refactor this project"));
}
