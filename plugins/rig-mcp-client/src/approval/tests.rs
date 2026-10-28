use rig_harness::rig_memory::{DirFiles, MemoryStore};
use serde_json::json;

use super::*;

#[test]
fn an_answer_keeps_the_answers_another_rig_gave_since() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("rig-mcp-approvals-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let data = PluginData {
        session: Arc::new(MemoryStore::default()),
        project: Arc::new(DirFiles::new(dir.join("project"))),
        user: Arc::new(DirFiles::new(dir.join("user"))),
    };
    let open = || Approvals::read(Some(&data), "/repo".to_owned(), dir.join("mcp.lock"));
    let (a, b) = (json!({ "command": "a" }), json!({ "command": "b" }));
    let mut first = open();
    assert_eq!(first.set("a", a.clone(), true), Ok(()));
    let (mut one, mut other) = (open(), open());
    assert_eq!(one.set("a", a.clone(), false), Ok(()));
    assert_eq!(other.set("b", b.clone(), true), Ok(()));
    let now = open();
    assert_eq!(
        (now.answer("a", &a), now.answer("b", &b)),
        (Some(false), Some(true))
    );
    // Another project's file, or a changed server, is no answer.
    let elsewhere = Approvals::read(Some(&data), "/other".to_owned(), dir.join("mcp.lock"));
    assert_eq!(elsewhere.answer("b", &b), None);
    assert_eq!(
        now.answer("b", &json!({ "command": "b", "args": ["x"] })),
        None
    );
    std::fs::remove_dir_all(dir).ok();
}
