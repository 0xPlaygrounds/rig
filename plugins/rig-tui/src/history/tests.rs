use bevy_app::prelude::*;

use super::*;
use crate::tests::{press, redacted, type_text, typed, view_app};

#[test]
fn up_and_down_browse_the_history_a_plugin_supplies() {
    let (mut app, _) = view_app();
    app.world_mut().resource_mut::<PromptHistory>().entries = vec!["one".into(), "two".into()];
    type_text(&mut app, "draft");
    press(&mut app, "up");
    assert_eq!(typed(&app), "two");
    press(&mut app, "up up");
    assert_eq!(typed(&app), "one");
    press(&mut app, "down down");
    assert_eq!(typed(&app), "draft");
}

#[test]
fn a_prompt_filtered_out_before_the_save_never_reaches_the_file() {
    let (mut app, _) = view_app();
    let files: Arc<dyn Files> = Arc::new(rig_harness::rig_memory::MemoryStore::default());
    app.world_mut().resource_mut::<PromptHistory>().file = Some(files.clone());
    app.add_systems(
        Update,
        (|mut history: ResMut<PromptHistory>| {
            history.entries.retain(|prompt| !prompt.contains("sk-"));
        })
        .after(HistorySystems::Record)
        .before(HistorySystems::Save),
    );
    for prompt in ["hello", "my key is sk-123", "bye"] {
        type_text(&mut app, prompt);
        press(&mut app, "enter");
    }
    let saved = files.read_file(HISTORY_FILE).unwrap_or_default();
    assert_eq!(saved, b"\"hello\"\n\"bye\"\n");
    assert_eq!(
        app.world().resource::<PromptHistory>().entries,
        ["hello", "bye"]
    );
    // Read back by the next session.
    assert_eq!(PromptHistory::load(files).entries, ["hello", "bye"]);
}

#[test]
fn a_session_rewrite_reaches_the_history_in_memory_and_on_disk() {
    let (mut app, _) = view_app();
    let files: Arc<dyn Files> = Arc::new(rig_harness::rig_memory::MemoryStore::default());
    files
        .write_file(HISTORY_FILE, b"\"sk-9 from another session\"\n")
        .unwrap_or_default();
    *app.world_mut().resource_mut::<PromptHistory>() = PromptHistory::load(files.clone());
    for prompt in ["my key is sk-9", "/redact sk-9"] {
        type_text(&mut app, prompt);
        press(&mut app, "enter");
    }
    app.world_mut().write_message(redacted("sk-9"));
    app.update();
    let expected = [
        "[REDACTED] from another session",
        "my key is [REDACTED]",
        "/redact [REDACTED]",
    ];
    assert_eq!(app.world().resource::<PromptHistory>().entries, expected);
    let saved = files.read_file(HISTORY_FILE).unwrap_or_default();
    assert!(!String::from_utf8_lossy(&saved).contains("sk-9"));
    assert_eq!(PromptHistory::load(files).entries, expected);
}
