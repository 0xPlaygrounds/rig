use bevy_app::prelude::*;

use super::*;
use crate::tests::{edit, press, type_text, typed, view_app};

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
        // No model answers it, so it comes back to the input.
        edit(&mut app, |input| drop(input.take()));
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
