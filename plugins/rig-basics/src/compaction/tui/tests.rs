use super::*;
use rig_memory::TrackedSet;
use rig_tui::{EntryRender, RenderInput, RenderKey, RenderOutput, RenderRequest};

#[test]
fn the_saved_summary_is_drawn_as_markdown_through_the_owned_render_path() {
    let mut app = App::new();
    app.init_resource::<Theme>().init_resource::<RenderBatch>();
    add(&mut app);
    let mut conversation = rig_core::conversation::Conversation::default();
    conversation.push(rig_core::completion::Message::user("old question"));
    let state = SummaryState {
        summary: "## Goal\n**Explain** the crate.".to_owned(),
        tracked: vec![TrackedSet {
            name: "read-files".to_owned(),
            values: ["a.rs".to_owned()].into(),
        }],
    };
    let change = rig_memory::apply_summary(&mut conversation, None, &state);
    assert!(change.is_some());
    let id = conversation.id(0);
    let agent = app.world_mut().spawn(Conversation(conversation)).id();
    let input = EntryRender {
        id,
        plugin: "compaction".to_owned(),
        title: "compaction".to_owned(),
        text: state.message().unwrap_or_default(),
    };
    assert_eq!(input.view().id, id);
    let key = RenderKey::Entry(0, 0);
    let mut batch = app.world_mut().resource_mut::<RenderBatch>();
    batch.agent = Some(agent);
    batch.width = 80;
    batch.requests.insert(
        key.clone(),
        RenderRequest {
            input: RenderInput::Entry(input),
            output: RenderOutput::default(),
        },
    );
    app.world_mut().run_schedule(TranscriptRender);
    let batch = app.world().resource::<RenderBatch>();
    let lines = batch
        .requests
        .get(&key)
        .map(|request| &request.output.lines);
    assert!(lines.is_some());
    let text = lines
        .into_iter()
        .flatten()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        text.contains("Goal") && text.contains("Read: a.rs"),
        "{text}"
    );
    assert!(
        !text.contains("##") && !text.contains("<summary>") && !text.contains("<read-files>"),
        "{text}"
    );
}

#[test]
fn a_summary_without_a_record_keeps_its_plain_text() {
    let entry = EntryView {
        id: None,
        plugin: "compaction",
        title: "compaction",
        text: "<summary>old</summary>",
    };
    let lines = summary_lines(&entry, None, &Theme::default(), 80);
    assert!(
        lines
            .iter()
            .any(|line| line.to_string().contains(entry.text))
    );
}
