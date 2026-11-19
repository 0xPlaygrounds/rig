use super::*;
use rig_tui::{EntryRender, RenderInput, RenderKey, RenderOutput, RenderRequest};

fn app() -> App {
    let mut app = App::new();
    app.init_resource::<Theme>().init_resource::<RenderBatch>();
    add(&mut app);
    app
}

fn state(summary: &str, read: &str) -> SummaryState {
    SummaryState {
        summary: summary.to_owned(),
        read_files: vec![read.to_owned()],
        modified_files: Vec::new(),
    }
}

/// What `agent`'s summary entry `id`, saying `text`, draws as.
fn drawn(app: &mut App, agent: Entity, id: Option<MessageId>, text: String) -> String {
    let input = EntryRender {
        id,
        plugin: "compaction".to_owned(),
        title: "compaction".to_owned(),
        text,
    };
    assert_eq!(input.id, id);
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
    lines
        .into_iter()
        .flatten()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_saved_summary_is_drawn_as_markdown_through_the_owned_render_path() {
    let mut app = app();
    let mut conversation = rig_core::conversation::Conversation::default();
    conversation.push(rig_core::completion::Message::user("old question"));
    let state = state("## Goal\n**Explain** the crate.", "a.rs");
    let read = conversation.messages().to_vec();
    let change = rig_memory::apply_summary(&mut conversation, &read, &state);
    assert!(change.is_some());
    let id = conversation.id(0);
    let agent = app.world_mut().spawn(Conversation(conversation)).id();
    let text = drawn(&mut app, agent, id, state.message().unwrap_or_default());
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
fn a_summary_a_later_compaction_took_out_is_still_drawn_from_its_state() {
    let mut app = app();
    let mut conversation = rig_core::conversation::Conversation::default();
    conversation.push(rig_core::completion::Message::user("old question"));
    let first = state("## Goal\nThe **first** goal.", "a.rs");
    let read = conversation.messages().to_vec();
    assert!(rig_memory::apply_summary(&mut conversation, &read, &first).is_some());
    let summary = conversation.id(0);
    conversation.push(rig_core::completion::Message::user("next question"));
    let before = Before::of(&conversation, 2);
    let second = state("## Goal\nThe second goal.", "b.rs");
    let read = conversation.messages().to_vec();
    let spliced = rig_memory::apply_summary(&mut conversation, &read, &second);
    let agent = app.world_mut().spawn(Conversation(conversation)).id();
    assert!(spliced.is_some());
    let Some(spliced) = spliced else {
        return;
    };
    before.keep(&mut app.world_mut().commands(), agent, &spliced);
    app.world_mut().flush();
    let replaced = app.world().get::<Replaced>(agent);
    let second_id = spliced.changed.first();
    let replaced = replaced.and_then(|replaced| replaced.0.get(second_id?));
    assert!(replaced.is_some_and(|entries| entries.iter().any(|entry| entry.id == summary)));
    let text = drawn(
        &mut app,
        agent,
        summary,
        first.message().unwrap_or_default(),
    );
    assert!(
        text.contains("The first goal.") && text.contains("Read: a.rs"),
        "{text}"
    );
    assert!(!text.contains("<summary>"), "{text}");
}

/// A summary is drawn from its entry alone: once a secret is taken out of the entry, nothing
/// else the session holds, such as an earlier version's aside with the state, brings it back.
#[test]
fn a_summary_is_drawn_from_its_entry_and_nothing_else() {
    use rig_core::conversation::{Aside, Author};
    let mut app = app();
    let mut conversation = rig_core::conversation::Conversation::default();
    conversation.push(rig_core::completion::Message::user("the key is hunter2"));
    let secret = state("The key is hunter2.", "hunter2.txt");
    let read = conversation.messages().to_vec();
    assert!(rig_memory::apply_summary(&mut conversation, &read, &secret).is_some());
    let id = conversation.id(0);
    let value = serde_json::json!({
        "summary": id,
        "replaced": [],
        "state": {"summary": secret.summary, "read_files": secret.read_files},
    });
    let author = rig_memory::summary_author();
    conversation.add_aside(Aside::new("rig.compaction", author, value).after(id));
    conversation.edit_each(Author::plugin("redact"), |entry| {
        let text = entry.message.rag_text()?;
        Some(rig_core::completion::Message::user(
            text.replace("hunter2", "[gone]"),
        ))
    });
    let text = conversation.messages().first().and_then(|it| it.rag_text());
    let agent = app.world_mut().spawn(Conversation(conversation)).id();
    let text = drawn(&mut app, agent, id, text.unwrap_or_default());
    assert!(
        text.contains("The key is [gone].") && text.contains("Read: [gone].txt"),
        "{text}"
    );
    assert!(!text.contains("hunter2"), "{text}");
}

#[test]
fn a_summary_without_a_record_keeps_its_plain_text() {
    let entry = EntryRender {
        id: None,
        plugin: "compaction".to_owned(),
        title: "compaction".to_owned(),
        text: "<summary>old</summary>".to_owned(),
    };
    let lines = summary_lines(&entry, None, &Theme::default(), 80);
    assert!(
        lines
            .iter()
            .any(|line| line.to_string().contains(&entry.text))
    );
}
