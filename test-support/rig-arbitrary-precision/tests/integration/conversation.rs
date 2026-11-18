use rig_core::conversation::{Aside, Author, ChangeRecord, Conversation, Entry, Halt};
use rig_core::message::{
    AssistantContent, AssistantMessage, Citation, Message, Source, SourceLocation, Text,
};
use rig_memory::{MemoryStore, load, save};

/// An answer citing a source with the provider's confidence.
fn cited(text: &str) -> Message {
    let source = Source::new(SourceLocation::Url {
        url: "https://example.com".into(),
    })
    .confidence(0.875);
    let text = Text::new(text).with_citations([Citation::new([source])]);
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::Text(text)]))
}

/// A conversation whose edits make every change a session saves but a reset.
fn edited() -> Conversation {
    let mut conversation = Conversation::new();
    conversation.push(Message::user("Where is it?"));
    conversation.push(cited("Here."));
    conversation.splice(1..2, [Entry::new(cited("There."))], Author::user());
    conversation.push(Message::user("And then?"));
    conversation.halt(Halt::Kept);
    conversation.add_aside(Aside {
        kind: "note".into(),
        after: None,
        author: Author::user(),
        value: serde_json::json!({"score": 0.25}),
        saved: true,
    });
    conversation
}

#[test]
fn a_saved_conversation_loads_as_it_was() -> anyhow::Result<()> {
    let store = MemoryStore::default();
    let mut conversation = edited();
    save(&store, "session", &mut conversation)?;
    let loaded = load(&store, "session")?;
    anyhow::ensure!(loaded == conversation, "{loaded:?}\n!=\n{conversation:?}");
    Ok(())
}

#[test]
fn every_kind_of_change_record_reads_back() -> anyhow::Result<()> {
    let mut conversation = edited();
    let mut records = conversation.take_unsaved();
    let mut reset: ChangeRecord = serde_json::from_value(serde_json::json!({
        "type": "reset",
        "state": serde_json::to_value(&conversation)?,
    }))?;
    reset.at = Some(1);
    records.push(reset);
    let kinds: Vec<_> = records
        .iter()
        .map(|record| serde_json::to_value(record).map(|value| value["type"].clone()))
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        kinds == ["push", "push", "splice", "push", "halt", "aside", "reset"],
        "{kinds:?}"
    );
    for record in records {
        let text = serde_json::to_string(&record)?;
        let read: ChangeRecord = serde_json::from_str(&text)?;
        anyhow::ensure!(read == record, "{text}");
    }
    Ok(())
}
