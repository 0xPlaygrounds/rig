//! Integers beyond u64 and i64, kept exactly in a `serde_json::Value`: the
//! reason a program turns `arbitrary_precision` on.
use rig_cassette::effect_log::jsonl;
use rig_core::conversation::Conversation;
use rig_core::effect::EffectRecord;
use rig_core::message::{AssistantContent, AssistantMessage, Message};
use rig_memory::{MemoryStore, load, save};

/// `u64::MAX + 1`.
const BIG: &str = "18446744073709551616";

/// A custom effect whose payload and answer hold [`BIG`].
fn custom_record() -> String {
    format!(
        r#"{{"tool_output":null,"id":1,"key":"host/note","kind":{{"effect":"custom","kind":"example","payload":{{"n":{BIG}}}}},"outcome":{{"Ok":{{"outcome":"custom","payload":{{"n":{BIG}}}}}}}}}"#
    )
}

#[test]
fn an_effect_record_keeps_a_big_integer() -> anyhow::Result<()> {
    let record: EffectRecord = serde_json::from_str(&custom_record())?;
    let written = serde_json::to_string(&record)?;
    anyhow::ensure!(written.matches(BIG).count() == 2, "{written}");
    Ok(())
}

#[test]
fn a_json_lines_effect_log_keeps_a_big_integer() -> anyhow::Result<()> {
    let log = jsonl::read_from(custom_record().as_bytes())?;
    let written = serde_json::to_string(&log.records)?;
    anyhow::ensure!(written.matches(BIG).count() == 2, "{written}");
    Ok(())
}

#[test]
fn a_saved_tool_call_keeps_a_big_integer() -> anyhow::Result<()> {
    let arguments: serde_json::Value = serde_json::from_str(&format!(r#"{{"n":{BIG}}}"#))?;
    let call = AssistantContent::tool_call("call-1", "count".try_into()?, arguments);
    let mut conversation = Conversation::new();
    conversation.push(Message::user("Count."));
    conversation.push(Message::Assistant(AssistantMessage::new(vec![call])));
    let store = MemoryStore::default();
    save(&store, "session", &mut conversation)?;
    let loaded = load(&store, "session")?;
    anyhow::ensure!(loaded == conversation, "{loaded:?}");
    let written = serde_json::to_string(loaded.messages())?;
    anyhow::ensure!(written.contains(BIG), "{written}");
    Ok(())
}
