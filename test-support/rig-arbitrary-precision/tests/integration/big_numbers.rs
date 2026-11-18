//! Integers beyond u64 and i64, kept exactly in a `serde_json::Value`: the
//! reason a program turns `arbitrary_precision` on.
use rig_cassette::effect_log::jsonl;
use rig_core::conversation::Conversation;
use rig_core::conversation::{Aside, Author};
use rig_core::effect::EffectRecord;
use rig_core::message::{AssistantContent, AssistantMessage, Message};
use rig_core::streaming::{Item, Part, StreamEvent, StreamRecord};
use rig_memory::{MemoryStore, Session, Strings, load, rewrite, save};

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

/// A tool call whose arguments hold [`BIG`] and `secret`.
fn big_call(secret: &str) -> anyhow::Result<AssistantContent> {
    let arguments: serde_json::Value =
        serde_json::from_str(&format!(r#"{{"n":{BIG},"secret":"{secret}"}}"#))?;
    Ok(AssistantContent::tool_call(
        "call-1",
        "count".try_into()?,
        arguments,
    ))
}

#[test]
fn a_stream_record_keeps_a_big_integer() -> anyhow::Result<()> {
    let items: Vec<_> = StreamEvent::whole(Part::new(0), &big_call("s")?)
        .into_iter()
        .map(Item::Event)
        .collect();
    let record = StreamRecord::parse(serde_json::to_value(&items)?)
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
    let written = serde_json::to_string(record.items())?;
    anyhow::ensure!(written.contains(BIG), "{written}");
    Ok(())
}

#[test]
fn a_rewrite_reaches_a_tool_call_holding_a_big_integer() -> anyhow::Result<()> {
    let store = MemoryStore::default();
    let mut conversation = Conversation::new();
    conversation.push(Message::user("Count."));
    conversation.push(Message::Assistant(AssistantMessage::new(vec![big_call(
        "sk-1",
    )?])));
    save(&store, "main", &mut conversation)?;
    let redact = Strings(|text: &mut String| {
        let hit = text.contains("sk-1");
        if hit {
            *text = text.replace("sk-1", "[REDACTED]");
        }
        hit
    });
    rewrite(&store, Author::plugin("redact"), &redact)?;
    let main = Session::read(&store)?
        .agents
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no agent"))?;
    let written = serde_json::to_string(main.conversation.messages())?;
    anyhow::ensure!(
        !written.contains("sk-1") && written.contains(BIG),
        "{written}"
    );
    Ok(())
}

/// A mark shaped like the ones plugins keep in asides: tagged, holding a
/// value.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Mark {
    Settled { value: serde_json::Value },
}

#[test]
fn an_aside_reads_a_tagged_value_holding_a_big_integer() -> anyhow::Result<()> {
    let value = serde_json::from_str(&format!(r#"{{"state":"settled","value":{{"n":{BIG}}}}}"#))?;
    let aside = Aside {
        kind: "mark".into(),
        after: None,
        author: Author::user(),
        value,
        saved: true,
    };
    let Some(Mark::Settled { value }) = aside.read::<Mark>("mark") else {
        anyhow::bail!("the mark does not read");
    };
    anyhow::ensure!(value.to_string().contains(BIG), "{value}");
    Ok(())
}

#[test]
fn tool_arguments_decode_into_a_tagged_type_holding_a_big_integer() -> anyhow::Result<()> {
    let args = serde_json::from_str(&format!(r#"{{"state":"settled","value":{{"n":{BIG}}}}}"#))?;
    let Mark::Settled { value } = rig_core::tool::decode_args::<Mark>(&args)?;
    anyhow::ensure!(value.to_string().contains(BIG), "{value}");
    Ok(())
}
