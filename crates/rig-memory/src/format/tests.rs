use super::LogLines;

/// The complete lines of `bytes` and the torn tail after them.
fn read(bytes: &[u8]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let log = LogLines::parse(bytes.to_vec());
    let lines = log.iter().map(|(_, _, line)| line.to_vec()).collect();
    (lines, log.tail().to_vec())
}

/// A torn tail a crash left is no line: bytes after the last newline, or a
/// last line that is not JSON. A blank last line is no tear.
#[test]
fn a_torn_tail_is_no_line() {
    let first = br#"{"a":1}"#.to_vec();
    assert_eq!(
        read(b"{\"a\":1}\n{\"b\":"),
        (vec![first.clone()], b"{\"b\":".to_vec())
    );
    assert_eq!(
        read(b"{\"a\":1}\n{\"b\":\n"),
        (vec![first.clone()], b"{\"b\":\n".to_vec())
    );
    assert_eq!(read(b"{\"a\":1}\n\n"), (vec![first], Vec::new()));
}

/// A key in a message is redacted in the line written, which still reads
/// back: the log never holds it.
#[test]
fn a_key_in_a_message_never_reaches_the_log() {
    use rig_core::completion::Message;
    use rig_core::conversation::Conversation;

    use crate::{MemoryStore, Store, load, save};

    let key = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789";
    let store = MemoryStore::default();
    let said = format!("my key is\n{key}, and \"OPENAI_API_KEY={key}\"");
    let mut conversation = Conversation::from_messages(vec![Message::user(said)]);
    assert!(save(&store, "agent", &mut conversation).is_ok());
    let bytes = store.read_log("agent").unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains("abcdefghijklmnop"), "{text}");
    assert!(text.contains(rig_core::redact::REDACTED), "{text}");
    let loaded = load(&store, "agent");
    assert!(loaded.is_ok_and(|loaded| loaded.messages().len() == 1));
}
