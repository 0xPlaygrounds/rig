use super::*;
use crate::completion::Message;

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str) -> Message {
    Message::assistant(text)
}

#[tokio::test]
async fn round_trip() {
    let mem = InMemoryConversationMemory::new();
    assert!(mem.load(&"c1".into()).await.unwrap().is_empty());

    mem.append(&"c1".into(), vec![user("hello"), assistant("hi")])
        .await
        .unwrap();

    let loaded = mem.load(&"c1".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
}
