use super::*;
use crate::completion::Message;

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str) -> Message {
    Message::assistant(text)
}

/// The `*_owned` companions return `Send + 'static` futures (spawnable on
/// any executor) and behave exactly like the borrowed methods, including
/// through an `Arc<dyn ConversationMemory>`.
#[tokio::test]
async fn owned_futures_are_static_and_match_borrowed_behavior() {
    fn assert_send_static<T: Send + 'static>(value: T) -> T {
        value
    }

    let mem: Arc<dyn ConversationMemory> = Arc::new(InMemoryConversationMemory::new());
    let id = ConversationId::from("c-owned");
    assert_send_static(mem.clone().append_owned(id.clone(), vec![user("hello")]))
        .await
        .unwrap();
    let loaded = assert_send_static(mem.clone().load_owned(id.clone()))
        .await
        .unwrap();
    assert_eq!(loaded.len(), 1);
    assert_send_static(mem.clone().clear_owned(id.clone()))
        .await
        .unwrap();
    assert!(mem.load(&id).await.unwrap().is_empty());
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

#[tokio::test]
async fn isolation_between_conversations() {
    let mem = InMemoryConversationMemory::new();
    mem.append(&"a".into(), vec![user("hi a")]).await.unwrap();
    mem.append(&"b".into(), vec![user("hi b")]).await.unwrap();

    assert_eq!(mem.load(&"a".into()).await.unwrap().len(), 1);
    assert_eq!(mem.load(&"b".into()).await.unwrap().len(), 1);
}

#[tokio::test]
async fn clear_removes_history() {
    let mem = InMemoryConversationMemory::new();
    mem.append(&"c".into(), vec![user("x")]).await.unwrap();
    mem.clear(&"c".into()).await.unwrap();
    assert!(mem.load(&"c".into()).await.unwrap().is_empty());
}

#[tokio::test]
async fn with_filter_transforms_loaded_messages() {
    let mem = InMemoryConversationMemory::new()
        .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).collect());

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();

    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2, "filter should retain only 2 messages");
}

#[tokio::test]
async fn arc_conversation_memory_forwards_to_inner() {
    let inner = Arc::new(InMemoryConversationMemory::new());
    let mem: Arc<dyn ConversationMemory> = inner.clone();

    mem.append(&"c".into(), vec![user("hello")]).await.unwrap();

    assert_eq!(inner.load(&"c".into()).await.unwrap().len(), 1);
    mem.clear(&"c".into()).await.unwrap();
    assert!(inner.load(&"c".into()).await.unwrap().is_empty());
}

#[tokio::test]
async fn boxed_conversation_memory_forwards_to_inner() {
    let mem: Box<dyn ConversationMemory> = Box::new(InMemoryConversationMemory::new());

    mem.append(&"c".into(), vec![user("hello")]).await.unwrap();

    assert_eq!(mem.load(&"c".into()).await.unwrap().len(), 1);
    mem.clear(&"c".into()).await.unwrap();
    assert!(mem.load(&"c".into()).await.unwrap().is_empty());
}
