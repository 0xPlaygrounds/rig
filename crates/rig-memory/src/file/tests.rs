use super::*;
use assert_fs::TempDir;
use std::collections::HashSet;
use std::time::Duration;

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str) -> Message {
    Message::assistant(text)
}

fn store() -> (TempDir, FileConversationMemory) {
    let dir = TempDir::new().unwrap();
    let memory = FileConversationMemory::new(dir.path().join("conversations"));
    (dir, memory)
}

fn file_error(error: MemoryError) -> FileMemoryError {
    let MemoryError::Backend(source) = error else {
        panic!("expected a backend error, got {error:?}");
    };
    *source
        .downcast::<FileMemoryError>()
        .unwrap_or_else(|source| panic!("expected a FileMemoryError, got {source:?}"))
}

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn round_trips_messages() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    let messages = vec![user("My name is Alice."), assistant("Hello, Alice!")];

    memory.append(&id, messages.clone()).await.unwrap();

    assert_eq!(memory.load(&id).await.unwrap(), messages);
    let contents = fs::read_to_string(memory.path(&id)).unwrap();
    assert_eq!(contents.lines().count(), 2);
    assert!(contents.ends_with('\n'));
}

#[tokio::test]
async fn loading_a_missing_conversation_is_empty() {
    let (_tmp, memory) = store();
    assert!(memory.load(&"missing".into()).await.unwrap().is_empty());
}

#[tokio::test]
async fn appends_persist_across_instances() {
    let (tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    memory.append(&id, vec![user("one")]).await.unwrap();
    memory
        .append(&id, vec![assistant("two"), user("three")])
        .await
        .unwrap();
    memory.append(&id, Vec::new()).await.unwrap();

    let reopened = FileConversationMemory::new(tmp.path().join("conversations"));
    assert_eq!(
        reopened.load(&id).await.unwrap(),
        vec![user("one"), assistant("two"), user("three")]
    );
}

#[tokio::test]
async fn skips_a_torn_last_line_and_appends_after_it() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    memory
        .append(&id, vec![user("one"), assistant("two")])
        .await
        .unwrap();
    let path = memory.path(&id);
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(br#"{"role":"user","content":[{"type":"te"#)
        .unwrap();
    drop(file);

    assert_eq!(
        memory.load(&id).await.unwrap(),
        vec![user("one"), assistant("two")]
    );

    memory.append(&id, vec![user("three")]).await.unwrap();
    assert_eq!(
        memory.load(&id).await.unwrap(),
        vec![user("one"), assistant("two"), user("three")]
    );
    let contents = fs::read_to_string(&path).unwrap();
    assert_eq!(contents.lines().count(), 3);
}

#[tokio::test]
async fn truncates_a_torn_line_that_is_the_whole_file() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    fs::create_dir_all(memory.dir()).unwrap();
    fs::write(memory.path(&id), br#"{"role":"us"#).unwrap();

    assert!(memory.load(&id).await.unwrap().is_empty());
    memory.append(&id, vec![user("one")]).await.unwrap();
    assert_eq!(memory.load(&id).await.unwrap(), vec![user("one")]);
}

#[tokio::test]
async fn a_malformed_middle_line_is_an_error() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    memory.append(&id, vec![user("one")]).await.unwrap();
    let path = memory.path(&id);
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"not a message\n").unwrap();
    drop(file);
    memory.append(&id, vec![user("three")]).await.unwrap();

    let error = file_error(memory.load(&id).await.unwrap_err());
    let FileMemoryError::MalformedLine {
        path: reported,
        line,
        ..
    } = error
    else {
        panic!("expected a malformed line, got {error:?}");
    };
    assert_eq!(reported, path);
    assert_eq!(line, 2);
}

#[tokio::test]
async fn replace_swaps_the_whole_history() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    memory
        .append(&id, vec![user("one"), assistant("two"), user("three")])
        .await
        .unwrap();

    memory
        .replace(&id, vec![assistant("summary")])
        .await
        .unwrap();
    assert_eq!(memory.load(&id).await.unwrap(), vec![assistant("summary")]);

    memory.append(&id, vec![user("four")]).await.unwrap();
    assert_eq!(
        memory.load(&id).await.unwrap(),
        vec![assistant("summary"), user("four")]
    );

    let fresh = ConversationId::from("fresh");
    memory.replace(&fresh, vec![user("new")]).await.unwrap();
    assert_eq!(memory.load(&fresh).await.unwrap(), vec![user("new")]);
    memory.replace(&fresh, Vec::new()).await.unwrap();
    assert!(memory.load(&fresh).await.unwrap().is_empty());

    assert_eq!(
        file_names(memory.dir()),
        vec!["fresh.jsonl".to_owned(), "thread-1.jsonl".to_owned()]
    );
}

#[tokio::test]
async fn lists_conversations_newest_first() {
    let (_tmp, memory) = store();
    assert!(memory.list().await.unwrap().is_empty());

    let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    for (offset, name) in [(10, "middle"), (20, "Newest"), (0, "oldest")] {
        let id = ConversationId::from(name);
        memory.append(&id, vec![user(name)]).await.unwrap();
        File::options()
            .write(true)
            .open(memory.path(&id))
            .unwrap()
            .set_modified(base + Duration::from_secs(offset))
            .unwrap();
    }
    fs::write(memory.dir().join("notes.txt"), "not a conversation").unwrap();
    fs::write(memory.dir().join("Foreign.jsonl"), "").unwrap();
    fs::write(memory.dir().join(".middle.jsonl.1-0.tmp"), "").unwrap();

    let listed = memory.list().await.unwrap();
    let ids: Vec<&str> = listed.iter().map(|entry| entry.id.as_str()).collect();
    assert_eq!(ids, vec!["Newest", "middle", "oldest"]);
    assert_eq!(
        listed.first().unwrap().modified,
        base + Duration::from_secs(20)
    );
}

#[tokio::test]
async fn clear_removes_the_conversation() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("thread-1");
    memory.clear(&id).await.unwrap();

    memory.append(&id, vec![user("one")]).await.unwrap();
    memory.clear(&id).await.unwrap();

    assert!(!memory.path(&id).exists());
    assert!(memory.load(&id).await.unwrap().is_empty());
    assert!(memory.list().await.unwrap().is_empty());
    memory.clear(&id).await.unwrap();
}

#[tokio::test]
async fn maps_every_id_to_its_own_file_inside_the_directory() {
    let (_tmp, memory) = store();
    let long_ascii = "x".repeat(300);
    let long_ascii_variant = format!("{}y", "x".repeat(299));
    let long_unicode = "会話".repeat(40);
    let ids = [
        "..",
        ".",
        "../escape",
        "../../etc/passwd",
        "/absolute",
        "a/b",
        "a\\b",
        "C:\\windows",
        "nul\0byte",
        "",
        "con",
        "CON",
        "Chat",
        "chat",
        "CHAT",
        "%63hat",
        "~hashed",
        "日本語",
        "café",
        "cafe\u{301}",
        long_ascii.as_str(),
        long_ascii_variant.as_str(),
        long_unicode.as_str(),
    ];

    for (index, id) in ids.iter().enumerate() {
        let id = ConversationId::from(*id);
        let path = memory.path(&id);
        assert_eq!(path.parent(), Some(memory.dir()), "{id:?}");
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.len() <= 255, "{id:?}");
        assert!(
            !name.bytes().any(|byte| byte.is_ascii_uppercase()),
            "{id:?}"
        );
        memory
            .append(&id, vec![user(&index.to_string())])
            .await
            .unwrap();
    }

    let mut names = HashSet::new();
    for (index, id) in ids.iter().enumerate() {
        let id = ConversationId::from(*id);
        assert_eq!(
            memory.load(&id).await.unwrap(),
            vec![user(&index.to_string())],
            "{id:?}"
        );
        let name = memory.path(&id).file_name().unwrap().to_owned();
        assert!(
            names.insert(name.to_string_lossy().to_lowercase()),
            "{id:?}"
        );
    }

    let listed: HashSet<String> = memory
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.id.into_string())
        .collect();
    let expected: HashSet<String> = ids.iter().map(|id| (*id).to_owned()).collect();
    assert_eq!(listed, expected);

    let parent_entries = file_names(memory.dir().parent().unwrap());
    assert_eq!(parent_entries, vec!["conversations".to_owned()]);
}

#[tokio::test]
async fn hashed_names_reject_a_different_id() {
    let (_tmp, memory) = store();
    let id = ConversationId::from("x".repeat(300));
    memory.append(&id, vec![user("one")]).await.unwrap();
    let id_file = memory.path(&id).with_extension(ID_EXTENSION);
    fs::write(&id_file, "someone else").unwrap();

    let error = file_error(memory.load(&id).await.unwrap_err());
    assert!(
        matches!(&error, FileMemoryError::IdMismatch { path } if *path == id_file),
        "{error:?}"
    );
    let error = file_error(memory.append(&id, vec![user("two")]).await.unwrap_err());
    assert!(
        matches!(error, FileMemoryError::IdMismatch { .. }),
        "{error:?}"
    );

    memory.clear(&id).await.unwrap();
    assert!(!id_file.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_appends_do_not_interleave() {
    let (_tmp, memory) = store();
    let memory = Arc::new(memory);
    let id = ConversationId::from("shared");

    let tasks: Vec<_> = (0..32)
        .map(|task| {
            let memory = Arc::clone(&memory);
            let id = id.clone();
            tokio::spawn(async move {
                let batch: Vec<Message> =
                    (0..4).map(|step| user(&format!("{task}:{step}"))).collect();
                memory.append(&id, batch).await.unwrap();
            })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }

    let loaded = memory.load(&id).await.unwrap();
    assert_eq!(loaded.len(), 128);
    let batch_of = |task: usize| -> Vec<Message> {
        (0..4).map(|step| user(&format!("{task}:{step}"))).collect()
    };
    let mut seen = HashSet::new();
    for batch in loaded.chunks(4) {
        let task = (0..32)
            .find(|&task| batch[0] == user(&format!("{task}:0")))
            .unwrap();
        assert_eq!(batch, batch_of(task).as_slice());
        assert!(seen.insert(task));
    }
    assert!(memory.locks.lock().unwrap().is_empty());
}

#[test]
fn works_without_a_tokio_runtime() {
    let tmp = TempDir::new().unwrap();
    let memory = FileConversationMemory::new(tmp.path());
    let id = ConversationId::from("thread-1");
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);

    let mut append = memory.append(&id, vec![user("one")]);
    assert!(matches!(
        append.as_mut().poll(&mut context),
        std::task::Poll::Ready(Ok(()))
    ));
    drop(append);
    let mut load = memory.load(&id);
    let std::task::Poll::Ready(Ok(loaded)) = load.as_mut().poll(&mut context) else {
        panic!("load did not complete synchronously");
    };
    assert_eq!(loaded, vec![user("one")]);
}
