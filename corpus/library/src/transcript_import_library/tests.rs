use std::path::PathBuf;

use rig_core::conversation::Author;
use rig_core::message::{AssistantContent, Message, UserContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::{AgentPlugin, Deliver, DeliveryMode, World};
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_protocol::{Home, SessionDir, SessionId};
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_memory::{DirStore, SavedAgent, SessionMeta};

use super::*;

const CHAT: &str = "\
Exported from a web chat about the cache.

## User
How should we cache the parsed configs?

## Assistant
Use an LRU keyed by path:

```rust
// A config file can quote a transcript:
## User
let cache = Lru::new(64);
```

**User:** And invalidation?

## Assistant
Watch the files. Here is what the tree looks like:

```tool:shell
$ ls config
a.toml
b.toml
```

Both are watched.

## User
Agreed, write it down.

**Assistant:** Plan: LRU of 64, invalidated by a watcher.
";

/// A fresh directory for this test under the target directory.
fn scratch(name: &str) -> Option<PathBuf> {
    let target = option_env!("CARGO_TARGET_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"),
        PathBuf::from,
    );
    let dir = target
        .join("corpus-ct-transcript-import-library")
        .join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The main agent of the session kept in `dir`, read strictly.
fn main_agent(dir: &std::path::Path) -> Option<SavedAgent> {
    let store = DirStore::open(dir).ok()?;
    Session::read(&store).ok()?.main_agent().cloned()
}

/// The one session of `home`.
fn only_session(home: &Home) -> Option<SessionId> {
    let mut ids = std::fs::read_dir(home.sessions())
        .ok()?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok()?.parse().ok());
    let id = ids.next();
    ids.next().is_none().then_some(id)?
}

fn text_of(message: &Message) -> String {
    match message {
        Message::System { content } => content.clone(),
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        Message::Assistant(assistant) => assistant.text(),
    }
}

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

#[test]
fn converts_a_transcript_and_loads_it_back() {
    let dir = scratch("convert");
    assert!(dir.is_some());
    let Some(dir) = dir else { return };
    let home = Home::at(dir.join("home"));
    let input = dir.join("design-chat.md");
    assert!(std::fs::write(&input, CHAT).is_ok());
    let input = input.to_string_lossy().into_owned();

    // Without --tools: one system note and six messages, the fence whole.
    let plain = dir.join("plain.session");
    let out = plain.to_string_lossy().into_owned();
    let said = run(
        &args(&[&input, "--name", "cache-design", "-o", &out]),
        &home,
    );
    assert!(said.is_ok(), "{said:?}");
    let agent = main_agent(&plain);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    assert_eq!(agent.model(), None);
    let messages = agent.conversation.messages();
    assert_eq!(messages.len(), 7);
    let author = agent.conversation.author(0).map(|author| &author.kind);
    assert_eq!(author, Some(&Author::plugin(IMPORTER).kind));
    let roles: Vec<&str> = messages
        .iter()
        .map(|message| match &**message {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant(_) => "assistant",
        })
        .collect();
    let expected = [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ];
    assert_eq!(roles, expected);
    let texts: Vec<String> = messages.iter().map(|message| text_of(message)).collect();
    assert_eq!(
        texts.first().map(String::as_str),
        Some("Exported from a web chat about the cache.")
    );
    let fenced = "```rust\n// A config file can quote a transcript:\n## User\nlet cache = Lru::new(64);\n```";
    assert!(texts.get(2).is_some_and(|text| text.ends_with(fenced)));
    assert_eq!(texts.get(3).map(String::as_str), Some("And invalidation?"));
    assert!(
        texts
            .get(4)
            .is_some_and(|text| text.contains("```tool:shell\n$ ls config"))
    );
    assert_eq!(
        texts.get(6).map(String::as_str),
        Some("Plan: LRU of 64, invalidated by a watcher.")
    );
    let paths = SessionDir::at(&plain);
    assert!(paths.is_saved());
    assert_eq!(paths.name().as_deref(), Some("cache-design"));
    let store = DirStore::open(&plain);
    let meta = store.as_ref().ok().and_then(SessionMeta::read);
    assert_eq!(
        meta.map(|meta| meta.title).as_deref(),
        Some("How should we cache the parsed configs?")
    );
    let origin = store.as_ref().ok().and_then(Origin::read);
    assert_eq!(
        origin.map(|origin| origin.source),
        Some(PathBuf::from(&input))
    );

    // With --tools, into the home: the shell fence is one call and its
    // result, on the assistant message that showed it.
    let said = run(
        &args(&[&input, "--tools", "--model", "anthropic/claude-x"]),
        &home,
    );
    assert!(
        said.as_ref()
            .is_ok_and(|said| said.contains("rig --resume")),
        "{said:?}"
    );
    let id = only_session(&home);
    assert!(id.is_some());
    let Some(id) = id else { return };
    let paths = home.session(&id);
    assert!(paths.is_saved());
    assert_eq!(
        home.find("How should we cache the parsed configs?").ok(),
        Some(id)
    );
    let agent = main_agent(paths.path());
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    assert_eq!(agent.model().as_deref(), Some("anthropic/claude-x"));
    let conversation = &agent.conversation;
    assert!(conversation.validate().is_ok());
    let messages = conversation.messages();
    assert_eq!(messages.len(), 9);
    let shown = match messages.get(4).map(|message| &**message) {
        Some(Message::Assistant(shown)) => Some(shown),
        _ => None,
    };
    let calls: Vec<_> = shown.iter().flat_map(|shown| shown.tool_calls()).collect();
    let [call] = calls.as_slice() else {
        return assert_eq!(calls.len(), 1);
    };
    assert_eq!(call.function.name.as_str(), "shell");
    assert_eq!(
        call.function.arguments.get("command"),
        Some(&serde_json::Value::from("ls config"))
    );
    assert!(
        matches!(shown.and_then(|shown| shown.content.first()), Some(AssistantContent::Text(text))
        if text.text == "Watch the files. Here is what the tree looks like:")
    );
    assert!(
        matches!(messages.get(5).map(|message| &**message), Some(Message::User { content }) if matches!(content.as_slice(), [UserContent::ToolResult(result)]
        if result.call == call.id
            && result.content.first().and_then(|item| item.as_text()) == Some("a.toml\nb.toml")))
    );
    assert_eq!(
        messages.get(6).map(|message| text_of(message)).as_deref(),
        Some("Both are watched.")
    );

    // No headings: an error, and nothing written.
    let bare = dir.join("bare.md");
    assert!(std::fs::write(&bare, "Just notes.\n\n```\n## User\n```\n").is_ok());
    let refused = dir.join("refused.session");
    let said = run(
        &args(&[&bare.to_string_lossy(), "-o", &refused.to_string_lossy()]),
        &home,
    );
    assert!(
        matches!(said, Err(ImportError::NoHeadings { .. })),
        "{said:?}"
    );
    assert!(!refused.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Runs `app` until a turn ended.
fn one_turn(world: &World) -> bool {
    !world.resource::<Ended>().0.is_empty()
}

#[test]
fn the_harness_resumes_it_with_the_whole_history() {
    let home = scratch_home("ct-transcript-import-resume");
    let input = home.root().join("design-chat.md");
    let written = std::fs::create_dir_all(home.root()).and_then(|()| std::fs::write(&input, CHAT));
    assert!(written.is_ok());
    let input = input.to_string_lossy().into_owned();
    let said = run(
        &args(&[
            &input,
            "--name",
            "cache-design",
            "--tools",
            "--model",
            "test/main",
        ]),
        &home,
    );
    assert!(said.is_ok(), "{said:?}");
    let id = home.find("cache-design");
    assert!(id.is_ok(), "{id:?}");
    let Ok(id) = id else { return };

    let model = MockCompletionModel::from_stream_turns(vec![reply("Implementing the LRU.")]);
    let (mut app, wakes) = app_in(&home, &id);
    app.insert_resource(scripted_models([("test/main", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    let line = Deliver::new(agent, "implement what we agreed", DeliveryMode::Queue)
        .with_author(Author::user());
    app.world_mut().trigger(line);
    run_until(&mut app, &wakes, one_turn);

    // The request carries the imported history, the tool exchange included.
    let requests = model.requests();
    let sent = requests
        .first()
        .map(|request| format!("{:?}", request.chat_history));
    for part in [
        "Exported from a web chat about the cache.",
        "How should we cache the parsed configs?",
        "## User\\nlet cache = Lru::new(64);",
        "a.toml\\nb.toml",
        "implement what we agreed",
    ] {
        assert!(
            sent.as_ref().is_some_and(|sent| sent.contains(part)),
            "{part}: {sent:?}"
        );
    }
    let messages = app
        .world()
        .get::<rig_harness::rig_ecs::agent::Conversation>(agent)
        .map(|conversation| conversation.messages().to_vec())
        .unwrap_or_default();
    assert_eq!(messages.len(), 11);
    assert_eq!(
        messages.last().map(|message| text_of(message)).as_deref(),
        Some("Implementing the LRU.")
    );
}
