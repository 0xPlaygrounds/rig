use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig::harness_protocol::Home;
use rig_core::conversation::{Aside, Author};
use rig_core::message::{Message, UserContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::{
    AgentPlugin, App, Deliver, DeliveryMode, Entity, ModelChoice, SessionLogPlugin,
};
use rig_harness::rig_ecs::agent::Conversation as AgentConversation;
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use serde_json::{Value, json};

use super::*;

const MODEL: &str = "test/model";

fn start(home: &Home, id: &SessionId, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(scripted_models([(MODEL, model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Edits the agent's live conversation, as a plugin's system would.
fn edit(app: &mut App, agent: Entity, change: impl FnOnce(&mut Conversation)) {
    if let Some(mut conversation) = app.world_mut().get_mut::<AgentConversation>(agent) {
        change(&mut conversation.0);
    }
    app.update();
}

/// The fixture: 2 shared turns, then branches of 3, 1 and 2 turns, the
/// second carrying a test plugin's aside. The session's id and the aside.
fn fixture(home: &Home) -> (SessionId, Aside) {
    let replies =
        ["s1", "s2", "a1", "a2", "a3", "b1", "c1", "c2"].map(|r| reply(&format!("re {r}")));
    let model = MockCompletionModel::from_stream_turns(replies);
    let id = SessionId::generate();
    let (mut app, wakes, agent) = start(home, &id, &model);
    app.world_mut()
        .entity_mut(agent)
        .insert(ModelChoice(MODEL.to_owned()));
    app.update();
    for text in ["q s1", "q s2", "q a1", "q a2", "q a3"] {
        ask(&mut app, &wakes, agent, text);
    }
    edit(&mut app, agent, |c| {
        drop(c.branch_off(4, "b", Author::user()))
    });
    ask(&mut app, &wakes, agent, "q b1");
    let pin = Aside::new(
        "test.pin",
        Author::plugin("pin"),
        json!({ "pinned": "only on b", "order": [3, 1, 2] }),
    );
    let mut placed = pin.clone();
    edit(&mut app, agent, |c| {
        placed = pin.after(c.entries().next_back().map(|entry| entry.id));
        c.add_aside(placed.clone());
    });
    edit(&mut app, agent, |c| {
        drop(c.branch_off(4, "c", Author::user()))
    });
    ask(&mut app, &wakes, agent, "q c1");
    ask(&mut app, &wakes, agent, "q c2");
    for _ in 0..3 {
        app.update();
    }
    drop(app);
    (id, placed)
}

fn args(words: &[&Path], flags: &[&str]) -> Vec<String> {
    let paths = words.iter().map(|path| path.display().to_string());
    paths
        .chain(flags.iter().map(|flag| (*flag).to_owned()))
        .collect()
}

fn explode_into(input: &Path, out: &Path, flags: &[&str]) -> Result<String, Error> {
    let mut printed = Vec::new();
    run(&args(&[input, out], flags), &mut printed)?;
    Ok(String::from_utf8_lossy(&printed).into_owned())
}

/// Every file under `dir` with its bytes.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut todo = vec![dir.to_owned()];
    while let Some(next) = todo.pop() {
        for entry in fs::read_dir(&next).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else if let Ok(bytes) = fs::read(&path) {
                files.insert(path, bytes);
            }
        }
    }
    files
}

fn text_of(message: &Message) -> String {
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        Message::Assistant(assistant) => assistant.text(),
        Message::System { content } => content.clone(),
    }
}

/// The output's main agent and its session store.
fn load(path: &Path) -> Option<(SavedAgent, DirStore)> {
    let store = DirStore::open(path).ok()?;
    let session = Session::read(&store).ok()?;
    Some((session.main_agent()?.clone(), store))
}

#[test]
fn every_branch_becomes_a_session_of_its_own() {
    let home = scratch_home("sp-branch-explode-library");
    let (id, pin) = fixture(&home);
    let input = home.session(&id).path().to_owned();
    let before = snapshot(&input);
    assert!(!before.is_empty());

    // Three outputs, in the sessions directory, so the agent resumes them.
    let out = home.sessions();
    let printed = explode_into(&input, &out, &[]);
    assert!(printed.is_ok(), "{printed:?}");
    let printed = printed.unwrap_or_default();
    let paths: Vec<PathBuf> = printed
        .lines()
        .filter_map(|line| line.split('\t').next().map(PathBuf::from))
        .collect();
    assert_eq!(paths.len(), 3, "{printed}");
    let pin_json = serde_json::to_string(&pin).unwrap_or_default();
    let expected = [
        (5, "q a3", "re a3", true),
        (3, "q b1", "re b1", false),
        (4, "q c2", "re c2", true),
    ];
    for (k, (path, (turns, last_q, last_re, no_pin))) in paths.iter().zip(expected).enumerate() {
        let loaded = load(path);
        assert!(loaded.is_some(), "{}", path.display());
        let Some((agent, store)) = loaded else { return };
        let conversation = &agent.conversation;
        let texts: Vec<String> = conversation
            .messages()
            .iter()
            .map(|message| text_of(message))
            .collect();
        assert_eq!(
            conversation.entries().filter(|e| e.is_request()).count(),
            turns
        );
        assert_eq!(texts.len(), turns * 2, "{texts:?}");
        assert_eq!(
            texts.get(..4),
            Some(&["q s1", "re s1", "q s2", "re s2"].map(String::from)[..])
        );
        assert_eq!(
            texts.get(texts.len().saturating_sub(2)..),
            Some(&[last_q.to_owned(), last_re.to_owned()][..])
        );
        assert!(
            conversation.branches().is_empty(),
            "no other branch rides along"
        );
        // The test plugin's aside, byte-identical, only in b's output.
        let pins: Vec<String> = conversation
            .asides()
            .filter(|(_, aside)| aside.kind == "test.pin")
            .filter_map(|(_, aside)| serde_json::to_string(aside).ok())
            .collect();
        assert_eq!(pins.is_empty(), no_pin, "{pins:?}");
        if !no_pin {
            assert_eq!(pins, std::slice::from_ref(&pin_json));
        }
        let logs = snapshot(&path.join("logs"));
        let raw = logs
            .values()
            .map(|bytes| String::from_utf8_lossy(bytes))
            .collect::<String>();
        assert_eq!(raw.contains("only on b"), !no_pin);
        // Title, origin and name.
        let meta = SessionMeta::read(&store).unwrap_or_default();
        assert!(
            meta.title.ends_with(&format!("[branch {} of 3]", k + 1)),
            "{}",
            meta.title
        );
        let from = ForkedFrom::read(&store).unwrap_or_default();
        assert_eq!(from.session, id.as_str());
        let leaf = conversation.entries().next_back().map(|entry| entry.id);
        assert_eq!(from.through.map(|(_, leaf)| leaf), leaf);
        assert!(SessionDir::at(path).id().is_some());
        let name = format!("{}-branch-{}", id.as_str(), k + 1);
        assert_eq!(SessionDir::at(path).name(), Some(name));
    }

    // Resuming b's output sends b's messages and no other branch's.
    let b = paths.get(1).and_then(|path| SessionDir::at(path).id());
    assert!(b.is_some());
    let model = MockCompletionModel::from_stream_turns([reply("re b2")]);
    if let Some(b) = b {
        let (mut app, wakes, agent) = start(&home, &b, &model);
        ask(&mut app, &wakes, agent, "q b2");
    }
    let sent = model
        .requests()
        .first()
        .map(|request| request.chat_history.clone());
    let sent: Vec<String> = sent
        .unwrap_or_default()
        .iter()
        .map(|message| text_of(message))
        .collect();
    assert_eq!(
        sent,
        ["q s1", "re s1", "q s2", "re s2", "q b1", "re b1", "q b2"].map(String::from)
    );

    // Exploding again never overwrites a name; `--min-turns 2` leaves b
    // out; `--json` lists the leaves.
    let again = explode_into(&input, &out, &["--min-turns", "2", "--json"]);
    let listed: Value = serde_json::from_str(&again.unwrap_or_default()).unwrap_or_default();
    let listed = listed.as_array().cloned().unwrap_or_default();
    assert_eq!(listed.len(), 2, "{listed:?}");
    let field = |at: usize, key: &str| listed.get(at).and_then(|item| item.get(key)).cloned();
    assert_eq!(field(0, "turns"), Some(json!(5)));
    assert_eq!(field(1, "turns"), Some(json!(4)));
    assert!(field(0, "leaf").is_some_and(|leaf| leaf.as_str().is_some_and(|l| l.starts_with('m'))));
    let renamed = format!("{}-branch-1-1", id.as_str());
    assert_eq!(field(0, "name"), Some(json!(renamed)));

    // The original is untouched.
    assert_eq!(snapshot(&input), before);

    // A truncated input: refused, nothing written.
    let broken = home.root().join("broken");
    for (path, bytes) in &before {
        let Ok(relative) = path.strip_prefix(&input) else {
            continue;
        };
        let to = broken.join(relative);
        let cut = if relative.starts_with("logs") {
            bytes.len().saturating_sub(7)
        } else {
            bytes.len()
        };
        let copied = to.parent().map(fs::create_dir_all);
        assert!(copied.is_some_and(|made| made.is_ok()));
        assert!(fs::write(&to, bytes.get(..cut).unwrap_or_default()).is_ok());
    }
    let nowhere = home.root().join("nowhere");
    let refused = explode_into(&broken, &nowhere, &[]);
    assert!(
        refused
            .as_ref()
            .is_err_and(|why| why.0.contains("unreadable")),
        "{refused:?}"
    );
    assert!(!nowhere.exists());
}
