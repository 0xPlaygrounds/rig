use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_basics::compaction::CompactionPlugin;
use rig_core::completion::Message as ChatMessage;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, connect, first_agent, log_ended, reply, run_until,
};

use super::*;

/// The kernel and this plugin, without the default compaction, on a store
/// in memory with `settings`; its agent answered by `model` with a
/// `window`-token context window.
fn start(model: &MockCompletionModel, window: u32, settings: &str) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, KeepLastCompactionPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(mut connection) = connect(model) {
        connection.spec = Arc::new(connection.spec.as_ref().clone().with_context_window(window));
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Sends `text` and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Each message of `agent`, as its text.
fn texts(app: &App, agent: Entity) -> Vec<String> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    let text = |entry: EntryRef| match entry.message {
        ChatMessage::Assistant(assistant) => assistant.text(),
        message => message.rag_text().unwrap_or_default(),
    };
    conversation.entries().map(text).collect()
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    requests
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect()
}

fn owned(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|text| (*text).to_owned()).collect()
}

#[test]
fn a_request_near_the_window_drops_the_oldest_turns_with_no_model_call() {
    let model =
        MockCompletionModel::from_stream_turns((1..=4).map(|turn| reply(&format!("A{turn}"))));
    // No reserve and a one-token window: every request is over it.
    let (mut app, wakes, agent) = start(&model, 1, "[compaction]\nlimits = { reserve = 0 }");
    for question in ["apples?", "bananas?", "cherries?", "dates?"] {
        ask(&mut app, &wakes, agent, question);
    }

    // One request per turn: no summary was asked for.
    let requests = sent(&model);
    assert_eq!(requests.len(), 4, "{requests:#?}");
    let last = requests.last().cloned().unwrap_or_default();
    assert!(
        !last.contains("apples") && !last.contains("bananas"),
        "{last}"
    );
    assert!(
        last.contains("cherries") && last.contains("dates"),
        "{last}"
    );
    assert_eq!(
        texts(&app, agent),
        owned(&["cherries?", "A3", "dates?", "A4"])
    );
}

#[test]
fn compact_drops_the_oldest_turns_now() {
    let model =
        MockCompletionModel::from_stream_turns((1..=3).map(|turn| reply(&format!("A{turn}"))));
    // A window no request nears: only `/compact` drops.
    let (mut app, wakes, agent) = start(&model, 1_000_000, "");
    for question in ["Q1", "Q2", "Q3"] {
        ask(&mut app, &wakes, agent, question);
    }
    assert_eq!(texts(&app, agent).len(), 6);

    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    app.update();
    app.update();
    assert_eq!(texts(&app, agent), owned(&["Q2", "A2", "Q3", "A3"]));
    assert_eq!(model.request_count(), 3);
}

/// The default `plugins.toml` with this plugin in place of the default
/// compaction: no default plugin adds that back, and this one's systems
/// are ordered against theirs.
#[test]
fn it_replaces_the_default_compaction_among_the_defaults() {
    let mut app = App::new();
    let home = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("keep-last-compaction");
    let print = Invocation {
        print: Some(String::new()),
        ..Invocation::default()
    };
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(Invoked {
            args: print,
            terminal: false,
        })
        .add_plugins((HeadlessPlugins, RigHarnessPlugins));
    let default = std::any::type_name::<CompactionPlugin>();
    for (_, add) in rig_defaults_test_support::DEFAULTS
        .iter()
        .filter(|(path, _)| *path != default)
    {
        add(&mut app);
    }
    app.add_plugins(KeepLastCompactionPlugin);
    // The session plugin sends panics to the session's log: keep them here.
    drop(std::panic::take_hook());
    app.finish();
    app.cleanup();

    assert!(!app.is_plugin_added::<CompactionPlugin>());
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
}
