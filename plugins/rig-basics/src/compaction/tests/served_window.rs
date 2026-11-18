//! A model whose catalog entry gives no context window compacts near the
//! window its local server says it serves.

use std::time::{Duration, Instant};

use super::*;
use rig_core::catalog::{Catalog, Connector};
use rig_core::http_client::DynHttpClient;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

/// The window the fake servers serve their model with.
const SERVED: u32 = 40_000;

/// An app whose connector reaches local servers through `server`, with its
/// first agent on the catalog model `reference`, answered by `model`, and
/// a conversation past [`SERVED`] tokens; its frames ran until the server
/// answered, or for a few seconds.
fn served(
    reference: &str,
    server: &SequencedHttpClient,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let mut connector = Connector::new(Catalog::builtin().clone());
    let reached = server.clone();
    connector.wrap_http(move |_| DynHttpClient::new(reached.clone()));
    let (mut app, wakes) = start_with(&MemoryStore::default(), |app| {
        app.insert_resource(Models(connector));
    });
    let spec = Catalog::builtin()
        .resolve(reference)
        .map(|found| found.shared());
    assert!(
        spec.as_ref().is_ok_and(|spec| spec.input_limit().is_none()),
        "the catalog lists {reference} with no window"
    );
    let Ok(spec) = spec else {
        return (app, wakes, Entity::PLACEHOLDER);
    };
    let handler =
        ModelAdapter::<rig_core::operation::Completion>::new(spec.reference(), model.clone());
    let connection = Connection {
        spec,
        handler: rig_ecs::effects::Handler(ErasedHandler::new(handler)),
    };
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    world.entity_mut(agent).insert(connection);
    let long = "word ".repeat(4_000);
    let mut messages = vec![Message::user("What does the crate do?")];
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    for message in messages {
        commit_message(world, agent, message).ok();
    }
    // Until the server was asked and its answer is in.
    let asked = |world: &mut World| {
        let mut asks = world.query::<&Name>();
        let asking = asks
            .iter(world)
            .any(|name| name.starts_with("context window"));
        !server.requests().is_empty() && !asking
    };
    let started = Instant::now();
    while !asked(app.world_mut()) && started.elapsed() < Duration::from_secs(3) {
        app.update();
        wakes.recv_timeout(Duration::from_millis(50)).ok();
    }
    (app, wakes, agent)
}

#[test]
fn a_model_with_no_catalog_window_compacts_near_the_window_its_server_serves() {
    let ollama = format!(
        r#"{{"parameters": "num_ctx {SERVED}", "model_info": {{"general.architecture": "qwen3", "qwen3.context_length": 262144}}}}"#
    );
    let llama_cpp = format!(r#"{{"default_generation_settings": {{"n_ctx": {SERVED}}}}}"#);
    for (reference, body, asked) in [
        ("ollama/qwen3", ollama, "/api/show"),
        ("llamacpp/LLaMA_CPP", llama_cpp, "/props"),
    ] {
        let server = SequencedHttpClient::new([MockHttpResponse::success(body)]);
        let model = MockCompletionModel::from_stream_turns([
            reply("## Goal\nAnswer the user's questions about the crate."),
            reply("done"),
        ]);
        let (mut app, wakes, agent) = served(reference, &server, &model);
        ask(&mut app, &wakes, agent);
        let requests = sent(&model);
        assert!(
            requests.first().is_some_and(|first| asks_summary(first)),
            "{reference}: the conversation is summarized before the request: {} requests",
            requests.len()
        );
        assert_eq!(answer(&app).as_deref(), Some("done"), "{reference}");
        // Once, at the server's root rather than under its `/v1`.
        let uris: Vec<String> = server.requests().into_iter().map(|it| it.uri).collect();
        let once = matches!(uris.as_slice(), [uri] if uri.ends_with(asked) && !uri.contains("/v1"));
        assert!(once, "{reference}: {uris:?}");
    }
}

#[test]
fn a_server_that_does_not_say_leaves_the_window_unknown() {
    let server = SequencedHttpClient::new([]);
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    let limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.input_limit());
    assert_eq!(limit, None);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert!(!sent(&model).iter().any(|request| asks_summary(request)));
}
