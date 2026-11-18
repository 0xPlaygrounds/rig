//! A model whose catalog entry gives no context window compacts near the
//! window its local server says it serves.

use super::*;
use rig_core::catalog::{Catalog, Connector};
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response,
    StreamingResponse,
};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_core::wasm_compat::WasmCompatSend;

/// The window the fake servers serve their model with.
const SERVED: u32 = 40_000;

/// A local server that answers with `replies` once the test opens it
/// ([`Self::open`]), and never if it does not.
#[derive(Clone)]
struct Server {
    replies: SequencedHttpClient,
    opened: async_channel::Receiver<()>,
    open: async_channel::Sender<()>,
}

impl Server {
    fn new(replies: impl IntoIterator<Item = MockHttpResponse>) -> Self {
        let (open, opened) = async_channel::bounded(1);
        Self {
            replies: SequencedHttpClient::new(replies),
            opened,
            open,
        }
    }

    /// Lets every answer, held or later, go out.
    fn open(&self) {
        self.open.close();
    }

    /// The paths of the requests the server was sent.
    fn asked(&self) -> Vec<String> {
        self.replies
            .requests()
            .into_iter()
            .map(|request| request.uri)
            .collect()
    }
}

impl HttpClientExt for Server {
    fn send(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody>>> + WasmCompatSend + 'static
    {
        let reply = self.replies.send(request);
        let opened = self.opened.clone();
        async move {
            opened.recv().await.ok();
            reply.await
        }
    }

    fn send_multipart(
        &self,
        request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody>>> + WasmCompatSend + 'static
    {
        self.replies.send_multipart(request)
    }

    fn send_streaming(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        self.replies.send_streaming(request)
    }
}

/// An app whose connector reaches local servers through `server`, with its
/// first agent on the catalog model `reference`, answered by `model`, and
/// a conversation past [`SERVED`] tokens. Nothing ran yet: the server is
/// asked in the first frame.
fn served(
    reference: &str,
    server: &Server,
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
    (app, wakes, agent)
}

/// Asks `agent` a question while its server holds its answer back, runs
/// frames until the turn's call is routed to its model, then lets the
/// server answer and runs until the turn ended.
fn ask_before_the_answer(app: &mut App, wakes: &Receiver<()>, agent: Entity, server: &Server) {
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    run_until(app, wakes, |world| {
        let ended = !world.resource::<Ended>().0.is_empty();
        let routed = world
            .iter_entities()
            .any(|entity| entity.contains::<TurnCall>() && entity.contains::<Adapted>());
        ended || routed
    });
    server.open();
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

/// Whether a window ask is still out in `world`.
fn asking(world: &World) -> bool {
    world.iter_entities().any(|entity| {
        entity
            .get::<Name>()
            .is_some_and(|name| name.starts_with("context window"))
    })
}

#[test]
fn a_model_with_no_catalog_window_compacts_near_the_window_its_server_serves() {
    let ollama = format!(
        r#"{{"parameters": "num_ctx {SERVED}", "model_info": {{"general.architecture": "qwen3", "qwen3.context_length": 262144}}}}"#
    );
    let llama_cpp = format!(r#"{{"default_generation_settings": {{"n_ctx": {SERVED}}}}}"#);
    for (reference, body, path) in [
        ("ollama/qwen3", ollama, "/api/show"),
        ("llamacpp/LLaMA_CPP", llama_cpp, "/props"),
    ] {
        let server = Server::new([MockHttpResponse::success(body)]);
        let model = MockCompletionModel::from_stream_turns([
            reply("## Goal\nAnswer the user's questions about the crate."),
            reply("done"),
        ]);
        let (mut app, wakes, agent) = served(reference, &server, &model);
        ask_before_the_answer(&mut app, &wakes, agent, &server);
        let requests = sent(&model);
        assert!(
            requests.first().is_some_and(|first| asks_summary(first)),
            "{reference}: the conversation is summarized before the request: {} requests",
            requests.len()
        );
        assert_eq!(answer(&app).as_deref(), Some("done"), "{reference}");
        // Once, at the server's root rather than under its `/v1`.
        let asked = server.asked();
        let once = matches!(asked.as_slice(), [uri] if uri.ends_with(path) && !uri.contains("/v1"));
        assert!(once, "{reference}: {asked:?}");
    }
}

/// Asks `agent` a question, then runs frames until `turns` turns ended.
fn ask_until(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() == turns
    });
}

#[test]
fn a_server_that_does_not_say_leaves_the_window_unknown() {
    let server = Server::new([MockHttpResponse::success(
        r#"{"parameters": "", "model_info": {}}"#,
    )]);
    server.open();
    let model = MockCompletionModel::from_stream_turns([reply("first"), reply("done")]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    ask_until(&mut app, &wakes, agent, 1);
    ask_until(&mut app, &wakes, agent, 2);
    let limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.input_limit());
    assert_eq!(limit, None);
    assert!(!sent(&model).iter().any(|request| asks_summary(request)));
    // Its answer is final: the second turn does not ask again.
    assert_eq!(server.asked().len(), 1, "{:?}", server.asked());
}

#[test]
fn a_server_that_fails_the_ask_is_asked_again_on_the_next_turn() {
    let ollama = format!(r#"{{"parameters": "num_ctx {SERVED}", "model_info": {{}}}}"#);
    let loading =
        || MockHttpResponse::error(http_client::StatusCode::INTERNAL_SERVER_ERROR, "loading");
    let server = Server::new([loading(), loading(), MockHttpResponse::success(ollama)]);
    server.open();
    let model = MockCompletionModel::from_stream_turns([
        reply("first"),
        reply("## Goal\nAnswer the user's questions about the crate."),
        reply("done"),
    ]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    // The ask made on connecting fails, and so does the first turn's.
    run_until(&mut app, &wakes, |world| !asking(world));
    ask_until(&mut app, &wakes, agent, 1);
    let first = sent(&model).len();
    ask_until(&mut app, &wakes, agent, 2);
    let requests = sent(&model);
    assert!(
        requests
            .get(first)
            .is_some_and(|request| asks_summary(request)),
        "the second turn summarizes first: {} requests, {first} in the first turn",
        requests.len()
    );
    assert_eq!(first, 1, "the first turn has no window to compact to");
    assert_eq!(server.asked().len(), 3, "{:?}", server.asked());
}

#[test]
fn a_server_that_never_answers_is_given_up_on() {
    let server = Server::new([]);
    let model = MockCompletionModel::from_stream_turns([reply("done"), reply("again")]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    run_until(&mut app, &wakes, |world| !asking(world));
    // Its silence is final: the next turn neither asks nor waits again.
    ask_until(&mut app, &wakes, agent, 2);
    assert!(!asking(app.world()));
    let limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.input_limit());
    assert_eq!(limit, None);
    assert_eq!(server.asked().len(), 1, "{:?}", server.asked());
}

#[test]
fn a_failed_ask_holds_one_call_of_a_turn_and_a_refusal_is_final() {
    let status = |code| move || MockHttpResponse::error(code, "no");
    let failing = status(http_client::StatusCode::INTERNAL_SERVER_ERROR);
    let refusing = status(http_client::StatusCode::NOT_FOUND);
    // A 500 is asked again once per turn: on connecting, then by the
    // turn's first call only. A 404 is the server's answer, asked once.
    for (replies, asks) in [(vec![failing(); 4], 2), (vec![refusing(); 4], 1)] {
        let server = Server::new(replies);
        server.open();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("l1", "look", serde_json::json!({})),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            reply("done"),
        ]);
        let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
        run_until(&mut app, &wakes, |world| !asking(world));
        ask_until(&mut app, &wakes, agent, 1);
        assert_eq!(
            model.request_count(),
            2,
            "the tool step makes a second call"
        );
        assert_eq!(server.asked().len(), asks, "{:?}", server.asked());
    }
}

/// On a hold [`hold_once`] keeps on a turn's first call.
#[derive(Component)]
struct OtherHold;

/// Holds the first turn's call admitted, as another plugin does in Prepare.
fn hold_once(
    calls: Query<Entity, (Added<Admitted>, With<TurnCall>)>,
    held: Query<(), With<OtherHold>>,
    mut done: Local<bool>,
    mut commands: Commands,
) {
    for call in &calls {
        if !*done && held.is_empty() {
            *done = true;
            commands.spawn((OtherHold, HoldOf(call)));
        }
    }
}

/// Leaves each turn's call to its agent's connection, as a plugin that
/// clears a route does.
fn route_to_agent(mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>) {
    for mut call in &mut calls {
        call.connection = None;
    }
}

#[test]
fn a_call_held_by_another_plugin_past_the_answer_compacts_to_the_window() {
    // Routed explicitly, and left to its agent's connection.
    for to_agent in [false, true] {
        held_past_the_answer(to_agent);
    }
}

fn held_past_the_answer(to_agent: bool) {
    let ollama = format!(r#"{{"parameters": "num_ctx {SERVED}", "model_info": {{}}}}"#);
    let server = Server::new([MockHttpResponse::success(ollama)]);
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the user's questions about the crate."),
        reply("done"),
    ]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    app.add_systems(AgentPass, hold_once.in_set(PrepareSystems::Edit));
    if to_agent {
        app.add_systems(AgentPass, route_to_agent.in_set(PrepareSystems::Edit));
    }
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    let held = |world: &mut World| {
        world
            .query_filtered::<(), With<OtherHold>>()
            .iter(world)
            .next()
            .is_some()
    };
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<OtherHold>())
    });
    server.open();
    run_until(&mut app, &wakes, |world| !asking(world));
    let world = app.world_mut();
    assert!(held(world), "the call is still held when the answer comes");
    let holds: Vec<Entity> = world
        .query_filtered::<Entity, With<OtherHold>>()
        .iter(world)
        .collect();
    for hold in holds {
        world.despawn(hold);
    }
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let requests = sent(&model);
    assert!(
        requests.first().is_some_and(|first| asks_summary(first)),
        "the conversation is summarized before the request: {} requests (to agent: {to_agent})",
        requests.len()
    );
}

/// The connection [`route_first`] sends a turn's first call to.
#[derive(Resource)]
struct RouteTo(Connection);

/// Routes each turn's call to [`RouteTo`], unless it took over another
/// call, as a router that chooses per request does.
fn route_first(
    mut calls: Query<
        &mut ModelRequest,
        (
            Added<Admitted>,
            With<TurnCall>,
            Without<rig_ecs::stage::Supersedes>,
        ),
    >,
    to: Res<RouteTo>,
) {
    for mut call in &mut calls {
        call.connection = Some(to.0.clone());
    }
}

#[test]
fn a_call_routed_to_a_served_model_keeps_its_route_once_the_window_comes() {
    let ollama = format!(r#"{{"parameters": "num_ctx {SERVED}", "model_info": {{}}}}"#);
    let server = Server::new([MockHttpResponse::success(ollama)]);
    let routed = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the user's questions about the crate."),
        reply("done"),
    ]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &routed);
    let own = MockCompletionModel::from_stream_turns([reply("not this model")]);
    let (Some(served), Some(own_connection)) =
        (app.world().get::<Connection>(agent).cloned(), connect(&own))
    else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(own_connection);
    app.insert_resource(RouteTo(served))
        .add_systems(AgentPass, route_first.in_set(PrepareSystems::Edit));
    ask_before_the_answer(&mut app, &wakes, agent, &server);
    // Compaction's resend after the summary is a call of its own, which
    // this router does not route.
    let own = sent(&own);
    assert!(!own.iter().any(|request| asks_summary(request)), "{own:?}");
    let requests = sent(&routed);
    assert!(
        requests.first().is_some_and(|first| asks_summary(first)),
        "the routed model summarizes first: {} requests",
        requests.len()
    );
}
