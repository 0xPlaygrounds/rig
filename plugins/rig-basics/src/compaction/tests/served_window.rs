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

#[test]
fn a_server_that_does_not_say_leaves_the_window_unknown() {
    let server = Server::new([]);
    server.open();
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    ask(&mut app, &wakes, agent);
    let limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.input_limit());
    assert_eq!(limit, None);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert!(!sent(&model).iter().any(|request| asks_summary(request)));
    assert_eq!(server.asked().len(), 1, "{:?}", server.asked());
}

#[test]
fn a_server_that_never_answers_is_given_up_on() {
    let server = Server::new([]);
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let (mut app, wakes, agent) = served("ollama/qwen3", &server, &model);
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    run_until(&mut app, &wakes, |world| !asking(world));
    let limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.input_limit());
    assert_eq!(limit, None);
    assert_eq!(server.asked().len(), 1, "{:?}", server.asked());
}
