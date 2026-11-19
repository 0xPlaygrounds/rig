use std::sync::atomic::{AtomicBool, Ordering};

use rig_core::http_client::{
    self, Bytes, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;

use super::*;

/// A transport whose every request waits forever, saying when one started and when its future
/// was dropped.
#[derive(Clone, Default)]
struct PendingHttp {
    started: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

/// Sets its flag when dropped.
struct Dropped(Arc<AtomicBool>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl PendingHttp {
    fn hang<T>(&self) -> impl Future<Output = T> + WasmCompatSend + 'static {
        let started = self.started.clone();
        let dropped = self.dropped.clone();
        async move {
            let _guard = Dropped(dropped);
            started.store(true, Ordering::SeqCst);
            std::future::pending().await
        }
    }
}

impl HttpClientExt for PendingHttp {
    fn send(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody>>> + WasmCompatSend + 'static
    {
        self.hang()
    }

    fn send_multipart(
        &self,
        _request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody>>> + WasmCompatSend + 'static
    {
        self.hang()
    }

    fn send_streaming(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        self.hang()
    }
}

/// The app's agent on the catalog's Claude model with a `window`-token context window, through
/// the real Anthropic wire over `http`.
fn connect_anthropic(app: &mut App, http: &PendingHttp, window: u32) -> Option<Entity> {
    let served = rig_core::providers::anthropic::AnthropicConfig::new("synthetic-key")
        .connect(http.clone())
        .completion("claude-opus-4-6");
    let spec = rig_core::catalog::Catalog::builtin()
        .resolve("anthropic/claude-opus-4-6")
        .ok()?
        .shared();
    let spec = Arc::new(spec.as_ref().clone().with_context_window(window));
    let handler = rig_core::serve::adapters::ModelAdapter::<rig_core::operation::Completion>::new(
        spec.reference(),
        served,
    );
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(Connection {
        spec,
        handler: rig_ecs::effects::Handler(rig_core::serve::ErasedHandler::new(handler)),
    });
    Some(agent)
}

/// Esc while a summary near the window waits on the network drops its request, releases its
/// hold on the turn's call and summarizes nothing; the session reopened holds the same messages.
#[test]
fn stopping_a_pending_network_summary_drops_the_future_and_releases_its_hold() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start_with(&store, 1_000, |_| {});
    let http = PendingHttp::default();
    let agent = connect_anthropic(&mut app, &http, 40_000);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let long = "word ".repeat(4_000);
    for turn in 0..6 {
        let question = Message::user(format!("question {turn}"));
        let answer = Message::assistant(format!("{turn}: {long}"));
        for message in [question, answer] {
            assert!(commit_message(app.world_mut(), agent, message).is_ok());
        }
    }
    let original = entries(app.world(), agent);
    let next = "Continue without restating the facts.";
    app.world_mut()
        .trigger(Deliver::new(agent, next, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| http.started.load(Ordering::SeqCst));
    assert!(!http.dropped.load(Ordering::SeqCst));
    assert!(
        app.world()
            .iter_entities()
            .any(|entity| entity.contains::<Summarizing>() && entity.contains::<HoldOf>())
    );

    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        http.dropped.load(Ordering::SeqCst)
            && !world.resource::<Ended>().0.is_empty()
            && !world
                .iter_entities()
                .any(|entity| entity.contains::<Summarizing>() || entity.contains::<HoldOf>())
    });
    assert!(
        stopped_by_user(&app),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    let stopped = entries(app.world(), agent);
    assert_eq!(stopped.get(..original.len()), Some(original.as_slice()));
    assert!(!stopped.iter().any(holds_summary), "{stopped:?}");
    drop(app);

    let (mut resumed, _) = start_with(&store, 1_000, |_| {});
    let world = resumed.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "one agent was restored");
    let Ok(agent) = agent else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), stopped);
}
