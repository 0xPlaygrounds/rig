use super::*;
use rig_core::http_client::{
    self, Bytes, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Default)]
struct PendingHttp {
    started: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

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
    fn send<T, U>(
        &self,
        _request: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.hang()
    }

    fn send_multipart<U>(
        &self,
        _request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.hang()
    }

    fn send_streaming<T>(
        &self,
        _request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.hang()
    }
}

#[test]
fn stopping_a_pending_network_summary_drops_the_future_and_releases_its_hold() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let http = PendingHttp::default();
    let Some(agent) = connect_anthropic(&mut app, &http, 40_000) else {
        return;
    };
    seed(&mut app, agent, 6);
    let original = entries(app.world(), agent);
    app.world_mut().trigger(Deliver::new(
        agent,
        "Continue without restating the facts.",
        DeliveryMode::Steer,
    ));
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
    assert!(matches!(
        app.world().resource::<Ended>().0.as_slice(),
        [TurnOutcome::Stopped(stop)] if *stop.cause() == StopCause::User
    ));
    let stopped = entries(app.world(), agent);
    assert_eq!(stopped.get(..original.len()), Some(original.as_slice()));
    assert_eq!(stopped.len(), original.len() + 1);
    assert!(stopped.last().is_some_and(
        |entry| *entry.message == Message::user("Continue without restating the facts.")
    ));
    drop(app);

    let (mut resumed, _) = start(&store);
    let agent = restored_agent(&mut resumed);
    assert!(agent.is_some());
    let Some(agent) = agent else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), stopped);
}
