//! The real SDK model is invoked by ECS workers and restored from host resources.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod support;

use google_cloud_aiplatform_v1::client::PredictionService;
use rig_core::{
    completion::CompletionRequestBuilder,
    driver::CompletionProvider,
    effect::{EffectKind, HandlerDescriptor},
    serve::{Dispatch, ErasedHandler, Reply, Serve, adapters::CompletionAdapter},
};
use rig_ecs::{
    bus::{EffectOutcome, Handlers, PendingEffect},
    checkpoint::{RestoreMode, load_world, save_world},
};
use rig_vertexai::Client;
use support::{LocalEndpoint, Reply as HttpReply, SentinelCredentials, text_response};

/// This host chooses context-bound polling for the unary-only Vertex model.
/// The application owns/drives the runtime; the ECS task owns the RPC future.
/// This is not a general streaming SDK wrapper.
struct HostedVertex {
    handler: ErasedHandler,
    runtime: tokio::runtime::Handle,
}
impl Serve for HostedVertex {
    type Family = rig_core::effect::family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        self.handler.descriptor()
    }
    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let mut future = self.handler.handle(kind, dispatch);
        std::future::poll_fn(|cx| {
            let _entered = self.runtime.enter();
            future.as_mut().poll(cx)
        })
        .await
    }
}

async fn assemble(endpoint: &LocalEndpoint, credentials: &SentinelCredentials) -> ErasedHandler {
    let service = PredictionService::builder()
        .with_endpoint(endpoint.url())
        .with_attempt_timeout(std::time::Duration::from_secs(60))
        .with_credentials(credentials.credentials())
        .build()
        .await
        .unwrap();
    let client = Client::builder()
        .with_project("host-project")
        .with_location("global")
        .with_prediction_service(service)
        .build()
        .unwrap();
    ErasedHandler::new(HostedVertex {
        handler: ErasedHandler::new(CompletionAdapter::new(
            "vertex",
            client.completion("gemini-test"),
        )),
        runtime: tokio::runtime::Handle::current(),
    })
}

fn app() -> bevy_app::App {
    let mut app = bevy_app::App::new();
    app.add_plugins(rig_ecs::RigPlugin::default());
    app.finish();
    app.cleanup();
    app
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_vertex_requests_run_on_ecs_workers_and_strictly_resume_with_new_credentials() {
    let endpoint = LocalEndpoint::spawn([
        HttpReply::ok(text_response("first")),
        HttpReply::ok(text_response("resumed")),
    ])
    .await;
    let credentials = SentinelCredentials::rotating("ecs-original");
    let mut live = app();
    let handler = assemble(&endpoint, &credentials).await;
    Handlers::with(live.world_mut(), |h| h.register_erased("model", handler))
        .unwrap()
        .unwrap();
    let checkpoint = save_world(live.world_mut()).unwrap();
    assert!(!checkpoint.to_json().unwrap().contains("ecs-original"));
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: CompletionRequestBuilder::unbound("hello").build(),
                stream: false,
            },
        ))
        .id();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while live.world().get::<EffectOutcome>(effect).is_none() {
            live.update();
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert!(live.world().get::<EffectOutcome>(effect).unwrap().0.is_ok());
    drop(live);

    let rotated = SentinelCredentials::rotating("ecs-rotated");
    let mut resumed = app();
    checkpoint.validate(resumed.world()).unwrap();
    let handler = assemble(&endpoint, &rotated).await;
    load_world(
        &checkpoint,
        resumed.world_mut(),
        RestoreMode::Strict,
        [("model".into(), handler)],
    )
    .unwrap();
    let effect = resumed
        .world_mut()
        .spawn(PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: CompletionRequestBuilder::unbound("again").build(),
                stream: false,
            },
        ))
        .id();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while resumed.world().get::<EffectOutcome>(effect).is_none() {
            resumed.update();
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert!(
        resumed
            .world()
            .get::<EffectOutcome>(effect)
            .unwrap()
            .0
            .is_ok()
    );
    let requests = endpoint.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(credentials.issued(), 1);
    assert_eq!(rotated.issued(), 1);
    assert!(
        requests[0]
            .header("authorization")
            .unwrap()
            .contains("ecs-original")
    );
    assert!(
        requests[1]
            .header("authorization")
            .unwrap()
            .contains("ecs-rotated")
    );
    // Drop operations/handlers before the shared endpoint and host runtime.
    drop(resumed);
    drop(endpoint);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_world_cancels_the_real_sdk_rpc() {
    let endpoint = LocalEndpoint::spawn([HttpReply::Hang]).await;
    let credentials = SentinelCredentials::rotating("ecs-cancelled");
    let mut live = app();
    let handler = assemble(&endpoint, &credentials).await;
    Handlers::with(live.world_mut(), |h| h.register_erased("model", handler))
        .unwrap()
        .unwrap();
    live.world_mut().spawn(PendingEffect::new(
        "model",
        EffectKind::Completion {
            request: CompletionRequestBuilder::unbound("held").build(),
            stream: false,
        },
    ));
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while endpoint.request_count() == 0 {
            live.update();
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    drop(live);
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        endpoint.wait_for_disconnects(1),
    )
    .await
    .unwrap();
    assert_eq!(endpoint.disconnects(), 1);
}
