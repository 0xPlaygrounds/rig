use std::collections::VecDeque;
use std::future::{Future, ready};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rig_core::catalog::{Catalog, Connector, ModelStatus};
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response,
    StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use rig_ecs::prelude::*;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{Ended, app_in, first_agent, log_ended, run_until, scratch_home};

use super::*;
use crate::models::FirstPick;

/// The ChatGPT backend as a test scripts it: each `GET /models` answers
/// the next of its lists, with its status; every other request is refused
/// as `model_not_found`. It counts both.
#[derive(Clone, Default)]
struct Backend {
    lists: Arc<Mutex<VecDeque<(u16, &'static str)>>>,
    listed: Arc<AtomicUsize>,
    sent: Arc<AtomicUsize>,
}

/// What the backend says to a model it does not serve.
const NOT_FOUND: &str = r#"{"error":{"message":"The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.","type":"invalid_request_error","code":"model_not_found"}}"#;

impl HttpClientExt for Backend {
    fn send<T, U>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        ready(Err(http_client::Error::StreamEnded))
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let listing = request.uri().path().ends_with("/models");
        let (status, body) = match listing {
            true => {
                self.listed.fetch_add(1, Ordering::SeqCst);
                let next = self
                    .lists
                    .lock()
                    .ok()
                    .and_then(|mut lists| lists.pop_front());
                next.unwrap_or((500, "no more lists"))
            }
            false => {
                self.sent.fetch_add(1, Ordering::SeqCst);
                (400, NOT_FOUND)
            }
        };
        async move {
            let (sender, chunks) = async_channel::bounded(1);
            sender
                .try_send(Ok(Bytes::from_static(body.as_bytes())))
                .ok();
            let chunks: http_client::BoxedStream = Box::pin(chunks);
            Ok(Response::builder().status(status).body(chunks)?)
        }
    }
}

/// An app signed in to ChatGPT in a scratch home named `name`, whose
/// catalog is the plan's models alone, served through `backend` once it
/// answers `lists`; its first frame ran. Returns its loop's wakes and the
/// agent.
fn signed_in(
    name: &str,
    backend: &Backend,
    lists: &[(u16, &'static str)],
) -> (App, Receiver<()>, Entity) {
    if let Ok(mut held) = backend.lists.lock() {
        held.extend(lists.iter().copied());
    }
    let home = scratch_home(name);
    let credential = r#"{"access_token":"test-token","refresh_token":"test-refresh","id_token":"test-id","expires_at":4102444800,"account_id":"test-account"}"#;
    std::fs::create_dir_all(home.auth()).ok();
    std::fs::write(home.auth().join("chatgpt.json"), credential).ok();
    let mut catalog = Catalog::default();
    let plan = Catalog::builtin()
        .iter()
        .filter(|spec| spec.provider.vendor() == PROVIDER);
    for spec in plan {
        catalog.insert(spec.clone());
    }
    let mut connector = Connector::new(catalog);
    let http = backend.clone();
    connector.wrap_http(move |_| DynHttpClient::new(http.clone()));
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(Models(connector))
        .add_plugins((AgentPlugin, LoginPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// The status of the plan's model `name`.
fn status(world: &World, name: &str) -> Option<ModelStatus> {
    let models = &world.resource::<Models>().0;
    let statuses = models.statuses().into_iter();
    statuses
        .filter(|(spec, _)| spec.id == name)
        .map(|(_, status)| status)
        .next()
}

const SOL: &str =
    r#"{"models":[{"slug":"gpt-6.1-sol","visibility":"list"},{"slug":"gpt-reserve"}]}"#;

#[test]
fn the_plans_models_are_those_the_backend_lists_for_the_account() {
    let backend = Backend::default();
    let (mut app, wakes, _) = signed_in("login-offered", &backend, &[(200, SOL)]);
    let world = app.world();
    assert_eq!(status(world, "gpt-6.1-sol"), Some(ModelStatus::Checking));
    assert_eq!(
        FirstPick::of(world.resource::<Models>()),
        FirstPick::Checking
    );
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") == Some(ModelStatus::Usable)
    });
    let world = app.world();
    let terra = status(world, "gpt-6.1-terra");
    assert!(
        matches!(terra, Some(ModelStatus::NotOffered(_))),
        "{terra:?}"
    );
    // The first row of the catalog is not offered; the pick skips it.
    let pick = FirstPick::of(world.resource::<Models>());
    assert_eq!(pick, FirstPick::Model("chatgpt/gpt-6.1-sol".to_owned()));
    let reachable: Vec<String> = (world.resource::<Models>().0.reachable().iter())
        .map(|spec| spec.reference())
        .collect();
    assert_eq!(reachable, ["chatgpt/gpt-6.1-sol"]);
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

#[test]
fn a_list_that_cannot_be_had_refuses_the_plans_models_and_says_why() {
    let backend = Backend::default();
    let (mut app, wakes, _) = signed_in("login-unlisted", &backend, &[(503, "busy")]);
    run_until(&mut app, &wakes, |world| {
        status(world, "gpt-6.1-sol") != Some(ModelStatus::Checking)
    });
    let pick = FirstPick::of(app.world().resource::<Models>());
    let said = match &pick {
        FirstPick::Nothing(Some(why)) => why.starts_with("could not ask ChatGPT which models"),
        _ => false,
    };
    assert!(said, "{pick:?}");
    assert_eq!(backend.listed.load(Ordering::SeqCst), 1);
}

#[test]
fn a_model_the_backend_did_not_find_has_the_list_asked_again_once() {
    let backend = Backend::default();
    let lists = [(200, SOL), (200, r#"{"models":[]}"#)];
    let (mut app, wakes, agent) = signed_in("login-again", &backend, &lists);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "chatgpt/gpt-6.1-sol".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    let ask = |app: &mut App, text: &str| {
        let ended = app.world().resource::<Ended>().0.len();
        let world = app.world_mut();
        world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() > ended
        });
    };
    ask(&mut app, "Hello.");
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
    run_until(&mut app, &wakes, |world| {
        matches!(
            status(world, "gpt-6.1-sol"),
            Some(ModelStatus::NotOffered(_))
        )
    });
    // The list now leaves it out: refused before anything is sent.
    ask(&mut app, "Again.");
    let ended = &app.world().resource::<Ended>().0;
    let refused = ended.last();
    assert!(
        matches!(refused, Some(TurnOutcome::Failed(why)) if why.contains("not offered to this ChatGPT account")),
        "{ended:?}"
    );
    assert_eq!(backend.sent.load(Ordering::SeqCst), 1);
    // Once in a run.
    assert_eq!(backend.listed.load(Ordering::SeqCst), 2);
}
