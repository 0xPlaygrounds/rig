use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};

use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared;

use rig_basics::telemetry::{SessionUsage, UsagePlugin};
use rig_core::catalog::{Catalog, Connector, ModelSpec, Pricing, Transport};
use rig_core::completion::Message;
use rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::adapters::ModelAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, calls, log_ended, reply, run_until};

use super::*;

const A: &str = "provider-a/model-x";
const B: &str = "provider-b/model-y";
const C: &str = "provider-c/model-z";

/// When a scripted member starts answering a request.
#[derive(Clone)]
enum Wait {
    /// At once.
    Now,
    /// Never: it waits until it is cancelled.
    Never,
    /// When the gate opens.
    Until(Shared<oneshot::Receiver<()>>),
}

/// A scripted model that starts answering each request when its wait says.
///
/// What it was sent is read from the stack (`rig_basics::telemetry`'s spending, the
/// race's tally), never counted here: a request the kernel sent and then
/// cancelled may be dropped before its future was ever polled, so a count
/// kept by the model would depend on when the pool got to it.
struct Paced {
    inner: ErasedHandler,
    waits: Arc<Mutex<VecDeque<Wait>>>,
}

impl Serve for Paced {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let wait = self
            .waits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .unwrap_or(Wait::Now);
        match wait {
            Wait::Now => {}
            Wait::Never => futures::future::pending().await,
            Wait::Until(gate) => {
                gate.await.ok();
            }
        }
        self.inner.handle(kind, dispatch).await
    }
}

/// The catalog with each member answered by its script when its waits
/// say, all priced.
fn members<const N: usize>(scripts: [(&str, &MockCompletionModel, Vec<Wait>); N]) -> Models {
    let mut connector = Connector::new(Catalog::builtin().clone());
    for (reference, model, waits) in scripts {
        let Some((vendor, id)) = reference.split_once('/') else {
            continue;
        };
        let Ok(vendor) = ProviderId::declare(vendor) else {
            continue;
        };
        let spec = ModelSpec::new(vendor, id)
            .with_tools(true)
            .with_pricing(Pricing::new(1.0, 2.0));
        connector.catalog_mut().insert(spec.clone());
        // Named as its member, as a provider's reply is.
        let model = model.clone().priced_as(spec);
        let waits = Arc::new(Mutex::new(VecDeque::from(waits)));
        connector.provide(
            vendor,
            move |spec: &ModelSpec,
                  _: &Catalog,
                  _: &Transport|
                  -> Result<ErasedHandler, ConnectError> {
                let inner = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
                let inner = ErasedHandler::new(inner);
                let waits = waits.clone();
                Ok(ErasedHandler::new(Paced { inner, waits }))
            },
        );
    }
    Models(connector)
}

/// How often `probe` ran.
#[derive(Resource, Default)]
struct Probes(usize);

/// Every notice's text.
#[derive(Resource, Default)]
struct Notices(Vec<String>);

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn probed(
    called: On<ToolCalled<NoArgs>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut probes: ResMut<Probes>,
) {
    probes.0 += 1;
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("probe"),
            ToolResult::success("probed".into()),
        );
    }
}

/// An app whose agent uses `race-fast`, raced over `over`.
fn race(models: Models, over: &[&str]) -> (App, Receiver<()>, Entity) {
    let mut races = RaceModels::default();
    let over = over.iter().map(|member| (*member).to_owned()).collect();
    races.models.insert("race-fast".to_owned(), over);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(models)
        .insert_resource(races)
        .add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin, RaceModelsPlugin))
        .add_open_tool("probe", "Probe.", ToolOptions::default(), probed)
        .init_resource::<Probes>()
        .init_resource::<Ended>()
        .init_resource::<Notices>()
        .add_observer(log_ended)
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut seen: ResMut<Notices>| {
                seen.0
                    .extend(notices.read().map(|notice| notice.text.clone()));
            },
        );
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .unwrap_or(Entity::PLACEHOLDER);
    world
        .entity_mut(agent)
        .insert(ModelChoice(format!("{VENDOR}/race-fast")));
    app.update();
    (app, wakes, agent)
}

/// Sends `text` and runs frames until one more turn ended and `also`.
fn ask(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    text: &str,
    also: impl Fn(&World) -> bool,
) {
    let ended = |world: &World| world.resource::<Ended>().0.len();
    let before = ended(app.world());
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| ended(world) > before && also(world));
}

/// The agent's messages as text, each with the model that wrote it.
fn stored(app: &App, agent: Entity) -> Vec<(String, Option<String>)> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .messages()
        .iter()
        .map(|message| match &**message {
            Message::Assistant(reply) => (
                format!("{:?}", reply.content),
                reply.origin.as_ref().map(|origin| format!("{origin:?}")),
            ),
            other => (format!("{other:?}"), None),
        })
        .collect()
}

fn last_reply(app: &App, agent: Entity) -> Option<(String, Option<String>)> {
    stored(app, agent)
        .into_iter()
        .rev()
        .find(|(_, origin)| origin.is_some())
}

#[test]
fn the_faster_member_answers_and_the_slower_are_cancelled_but_paid_for() {
    let a = MockCompletionModel::from_stream_turns([reply("from A"), reply("from A")]);
    let b = MockCompletionModel::from_stream_turns([
        calls("c1", "probe", serde_json::json!({})),
        reply("from B"),
    ]);
    let c = MockCompletionModel::from_stream_turns([reply("from C"), reply("from C")]);
    let slow = || vec![Wait::Never, Wait::Never];
    let models = members([
        (A, &a, slow()),
        (B, &b, vec![Wait::Now, Wait::Now]),
        (C, &c, slow()),
    ]);
    let (mut app, wakes, agent) = race(models, &[A, B, C]);
    // Until the race has counted A's and C's cancelled requests: they
    // settle once the kernel cancelled them, which may be after the turn.
    ask(&mut app, &wakes, agent, "Go.", |world| {
        tally(world).cancelled == 4
    });

    // B answered both requests; A and C were sent each and never answered.
    assert_eq!(b.request_count(), 2);
    assert_eq!(
        a.request_count() + c.request_count(),
        0,
        "A and C never answered"
    );
    // B's tool call ran once, and the next request raced all again.
    assert_eq!(app.world().resource::<Probes>().0, 1);
    let messages = stored(&app, agent);
    assert!(
        messages.iter().all(|(text, _)| !text.contains("from A")),
        "{messages:?}"
    );
    let (text, origin) = last_reply(&app, agent).unwrap_or_default();
    assert!(text.contains("from B"), "{text}");
    let origin = origin.unwrap_or_default();
    assert!(origin.contains("model-y"), "the reply names B: {origin}");
    // The cancelled requests are counted at A's price.
    let session = &app.world().resource::<SessionUsage>().0;
    let spent_a = session.models.get(A).cloned().unwrap_or_default();
    assert_eq!(spent_a.calls, 2, "{session:?}");
    assert!(spent_a.tokens.input_tokens.unwrap_or_default() > 0);
    assert!(spent_a.cost > 0.0);
    let calls_to = |model: &str| session.models.get(model).map(|spent| spent.calls);
    assert_eq!(calls_to(C), Some(2), "{session:?}");
    assert_eq!(
        session.total.calls, 6,
        "B's two replies and A's and C's two cancelled requests each"
    );
    let tally = tally(app.world());
    assert_eq!(tally.members.get(B).map(|b| b.wins), Some(2));
    assert_eq!(tally.cancelled, 4);
}

/// The race model's tally.
fn tally(world: &World) -> RaceTally {
    let stats = &world.resource::<RaceStats>().0;
    stats.get("race-fast").cloned().unwrap_or_default()
}

#[test]
fn a_member_that_fails_first_drops_out_and_both_failing_fails_once() {
    let a = MockCompletionModel::from_stream_turns([
        reply("from A"),
        vec![MockStreamEvent::error("A broke")],
    ]);
    let b = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::error("B broke")],
        vec![MockStreamEvent::error("B broke")],
    ]);
    // A answers the first request only once B has dropped out.
    let (open, gate) = oneshot::channel();
    let models = members([
        (A, &a, vec![Wait::Until(gate.shared()), Wait::Now]),
        (B, &b, vec![Wait::Now, Wait::Now]),
    ]);
    let (mut app, wakes, agent) = race(models, &[A, B]);
    let open = Mutex::new(Some(open));
    app.add_systems(Update, move |dropped: Query<(), Added<DroppedOut>>| {
        if !dropped.is_empty()
            && let Some(open) = open.lock().unwrap_or_else(PoisonError::into_inner).take()
        {
            open.send(()).ok();
        }
    });

    // B fails at once; A's reply is used, and no error is shown.
    ask(&mut app, &wakes, agent, "Go.", |_| true);
    let (text, origin) = last_reply(&app, agent).unwrap_or_default();
    assert!(text.contains("from A"), "{text}");
    let origin = origin.unwrap_or_default();
    assert!(origin.contains("model-x"), "{origin}");
    let notices = app.world().resource::<Notices>().0.clone();
    assert!(
        notices.iter().all(|text| !text.contains("broke")),
        "{notices:?}"
    );
    assert!(matches!(
        app.world().resource::<Ended>().0.last(),
        Some(TurnOutcome::Answered(_))
    ));

    // Both fail: the request fails once, listing both errors.
    ask(&mut app, &wakes, agent, "Again.", |_| true);
    let ended = &app.world().resource::<Ended>().0;
    assert_eq!(ended.len(), 2, "{ended:?}");
    let why = match ended.last() {
        Some(TurnOutcome::Failed(why)) => why.clone(),
        other => format!("not failed: {other:?}"),
    };
    assert!(why.contains("A broke") && why.contains("B broke"), "{why}");
    let notices = &app.world().resource::<Notices>().0;
    let failures: Vec<&String> = notices
        .iter()
        .filter(|text| text.contains("broke"))
        .collect();
    assert_eq!(failures.len(), 1, "one failure shown: {notices:?}");
}

#[test]
fn a_winner_that_fails_mid_stream_is_replaced_by_the_other_member_alone() {
    let a = MockCompletionModel::from_stream_turns([reply("the whole answer from A")]);
    let b = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("partial"),
        MockStreamEvent::error("cut off"),
    ]]);
    let models = members([
        (A, &a, vec![Wait::Never, Wait::Now]),
        (B, &b, vec![Wait::Now]),
    ]);
    let (mut app, wakes, agent) = race(models, &[A, B]);
    ask(&mut app, &wakes, agent, "Go.", |_| true);

    // A was cancelled, then asked alone: only that request reached it.
    assert_eq!(a.request_count(), 1, "A answered once");
    let (text, _) = last_reply(&app, agent).unwrap_or_default();
    assert!(text.contains("the whole answer from A"), "{text}");
    let messages = stored(&app, agent);
    assert!(
        messages.iter().all(|(text, _)| !text.contains("partial")),
        "{messages:?}"
    );
    let notices = &app.world().resource::<Notices>().0;
    assert!(
        notices
            .iter()
            .any(|text| text == &format!("restarting on {A}")),
        "{notices:?}"
    );
    // B's cut-off reply and A's cancelled request are paid for too.
    let session = &app.world().resource::<SessionUsage>().0;
    let calls_to = |model: &str| session.models.get(model).map(|spent| spent.calls);
    assert_eq!(
        (calls_to(A), calls_to(B)),
        (Some(2), Some(1)),
        "{session:?}"
    );
    assert!(matches!(
        app.world().resource::<Ended>().0.last(),
        Some(TurnOutcome::Answered(_))
    ));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RaceModelsPlugin);
}
