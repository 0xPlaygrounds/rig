use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};

use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared as Opened;

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
use rig_telemetry::UsagePlugin;

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
    Until(Opened<oneshot::Receiver<()>>),
}

/// What a scripted member saw: the requests sent to it, and those
/// cancelled before it streamed anything.
#[derive(Default, Debug)]
struct Seen {
    requests: usize,
    cancelled: usize,
}

type Shared = Arc<Mutex<Seen>>;

fn seen(shared: &Shared) -> (usize, usize) {
    let seen = shared.lock().unwrap_or_else(PoisonError::into_inner);
    (seen.requests, seen.cancelled)
}

/// A scripted model that starts answering each request when its wait says.
struct Paced {
    inner: ErasedHandler,
    waits: Arc<Mutex<VecDeque<Wait>>>,
    seen: Shared,
}

/// Counts a request dropped while it waited to answer as cancelled.
struct Waiting(Shared, bool);

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.1 {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .cancelled += 1;
        }
    }
}

impl Serve for Paced {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .requests += 1;
        let wait = self
            .waits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .unwrap_or(Wait::Now);
        let mut waiting = Waiting(self.seen.clone(), true);
        match wait {
            Wait::Now => {}
            Wait::Never => futures::future::pending().await,
            Wait::Until(gate) => {
                gate.await.ok();
            }
        }
        waiting.1 = false;
        self.inner.handle(kind, dispatch).await
    }
}

/// The catalog with each member answered by its script when its waits
/// say, all priced, and what each saw.
fn members<const N: usize>(
    scripts: [(&str, &MockCompletionModel, Vec<Wait>); N],
) -> (Models, [Shared; N]) {
    let mut connector = Connector::new(Catalog::builtin().clone());
    let shared: [Shared; N] = std::array::from_fn(|_| Shared::default());
    for ((reference, model, waits), seen) in scripts.into_iter().zip(shared.clone()) {
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
                let (waits, seen) = (waits.clone(), seen.clone());
                Ok(ErasedHandler::new(Paced { inner, waits, seen }))
            },
        );
    }
    (Models(connector), shared)
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
        .add_plugins((AgentPlugin, JournalPlugin, UsagePlugin, RaceModelsPlugin))
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
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, also: impl Fn() -> bool) {
    let ended = |world: &World| world.resource::<Ended>().0.len();
    let before = ended(app.world());
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| ended(world) > before && also());
}

/// The agent's messages as text, each with the model that wrote it.
fn stored(app: &App, agent: Entity) -> Vec<(String, Option<String>)> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .messages()
        .iter()
        .map(|message| match message {
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
    let (models, [seen_a, seen_b, seen_c]) = members([
        (A, &a, slow()),
        (B, &b, vec![Wait::Now, Wait::Now]),
        (C, &c, slow()),
    ]);
    let (mut app, wakes, agent) = race(models, &[A, B, C]);
    let (cancelled_a, cancelled_c) = (seen_a.clone(), seen_c.clone());
    ask(&mut app, &wakes, agent, "Go.", move || {
        seen(&cancelled_a).1 == 2 && seen(&cancelled_c).1 == 2
    });

    // All got each request; A's and C's were cancelled before they streamed.
    assert_eq!(seen(&seen_a), (2, 2), "A got both requests, both cancelled");
    assert_eq!(seen(&seen_c), (2, 2), "C got both requests, both cancelled");
    assert_eq!(seen(&seen_b), (2, 0), "B got both requests and answered");
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
    let session = &app.world().resource::<SessionSpending>().0;
    let spent_a = session.models.get(A).cloned().unwrap_or_default();
    assert_eq!(spent_a.calls, 2, "{session:?}");
    assert!(spent_a.tokens.input_tokens.unwrap_or_default() > 0);
    assert!(spent_a.cost > 0.0);
    assert_eq!(
        session.total.calls, 6,
        "B's two replies and A's and C's two cancelled requests each"
    );
    let stats = &app.world().resource::<RaceStats>().0;
    let tally = stats.get("race-fast").cloned().unwrap_or_default();
    assert_eq!(tally.members.get(B).map(|b| b.wins), Some(2));
    assert_eq!(tally.cancelled, 4);
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
    let (models, _) = members([
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
    ask(&mut app, &wakes, agent, "Go.", || true);
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
    ask(&mut app, &wakes, agent, "Again.", || true);
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
    let (models, [seen_a, _]) = members([
        (A, &a, vec![Wait::Never, Wait::Now]),
        (B, &b, vec![Wait::Now]),
    ]);
    let (mut app, wakes, agent) = race(models, &[A, B]);
    ask(&mut app, &wakes, agent, "Go.", || true);

    assert_eq!(seen(&seen_a), (2, 1), "A was cancelled, then asked alone");
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
    assert!(matches!(
        app.world().resource::<Ended>().0.last(),
        Some(TurnOutcome::Answered(_))
    ));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RaceModelsPlugin);
}
