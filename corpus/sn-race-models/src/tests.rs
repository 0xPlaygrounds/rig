use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};

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

/// A scripted model that starts answering `delay` after each request.
struct Paced {
    inner: ErasedHandler,
    delay: Duration,
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

async fn sleep(by: Duration) {
    let (done, wait) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(by);
        done.send(()).ok();
    });
    wait.await.ok();
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
        let mut waiting = Waiting(self.seen.clone(), true);
        sleep(self.delay).await;
        waiting.1 = false;
        self.inner.handle(kind, dispatch).await
    }
}

/// The catalog with A and B answered by their scripts after their delays,
/// both priced, and what each saw.
fn members(
    a: (&MockCompletionModel, u64),
    b: (&MockCompletionModel, u64),
) -> (Models, Shared, Shared) {
    let mut connector = Connector::new(Catalog::builtin().clone());
    let mut shared = Vec::new();
    for (reference, (model, delay)) in [(A, a), (B, b)] {
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
        let seen = Shared::default();
        shared.push(seen.clone());
        // Named as its member, as a provider's reply is.
        let model = model.clone().priced_as(spec);
        let delay = Duration::from_millis(delay);
        connector.provide(
            vendor,
            move |spec: &ModelSpec,
                  _: &Catalog,
                  _: &Transport|
                  -> Result<ErasedHandler, ConnectError> {
                let inner = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
                let inner = ErasedHandler::new(inner);
                let seen = seen.clone();
                Ok(ErasedHandler::new(Paced { inner, delay, seen }))
            },
        );
    }
    let b = shared.pop().unwrap_or_default();
    let a = shared.pop().unwrap_or_default();
    (Models(connector), a, b)
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

/// An app whose agent uses `race-fast`, raced over A and B.
fn race(models: Models) -> (App, Receiver<()>, Entity) {
    let mut races = RaceModels::default();
    races
        .models
        .insert("race-fast".to_owned(), vec![A.to_owned(), B.to_owned()]);
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
fn the_faster_member_answers_and_the_slower_is_cancelled_but_paid_for() {
    let a = MockCompletionModel::from_stream_turns([reply("from A"), reply("from A")]);
    let b = MockCompletionModel::from_stream_turns([
        calls("c1", "probe", serde_json::json!({})),
        reply("from B"),
    ]);
    let (models, seen_a, seen_b) = members((&a, 500), (&b, 50));
    let (mut app, wakes, agent) = race(models);
    let cancelled = seen_a.clone();
    ask(&mut app, &wakes, agent, "Go.", move || {
        seen(&cancelled).1 == 2
    });

    // Both got each request; A's two were cancelled before it streamed.
    assert_eq!(seen(&seen_a), (2, 2), "A got both requests, both cancelled");
    assert_eq!(seen(&seen_b), (2, 0), "B got both requests and answered");
    assert_eq!(a.request_count(), 0, "A never started answering");
    // B's tool call ran once, and the next request raced both again.
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
        session.total.calls, 4,
        "B's two replies and A's two cancelled requests"
    );
    let stats = &app.world().resource::<RaceStats>().0;
    let tally = stats.get("race-fast").cloned().unwrap_or_default();
    assert_eq!(tally.members.get(B).map(|b| b.wins), Some(2));
    assert_eq!(tally.cancelled, 2);
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
    let (models, _, _) = members((&a, 200), (&b, 0));
    let (mut app, wakes, agent) = race(models);

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
    let (models, seen_a, _) = members((&a, 300), (&b, 0));
    let (mut app, wakes, agent) = race(models);
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
