use std::sync::Mutex;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::completion::Usage;
use rig_core::message::{AssistantContent, Origin as Made, ToolName};
use rig_harness::clock::ClockPlugin;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, log_ended, log_tool_outputs, pass_wall_clock, run_until,
    scripted_models,
};

use super::*;

/// What the fake batch endpoint was sent and what the test released.
#[derive(Default)]
struct Endpoint {
    /// Each batch submitted, in order.
    batches: Vec<Vec<BatchItem>>,
    /// Each item's state once released, by custom id.
    released: HashMap<String, ItemState>,
    /// The batches polled, in order.
    polls: Vec<String>,
    /// The items cancelled.
    cancelled: Vec<String>,
}

/// A batch endpoint that answers an item only once the test released it.
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Endpoint>>);

impl Fake {
    fn with<T>(&self, read: impl FnOnce(&mut Endpoint) -> T) -> T {
        read(
            &mut self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    fn release(&self, custom_id: &str, state: ItemState) {
        self.with(|endpoint| endpoint.released.insert(custom_id.to_owned(), state));
    }

    /// The custom id of the item of `batch` sent for the agent `id`.
    fn item_of(&self, batch: usize, id: &str) -> String {
        self.with(|endpoint| {
            endpoint
                .batches
                .get(batch)
                .into_iter()
                .flatten()
                .find(|item| item.custom_id.starts_with(id))
                .map(|item| item.custom_id.clone())
                .unwrap_or_default()
        })
    }

    fn batches(&self) -> usize {
        self.with(|endpoint| endpoint.batches.len())
    }
}

impl BatchApi for Fake {
    fn submit(&self, items: Vec<BatchItem>) -> BoxFuture<'static, Result<String, String>> {
        let id = self.with(|endpoint| {
            endpoint.batches.push(items);
            format!("batch-{}", endpoint.batches.len())
        });
        Box::pin(async move { Ok(id) })
    }

    fn poll(&self, batch: String) -> BoxFuture<'static, Result<Vec<(String, ItemState)>, String>> {
        let states = self.with(|endpoint| {
            endpoint.polls.push(batch.clone());
            let number: usize = batch.trim_start_matches("batch-").parse().unwrap_or(0);
            let items = endpoint.batches.get(number.saturating_sub(1));
            items
                .into_iter()
                .flatten()
                .map(|item| {
                    let state = endpoint.released.get(&item.custom_id).cloned();
                    (item.custom_id.clone(), state.unwrap_or(ItemState::Pending))
                })
                .collect()
        });
        Box::pin(async move { Ok(states) })
    }

    fn cancel(&self, _: String, custom_id: String) -> BoxFuture<'static, Result<(), String>> {
        self.with(|endpoint| endpoint.cancelled.push(custom_id));
        Box::pin(async { Ok(()) })
    }
}

/// A reply of `content`, 1000 tokens in and 1000 out.
fn reply_of(content: AssistantContent) -> ItemState {
    let mut usage = Usage::new();
    usage.input_tokens = Some(1_000);
    usage.output_tokens = Some(1_000);
    let made = Made::new("anthropic", "anthropic", "test");
    let reply = CompletionResponse::new(vec![content], usage, made, serde_json::Value::Null);
    ItemState::Succeeded(Box::new(reply))
}

/// The cost of each model call that settled with a reply.
#[derive(Resource, Default)]
struct Costs(Vec<f64>);

fn log_costs(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, With<ModelRequest>>,
    mut costs: ResMut<Costs>,
) {
    let cost = calls
        .get(settled.entity)
        .ok()
        .and_then(Outcome::completion)
        .and_then(|reply| reply.usage.cost)
        .map(|cost| cost.total);
    costs.0.extend(cost);
}

/// A `read` that answers with the file's made-up text.
fn read(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success("contents of a.txt".into()),
        );
    }
}

/// The kernel and the plugin on `store`, with `fake` as the Anthropic
/// batch endpoint and `anthropic/test` at $2 and $4 per million tokens,
/// after the first frame restored the session.
fn start(store: &MemoryStore, fake: &Fake) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let mut models = scripted_models([]);
    if let Ok(anthropic) = ProviderId::declare("anthropic") {
        let spec = ModelSpec::new(anthropic, "test")
            .with_tools(true)
            .with_pricing(Pricing::new(2.0, 4.0));
        models.0.catalog_mut().insert(spec);
    }
    let api: Arc<dyn BatchApi> = Arc::new(fake.clone());
    let endpoints = BatchEndpoints(HashMap::from([("anthropic-batch".to_owned(), api)]));
    app.insert_resource(models)
        .insert_resource(endpoints)
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ClockPlugin,
            ExitPlugin,
            BatchProviderPlugin,
        ))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Costs>()
        .add_observer(log_costs);
    app.finish();
    app.update();
    (app, wakes)
}

fn agents(app: &mut App) -> Vec<(Entity, String)> {
    let world = app.world_mut();
    let mut agents: Vec<(Entity, String)> = world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .map(|(agent, id)| (agent, id.0.clone()))
        .collect();
    agents.sort_by(|a, b| a.1.cmp(&b.1));
    agents
}

/// Frames run, each after the next wait passed on the wall clock, until
/// `done` or a deadline.
fn until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&World) -> bool) {
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(10) && !done(app.world()) {
        pass_wall_clock(app.world_mut(), Duration::from_secs(400));
        app.update();
        wakes.recv_timeout(Duration::from_millis(20)).ok();
    }
}

/// The batched calls not settled yet.
fn waiting(world: &World) -> usize {
    world
        .try_query_filtered::<(), (With<Batched>, Without<Settled>)>()
        .map_or(0, |mut calls| calls.iter(world).count())
}

fn ended(world: &World) -> usize {
    world.resource::<Ended>().0.len()
}

#[test]
fn batched_turns_group_run_tools_survive_a_restart_and_resend_an_expired_item_once() {
    let store = MemoryStore::default();
    let fake = Fake::default();
    let (mut app, wakes) = start(&store, &fake);
    app.world_mut().spawn(Agent);
    app.update();
    let found = agents(&mut app);
    assert_eq!(found.len(), 2, "{found:?}");
    let [(a, a_id), (b, b_id)] = found.as_slice() else {
        return;
    };
    let (a, b) = (*a, *b);
    let b_id = b_id.clone();
    for agent in [a, b] {
        let model = "anthropic-batch/test".to_owned();
        app.world_mut().trigger(SetModel {
            entity: agent,
            model,
        });
        app.world_mut().trigger(Deliver::new(
            agent,
            "Upgrade the deps.",
            DeliveryMode::Queue,
        ));
    }
    run_until(&mut app, &wakes, |world| waiting(world) == 2);
    assert_eq!(fake.batches(), 0, "nothing is sent within the window");

    // Both requests went within the window: one batch of two items.
    until(&mut app, &wakes, |_| fake.batches() == 1);
    let sizes = fake.with(|endpoint| endpoint.batches.iter().map(Vec::len).collect::<Vec<_>>());
    assert_eq!(sizes, [2]);
    let (item_a, item_b) = (fake.item_of(0, a_id), fake.item_of(0, &b_id));
    assert!(!item_a.is_empty() && !item_b.is_empty());

    // Item a asks for `read`: it runs here, and its result goes in a new
    // batch.
    let name = ToolName::new("read");
    assert!(name.is_ok(), "a tool name");
    let Ok(name) = name else {
        return;
    };
    let args = serde_json::json!({});
    let read_call = AssistantContent::tool_call("c1", name, args);
    fake.release(&item_a, reply_of(read_call));
    until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("c1")
    });
    until(&mut app, &wakes, |_| fake.batches() == 2);
    let resent = fake.with(|endpoint| format!("{:?}", endpoint.batches.get(1)));
    assert!(resent.contains("contents of a.txt"), "{resent}");
    // Half of 1000 tokens at $2 and 1000 at $4 per million.
    let costs = app.world().resource::<Costs>().0.clone();
    assert!(
        matches!(costs.as_slice(), [cost] if (cost - 0.003).abs() < 1e-9),
        "{costs:?}"
    );

    // The harness stops while both items wait, and resumes.
    app.world_mut().trigger(Exit(ExitReason::Quit));
    app.update();
    app.update();
    drop(app);
    let polls_before = fake.with(|endpoint| endpoint.polls.len());
    let (mut app, wakes) = start(&store, &fake);
    run_until(&mut app, &wakes, |world| waiting(world) == 2);
    until(&mut app, &wakes, |_| {
        fake.with(|endpoint| endpoint.polls.len()) >= polls_before + 2
    });
    assert_eq!(fake.batches(), 2, "nothing was sent again");
    let polled = fake.with(|endpoint| endpoint.polls.get(polls_before..).map(<[String]>::to_vec));
    assert!(
        polled
            .as_deref()
            .is_some_and(|polled| polled.contains(&"batch-1".to_owned())),
        "{polled:?}"
    );

    fake.release(&item_b, reply_of(AssistantContent::text("deps upgraded")));
    until(&mut app, &wakes, |world| ended(world) == 1);
    let answer = match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => final_answer(message),
        other => Some(format!("{other:?}")),
    };
    assert_eq!(answer.as_deref(), Some("deps upgraded"));

    // Item a's second request expires: sent once more, then it fails.
    let item_a2 = fake.item_of(1, a_id);
    fake.release(&item_a2, ItemState::Expired);
    until(&mut app, &wakes, |_| fake.batches() == 3);
    assert_eq!(fake.item_of(2, a_id), item_a2, "the same request again");
    until(&mut app, &wakes, |world| ended(world) == 2);
    let outcome = format!("{:?}", app.world().resource::<Ended>().0.get(1));
    assert!(outcome.contains("expired twice"), "{outcome}");
    assert_eq!(fake.batches(), 3, "sent again only once");

    // Interrupting a waiting turn cancels its item at the provider.
    let restored = agents(&mut app);
    let b = restored
        .iter()
        .find(|(_, id)| *id == b_id)
        .map(|(agent, _)| *agent);
    let b = b.unwrap_or(Entity::PLACEHOLDER);
    app.world_mut()
        .trigger(Deliver::new(b, "And the warnings.", DeliveryMode::Queue));
    until(&mut app, &wakes, |_| fake.batches() == 4);
    let item_b2 = fake.item_of(3, &b_id);
    app.world_mut().trigger(Interrupt {
        entity: b,
        reason: None,
    });
    run_until(&mut app, &wakes, |_| {
        fake.with(|endpoint| endpoint.cancelled.contains(&item_b2))
    });
    assert!(fake.with(|endpoint| endpoint.cancelled == [item_b2.clone()]));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(BatchProviderPlugin);
}
