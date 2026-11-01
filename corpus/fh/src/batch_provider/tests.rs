use std::future::{self, Future};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rig_core::catalog::Pricing;
use rig_core::http_client::{
    self, LazyBody, MultipartForm, Response, StatusCode, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use rig_harness::clock::ClockPlugin;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, log_ended, log_tool_outputs, pass_wall_clock, run_until,
    scripted_models,
};

use super::*;

/// One item sent in a batch.
#[derive(Clone, Debug)]
struct Item {
    batch: String,
    custom_id: String,
    params: String,
}

/// What the fake Message Batches API was sent, and what the test released.
#[derive(Default)]
struct Api {
    items: Vec<Item>,
    batches: usize,
    /// Each item's result line, by batch and custom id.
    released: HashMap<(String, String), Value>,
    polls: Vec<String>,
    cancelled: Vec<String>,
}

/// An HTTP client serving the Message Batches API, which ends a batch only
/// once the test released every item of it.
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Api>>);

impl Fake {
    fn with<T>(&self, read: impl FnOnce(&mut Api) -> T) -> T {
        read(
            &mut self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Every item sent whose request holds `text`, in order.
    fn sent(&self, text: &str) -> Vec<Item> {
        self.with(|api| {
            let items = api.items.iter().filter(|item| item.params.contains(text));
            items.cloned().collect()
        })
    }

    fn items(&self) -> usize {
        self.with(|api| api.items.len())
    }

    fn release(&self, item: &Item, result: Value) {
        let key = (item.batch.clone(), item.custom_id.clone());
        self.with(|api| api.released.insert(key, result));
    }

    fn answer(&self, method: &str, url: &str, body: &[u8]) -> Value {
        self.with(|api| {
            let batch = url.rsplit('/').next().unwrap_or_default().to_owned();
            if url.ends_with("/v1/messages/batches") && method == "POST" {
                api.batches += 1;
                let id = format!("batch-{}", api.batches);
                let body: Value = serde_json::from_slice(body).unwrap_or_default();
                let requests = body.get("requests").and_then(Value::as_array);
                for request in requests.into_iter().flatten() {
                    let field = |name: &str| request.get(name).cloned().unwrap_or_default();
                    api.items.push(Item {
                        batch: id.clone(),
                        custom_id: field("custom_id").as_str().unwrap_or_default().into(),
                        params: field("params").to_string(),
                    });
                }
                json!({"id": id, "processing_status": "in_progress", "results_url": null})
            } else if url.ends_with("/cancel") {
                let batch = url.trim_end_matches("/cancel").rsplit('/').next();
                api.cancelled.extend(batch.map(str::to_owned));
                json!({})
            } else if url.starts_with("https://results/") {
                let lines = api
                    .items
                    .iter()
                    .filter(|item| item.batch == batch)
                    .map(|item| {
                        let key = (batch.clone(), item.custom_id.clone());
                        let result = api.released.get(&key).cloned().unwrap_or_default();
                        json!({"custom_id": item.custom_id, "result": result}).to_string()
                    });
                Value::String(lines.collect::<Vec<_>>().join("\n"))
            } else {
                api.polls.push(batch.clone());
                let mut items = api.items.iter().filter(|item| item.batch == batch);
                let ended = items.all(|item| {
                    let key = (batch.clone(), item.custom_id.clone());
                    api.released.contains_key(&key)
                });
                let (status, url) = match ended {
                    true => ("ended", json!(format!("https://results/{batch}"))),
                    false => ("in_progress", Value::Null),
                };
                json!({"id": batch, "processing_status": status, "results_url": url})
            }
        })
    }
}

impl HttpClientExt for Fake {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let answer = self.answer(parts.method.as_str(), &parts.uri.to_string(), &body);
        let bytes = match answer {
            Value::String(lines) => Bytes::from(lines),
            other => Bytes::from(other.to_string()),
        };
        let body: LazyBody<U> = Box::pin(async move { Ok(U::from(bytes)) });
        let response = Response::builder().status(StatusCode::OK).body(body);
        future::ready(response.map_err(http_client::Error::Protocol))
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(http_client::Error::NoHeaders))
    }

    fn send_streaming<T>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        future::ready(Err(http_client::Error::NoHeaders))
    }
}

/// A successful result: an Anthropic message of `content`, 1000 tokens in
/// and 1000 out.
fn succeeded(content: Value, stop_reason: &str) -> Value {
    let message = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "test",
        "content": [content], "stop_reason": stop_reason, "stop_sequence": null,
        "usage": {"input_tokens": 1000, "output_tokens": 1000},
    });
    json!({"type": "succeeded", "message": message})
}

fn expired() -> Value {
    json!({"type": "expired"})
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
fn read(called: On<ToolCalled<Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success("contents of a.txt".into()),
        );
    }
}

/// The kernel and the plugin on `store`, every HTTP request served by
/// `fake`, and `anthropic/test` at $2 and $4 per million tokens, after the
/// first frame restored the session.
fn start(store: &MemoryStore, fake: &Fake) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let mut models = scripted_models([]);
    if let Ok(anthropic) = ProviderId::declare("anthropic") {
        let spec = ModelSpec::new(anthropic, "test")
            .with_tools(true)
            .with_max_output_tokens(1024)
            .with_pricing(Pricing::new(2.0, 4.0));
        models.0.catalog_mut().insert(spec);
    }
    let fake = fake.clone();
    models
        .0
        .wrap_http(move |_| DynHttpClient::new(fake.clone()));
    app.insert_resource(models)
        .insert_resource(BatchAccount(AnthropicConfig::new("test-key")))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
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

fn ended(world: &World) -> String {
    format!("{:?}", world.resource::<Ended>().0)
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
    let [(a, a_id), (b, _)] = found.as_slice() else {
        return;
    };
    let (a, a_id) = (*a, a_id.clone());
    for (agent, text) in [(a, "Upgrade the deps."), (*b, "Fix the warnings.")] {
        let model = "anthropic-batch/test".to_owned();
        app.world_mut().trigger(SetModel {
            entity: agent,
            model,
            kind: ChoiceKind::ForThisRun,
        });
        let deliver = Deliver::new(agent, text, DeliveryMode::Queue);
        app.world_mut().trigger(deliver);
    }
    run_until(&mut app, &wakes, |world| waiting(world) == 2);
    assert_eq!(fake.items(), 0, "nothing is sent within the window");

    // Both requests went within the window: one batch of two items.
    until(&mut app, &wakes, |_| fake.items() == 2);
    let (deps, warnings) = (
        fake.sent("Upgrade the deps."),
        fake.sent("Fix the warnings."),
    );
    assert_eq!((deps.len(), warnings.len()), (1, 1), "both requests sent");
    let (Some(deps), Some(warnings)) = (deps.first(), warnings.first()) else {
        return;
    };
    assert_eq!(fake.with(|api| api.batches), 1);
    assert_eq!(deps.batch, warnings.batch);

    // The batch ends: a's reply asks for `read`, which runs here, and its
    // result goes in a new batch; b's item expired and is sent once more.
    let call = json!({"type": "tool_use", "id": "c1", "name": "read", "input": {}});
    fake.release(deps, succeeded(call, "tool_use"));
    fake.release(warnings, expired());
    until(&mut app, &wakes, |_| fake.items() == 4);
    assert!(app.world().resource::<ToolOutputs>().0.contains_key("c1"));
    let read_result = fake.sent("contents of a.txt");
    let resent = fake.sent("Fix the warnings.");
    assert_eq!((read_result.len(), resent.len()), (1, 2), "sent again once");
    // Half of 1000 tokens at $2 and 1000 at $4 per million.
    let costs = app.world().resource::<Costs>().0.clone();
    assert!(
        matches!(costs.as_slice(), [cost] if (cost - 0.003).abs() < 1e-9),
        "{costs:?}"
    );

    // The harness stops while both items wait, and resumes: nothing is
    // sent again, and their batches are polled.
    app.world_mut().trigger(Exit(ExitReason::Quit));
    app.update();
    app.update();
    drop(app);
    let polls_before = fake.with(|api| api.polls.len());
    let (mut app, wakes) = start(&store, &fake);
    run_until(&mut app, &wakes, |world| waiting(world) == 2);
    let (Some(read_result), Some(resent)) = (read_result.first(), resent.get(1)) else {
        return;
    };
    let polled =
        |batch: &str| fake.with(|api| api.polls.iter().skip(polls_before).any(|b| b == batch));
    until(&mut app, &wakes, |_| {
        polled(&read_result.batch) && polled(&resent.batch)
    });
    assert!(polled(&read_result.batch) && polled(&resent.batch));
    assert_eq!(fake.items(), 4, "nothing was sent again");

    // a's resumed turn goes on as any turn: its batch asks for `read`
    // again, which runs, and the result goes in a new batch; b's item
    // expires again and fails.
    let call = json!({"type": "tool_use", "id": "c2", "name": "read", "input": {}});
    fake.release(read_result, succeeded(call, "tool_use"));
    fake.release(resent, expired());
    until(&mut app, &wakes, |_| fake.items() == 5);
    assert!(app.world().resource::<ToolOutputs>().0.contains_key("c2"));
    let read_again = fake.sent("contents of a.txt");
    let Some(read_again) = read_again.last() else {
        return;
    };

    // a's turn ends with its answer.
    let answer = json!({"type": "text", "text": "deps upgraded"});
    fake.release(read_again, succeeded(answer, "end_turn"));
    until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let outcomes = ended(app.world());
    assert!(outcomes.contains("deps upgraded"), "{outcomes}");
    assert!(outcomes.contains("expired twice"), "{outcomes}");
    assert_eq!(fake.items(), 5, "sent again only once");

    // Interrupting a waiting turn cancels its batch at the provider.
    let a = agents(&mut app)
        .into_iter()
        .find(|(_, id)| *id == a_id)
        .map_or(Entity::PLACEHOLDER, |(agent, _)| agent);
    let deliver = Deliver::new(a, "And the warnings.", DeliveryMode::Queue);
    app.world_mut().trigger(deliver);
    until(&mut app, &wakes, |_| fake.items() == 6);
    let last = fake.sent("And the warnings.");
    app.world_mut().trigger(StopTurn::new(a));
    let batch = last
        .first()
        .map(|item| item.batch.clone())
        .unwrap_or_default();
    run_until(&mut app, &wakes, |_| {
        fake.with(|api| api.cancelled.contains(&batch))
    });
    assert_eq!(fake.with(|api| api.cancelled.clone()), [batch]);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(BatchProviderPlugin);
}
