use std::sync::mpsc::Receiver;

use rig_basics::models::ModelsPlugin;
use rig_basics::subagents::SubagentsPlugin;
use rig_core::completion::CompletionRequest;
use rig_core::http_client::{HeaderMap, StatusCode};
use rig_core::providers::registry::ConnectOptions;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, calls, collect, first_agent, log_ended, reply, run_until,
    scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// The user's `models.toml`; line 13 holds a temperature that does not fit.
const MODELS: &str = r#"["scripted-*"]
temperature = 0.2
max_output_tokens = 4000
top_p = 0.9

["scripted-strong"]
temperature = 0.7
extra_body = { reasoning = { effort = "high" } }
headers = { "x-beta" = "1" }
system_suffix = "Answer tersely."

["scripted-broken"]
temperature = "hot"
"#;

/// An app on `store` with `home` as `RIG_HOME`, the scripted models
/// `fast` and `strong`, `/model`, `task` and the plugin, after its first
/// frame, and its agent.
fn start(
    store: &MemoryStore,
    home: &Home,
    fast: &MockCompletionModel,
    strong: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let models = scripted_models([
        ("scripted/scripted-fast", fast),
        ("scripted/scripted-strong", strong),
    ]);
    let (mut app, wakes) = app_on(store);
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, ModelsPlugin, SubagentsPlugin))
        .add_plugins(ModelParamsPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// Runs `/line` for `agent`.
fn run(app: &mut App, agent: Entity, line: &str) {
    app.world_mut()
        .trigger(RunCommand::new(agent, line.to_owned()));
    app.update();
}

/// Sends `agent` a message and waits until `turns` turns ended in all.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    let ask = Deliver::new(agent, "Go on.", DeliveryMode::Queue);
    app.world_mut().trigger(ask);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// The request's system prompt.
fn system(request: &CompletionRequest) -> &str {
    request.system_instructions().unwrap_or_default()
}

/// The request's `reasoning.effort` body field.
fn effort(request: &CompletionRequest) -> Option<&str> {
    let params = request.additional_params.as_ref()?;
    params.pointer("/reasoning/effort")?.as_str()
}

/// The last notice's text.
fn last_notice(app: &App) -> String {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .last()
        .map(|notice| notice.text.clone())
        .unwrap_or_default()
}

/// The project's file sets `top_p` over the user's.
const PROJECT: &str = r#"["scripted-*"]
top_p = 0.5
"#;

/// The file's glob and exact tables apply by model, the project's file
/// over the user's, `/model` switches them at the next request, `/params
/// set` overrides one and is back after a restart, a `task` subagent's
/// requests get the same, and a table that does not fit is reported by
/// line and left out.
#[test]
fn each_request_gets_its_models_parameters() {
    let home = scratch_home("sn-model-params");
    std::fs::create_dir_all(home.root()).ok();
    std::fs::write(home.root().join(FILE), MODELS).ok();
    let project = home.root().join("project/.rig");
    std::fs::create_dir_all(&project).ok();
    std::fs::write(project.join(FILE), PROJECT).ok();
    let store = MemoryStore::default();
    let fast = MockCompletionModel::from_stream_turns([reply("Fast.")]);
    let strong = MockCompletionModel::from_stream_turns([reply("Strong."), reply("Cold.")]);
    let (mut app, wakes, agent) = start(&store, &home, &fast, &strong);

    let problems = &app.world().resource::<Collected<Notice>>().0;
    let problem = problems
        .iter()
        .find(|notice| notice.text.contains("scripted-broken"));
    let problem = problem
        .map(|notice| notice.text.as_str())
        .unwrap_or_default();
    assert!(problem.contains("models.toml:13:"), "{problem}");
    assert!(problem.contains("temperature"), "{problem}");

    run(&mut app, agent, "model scripted/scripted-fast");
    turn(&mut app, &wakes, agent, 1);
    let sent = fast.requests();
    let first = sent.first();
    assert_eq!(first.and_then(|it| it.temperature), Some(0.2));
    assert_eq!(first.and_then(|it| it.max_tokens), Some(4000));
    assert_eq!(
        first.and_then(|it| it.options.top_p),
        Some(0.5),
        "the project's"
    );
    assert_eq!(first.and_then(effort), None, "the glob sets no body field");

    run(&mut app, agent, "model scripted/scripted-strong");
    turn(&mut app, &wakes, agent, 2);
    let sent = strong.requests();
    let first = sent.first();
    assert_eq!(first.and_then(|it| it.temperature), Some(0.7));
    assert_eq!(
        first.and_then(|it| it.max_tokens),
        Some(4000),
        "from the glob"
    );
    assert_eq!(first.and_then(effort), Some("high"));
    let prompt = first.map(system).unwrap_or_default();
    let ending = format!("<{SUFFIX_TAG}>\nAnswer tersely.\n</{SUFFIX_TAG}>");
    assert!(prompt.ends_with(&ending), "{prompt}");

    run(&mut app, agent, "params");
    let shown = last_notice(&app);
    assert!(shown.contains("temperature: 0.7 (from "), "{shown}");
    assert!(shown.contains("headers.x-beta: \"1\" (from "), "{shown}");
    assert!(shown.contains("seed: (provider default)"), "{shown}");

    run(&mut app, agent, "params set temperature 0");
    turn(&mut app, &wakes, agent, 3);
    let sent = strong.requests();
    assert_eq!(sent.get(1).and_then(|it| it.temperature), Some(0.0));
    app.update();
    drop(app);

    // A restart keeps the override; a `task` subagent on the same model
    // gets the same parameters.
    let task = json!({ "description": "Look around", "prompt": "Look around the crate." });
    let strong = MockCompletionModel::from_stream_turns([
        calls("t1", "task", task),
        reply("Done."),
        reply("Done."),
        reply("Done."),
        reply("Done."),
    ]);
    let (mut app, wakes, agent) = start(&store, &home, &fast, &strong);
    let overridden = app
        .world()
        .resource::<ParamOverrides>()
        .0
        .get("scripted/scripted-strong");
    assert_eq!(
        overridden
            .and_then(|values| values.get("temperature"))
            .map(String::as_str),
        Some("0.0")
    );
    app.world_mut()
        .trigger(Deliver::new(agent, "Look.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        let subagent_done = world.resource::<Ended>().0.len() >= 2;
        let mut running = world.try_query::<&TurnOf>();
        let idle = running
            .as_mut()
            .is_some_and(|turns| turns.iter(world).next().is_none());
        subagent_done && idle && strong.requests().len() >= 3
    });
    let sent = strong.requests();
    let briefed = |request: &&CompletionRequest| {
        format!("{:?}", request.chat_history).contains("Look around the crate.")
    };
    let by_subagent = sent.iter().filter(briefed).count();
    assert!(
        by_subagent >= 1,
        "the subagent asked the model: {}",
        sent.len()
    );
    for request in &sent {
        assert_eq!(request.temperature, Some(0.0));
        assert_eq!(request.max_tokens, Some(4000));
        assert_eq!(effort(request), Some("high"));
        assert!(system(request).ends_with(&ending), "{}", system(request));
    }
}

/// An Anthropic model, whose wire has no `seed`.
const ANTHROPIC: &str = "anthropic/claude-sonnet-4-5";

/// What the test's HTTP client saw: each request's headers and body.
#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<(HeaderMap, Value)>>>);

/// An HTTP client that keeps each request and refuses it.
struct Refusing(Seen);

impl Refusing {
    fn refuse<T>(&self, request: Request<Bytes>) -> http_client::Result<T> {
        let body = serde_json::from_slice(request.body()).unwrap_or_default();
        let kept = (request.headers().clone(), body);
        self.0
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(kept);
        Err(http_client::Error::InvalidStatusCodeWithDetails {
            status: StatusCode::BAD_REQUEST,
            body: "refused by the test".to_owned(),
            headers: HeaderMap::new(),
        })
    }
}

impl HttpClientExt for Refusing {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(self.refuse(req.map(Into::into)))
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        std::future::ready(self.refuse(req.map(Into::into)))
    }
}

/// Through a real provider's wire, a model's headers reach the HTTP
/// request and its `extra_body` is deep-merged into the body; a `seed`
/// Anthropic's wire does not take is left out, warned of once.
#[test]
fn headers_and_extra_body_reach_the_wire() {
    let home = scratch_home("sn-model-params-wire");
    std::fs::create_dir_all(home.root()).ok();
    let file = r#"["deepseek/*"]
headers = { "x-beta" = "1" }
extra_body = { thinking = { type = "enabled" } }

["anthropic/*"]
seed = 7
"#;
    std::fs::write(home.root().join(FILE), file).ok();
    let seen = Seen::default();
    let mut models = Models::default();
    let kept = seen.clone();
    models
        .0
        .wrap_http(move |_| DynHttpClient::new(Refusing(kept.clone())));
    for model in ["deepseek/deepseek-flash", ANTHROPIC] {
        let vendor = models.0.catalog().resolve(model);
        if let Ok(found) = vendor.map(|found| found.spec.provider) {
            models
                .0
                .provide(found, ConnectOptions::new().api_key("sk-test"));
        }
    }
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .insert_resource(models)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ModelsPlugin,
            ModelParamsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    run(&mut app, agent, "model deepseek/deepseek-flash");
    turn(&mut app, &wakes, agent, 1);
    run(&mut app, agent, &format!("model {ANTHROPIC}"));
    turn(&mut app, &wakes, agent, 2);
    turn(&mut app, &wakes, agent, 3);

    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned = notices
        .iter()
        .filter(|notice| notice.text.contains("does not take seed"));
    let warned: Vec<_> = warned.map(|notice| notice.text.as_str()).collect();
    assert_eq!(warned.len(), 1, "{warned:?}");
    assert!(
        warned.iter().all(|text| text.contains(ANTHROPIC)),
        "{warned:?}"
    );
    let seen = seen.0.lock().unwrap_or_else(PoisonError::into_inner);
    let to_anthropic: Vec<_> = seen
        .iter()
        .filter(|(_, body)| {
            body.get("model")
                .and_then(Value::as_str)
                .is_some_and(|m| m.starts_with("claude"))
        })
        .collect();
    assert!(
        to_anthropic.len() >= 2,
        "both turns reached the wire: {}",
        seen.len()
    );
    assert!(
        to_anthropic
            .iter()
            .all(|(_, body)| body.get("seed").is_none())
    );
    let first = seen.first();
    let beta = first.and_then(|(headers, _)| headers.get("x-beta")?.to_str().ok());
    assert_eq!(beta, Some("1"), "{} requests", seen.len());
    let thinking = first.and_then(|(_, body)| body.pointer("/thinking/type")?.as_str());
    assert_eq!(thinking, Some("enabled"));
    let streams = first.and_then(|(_, body)| body.get("stream")?.as_bool());
    assert_eq!(streams, Some(true), "the wire's own fields stay");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ModelParamsPlugin);
}
