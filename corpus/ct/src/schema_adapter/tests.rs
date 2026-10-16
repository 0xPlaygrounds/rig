use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};

use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, connect, first_agent, log_ended, reply,
    run_until,
};
use serde_json::{Value, json};

use super::*;

/// What the rule table says: `strict-model` gets `strict-subset`,
/// `few-model` `few-tools` with three tools, `t8` always among them.
const SETTINGS: &str = r#"
[[schema_adapter.models]]
pattern = "*/strict-model"
profiles = ["strict-subset"]

[[schema_adapter.models]]
pattern = "few-model"
profiles = ["few-tools"]
max_tools = 3
always = ["t8"]
"#;

/// The keywords the strict provider refuses.
const FORBIDDEN: [&str; 3] = ["oneOf", "$ref", "format"];

/// `pick`'s parameters: a target that is a point (by `$ref`) or a name.
fn pick_schema() -> Value {
    json!({
        "type": "object",
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
                "additionalProperties": false
            }
        },
        "properties": {
            "target": {
                "description": "Where to go.",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"at": {"$ref": "#/$defs/point"}},
                        "required": ["at"],
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "properties": {"name": {"type": "string", "format": "hostname"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                ]
            }
        },
        "required": ["target"],
        "additionalProperties": false
    })
}

/// The first forbidden keyword used as a key anywhere in `value`.
fn forbidden(value: &Value) -> Option<&'static str> {
    match value {
        Value::Object(map) => FORBIDDEN
            .into_iter()
            .find(|keyword| map.contains_key(*keyword))
            .or_else(|| map.values().find_map(forbidden)),
        Value::Array(list) => list.iter().find_map(forbidden),
        _ => None,
    }
}

/// A provider that records every request and refuses one whose tools use
/// a forbidden keyword, naming the tool, as strict providers do.
#[derive(Clone, Default)]
struct StrictProvider(Arc<Mutex<Vec<CompletionRequest>>>);

impl StrictProvider {
    fn requests(&self) -> Vec<CompletionRequest> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Intercept for StrictProvider {
    fn name(&self) -> String {
        "strict-provider".to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> serve::BeforeServe {
        let EffectKind::Completion { request, .. } = kind else {
            return serve::BeforeServe::Proceed;
        };
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((**request).clone());
        let refused = request
            .tools
            .iter()
            .find_map(|tool| Some((tool.name.as_str(), forbidden(&tool.parameters)?)));
        match refused {
            Some((tool, keyword)) => serve::BeforeServe::deny(format!(
                "invalid schema for function `{tool}`: `{keyword}` is not supported"
            )),
            None => serve::BeforeServe::Proceed,
        }
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// `model` as the catalog's model `id`, behind `provider` when given.
fn served(
    model: &MockCompletionModel,
    id: &str,
    provider: Option<&StrictProvider>,
) -> Option<Connection> {
    let mut connection = connect(model)?;
    let mut spec = connection.spec.as_ref().clone();
    spec.id = id.to_owned();
    connection.spec = Arc::new(spec);
    if let Some(provider) = provider {
        connection.handler = Handler(connection.handler.0.layered(provider.clone()));
    }
    Some(connection)
}

/// The arguments each tool call got, in order.
#[derive(Resource, Default)]
struct Called(Vec<(String, Value)>);

/// The kernel and the plugin with the rule table and stand-in tools
/// `names`, each answering "ok", after the first frame.
fn start(store: &MemoryStore, names: &[&str]) -> (App, Receiver<()>, Option<Entity>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(SETTINGS))
        .add_plugins((AgentPlugin, SessionLogPlugin, SchemaAdapterPlugin))
        .init_resource::<Called>()
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    for name in names {
        let tool = (*name).to_owned();
        let answer = move |called: On<ToolCalled<Value>>,
                           mut log: ResMut<Called>,
                           mut calls: Query<&mut Decision, ToolCalls>| {
            log.0.push((tool.clone(), called.args.clone()));
            if let Ok(mut call) = calls.get_mut(called.call) {
                call.decide(Decider::plugin("test"), ToolResult::success("ok".into()));
            }
        };
        app.add_open_tool(
            name,
            "A stand-in. It does a thing.",
            ToolOptions::default(),
            answer,
        );
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    (app, wakes, agent)
}

/// Answers `agent` with `connection`, sends it a message and waits until
/// `turns` turns ended.
fn turn(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    connection: Option<Connection>,
    turns: usize,
) {
    assert!(connection.is_some(), "the catalog's model");
    if let Some(connection) = connection {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    app.world_mut()
        .trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
}

/// The parameters of the tool `name` in `request`.
fn parameters<'a>(request: &'a CompletionRequest, name: &str) -> Option<&'a Value> {
    let tool = request.tools.iter().find(|tool| tool.name.as_str() == name);
    tool.map(|tool| &tool.parameters)
}

#[test]
fn a_strict_model_gets_a_strict_schema_and_its_calls_are_mapped_back() {
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &["pick"]);
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else { return };
    let world = app.world_mut();
    let mut tools = world.query::<&mut ToolDef>();
    for mut tool in tools.iter_mut(world) {
        std::sync::Arc::make_mut(&mut tool.0).parameters = pick_schema();
    }

    // The model calls `pick` with the flattened shape, as strict models do.
    let flattened = json!({"target": {"name": "home", "at": null}});
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "pick", flattened),
        reply("Went home."),
    ]);
    let provider = StrictProvider::default();
    let strict = served(&model, "strict-model", Some(&provider));
    turn(&mut app, &wakes, agent, strict, 1);

    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
    let requests = provider.requests();
    assert_eq!(requests.len(), 2, "both requests went through");
    for request in &requests {
        let sent = parameters(request, "pick");
        assert_eq!(sent.and_then(forbidden), None, "{sent:?}");
        let target = sent.and_then(|it| it.pointer("/properties/target/properties"));
        assert!(target.is_some_and(|it| it.get("at").is_some() && it.get("name").is_some()));
    }
    // The tool got the right variant of its declared schema.
    let called = &app.world().resource::<Called>().0;
    let home = json!({"target": {"name": "home"}});
    assert_eq!(called.as_slice(), [("pick".to_owned(), home.clone())]);
    // The tool's definition and the conversation keep the declared shapes.
    let world = app.world_mut();
    let defs: Vec<Value> = world
        .query::<&ToolDef>()
        .iter(world)
        .map(|it| it.0.parameters.clone())
        .collect();
    assert_eq!(defs, [pick_schema()]);
    let conversation = world.get::<Conversation>(agent);
    let kept: Vec<Value> = conversation
        .into_iter()
        .flat_map(|it| it.messages())
        .filter_map(|message| match &**message {
            Message::Assistant(assistant) => Some(assistant.tool_calls()),
            _ => None,
        })
        .flatten()
        .map(|call| call.function.arguments_value())
        .collect();
    assert_eq!(kept, [home]);

    // What the strict model receives, explained.
    app.world_mut()
        .trigger(RunCommand::new(agent, "schema-adapter show pick"));
    app.update();
    let shown = app
        .world()
        .resource::<Collected<Notice>>()
        .0
        .iter()
        .any(|notice| {
            notice.text.contains("flattened oneOf") && notice.text.contains("inlined $ref")
        });
    assert!(shown, "{:?}", app.world().resource::<Collected<Notice>>().0);

    // On a model no rule names, the request carries the declared schema,
    // the strict provider refuses it, and that is reported once.
    let unused = MockCompletionModel::from_stream_turns([reply("unused")]);
    let plain = served(&unused, "plain-model", Some(&provider));
    turn(&mut app, &wakes, agent, plain, 2);
    let requests = provider.requests();
    let last = requests
        .last()
        .and_then(|request| parameters(request, "pick"));
    assert_eq!(last, Some(&pick_schema()));
    let reports: Vec<&Notice> = app
        .world()
        .resource::<Collected<Notice>>()
        .0
        .iter()
        .filter(|notice| {
            notice
                .text
                .contains("rejected the request over tool `pick`")
        })
        .collect();
    assert_eq!(reports.len(), 1, "{reports:?}");
}

#[test]
fn few_tools_sends_the_always_and_the_recent_ones_and_names_the_rest() {
    let store = MemoryStore::default();
    let names = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"];
    let (mut app, wakes, agent) = start(&store, &names);
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else { return };
    let model =
        MockCompletionModel::from_stream_turns([calls("c1", "t5", json!({})), reply("done")]);
    let few = served(&model, "few-model", None);
    turn(&mut app, &wakes, agent, few, 1);

    let requests = model.requests();
    let second = requests.get(1);
    let mut sent: Vec<&str> = second
        .into_iter()
        .flat_map(|request| &request.tools)
        .map(|tool| tool.name.as_str())
        .collect();
    sent.sort_unstable();
    assert_eq!(sent, ["t1", "t5", "t8"]);
    let prompt = second
        .and_then(CompletionRequest::system_instructions)
        .unwrap_or_default();
    assert!(
        prompt.contains("left out of this request: t2, t3, t4, t6, t7."),
        "{prompt}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SchemaAdapterPlugin);
}
