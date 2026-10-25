use super::*;

#[derive(Resource, Default)]
struct Executed(Vec<String>);

fn restricted_run(
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
) -> (App, Receiver<()>, Lines, Sender<String>) {
    restricted_run_with(model, tools, access, |_| {})
}

fn restricted_run_with(
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
    configure: impl FnOnce(&mut App),
) -> (App, Receiver<()>, Lines, Sender<String>) {
    restricted_on(&MemoryStore::default(), model, tools, access, configure)
}

fn restricted_on(
    store: &MemoryStore,
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
    configure: impl FnOnce(&mut App),
) -> (App, Receiver<()>, Lines, Sender<String>) {
    let lines = Lines::default();
    let (input, requests) = JsonIn::channel();
    let mut flags = vec![(JSON.to_owned(), None)];
    flags.extend(tools.map(|names| (TOOLS.to_owned(), Some(names.to_owned()))));
    let invoked = Invocation {
        flags,
        ..Invocation::default()
    };
    let (mut app, wakes) = noting(store, invoked);
    app.insert_resource(JsonOut::new(lines.clone()))
        .insert_resource(requests)
        .insert_resource(scripted_models([(MODEL, model)]))
        .init_resource::<Executed>()
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin));
    for name in ["allowed", "denied"] {
        app.add_open_tool(
            name,
            "Record execution.",
            ToolOptions::default(),
            |called: On<ToolCalled<Value>>,
             mut executed: ResMut<Executed>,
             mut commands: Commands| {
                executed
                    .0
                    .push(called.tool_call.function.name.as_str().to_owned());
                commands.entity(called.call).insert(Outcome::from(
                    rig_core::tool::ToolResult::success("Executed.".into()),
                ));
            },
        );
    }
    configure(&mut app);
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{conflicts:#?}");
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.entity_mut(agent).insert(access);
    (app, wakes, lines, input)
}

#[test]
fn cli_allowlists_filter_the_first_request_and_enforce_actual_dispatch() {
    // A call of a tool no handler serves is an error, not a refusal.
    for (tools, expected, called, executes, refused) in [
        (None, vec!["allowed", "denied"], "denied", true, false),
        (Some(""), vec![], "allowed", false, true),
        (Some("allowed"), vec!["allowed"], "allowed", true, false),
        (Some("allowed"), vec!["allowed"], "denied", false, true),
        (Some("unknown"), vec![], "allowed", false, true),
        (
            Some("unknown, allowed"),
            vec!["allowed"],
            "denied",
            false,
            true,
        ),
        (None, vec!["allowed", "denied"], "missing", false, false),
    ] {
        let model =
            MockCompletionModel::from_stream_turns([calls("c1", called, json!({})), reply("done")]);
        let (mut app, wakes, lines, input) = restricted_run(&model, tools, ToolAccess::All);
        send(&input, json!({"v": 1, "type": "send", "text": "Run it."}));
        test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
        assert_eq!(model.request_count(), 2, "{tools:?}");
        let requests = model.requests();
        let mut advertised: Vec<_> = requests
            .first()
            .into_iter()
            .flat_map(|request| &request.tools)
            .map(|tool| tool.name.as_str())
            .collect();
        advertised.sort_unstable();
        assert_eq!(advertised, expected, "{tools:?}");
        let executed = &app.world().resource::<Executed>().0;
        let expected_calls: Vec<_> = executes.then_some(called).into_iter().collect();
        assert_eq!(executed, &expected_calls, "{tools:?}");
        assert_eq!(lines.of("turn_end").len(), 1, "{tools:?}");
        // A call not run is announced before its result, which says
        // whether it was refused.
        let kinds: Vec<Value> = lines
            .parsed()
            .iter()
            .map(|l| at(l, "type").clone())
            .collect();
        let call = kinds.iter().position(|kind| kind == "tool_call");
        let result = kinds.iter().position(|kind| kind == "tool_result");
        assert!(call.zip(result).is_some_and(|(c, r)| c < r), "{kinds:?}");
        let said = lines
            .of("tool_result")
            .first()
            .map(|r| at(r, "refused").clone());
        assert_eq!(said, Some(Value::Bool(refused)), "{tools:?}");
    }
}

#[cfg(feature = "tui")]
#[test]
fn actual_tui_policy_compaction_and_persistence_have_no_stage_conflicts() {
    let (mut app, _) = noting(
        &MemoryStore::default(),
        Invocation {
            flags: vec![(TOOLS.to_owned(), Some("allowed".to_owned()))],
            ..Invocation::default()
        },
    );
    app.world_mut().resource_mut::<Invoked>().terminal = true;
    app.add_plugins((
        SessionLogPlugin,
        ExitPlugin,
        rig_tui::TuiPlugin,
        PrintPlugin,
        crate::compaction::CompactionPlugin,
        crate::subagents::SubagentsPlugin,
    ));
    app.finish();
    app.cleanup();
    assert!(app.runs_front(rig_tui::FRONT));
    assert!(app.world().contains_resource::<Tools>());
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{conflicts:#?}");
}

#[test]
fn policy_and_persistence_are_ordered_for_every_front() {
    for front in ["tui", FRONT, JSON] {
        for tools in [None, Some(""), Some("allowed"), Some("unknown")] {
            let model = MockCompletionModel::from_stream_turns([
                calls("c1", "denied", json!({})),
                reply("done"),
            ]);
            let (mut app, wakes, _lines, _input) = restricted_run_with(
                &model,
                tools,
                ToolAccess::Only(vec!["denied".to_owned()]),
                |app| {
                    if front != JSON {
                        let mut invoked = app.world_mut().resource_mut::<Invoked>();
                        invoked.args.flags.retain(|(name, _)| name != JSON);
                        invoked.terminal = front == "tui";
                    }
                    app.add_front(Front::new("tui").on_terminal());
                },
            );
            let world = app.world_mut();
            let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
            if let Some(connection) = connect(&model) {
                world.entity_mut(agent).insert(connection);
            }
            world.trigger(Deliver::new(agent, "Run it.", DeliveryMode::Steer));
            test_support::run_until(&mut app, &wakes, |world| {
                world.get::<ActiveTurn>(agent).is_none() && model.request_count() == 2
            });
            let requests = model.requests();
            assert_eq!(requests.len(), 2, "{front} {tools:?}");
            let advertised: Vec<_> = requests
                .first()
                .into_iter()
                .flat_map(|request| &request.tools)
                .map(|tool| tool.name.as_str())
                .collect();
            let expected = if tools.is_none() {
                vec!["denied"]
            } else {
                vec![]
            };
            assert_eq!(advertised, expected, "{front} {tools:?}");
            assert_eq!(
                app.world().resource::<Executed>().0,
                expected,
                "{front} {tools:?}"
            );
        }
    }
}

#[test]
fn cli_allowlist_intersects_agent_specific_access_before_requests() {
    let model = answering(&["done"]);
    let (mut app, wakes, lines, input) = restricted_run(
        &model,
        Some("allowed,denied"),
        ToolAccess::Only(vec!["allowed".to_owned()]),
    );
    send(&input, json!({"v": 1, "type": "send", "text": "Run it."}));
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    let requests = model.requests();
    let advertised: Vec<_> = requests
        .first()
        .into_iter()
        .flat_map(|request| &request.tools)
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(advertised, ["allowed"]);
    assert_eq!(model.request_count(), 1);
}

#[test]
fn restored_agents_cannot_broaden_the_cli_limit() {
    let store = MemoryStore::default();
    let first = answering(&["saved"]);
    let (mut app, wakes, lines, input) = restricted_on(
        &store,
        &first,
        None,
        ToolAccess::Only(vec!["allowed".to_owned()]),
        |_| {},
    );
    send(&input, json!({"v": 1, "type": "send", "text": "Save it."}));
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let second = answering(&["restored"]);
    let (mut app, wakes) = noting(
        &store,
        Invocation {
            flags: vec![
                (JSON.to_owned(), None),
                (TOOLS.to_owned(), Some("denied".to_owned())),
            ],
            ..Invocation::default()
        },
    );
    let lines = Lines::default();
    let (input, requests) = JsonIn::channel();
    app.insert_resource(JsonOut::new(lines.clone()))
        .insert_resource(requests)
        .insert_resource(scripted_models([(MODEL, &second)]))
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin));
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{conflicts:#?}");
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    assert!(
        matches!(world.get::<ToolAccess>(agent), Some(ToolAccess::Only(names)) if names == &["allowed"])
    );
    send(
        &input,
        json!({"v": 1, "type": "send", "text": "Recall it."}),
    );
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    assert_eq!(second.request_count(), 1);
    assert!(
        matches!(app.world().get::<ToolAccess>(agent), Some(ToolAccess::Only(names)) if names.is_empty())
    );
    assert!(
        second
            .requests()
            .iter()
            .all(|request| request.tools.is_empty())
    );
}

#[test]
fn cli_allowlist_reaches_anthropic_http_output() {
    use rig_core::catalog::Catalog;
    use rig_core::operation::Completion;
    use rig_core::providers::anthropic::{AnthropicConfig, CLAUDE_SONNET_4_6};
    use rig_core::serve::ErasedHandler;
    use rig_core::serve::adapters::ModelAdapter;
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
    use rig_harness::rig_ecs::effects::Handler;

    for (tools, expected) in [("", vec![]), ("allowed", vec!["allowed"])] {
        let response = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
            response,
            "text/event-stream",
        )]);
        let model = rig_core::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            http.clone(),
        );
        let (mut app, wakes, lines, input) =
            restricted_run(&silent(), Some(tools), ToolAccess::All);
        let spec = Catalog::builtin().resolve("anthropic/claude-sonnet-4-6");
        assert!(spec.is_ok());
        let Ok(spec) = spec else { return };
        let spec = spec.shared();
        let adapter = ModelAdapter::<Completion>::new(spec.reference(), model);
        let world = app.world_mut();
        let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
        world.entity_mut(agent).insert(Connection {
            spec,
            handler: Handler(ErasedHandler::new(adapter)),
        });
        send(&input, json!({"v": 1, "type": "send", "text": "Run it."}));
        test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
        let requests = http.requests();
        assert_eq!(requests.len(), 1);
        let body = requests
            .first()
            .and_then(|request| serde_json::from_slice::<Value>(&request.body).ok())
            .unwrap_or_default();
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some(CLAUDE_SONNET_4_6)
        );
        let names: Vec<_> = body
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, expected, "{body}");
        let turns = lines.of("turn_end");
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns
                .first()
                .and_then(|turn| turn.get("answer"))
                .and_then(Value::as_str),
            Some("done")
        );
    }
}

#[test]
fn delegated_subagents_obey_inherited_and_cli_limits() {
    use crate::subagents::{SubagentPolicy, SubagentsPlugin};

    for called in ["denied", "message", "wait", "allowed"] {
        let executes = called == "allowed";
        let tools = if executes {
            "allowed,denied,task,message,wait"
        } else {
            "allowed,denied,task"
        };
        let parent_model = silent();
        let child_model = MockCompletionModel::from_stream_turns([
            calls("child-call", called, json!({})),
            reply("child done"),
        ]);
        let (mut app, wakes, lines, _input) = restricted_run_with(
            &parent_model,
            Some(tools),
            ToolAccess::Only(vec!["allowed".to_owned(), "task".to_owned()]),
            |app| {
                app.insert_resource(scripted_models([
                    (MODEL, &parent_model),
                    ("test/child", &child_model),
                ]));
                rig_harness::load::<SubagentsPlugin>(app, "rig-basics", "path plugins/rig-basics");
                app.insert_resource(SubagentPolicy { max_depth: 2 });
            },
        );
        let world = app.world_mut();
        let parent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
        let connection = connect(&parent_model);
        assert!(connection.is_some());
        if let Some(connection) = connection {
            let choice = ModelChoice(connection.spec.reference());
            world.entity_mut(parent).insert((connection, choice));
        }
        let task = test_support::start_call(
            world,
            parent,
            ("task", "delegate"),
            json!({"description": "Test child", "prompt": "Run it.", "model": "test/child"}),
        );
        assert!(task.is_some());
        test_support::run_until(&mut app, &wakes, |_| {
            lines
                .of("turn_end")
                .iter()
                .any(|turn| turn.get("answer").and_then(Value::as_str) == Some("child done"))
        });
        let requests = child_model.requests();
        assert_eq!(requests.len(), 2, "{called}");
        let mut advertised: Vec<_> = requests
            .first()
            .into_iter()
            .flat_map(|request| &request.tools)
            .map(|tool| tool.name.as_str())
            .collect();
        advertised.sort_unstable();
        let expected = if executes {
            vec!["allowed", "message", "task", "wait"]
        } else {
            vec!["allowed", "task"]
        };
        assert_eq!(advertised, expected, "{called}");
        let world = app.world_mut();
        let child = world
            .get::<Spawned>(parent)
            .and_then(|children| children.iter().next());
        assert!(child.is_some());
        let child_id = child
            .and_then(|child| world.get::<AgentId>(child))
            .map(|id| id.0.as_str());
        let expected_calls: Vec<_> = executes.then_some(called).into_iter().collect();
        assert_eq!(world.resource::<Executed>().0, expected_calls, "{called}");
        let results = lines.of("tool_result");
        assert!(
            results.iter().any(|result| {
                result.get("name").and_then(Value::as_str) == Some(called)
                    && result.get("agent").and_then(Value::as_str) == child_id
                    && result.get("ok").and_then(Value::as_bool) == Some(executes)
            }),
            "{called}"
        );
    }
}

#[test]
fn agents_spawned_during_a_pass_are_restricted_before_dispatch() {
    let mut app = App::new();
    app.insert_resource(Tools(vec!["allowed".to_owned()]));
    app.add_systems(
        AgentPass,
        (
            (|mut commands: Commands| {
                commands.spawn((Agent, ToolAccess::Only(vec!["denied".to_owned()])));
                commands.spawn(Agent);
            })
            .in_set(Stage::Commit),
            limit_tools
                .after(Stage::Commit)
                .in_set(PrepareSystems::Input),
        ),
    );
    app.world_mut().run_schedule(AgentPass);
    let world = app.world_mut();
    let accesses: Vec<_> = world
        .query::<&ToolAccess>()
        .iter(world)
        .map(|access| (access.allows("allowed"), access.allows("denied")))
        .collect();
    assert_eq!(accesses.len(), 2);
    assert!(accesses.contains(&(false, false)));
    assert!(accesses.contains(&(true, false)));
}
