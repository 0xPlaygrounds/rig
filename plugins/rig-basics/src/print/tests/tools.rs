use rig_harness::run::TOOLS;

use super::*;

#[derive(Resource, Default)]
struct Executed(Vec<String>);

/// Adds the tools `names`, each recording that it ran in [`Executed`].
fn recording(app: &mut App, names: &[&str]) {
    app.init_resource::<Executed>();
    for name in names {
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
}

fn restricted_run(
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
) -> (App, Receiver<()>, Lines, Sender<RequestLine>) {
    restricted_run_with(model, tools, access, |_| {})
}

fn restricted_run_with(
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
    configure: impl FnOnce(&mut App),
) -> (App, Receiver<()>, Lines, Sender<RequestLine>) {
    restricted_on(&MemoryStore::default(), model, tools, access, configure)
}

fn restricted_on(
    store: &MemoryStore,
    model: &MockCompletionModel,
    tools: Option<&str>,
    access: ToolAccess,
    configure: impl FnOnce(&mut App),
) -> (App, Receiver<()>, Lines, Sender<RequestLine>) {
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
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin));
    recording(&mut app, &["allowed", "denied"]);
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

/// Adds the tool `late` as it finishes, after the print plugin did.
struct LatePlugin;

impl Plugin for LatePlugin {
    fn build(&self, _: &mut App) {}

    fn finish(&self, app: &mut App) {
        recording(app, &["late"]);
    }
}

#[test]
fn unknown_cli_tool_names_are_refused_before_any_model_call() {
    let listed = "--tools: unknown tool `bogus`; the tools are allowed, denied, late";
    for (tools, why) in [
        ("bogus", listed),
        ("bogus, allowed", listed),
        (
            "alowed",
            "--tools: unknown tool `alowed`; did you mean `allowed`?",
        ),
        // Two letters swapped are one edit.
        (
            "dneied",
            "--tools: unknown tool `dneied`; did you mean `denied`?",
        ),
        (
            "alowed,bogus",
            "--tools: unknown tools `alowed`, `bogus`; the tools are allowed, denied, late",
        ),
    ] {
        let model = answering(&["done"]);
        let stderr = Lines::default();
        let setup = |app: &mut App| {
            app.insert_resource(Printed::to(stderr.clone()))
                .add_plugins(LatePlugin);
            recording(app, &["allowed", "denied"]);
            let flag = (TOOLS.to_owned(), Some(tools.to_owned()));
            app.world_mut()
                .resource_mut::<Invoked>()
                .args
                .flags
                .push(flag);
        };
        let (app, lines) = one_shot("hi", Some(MODEL), &model, true, setup);
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)), "{tools}");
        let kinds: Vec<Value> = lines
            .parsed()
            .iter()
            .map(|l| at(l, "type").clone())
            .collect();
        assert_eq!(kinds, ["ready", "exit"], "{tools}");
        let exit = lines.of("exit").first().cloned().unwrap_or_default();
        assert_eq!(
            (at(&exit, "code"), at(&exit, "error")),
            (&json!(2), &json!(why))
        );
        // Plain: said once on stderr.
        let (app, _) = one_shot("hi", Some(MODEL), &model, false, setup);
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)), "{tools}");
        assert_eq!(stderr.text(), [format!("rig: {why}")]);
        // A long-lived run whose driver sent a message at once: no model
        // call either. The run may have stopped reading already: it exits
        // in its first frame.
        let configure = |app: &mut App| {
            app.add_plugins(LatePlugin);
        };
        let (mut app, wakes, lines, input) =
            restricted_run_with(&model, Some(tools), ToolAccess::All, configure);
        let line = json!({"v": 1, "type": "send", "text": "Run it."});
        input.try_send(Ok(line.to_string().into_bytes())).ok();
        test_support::run_until(&mut app, &wakes, |_| !lines.of("exit").is_empty());
        let exit = lines.of("exit").first().cloned().unwrap_or_default();
        assert_eq!(at(&exit, "code"), &json!(2), "{tools}");
        assert_eq!(model.request_count(), 0, "{tools}");
    }
}

#[test]
fn a_tool_a_later_plugin_registers_can_be_named() {
    let model =
        MockCompletionModel::from_stream_turns([calls("c1", "late", json!({})), reply("done")]);
    let (app, lines) = one_shot("Run it.", Some(MODEL), &model, true, |app| {
        app.add_plugins(LatePlugin);
        let flag = (TOOLS.to_owned(), Some("late".to_owned()));
        app.world_mut()
            .resource_mut::<Invoked>()
            .args
            .flags
            .push(flag);
    });
    let exit = lines.of("exit").first().cloned().unwrap_or_default();
    assert_eq!(at(&exit, "answer"), &json!("done"), "{exit}");
    let first = model
        .requests()
        .first()
        .map(|request| request.tools.clone());
    let offered: Vec<String> = first
        .into_iter()
        .flatten()
        .map(|tool| tool.name.to_string())
        .collect();
    assert_eq!(offered, ["late"]);
    assert_eq!(app.world().resource::<Executed>().0, ["late"]);
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
    recording(&mut app, &["allowed"]);
    app.finish();
    app.cleanup();
    assert!(app.runs_front(rig_tui::FRONT));
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{conflicts:#?}");
}

#[test]
fn policy_and_persistence_are_ordered_for_every_front() {
    for front in ["tui", FRONT, JSON] {
        for tools in [None, Some(""), Some("allowed")] {
            let model = MockCompletionModel::from_stream_turns([
                calls("c1", "denied", json!({})),
                reply("done"),
            ]);
            let (mut app, wakes, _lines, _input) =
                restricted_run_with(&model, tools, ToolAccess::All.narrow(["denied"]), |app| {
                    if front != JSON {
                        let mut invoked = app.world_mut().resource_mut::<Invoked>();
                        invoked.args.flags.retain(|(name, _)| name != JSON);
                        invoked.terminal = front == "tui";
                    }
                    app.add_front(Front::new("tui").on_terminal());
                });
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
        ToolAccess::All.narrow(["allowed"]),
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
        ToolAccess::All.narrow(["allowed"]),
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
    recording(&mut app, &["allowed", "denied"]);
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{conflicts:#?}");
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    // Kept to `--tools` as the start ended, before any request: the saved
    // `allowed` and the limit's `denied` leave nothing.
    assert!(world.get::<ToolAccess>(agent).and_then(ToolAccess::names) == Some(&[][..]));
    send(
        &input,
        json!({"v": 1, "type": "send", "text": "Recall it."}),
    );
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    assert_eq!(second.request_count(), 1);
    assert!(
        app.world()
            .get::<ToolAccess>(agent)
            .and_then(ToolAccess::names)
            == Some(&[][..])
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
            ToolAccess::All.narrow(["allowed", "task"]),
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
        // The parent has neither `message` nor `wait`, so its child gets
        // neither, whatever `--tools` allows.
        assert_eq!(advertised, ["allowed", "task"], "{called}");
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
fn a_refused_tool_name_names_itself_on_a_carried_turn_it_cuts_off() {
    // A crash left a turn, which a plugin carries on after the restart.
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let second = answering(&["hello"]);
    let refused = |app: &mut App| {
        recording(app, &["read", "write"]);
        app.add_systems(AgentPass, carry_on.in_set(Stage::Gate));
        let flag = (TOOLS.to_owned(), Some("read,bogus".to_owned()));
        app.world_mut()
            .resource_mut::<Invoked>()
            .args
            .flags
            .push(flag);
    };
    let (app, lines) = one_shot_on(&store, "", Some(MODEL), &second, true, refused);
    assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
    assert_eq!(second.request_count(), 0);
    assert!(app.world().resource::<Executed>().0.is_empty());
    // The turn the refusal cut off says why, on its line and in the session.
    let why = "--tools: unknown tool `bogus`; the tools are read, write";
    let ends = lines.of("turn_end");
    let reasons: Vec<_> = ends.iter().map(|end| at(end, "error").clone()).collect();
    assert_eq!(reasons, [json!(why)], "{:?}", lines.parsed());
    drop(app);
    let saved = rig_harness::rig_memory::Session::read(&store).ok();
    let main = saved.as_ref().and_then(|saved| saved.main_agent());
    let ended = main.and_then(|agent| {
        agent
            .conversation
            .turns()
            .last()
            .and_then(|(_, record)| record.end)
    });
    assert!(
        matches!(&ended, Some(rig_ecs::prelude::TurnEnd::Stopped { why: Some(said), .. }) if said == why),
        "{ended:?}"
    );
}

#[test]
fn a_refused_tool_name_runs_nothing_of_a_resumed_session() {
    for pending_tool in [false, true] {
        // A session that exited before its model call, or its write,
        // started.
        let first = match pending_tool {
            true => MockCompletionModel::from_stream_turns([calls("c1", "write", json!({}))]),
            false => silent(),
        };
        let store = MemoryStore::default();
        let (mut app, _) = app_on(&store);
        app.insert_resource(scripted_models([(MODEL, &first)]))
            .add_plugins((AgentPlugin, SessionLogPlugin));
        recording(&mut app, &["read", "write"]);
        if pending_tool {
            app.add_systems(
                AgentPass,
                (|calls: Query<Entity, (ReadyToStart, ToolCalls)>, mut commands: Commands| {
                    for call in &calls {
                        commands.spawn(HoldOf(call));
                    }
                })
                .in_set(Stage::Gate),
            );
        } else {
            app.add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
        }
        app.finish();
        app.cleanup();
        app.update();
        let world = app.world_mut();
        let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
        let model = MODEL.to_owned();
        world.trigger(SetModel {
            entity: agent,
            model,
            kind: ChoiceKind::ForThisRun,
        });
        world.trigger(Deliver::new(agent, "hi", DeliveryMode::Steer));
        run_until(&mut app, |world| match pending_tool {
            true => world
                .query_filtered::<(), (With<ToolRequest>, With<Holds>)>()
                .iter(world)
                .next()
                .is_some(),
            false => world.get::<ActiveTurn>(agent).is_some(),
        });
        assert_eq!(first.request_count(), usize::from(pending_tool));
        assert!(app.world().resource::<Executed>().0.is_empty());
        app.world_mut().write_message(AppExit::Success);
        app.update();
        drop(app);

        // Refusal must run neither a model nor a restored tool call; the
        // exit stopped the turn, so it has none to close.
        let second = answering(&["hello"]);
        let refused = |app: &mut App| {
            recording(app, &["read", "write"]);
            let flag = (TOOLS.to_owned(), Some("read,bogus".to_owned()));
            app.world_mut()
                .resource_mut::<Invoked>()
                .args
                .flags
                .push(flag);
        };
        let (app, lines) = one_shot_on(&store, "", Some(MODEL), &second, true, refused);
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
        assert_eq!(second.request_count(), 0);
        assert!(app.world().resource::<Executed>().0.is_empty());
        let why = "--tools: unknown tool `bogus`; the tools are read, write";
        assert!(lines.of("turn_end").is_empty(), "{:?}", lines.parsed());
        let exit = lines.of("exit").first().cloned().unwrap_or_default();
        assert_eq!(at(&exit, "error"), &json!(why));
        drop(app);

        // So must a run someone sits at, which is refused before its view
        // starts.
        let fourth = answering(&["hello"]);
        let invoked = Invocation {
            flags: vec![(TOOLS.to_owned(), Some("read,bogus".to_owned()))],
            ..Invocation::default()
        };
        let (mut app, _) = noting(&store, invoked);
        app.world_mut().resource_mut::<Invoked>().terminal = true;
        app.insert_resource(scripted_models([(MODEL, &fourth)]))
            .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin))
            .add_front(Front::new("view").on_terminal());
        recording(&mut app, &["read", "write"]);
        app.finish();
        app.cleanup();
        run_until(&mut app, |_| false);
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
        assert_eq!(fourth.request_count(), 0);
        assert!(app.world().resource::<Executed>().0.is_empty());
        drop(app);

        // So must a run whose launcher is gone while it waits for its
        // prompt on a stdin that stays open: it exits as on a signal, with
        // no turn to close.
        let fifth = answering(&["hello"]);
        let Ok((silent, writer)) = std::io::pipe() else {
            return;
        };
        let home = rig_harness::harness_protocol::Home::at(std::env::temp_dir().join(format!(
            "rig-gone-launcher-{}-{pending_tool}",
            std::process::id()
        )));
        // A run directory whose launcher holds no lock: it is gone.
        let run = home.run(1);
        std::fs::create_dir_all(run.path()).ok();
        let gone = |app: &mut App| {
            recording(app, &["read", "write"]);
            app.insert_resource(PipedIn(Box::new(silent)))
                .insert_resource(launcher::LauncherRun(run))
                .add_plugins(launcher::LauncherPlugin);
        };
        let (app, lines) = one_shot_on(&store, "", Some(MODEL), &fifth, true, gone);
        drop(writer);
        std::fs::remove_dir_all(home.root()).ok();
        assert_eq!(app.should_exit(), Some(AppExit::from_code(130)));
        assert_eq!(fifth.request_count(), 0);
        assert!(app.world().resource::<Executed>().0.is_empty());
        let exit = lines.parsed().last().cloned().unwrap_or_default();
        assert_eq!(
            (at(&exit, "type"), at(&exit, "code")),
            (&json!("exit"), &json!(130))
        );
        assert!(lines.of("turn_end").is_empty(), "{:?}", lines.parsed());
        drop(app);

        // A valid resume keeps the access and answers its own prompt; the
        // write the exit stopped does not run.
        let third = answering(&["hello"]);
        let (mut app, lines) = one_shot_on(&store, "go on", Some(MODEL), &third, true, |app| {
            recording(app, &["read", "write"]);
        });
        let exit = lines.of("exit").first().cloned().unwrap_or_default();
        assert_eq!(
            (at(&exit, "code"), at(&exit, "answer")),
            (&json!(0), &json!("hello"))
        );
        assert!(app.world().resource::<Executed>().0.is_empty());
        assert_eq!(third.request_count(), 1);
        let world = app.world_mut();
        let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
        assert!(matches!(
            world.get::<ToolAccess>(agent),
            Some(ToolAccess::All)
        ));
    }
}
