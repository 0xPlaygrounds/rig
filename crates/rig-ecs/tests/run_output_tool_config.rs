//! Custom output tools through the public native schedule and persistence APIs.
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{
        Failed, Failure, Grant, MaxTurns, Output, OutputKind, OutputRetries, OutputToolConfig,
        RunResult, Settled,
    },
    systems::RunCommands,
};
use run_support::*;

const MODEL: &str = "custom/model";
fn config() -> OutputToolConfig {
    OutputToolConfig {
        name: Some("submit".into()),
        description: Some("Extract this data.".into()),
        augment_preamble: false,
    }
}
fn schema() -> Output {
    Output {
        mode: OutputKind::Tool,
        schema: Some(serde_json::json!({
            "type":"object", "properties":{"answer":{"type":"integer"}}, "required":["answer"]
        })),
    }
}
fn settle(app: &mut bevy_app::App, run: Entity) {
    tick_until(app, "output tool settlement", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(
        app.world().get::<Failed>(run).is_none(),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    assert_eq!(
        app.world().get::<RunResult>(run).unwrap().0,
        "{\"answer\":42}"
    );
}

#[test]
fn output_tool_history_preserves_reasoning_and_commits_arguments_as_text() {
    use rig_core::message::Reasoning;
    use rig_ecs::agent::{MessageParts, Utterance};

    let reasoning = AssistantContent::Reasoning(Reasoning::new("private reasoning")).with_native(
        serde_json::json!({"id": "reasoning-id", "signature": "signature", "encrypted_content": "encrypted"}),
    );
    let mut app = app();
    let (agent, _) = scripted_agent(
        &mut app,
        MODEL,
        vec![vec![
            reasoning.clone(),
            call("c", "submit", serde_json::json!({"answer":42})),
        ]],
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, Some(1));
    settle(&mut app, run);

    let assistant: Vec<_> = app
        .world_mut()
        .query_filtered::<(Entity, &ChildOf), With<Utterance>>()
        .iter(app.world())
        .filter_map(|(entity, parent)| {
            match rig_ecs::agent::content::parts::read_message(app.world(), entity)
                .expect("valid assistant graph")
            {
                MessageParts::Assistant(rig_core::message::AssistantMessage {
                    content, ..
                }) if parent.parent() == run => Some(content.clone()),
                _ => None,
            }
        })
        .collect();
    assert_eq!(
        assistant,
        [vec![reasoning, AssistantContent::text("{\"answer\":42}")]]
    );
}

#[test]
fn run_configuration_overrides_and_can_reset_the_agent_configuration() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![vec![call(
            "c",
            "final_result",
            serde_json::json!({"answer":42}),
        )]],
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    app.world_mut()
        .entity_mut(run)
        .insert(OutputToolConfig::default());
    settle(&mut app, run);
    let requests = requests.lock().unwrap();
    assert_eq!(requests[0].tools[0].name, "final_result");
    assert_eq!(
        requests[0].tools[0].description,
        rig_core::structured_output::OUTPUT_TOOL_DESCRIPTION
    );
    assert_eq!(
        requests[0].system_instructions().unwrap(),
        format!(
            "You are terse.\n\n{}",
            rig_core::structured_output::output_tool_augmentation("final_result")
        )
    );
    assert_eq!(app.world().get::<OutputToolConfig>(agent), Some(&config()));
}

#[test]
fn reserved_name_collision_fails_before_provider_or_tool_dispatch() {
    let mut app = app();
    let (agent, requests) = scripted_agent(&mut app, MODEL, vec![]);
    let tool = register(
        &mut app,
        "real-submit",
        NeverCalled {
            name: "submit".into(),
        },
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    tick_until(&mut app, "collision refusal", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert_eq!(
        app.world().get::<Failed>(run),
        Some(&Failed(Failure::OutputToolCollision {
            name: "submit".into()
        }))
    );
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(
        app.world_mut()
            .query::<&rig_ecs::bus::PendingEffect>()
            .iter(app.world())
            .count(),
        0
    );
}

#[test]
fn a_committed_name_survives_later_configuration_changes() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![
            vec![AssistantContent::text("Use the tool next")],
            vec![call("c", "submit", serde_json::json!({"answer":42}))],
        ],
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config(), MaxTurns(2)));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    let observed = requests.clone();
    app.add_systems(
        rig_ecs::bus::RigSchedule,
        (move |retries: Query<&OutputRetries>, mut configs: Query<&mut OutputToolConfig>| {
            let mut config = configs.get_mut(agent).unwrap();
            if retries.get(run).is_ok_and(|retries| retries.0 == 1)
                && config.name.as_deref() == Some("submit")
            {
                assert_eq!(observed.lock().unwrap().len(), 1);
                config.name = Some("renamed".into());
            }
        })
        .after(rig_ecs::systems::RigSet::Select)
        .before(rig_ecs::systems::RigSet::Assemble),
    );
    settle(&mut app, run);
    assert_eq!(
        app.world()
            .get::<OutputToolConfig>(agent)
            .unwrap()
            .name
            .as_deref(),
        Some("renamed")
    );
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.tools[0].name == "submit")
    );
}

#[test]
fn output_configuration_without_a_schema_does_not_create_a_tool() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "plain");
    app.world_mut().entity_mut(agent).insert(config());
    let run = app.world_mut().spawn_run(agent, &[], "go", false, None);
    tick_until(&mut app, "plain settlement", |world| {
        world.get::<Settled>(run).is_some()
    });
    assert!(requests.lock().unwrap()[0].tools.is_empty());
    assert_eq!(app.world().get::<RunResult>(run).unwrap().0, "plain");
}

#[test]
fn an_output_tool_call_whose_arguments_are_not_an_object_fails_the_run() {
    use rig_core::message::{ToolCall, ToolFunction, ToolName};
    let malformed = AssistantContent::ToolCall(ToolCall::from_wire(
        "c",
        ToolFunction::parse(ToolName::new("submit").unwrap(), "{\"answer\":"),
    ));
    let mut app = app();
    let (agent, _) = scripted_agent(&mut app, MODEL, vec![vec![malformed]]);
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, Some(1));
    tick_until(&mut app, "output tool settlement", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(app.world().get::<RunResult>(run).is_none());
    match app.world().get::<Failed>(run) {
        Some(Failed(Failure::Provider(report))) => {
            assert!(report.to_string().contains("not a JSON object"), "{report}");
        }
        other => panic!("the run did not fail on the arguments: {other:?}"),
    }
}

#[test]
fn an_output_tool_call_whose_arguments_are_not_an_object_is_reprompted() {
    use rig_core::message::{ToolCall, ToolFunction, ToolName};
    let malformed = AssistantContent::ToolCall(ToolCall::from_wire(
        "c",
        ToolFunction::parse(ToolName::new("submit").unwrap(), "{\"answer\":"),
    ));
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![
            vec![malformed],
            vec![call("d", "submit", serde_json::json!({"answer":42}))],
        ],
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, Some(2));
    settle(&mut app, run);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let retry = serde_json::to_string(&requests[1].chat_history).unwrap();
    assert!(retry.contains("not a JSON object"), "{retry}");
}
