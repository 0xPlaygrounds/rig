//! Custom output tools through the public native schedule and persistence APIs.
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::message::{AssistantContent, ToolChoice};
use rig_ecs::{
    agent::{
        Failed, Failure, Grant, MaxTurns, Order, Output, OutputKind, OutputRetries,
        OutputToolConfig, OutputToolName, Run, RunResult, Settled, ToolChoiceSpec, scene::RunScene,
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
fn custom_output_tool_is_advertised_and_finalizes_without_executing_a_tool() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![vec![call("c", "submit", serde_json::json!({"answer":42}))]],
    );
    app.world_mut().entity_mut(agent).insert((
        schema(),
        config(),
        ToolChoiceSpec(Some(ToolChoice::Required)),
    ));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, Some(1));
    settle(&mut app, run);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "submit");
    assert_eq!(requests[0].tools[0].description, "Extract this data.");
    assert_eq!(requests[0].tools[0].parameters, schema().schema.unwrap());
    assert!(requests[0].output_schema.is_none());
    assert_eq!(requests[0].system_instructions(), Some("You are terse."));
    assert_eq!(
        app.world().get::<OutputToolName>(run).unwrap().0.as_deref(),
        Some("submit")
    );
    assert_eq!(
        app.world_mut()
            .query::<&rig_ecs::agent::ToolCallSlot>()
            .iter(app.world())
            .count(),
        0
    );
}

#[test]
fn output_tool_history_preserves_reasoning_and_commits_arguments_as_text() {
    use rig_core::message::{Reasoning, ReasoningContent};
    use rig_ecs::agent::{MessageParts, Utterance};

    let reasoning = AssistantContent::Reasoning(Reasoning {
        id: Some("reasoning-id".into()),
        content: vec![
            ReasoningContent::Text {
                text: "private reasoning".into(),
                signature: Some("signature".into()),
            },
            ReasoningContent::Encrypted("encrypted".into()),
        ],
    });
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
                MessageParts::Assistant { content, .. } if parent.parent() == run => {
                    Some(content.clone())
                }
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
        rig_ecs::policy::text::OUTPUT_TOOL_DESCRIPTION
    );
    assert_eq!(
        requests[0].system_instructions().unwrap(),
        format!(
            "You are terse.\n\n{}",
            rig_ecs::policy::text::output_tool_augmentation("final_result")
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
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
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
fn scene_restores_custom_configuration_in_a_fresh_world() {
    let mut original = app();
    let (agent, _) = scripted_agent(&mut original, MODEL, vec![]);
    original
        .world_mut()
        .entity_mut(agent)
        .insert((schema(), OutputToolConfig::default()));
    let run = original
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    original.world_mut().entity_mut(run).insert(config());
    let scene = RunScene::save(original.world_mut()).unwrap();
    let scene: RunScene = serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
    drop(original);
    let mut restored = app();
    let (model, requests) = Scripted::new(
        MODEL,
        vec![vec![call("c", "submit", serde_json::json!({"answer":42}))]],
    );
    register(&mut restored, MODEL, model);
    scene.load(restored.world_mut()).unwrap();
    let run = restored
        .world_mut()
        .query_filtered::<Entity, With<Run>>()
        .single(restored.world())
        .unwrap();
    assert_eq!(
        restored.world().get::<OutputToolConfig>(run),
        Some(&config())
    );
    settle(&mut restored, run);
    assert_eq!(
        requests.lock().unwrap()[0].tools[0].description,
        "Extract this data."
    );
}
#[test]
fn replay_identity_includes_each_effective_output_tool_setting() {
    let mut app = app();
    let (agent, _) = scripted_agent(&mut app, MODEL, vec![]);
    app.world_mut()
        .entity_mut(agent)
        .insert((schema(), config()));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    let inherited = rig_ecs::replay::spec_hash(app.world_mut(), run).unwrap();
    for changed in [
        OutputToolConfig {
            name: Some("different".into()),
            ..config()
        },
        OutputToolConfig {
            description: Some("different".into()),
            ..config()
        },
        OutputToolConfig {
            augment_preamble: true,
            ..config()
        },
    ] {
        app.world_mut().entity_mut(run).insert(changed);
        assert_ne!(
            rig_ecs::replay::spec_hash(app.world_mut(), run).unwrap(),
            inherited
        );
    }
    app.world_mut().entity_mut(run).insert(config());
    assert_eq!(
        rig_ecs::replay::spec_hash(app.world_mut(), run).unwrap(),
        inherited
    );
}

#[test]
fn reserved_name_commits_tool_mode_despite_native_or_auto_preference() {
    for mode in [OutputKind::Auto, OutputKind::Native] {
        let mut app = app();
        let (agent, requests) = scripted_agent(
            &mut app,
            MODEL,
            vec![vec![call("c", "submit", serde_json::json!({"answer":42}))]],
        );
        app.world_mut()
            .entity_mut(agent)
            .insert((Output { mode, ..schema() }, config()));
        let run = app
            .world_mut()
            .spawn_run(agent, &[], "extract", false, Some(1));
        settle(&mut app, run);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].output_schema.is_none());
        assert_eq!(requests[0].tools[0].name, "submit");
    }
}

#[test]
fn description_only_configuration_keeps_collision_safe_automatic_naming() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![vec![call(
            "c",
            "final_result_1",
            serde_json::json!({"answer":42}),
        )]],
    );
    let real = register(
        &mut app,
        "real-final",
        NeverCalled {
            name: "final_result".into(),
        },
    );
    app.world_mut().entity_mut(agent).insert((
        schema(),
        OutputToolConfig {
            name: None,
            ..config()
        },
    ));
    app.world_mut()
        .spawn((Grant(real), Order(0), ChildOf(agent)));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, Some(1));
    settle(&mut app, run);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 2);
    let output = requests[0]
        .tools
        .iter()
        .find(|tool| tool.name == "final_result_1")
        .unwrap();
    assert_eq!(output.description, "Extract this data.");
    assert_eq!(
        app.world().get::<OutputToolName>(run).unwrap().0.as_deref(),
        Some("final_result_1")
    );
}

/// A provider that never carries `output_schema` to the wire cannot enforce a
/// schema natively. The runtime used to read only
/// `composes_native_output_with_tools` — a question about *tools*, not about
/// schema support — resolve `Native`, and settle the run on whatever text came
/// back, reporting unvalidated prose as the structured answer. It now takes
/// the output tool, whose arguments are validated.
#[test]
fn a_provider_that_drops_the_schema_gets_the_output_tool_not_unvalidated_text() {
    use rig_core::{
        completion::{CompletionResponse, ModelRef, ProviderCapabilities, Usage},
        effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
        serve::{Dispatch, Reply, Serve},
    };
    use std::sync::{Arc, Mutex};

    /// Answers with the output tool's call when one is advertised, and with
    /// bare prose when none is.
    struct Dropper {
        advertised: Arc<Mutex<Vec<String>>>,
    }

    impl Serve for Dropper {
        type Family = rig_core::effect::family::Completion;

        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from(MODEL),
                family: FamilyDescriptor::Completion {
                    model: ModelRef::new(MODEL),
                    // The whole point: this provider drops the schema.
                    capabilities: ProviderCapabilities::default().with_native_output_schema(false),
                },
                layers: Vec::new(),
            }
        }

        async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
            let EffectKind::Completion { request, .. } = kind else {
                unreachable!("a completion handler is asked for completions")
            };
            let names: Vec<String> = request.tools.iter().map(|tool| tool.name.clone()).collect();
            self.advertised
                .lock()
                .expect("the advertised names are readable")
                .extend(names.iter().cloned());

            let content = match names.first() {
                Some(name) => vec![AssistantContent::tool_call(
                    "call-1",
                    name,
                    serde_json::json!({"answer": 42}),
                )],
                None => vec![AssistantContent::text("forty two, roughly")],
            };
            Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
                content,
                Usage::new(),
                "dropper",
            ))))
        }
    }

    let mut app = app();
    let advertised = Arc::new(Mutex::new(Vec::new()));
    let model = register(
        &mut app,
        MODEL,
        Dropper {
            advertised: Arc::clone(&advertised),
        },
    );
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut().entity_mut(agent).insert((
        Output {
            // Asking for native enforcement from a provider that cannot give
            // it is exactly the case that used to pass unvalidated text off as
            // the structured answer.
            mode: OutputKind::Native,
            schema: schema().schema,
        },
        MaxTurns(2),
    ));

    let run = app.world_mut().spawn_run(agent, &[], "answer", false, None);
    settle(&mut app, run);

    assert_eq!(
        advertised
            .lock()
            .expect("the advertised names are readable")
            .as_slice(),
        ["final_result"],
        "the run asked through an output tool rather than a schema the \
         provider discards"
    );
    assert_eq!(
        app.world()
            .get::<OutputToolName>(run)
            .and_then(|n| n.0.clone()),
        Some("final_result".to_owned()),
        "the output tool is the committed mode for the run"
    );
}
