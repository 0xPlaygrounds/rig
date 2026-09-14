//! A tool result's status is graph data beside `ToolResultPart` (CONTRACT
//! §8.1): the batch lands it, a scene keeps it, and the DTO never shows it.
//!
//! | claim | test |
//! |---|---|
//! | every outcome the batch distinguishes lands as its status, in call order, and `read_message` is what the model saw | `every_outcome_lands_as_its_status_and_the_dto_is_unchanged` |
//! | a skipped invalid call and its peers land `Skipped` | `an_invalid_call_skipped_by_a_system_lands_skipped_results` |
//! | a scene round-trips the status, and refuses one off a tool-result part | `a_scene_keeps_the_status_and_refuses_it_off_a_result_part` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

use crate::run_support;

use std::{collections::BTreeMap, sync::Mutex};

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message},
    serve::{Dispatch, Reply, Serve},
    tool::{ToolExecutionError, ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{
        Failed, Grant, InvalidCall, InvalidCalls, Order, Resolution, Settled,
        content::parts::{ToolResultPart, ToolResultStatus, read_message},
        scene::{RunScene, SceneKind},
    },
    bus::{EffectLogResource, RigSchedule},
    systems::{RigSet, RunCommands},
};
use rig_effect_log::EffectLogRecorder;
use run_support::*;

const MODEL: &str = "t/model:default";
const PROBE: &str = "t/tool:probe#0";

/// A tool whose answer is scripted per call by the call's `n` argument.
struct Probe {
    replies: Mutex<BTreeMap<i64, Result<Outcome, ErrorReport>>>,
}

impl Serve for Probe {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(PROBE),
            family: FamilyDescriptor::Tool {
                name: "probe".to_owned(),
                description: "answers as scripted".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { args, .. } = kind else {
            panic!("the probe serves tool calls");
        };
        let args: serde_json::Value = serde_json::from_str(&args).unwrap();
        let reply = self
            .replies
            .lock()
            .unwrap()
            .remove(&args["n"].as_i64().unwrap())
            .expect("one scripted reply per call");
        Reply::Outcome(reply)
    }
}

fn probe_call(n: i64) -> AssistantContent {
    call(&format!("c{n}"), "probe", serde_json::json!({"n": n}))
}

fn tooling(
    turns: Vec<Vec<AssistantContent>>,
    replies: BTreeMap<i64, Result<Outcome, ErrorReport>>,
) -> (bevy_app::App, Entity, RequestsSeen) {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, requests) = Scripted::new(MODEL, turns);
    let model = register(&mut app, MODEL, model);
    let tool = register(
        &mut app,
        PROBE,
        Probe {
            replies: Mutex::new(replies),
        },
    );
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(4));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    (app, agent, requests)
}

type RequestsSeen = std::sync::Arc<Mutex<Vec<rig_core::completion::CompletionRequest>>>;

fn ended(app: &mut bevy_app::App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
}

/// The run's tool-result parts in utterance and sibling order, with their
/// statuses and the utterance each belongs to.
fn results(world: &mut World) -> Vec<(Entity, String, Option<ToolResultStatus>)> {
    let mut found: Vec<_> = world
        .query::<(&ChildOf, &Order, &ToolResultPart, Option<&ToolResultStatus>)>()
        .iter(world)
        .map(|(parent, order, part, status)| {
            let utterance = parent.parent();
            let utterance_order = world.get::<Order>(utterance).map(|o| o.0).unwrap_or(0);
            (
                (utterance_order, order.0),
                (utterance, part.name.clone(), status.copied()),
            )
        })
        .collect();
    found.sort_by_key(|(key, _)| *key);
    found.into_iter().map(|(_, value)| value).collect()
}

#[test]
fn every_outcome_lands_as_its_status_and_the_dto_is_unchanged() {
    let replies: BTreeMap<i64, Result<Outcome, ErrorReport>> = [
        (
            1,
            Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("fine")),
            }),
        ),
        (
            2,
            Ok(Outcome::ToolResult {
                result: ToolResult::failed(ToolExecutionError::other("broken")),
            }),
        ),
        (
            3,
            Ok(Outcome::ToolResult {
                result: ToolResult::failed(ToolExecutionError::refused("not for you")),
            }),
        ),
        (
            4,
            Ok(Outcome::ToolResult {
                result: ToolResult::skipped("skipped by policy"),
            }),
        ),
        (5, Err(ErrorReport::new(ErrorKind::Denied, "not today"))),
        (
            6,
            Ok(Outcome::Custom {
                payload: serde_json::json!({"not": "a tool result"}),
            }),
        ),
        (
            7,
            Err(ErrorReport::new(ErrorKind::Internal, "a layer broke")),
        ),
    ]
    .into_iter()
    .collect();
    let (mut app, agent, requests) = tooling(
        vec![
            (1..=7).map(probe_call).collect(),
            vec![AssistantContent::text("done")],
        ],
        replies,
    );
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "probe everything", false, None);
    ended(&mut app, run, "answered");
    assert!(app.world().get::<Settled>(run).is_some());

    let landed = results(app.world_mut());
    let statuses: Vec<_> = landed.iter().map(|(_, _, status)| *status).collect();
    assert_eq!(
        statuses,
        [
            Some(ToolResultStatus::Ok),
            Some(ToolResultStatus::Error),
            Some(ToolResultStatus::Refused),
            Some(ToolResultStatus::Skipped),
            Some(ToolResultStatus::Denied),
            Some(ToolResultStatus::WrongFamily),
            Some(ToolResultStatus::Error),
        ],
        "one status per call, in call order"
    );
    let utterance = landed[0].0;
    assert!(landed.iter().all(|(owner, _, _)| *owner == utterance));

    // The DTO the graph reconstructs is exactly what the model saw: the
    // status is graph data, not a content item.
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let seen = &requests[1].chat_history;
    let Message::User { content } = &seen[3] else {
        panic!("the results utterance: {seen:?}");
    };
    assert_eq!(content.len(), 7);
    assert_eq!(
        read_message(app.world(), utterance).unwrap().to_message(),
        seen[3]
    );
}

fn skip_invalid_calls(
    invalid: Query<Entity, (With<InvalidCall>, Without<Resolution>)>,
    mut commands: Commands,
) {
    for entity in &invalid {
        commands.entity(entity).insert(Resolution::Skip {
            reason: "no such tool; skipped".to_owned(),
        });
    }
}

#[test]
fn an_invalid_call_skipped_by_a_system_lands_skipped_results() {
    let (mut app, agent, requests) = tooling(
        vec![
            vec![call("c1", "nope", serde_json::json!({})), probe_call(2)],
            vec![AssistantContent::text("done")],
        ],
        BTreeMap::new(),
    );
    app.world_mut().entity_mut(agent).insert(InvalidCalls {
        retries: 1,
        ..Default::default()
    });
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, skip_invalid_calls.in_set(RigSet::Judge));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "call nothing", false, None);
    ended(&mut app, run, "answered");
    assert!(app.world().get::<Settled>(run).is_some());
    let landed = results(app.world_mut());
    assert_eq!(
        landed
            .iter()
            .map(|(_, name, status)| (name.as_str(), *status))
            .collect::<Vec<_>>(),
        [
            ("nope", Some(ToolResultStatus::Skipped)),
            ("probe", Some(ToolResultStatus::Skipped)),
        ],
        "nothing ran: the skipped call and its peer"
    );
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[test]
fn a_scene_keeps_the_status_and_refuses_it_off_a_result_part() {
    let replies: BTreeMap<i64, Result<Outcome, ErrorReport>> = [
        (
            1,
            Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("fine")),
            }),
        ),
        (
            2,
            Ok(Outcome::ToolResult {
                result: ToolResult::failed(ToolExecutionError::refused("no")),
            }),
        ),
    ]
    .into_iter()
    .collect();
    let (mut app, agent, _) = tooling(
        vec![
            vec![probe_call(1), probe_call(2)],
            vec![AssistantContent::text("done")],
        ],
        replies,
    );
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "probe twice", false, None);
    ended(&mut app, run, "answered");
    let before = results(app.world_mut());
    let statuses = |found: &[(Entity, String, Option<ToolResultStatus>)]| {
        found
            .iter()
            .map(|(_, _, status)| *status)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        statuses(&before),
        [Some(ToolResultStatus::Ok), Some(ToolResultStatus::Refused)]
    );
    let original = read_message(app.world(), before[0].0).unwrap();

    let scene = RunScene::save(app.world_mut()).unwrap();
    let encoded = serde_json::to_string(&scene).unwrap();
    assert_eq!(encoded.matches("\"tool_result_status\"").count(), 2);
    let scene: RunScene = serde_json::from_str(&encoded).unwrap();
    // Loaded beside the original, into the world its handlers are bound in.
    let loaded = scene.load(app.world_mut()).unwrap();
    let after: Vec<_> = results(app.world_mut())
        .into_iter()
        .filter(|(utterance, _, _)| loaded.contains(utterance))
        .collect();
    assert_eq!(after.len(), 2);
    assert_eq!(
        statuses(&after),
        statuses(&before),
        "the scene keeps the status"
    );
    let utterance = after[0].0;
    assert!(!before.iter().any(|(owner, _, _)| *owner == utterance));
    assert_eq!(read_message(app.world(), utterance).unwrap(), original);

    // A status on a part that is not a tool result is refused before the
    // destination changes.
    let mut misplaced = scene.clone();
    let text = misplaced
        .entities
        .iter_mut()
        .find(|entity| {
            entity.kind == SceneKind::ContentPart && entity.components.contains_key("text_part")
        })
        .unwrap();
    text.components
        .insert("tool_result_status".into(), serde_json::json!("ok"));
    let count = app.world().entities().len();
    let error = misplaced.load(app.world_mut()).unwrap_err();
    assert!(
        error.message.contains("tool result status"),
        "{}",
        error.message
    );
    assert_eq!(app.world().entities().len(), count);
}
