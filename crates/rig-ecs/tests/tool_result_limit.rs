//! `ToolResultLimit` shapes the request, never the graph (CONTRACT §8.1).
//!
//! | claim | test |
//! |---|---|
//! | no limit is verbatim: the request is byte-identical to the graph's DTOs | `no_limit_is_verbatim` |
//! | a run's limit cuts the request's tool-result text around the marker; the graph and a saved scene keep the full text | `a_limit_cuts_the_request_and_keeps_the_graph_and_scene_verbatim` |
//! | the cut lands on UTF-8 character boundaries | `the_cut_lands_on_character_boundaries` |
//! | a JSON item is never cut, nor an ordinary user text | `json_items_and_user_text_are_never_cut` |
//! | a limit set between turns affects only later turns; the agent's applies under the run's absence | `a_limit_change_between_turns_affects_only_later_turns` |
//! | a `RequestPartEdit` is applied first, then the limit | `an_edit_is_applied_before_the_limit` |

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor},
    message::{AssistantContent, Message, ToolResultContent, UserContent},
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::{
        MessageParts, Order, Owner, Turn, UsesModel, Utterance,
        content::parts::{
            EditTarget, RequestPartEdit, TOOL_RESULT_LIMIT_MARKER, TextPart, ToolResultLimit,
            ToolResultPart, read_message,
        },
        scene::RunScene,
    },
    bus::{Bus, Handlers, PendingEffect, RigSchedule},
    systems::{Fresh, RunCommands, install_agent},
};

const LONG: &str = "0123456789abcdefghijklmnopqrstuvwxyz";

fn history(text: &str) -> Vec<MessageParts> {
    vec![
        MessageParts::Assistant {
            id: None,
            content: vec![AssistantContent::tool_call(
                "c1",
                "probe",
                serde_json::json!({}),
            )],
        },
        MessageParts::User {
            content: vec![UserContent::ToolResult(rig_core::message::ToolResult {
                call: rig_core::message::ToolCallId::new("c1").unwrap(),
                provider: None,
                name: "probe".to_owned(),
                content: vec![
                    ToolResultContent::text(text),
                    ToolResultContent::Json {
                        value: serde_json::json!({"payload": LONG}),
                    },
                ],
            })],
        },
    ]
}

/// A world with an open model, one agent, and a run whose history holds a
/// tool result of one text and one JSON item, with a fresh turn.
fn fixture(text: &str) -> (World, Entity, Entity, Entity) {
    let mut world = World::new();
    Bus::with_policy(ServingPolicy::default()).install(&mut world);
    install_agent(&mut world);
    let model = Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "model",
            FamilyDescriptor::Completion {
                model: ModelRef::new("model"),
                capabilities: ProviderCapabilities::default(),
            },
        )
    })
    .unwrap()
    .unwrap();
    let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
    let run = world.spawn_run(agent, &history(text), "next", false, None);
    let turn = world.spawn((Turn, Fresh, Order(100), ChildOf(run))).id();
    (world, agent, run, turn)
}

/// The utterances of `run` in order, as the DTOs the graph reconstructs.
fn graph_messages(world: &mut World, run: Entity) -> Vec<Message> {
    let mut utterances: Vec<_> = world
        .query_filtered::<(Entity, &ChildOf, &Order), With<Utterance>>()
        .iter(world)
        .filter(|(_, parent, _)| parent.parent() == run)
        .map(|(entity, _, order)| (order.0, entity))
        .collect();
    utterances.sort();
    utterances
        .into_iter()
        .map(|(_, entity)| read_message(world, entity).unwrap().to_message())
        .collect()
}

/// The completion requests folded under `run`, by turn order.
fn requests(world: &mut World, run: Entity) -> Vec<rig_core::completion::CompletionRequest> {
    let mut found: Vec<_> = world
        .query::<(&PendingEffect, &ChildOf)>()
        .iter(world)
        .filter_map(|(effect, parent)| match &effect.kind {
            EffectKind::Completion { request, .. } => {
                let turn = parent.parent();
                (world.get::<ChildOf>(turn)?.parent() == run)
                    .then(|| (world.get::<Order>(turn).map(|o| o.0), request.clone()))
            }
            _ => None,
        })
        .collect();
    found.sort_by_key(|(order, _)| *order);
    found.into_iter().map(|(_, request)| request).collect()
}

fn result_items(message: &Message) -> Vec<ToolResultContent> {
    let Message::User { content } = message else {
        panic!("a user message: {message:?}");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("a tool result: {content:?}");
    };
    result.content.to_vec()
}

fn text_of(item: &ToolResultContent) -> &str {
    let ToolResultContent::Text(text) = item else {
        panic!("a text item: {item:?}");
    };
    &text.text
}

#[test]
fn no_limit_is_verbatim() {
    let (mut world, _, run, _) = fixture(LONG);
    let before = graph_messages(&mut world, run);
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].chat_history, before);
    assert_eq!(
        serde_json::to_vec(&requests[0].chat_history).unwrap(),
        serde_json::to_vec(&before).unwrap(),
        "byte-identical to the graph's DTOs"
    );
    assert_eq!(graph_messages(&mut world, run), before);
}

#[test]
fn a_limit_cuts_the_request_and_keeps_the_graph_and_scene_verbatim() {
    let (mut world, _, run, _) = fixture(LONG);
    let before = graph_messages(&mut world, run);
    world.entity_mut(run).insert(ToolResultLimit::new(10));
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    assert_eq!(requests.len(), 1);
    let items = result_items(&requests[0].chat_history[1]);
    let expected = format!(
        "01234{}vwxyz",
        TOOL_RESULT_LIMIT_MARKER.replace("{omitted}", "26")
    );
    assert_eq!(text_of(&items[0]), expected);
    assert_eq!(
        items[1],
        ToolResultContent::Json {
            value: serde_json::json!({"payload": LONG})
        },
        "the JSON item is verbatim"
    );
    // Everything but the cut text is what the graph holds.
    let mut cut = requests[0].chat_history.clone();
    let Message::User { content } = &mut cut[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &mut content[0] else {
        panic!("the result");
    };
    result.content = history(LONG)
        .pop()
        .map(|parts| match parts.to_message() {
            Message::User { content } => match content.into_iter().next().unwrap() {
                UserContent::ToolResult(result) => result.content,
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        })
        .unwrap();
    assert_eq!(cut, before);

    // The graph and a scene keep the full text.
    assert_eq!(graph_messages(&mut world, run), before);
    let parts: Vec<_> = world
        .query::<&TextPart>()
        .iter(&world)
        .map(|part| part.0.text.clone())
        .collect();
    assert!(parts.contains(&LONG.to_owned()));
    assert!(parts.iter().all(|text| !text.contains("omitted")));
    let scene = serde_json::to_string(&RunScene::save(&mut world).unwrap()).unwrap();
    assert!(scene.contains(LONG));
    assert!(
        !scene.contains("26 bytes omitted"),
        "the cut text is nowhere in the scene"
    );
    assert!(
        scene.contains("\"tool_result_limit\""),
        "the limit is saved with the run"
    );
}

#[test]
fn the_cut_lands_on_character_boundaries() {
    // Two-byte characters: no byte offset up to the limit is a boundary
    // for the odd half, and the tail start rounds forward.
    let text = "éèêëàâäîïôöùûüÿ".repeat(4);
    assert!(text.len() > 20);
    let (mut world, _, run, _) = fixture(&text);
    world.entity_mut(run).insert(ToolResultLimit {
        max_bytes: 7,
        marker: "<{omitted}>".to_owned(),
    });
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    let items = result_items(&requests[0].chat_history[1]);
    let cut = text_of(&items[0]);
    // head: 3 bytes floors to "é" (2); tail: 4 bytes is a boundary ("üÿ").
    let omitted = text.len() - 2 - 4;
    assert_eq!(cut, format!("é<{omitted}>üÿ"));
    assert!(std::str::from_utf8(cut.as_bytes()).is_ok());
    assert!(text.starts_with(&cut[..2]) && text.ends_with(&cut[cut.len() - 4..]));
}

#[test]
fn json_items_and_user_text_are_never_cut() {
    let (mut world, _, run, _) = fixture("short");
    // The prompt's own text is long: an ordinary user text, not a result.
    let prompt = world
        .query_filtered::<(Entity, &ChildOf, &Order), With<Utterance>>()
        .iter(&world)
        .filter(|(_, parent, _)| parent.parent() == run)
        .max_by_key(|(_, _, order)| order.0)
        .map(|(entity, _, _)| entity)
        .unwrap();
    rig_ecs::agent::content::parts::write_message(
        &mut world,
        prompt,
        MessageParts::User {
            content: vec![UserContent::text(LONG)],
        },
    )
    .unwrap();
    let before = graph_messages(&mut world, run);
    world.entity_mut(run).insert(ToolResultLimit::new(8));
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    assert_eq!(
        requests[0].chat_history, before,
        "a short text, a long JSON item and a long user text: nothing to cut"
    );
}

#[test]
fn a_limit_change_between_turns_affects_only_later_turns() {
    let (mut world, agent, run, _) = fixture(LONG);
    let before = graph_messages(&mut world, run);
    world.run_schedule(RigSchedule);
    // The agent's limit applies to the second turn; the run's, once set,
    // to the third. The first request, already folded, is untouched.
    world.entity_mut(agent).insert(ToolResultLimit::new(20));
    world.spawn((Turn, Fresh, Order(101), ChildOf(run)));
    world.run_schedule(RigSchedule);
    world.entity_mut(run).insert(ToolResultLimit::new(10));
    world.spawn((Turn, Fresh, Order(102), ChildOf(run)));
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].chat_history, before,
        "verbatim before any limit"
    );
    let texts: Vec<String> = requests
        .iter()
        .map(|request| text_of(&result_items(&request.chat_history[1])[0]).to_owned())
        .collect();
    assert_eq!(texts[0], LONG);
    assert_eq!(
        texts[1],
        format!(
            "0123456789{}qrstuvwxyz",
            TOOL_RESULT_LIMIT_MARKER.replace("{omitted}", "16")
        ),
        "the agent's limit"
    );
    assert_eq!(
        texts[2],
        format!(
            "01234{}vwxyz",
            TOOL_RESULT_LIMIT_MARKER.replace("{omitted}", "26")
        ),
        "the run's limit over the agent's"
    );
    assert_eq!(graph_messages(&mut world, run), before);
}

#[test]
fn an_edit_is_applied_before_the_limit() {
    let (mut world, _, run, turn) = fixture("short");
    let before = graph_messages(&mut world, run);
    let result = world
        .query::<(Entity, &ToolResultPart)>()
        .iter(&world)
        .map(|(entity, _)| entity)
        .next()
        .unwrap();
    let text_item = world
        .get::<Children>(result)
        .unwrap()
        .iter()
        .find(|child| world.get::<TextPart>(*child).is_some())
        .unwrap();
    world.spawn((
        RequestPartEdit::Text(LONG.to_owned()),
        EditTarget(text_item),
        Order(0),
        ChildOf(turn),
    ));
    world.entity_mut(run).insert(ToolResultLimit::new(10));
    world.run_schedule(RigSchedule);
    let requests = requests(&mut world, run);
    assert_eq!(requests.len(), 1);
    let items = result_items(&requests[0].chat_history[1]);
    assert_eq!(
        text_of(&items[0]),
        format!(
            "01234{}vwxyz",
            TOOL_RESULT_LIMIT_MARKER.replace("{omitted}", "26")
        ),
        "the edited text is what the limit cuts"
    );
    assert_eq!(graph_messages(&mut world, run), before);
    assert_eq!(world.get::<TextPart>(text_item).unwrap().0.text, "short");
}
