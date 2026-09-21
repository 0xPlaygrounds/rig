//! `ToolResultLimit` shapes the request, never the graph (CONTRACT §8.1).
//!
//! | claim | test |
//! |---|---|
//! | no limit is verbatim: the request is byte-identical to the graph's DTOs | `no_limit_is_verbatim` |
//! | a run's limit cuts the request's tool-result text around the marker; the graph and a saved checkpoint keep the full text | `a_limit_cuts_the_request_and_keeps_the_graph_and_checkpoint_verbatim` |
//! | the cut lands on UTF-8 character boundaries | `the_cut_lands_on_character_boundaries` |
//! | a JSON item is never cut, nor an ordinary user text | `json_items_and_user_text_are_never_cut` |
//! | a limit set between turns affects only later turns; the agent's applies under the run's absence | `a_limit_change_between_turns_affects_only_later_turns` |
//! | a `RequestPartEdit` is applied first, then the limit | `an_edit_is_applied_before_the_limit` |

use crate::run_support::{graph_messages, open_model_world, requests, utterances_of};

use bevy_ecs::prelude::*;
use rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig_ecs::{
    agent::{
        MessageParts, Turn,
        content::parts::{
            ContentPart, EditTarget, RequestPartEdit, TOOL_RESULT_LIMIT_MARKER, ToolResultLimit,
        },
    },
    bus::{PendingEffect, RigSchedule},
    checkpoint::save_world,
    systems::{Fresh, RunCommands},
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
    let (mut world, agent) = open_model_world();
    let run = world.spawn_run(agent, &history(text), "next", false, None);
    let turn = world.spawn((Turn, Fresh, ChildOf(run))).id();
    (world, agent, run, turn)
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
fn a_limit_cuts_the_request_and_keeps_the_graph_and_checkpoint_verbatim() {
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

    // The graph and a checkpoint keep the full text.
    assert_eq!(graph_messages(&mut world, run), before);
    let parts: Vec<_> = world
        .query::<&ContentPart>()
        .iter(&world)
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect();
    assert!(parts.contains(&LONG.to_owned()));
    assert!(parts.iter().all(|text| !text.contains("omitted")));
    let checkpoint = save_world(&mut world).unwrap();
    let json = checkpoint.to_json().unwrap();
    assert!(json.contains(LONG));
    assert!(
        json.contains(std::any::type_name::<ToolResultLimit>()),
        "the limit is saved with the run"
    );
    // The cut text is in the pending request alone, never in the graph.
    for entity in &checkpoint.entities {
        let entity_json = serde_json::to_string(entity).unwrap();
        if entity_json.contains("26 bytes omitted") {
            assert!(
                entity.contains_key(std::any::type_name::<PendingEffect>()),
                "the cut text is only in the request: {entity_json}"
            );
        }
    }
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
    let prompt = *utterances_of(&mut world, run).last().unwrap();
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
    world.spawn((Turn, Fresh, ChildOf(run)));
    world.run_schedule(RigSchedule);
    world.entity_mut(run).insert(ToolResultLimit::new(10));
    world.spawn((Turn, Fresh, ChildOf(run)));
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
        .query::<(Entity, &ContentPart)>()
        .iter(&world)
        .find_map(|(entity, part)| matches!(part, ContentPart::ToolResult { .. }).then_some(entity))
        .unwrap();
    let text_item = world
        .get::<Children>(result)
        .unwrap()
        .iter()
        .find(|child| matches!(world.get::<ContentPart>(*child), Some(ContentPart::Text(_))))
        .unwrap();
    world.spawn((
        RequestPartEdit::Text(LONG.to_owned()),
        EditTarget(text_item),
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
    assert!(
        matches!(world.get::<ContentPart>(text_item), Some(ContentPart::Text(text)) if text.text == "short")
    );
}
