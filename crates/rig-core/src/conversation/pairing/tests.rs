use super::*;
use crate::message::{
    AssistantContent, StopReason, ToolCall, ToolFunction, ToolResult, ToolResultContent,
};

fn tool_call(id: &str) -> ToolCall {
    ToolCall {
        id: crate::message::CallId::from_wire(id),
        function: ToolFunction::new(
            crate::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
        native: None,
    }
}
fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(tool_call(id))
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        is_error: false,
        call: crate::message::CallId::from_wire(id),
        name: crate::message::ToolName::new("add").expect("tool name"),
        content: vec![ToolResultContent::text("3")],
    })
}
fn results(ids: &[&str]) -> Message {
    Message::User {
        content: ids.iter().map(|id| result(id)).collect(),
    }
}
fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(content))
}
fn id(id: &str) -> CallId {
    CallId::from_wire(id)
}
fn unanswered(index: usize, call: &str) -> PairingError {
    PairingError::UnansweredToolCall {
        index,
        call_id: id(call),
    }
}
fn orphan(index: usize, call: &str) -> PairingError {
    PairingError::OrphanToolResult {
        index,
        call_id: id(call),
    }
}
/// What `validate` says agrees with what `pair` repaired.
fn repairs(history: &[Message]) -> Vec<PairingError> {
    let repairs = repair(history.to_vec()).repairs;
    assert_eq!(
        validate(history),
        repairs.first().cloned().map_or(Ok(()), Err)
    );
    repairs
}

#[test]
fn unanswered_and_orphan_results_are_rejected() {
    let unanswered_call = vec![assistant(vec![call("c1")]), Message::user("no result")];
    assert!(matches!(
        validate(&unanswered_call),
        Err(PairingError::UnansweredToolCall { .. })
    ));
    assert_eq!(repairs(&unanswered_call), vec![unanswered(0, "c1")]);
    let orphan_result = vec![
        Message::user("hi"),
        Message::User {
            content: vec![result("ghost")],
        },
    ];
    assert!(matches!(
        validate(&orphan_result),
        Err(PairingError::OrphanToolResult { .. })
    ));
    assert_eq!(repairs(&orphan_result), vec![orphan(1, "ghost")]);
    let trailing = vec![assistant(vec![call("c1")])];
    assert!(matches!(
        validate(&trailing),
        Err(PairingError::UnansweredToolCall { .. })
    ));
    assert_eq!(repairs(&trailing), vec![unanswered(0, "c1")]);
    let consecutive = vec![Message::assistant("a"), Message::assistant("b")];
    assert_eq!(
        repairs(&consecutive),
        vec![PairingError::ConsecutiveAssistant { index: 1 }]
    );
}

#[test]
fn repeated_ids_are_answered_once_per_occurrence() {
    let calls = assistant(vec![call("c1"), call("c1"), call("c2")]);
    let answered = vec![
        Message::user("go"),
        calls.clone(),
        results(&["c2", "c1", "c1"]),
    ];
    assert_eq!(repairs(&answered), Vec::new());
    assert_eq!(repair(answered.clone()).messages, answered);

    // One result for two occurrences leaves the second unanswered.
    let short = vec![calls.clone(), results(&["c1", "c2"])];
    assert_eq!(repairs(&short), vec![unanswered(0, "c1")]);
    assert_eq!(
        repair(short).messages[1],
        Message::User {
            content: vec![
                result("c1"),
                result("c2"),
                tool_result_message(
                    id("c1"),
                    crate::message::ToolName::new("add").expect("tool name"),
                    NO_RESULT_PROVIDED.to_owned()
                ),
            ]
        }
    );

    // A result past the occurrences is an orphan, and is dropped.
    let extra = vec![calls, results(&["c1", "c1", "c1", "c2"])];
    assert_eq!(repairs(&extra), vec![orphan(1, "c1")]);
    assert_eq!(repair(extra).messages[1], results(&["c1", "c1", "c2"]));
}

#[test]
fn a_failed_turn_owes_no_results_and_its_results_are_orphans() {
    for stop in [
        StopReason::Error("boom".to_owned()),
        StopReason::Aborted("stopped".to_owned()),
    ] {
        let failed = Message::Assistant(
            crate::message::AssistantMessage::new(vec![call("c1")]).with_stop(stop),
        );
        let history = vec![Message::user("go"), failed.clone(), Message::user("again")];
        assert_eq!(repairs(&history), Vec::new());
        assert_eq!(repair(history.clone()).messages, history);

        let answered = vec![Message::user("go"), failed, results(&["c1"])];
        assert_eq!(repairs(&answered), vec![orphan(2, "c1")]);
    }
}

#[test]
fn results_may_split_around_system_and_user_messages() {
    let split = vec![
        assistant(vec![call("c1"), call("c2")]),
        results(&["c1"]),
        Message::system("context"),
        results(&["c2"]),
        Message::assistant("done"),
    ];
    assert_eq!(repairs(&split), Vec::new());
    let adjacent = vec![
        assistant(vec![call("c1"), call("c2")]),
        results(&["c1"]),
        results(&["c2"]),
    ];
    assert_eq!(repairs(&adjacent), Vec::new());

    // A user message with more than results ends the turn's results.
    let ended = vec![
        assistant(vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c1"), UserContent::text("and")],
        },
        Message::system("context"),
        results(&["c2"]),
    ];
    assert_eq!(repairs(&ended), vec![unanswered(0, "c2"), orphan(3, "c2")]);
}

#[test]
fn a_duplicate_result_for_an_answered_call_is_an_orphan() {
    let history = vec![
        assistant(vec![call("c1")]),
        results(&["c1"]),
        Message::assistant("ok"),
        results(&["c1"]),
    ];
    assert_eq!(repairs(&history), vec![orphan(3, "c1")]);
}

#[test]
fn answers_matches_a_batch_as_a_multiset() {
    let pending = [id("c1"), id("c1"), id("c2")];
    let batch = |ids: &[&str]| ids.iter().map(|id| result(id)).collect::<Vec<_>>();
    assert_eq!(answers(&pending, &batch(&["c2", "c1", "c1"])), Ok(()));
    assert_eq!(
        answers(&pending, &batch(&["c1", "c1", "c1"])),
        Err(AnswerError::Unknown(id("c1")))
    );
    assert_eq!(
        answers(&pending, &batch(&["c1"])),
        Err(AnswerError::Unanswered(vec![id("c1"), id("c2")]))
    );
    assert_eq!(
        answers(&pending, &[UserContent::text("hi")]),
        Err(AnswerError::NotAResult)
    );
}

#[test]
fn close_pending_answers_each_occurrence_in_call_order() {
    let calls = [tool_call("c1"), tool_call("c2"), tool_call("c1")];
    assert_eq!(
        close_pending(&calls),
        close_pending_with(&calls, NO_RESULT_PROVIDED)
    );
    let Message::User { content } = close_pending_with(&calls, "skipped") else {
        panic!("a closure is a user message");
    };
    let ids: Vec<CallId> = content
        .iter()
        .filter_map(|part| match part {
            UserContent::ToolResult(result) => {
                assert!(result.is_error);
                assert_eq!(result.content, vec![ToolResultContent::text("skipped")]);
                Some(result.call.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec![id("c1"), id("c2"), id("c1")]);
}

#[test]
fn repair_closes_any_history_into_a_canonical_one() {
    let broken = vec![
        Message::user("q"),
        assistant(vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c2"), result("ghost")],
        },
        assistant(vec![AssistantContent::text("a")]),
        Message::User {
            content: vec![result("ghost")],
        },
        assistant(vec![AssistantContent::text("b"), call("c3"), call("c3")]),
        Message::system("s"),
    ];
    let closed = repair(broken).messages;
    assert_eq!(validate(&closed), Ok(()), "{closed:?}");
    assert_eq!(repair(closed.clone()).messages, closed);
    let Message::User { content } = &closed[2] else {
        panic!("results follow the turn: {closed:?}");
    };
    assert!(matches!(&content[..], [_, UserContent::ToolResult(r)] if r.call == id("c1")));
    let Message::Assistant(merged) = &closed[3] else {
        panic!("the turns an orphan result parted are merged: {closed:?}");
    };
    assert_eq!(merged.content.len(), 4);
    assert_eq!(
        closed[4],
        close_pending([&tool_call("c3"), &tool_call("c3")])
    );
    assert_eq!(closed[5], Message::system("s"));
}

#[test]
fn stored_results_are_kept_once_per_id_and_an_empty_user_message_stands() {
    let paired = pair(
        vec![
            Some(results(&["s1"])),
            Some(results(&["s1"])),
            Some(Message::User {
                content: Vec::new(),
            }),
            Some(Message::user("next")),
            Some(assistant(vec![AssistantContent::text("a")])),
            None,
        ],
        Pairing {
            stored: true,
            answers: true,
        },
    );
    assert_eq!(paired.repairs, vec![orphan(1, "s1")]);
    // The empty message stands, and the user message after it is not
    // merged into it.
    assert_eq!(
        paired.messages,
        vec![
            results(&["s1"]),
            Message::User {
                content: Vec::new()
            },
            Message::user("next"),
            assistant(vec![AssistantContent::text("a")]),
        ]
    );
}

#[test]
fn pending_calls_are_the_last_turns_unanswered_calls() {
    let history = vec![
        assistant(vec![call("old")]),
        results(&["old"]),
        assistant(vec![call("c1"), call("c2"), call("c3")]),
        results(&["c2"]),
    ];
    let pending: Vec<CallId> = pending_calls(&history)
        .into_iter()
        .map(|call| call.id)
        .collect();
    assert_eq!(pending, vec![id("c1"), id("c3")]);
    assert!(pending_calls(&[Message::user("hi")]).is_empty());
    // A stopped or failed turn's calls never run, so it owes nothing.
    for stop in [
        StopReason::Aborted("stopped".into()),
        StopReason::Error("lost".into()),
    ] {
        let cut = AssistantMessage::new(vec![call("cut")]).with_stop(stop);
        assert!(pending_calls(&[Message::Assistant(cut)]).is_empty());
    }
}

#[test]
fn the_nth_result_with_an_id_answers_the_nth_call_with_it() {
    let (c1, c2) = (id("c1"), id("c2"));
    let calls = [&c1, &c2, &c1];
    let answers = [(&c1, 'a'), (&c1, 'b'), (&c2, 'c'), (&c1, 'd')];
    assert_eq!(
        answered_by(calls, answers),
        vec![Some('a'), Some('c'), Some('b')]
    );
    // One result for a repeated id answers its first call; the second is
    // still owed.
    let history = vec![
        assistant(vec![call("c1"), call("c2"), call("c1")]),
        results(&["c2", "c1"]),
    ];
    let pending = pending(&history).map(|(_, calls)| {
        let calls = calls.into_iter();
        calls
            .map(|(at, call)| (at, call.id.clone()))
            .collect::<Vec<_>>()
    });
    assert_eq!(pending, Some(vec![(2, c1)]));
}

#[test]
fn a_window_starts_past_results_whose_calls_it_cuts_off() {
    let history = vec![
        assistant(vec![call("c1")]),
        results(&["c1"]),
        Message::user("next"),
        results(&["c0"]),
        assistant(vec![call("c2")]),
        results(&["c2"]),
    ];
    assert_eq!(safe_start(&history, 0), 0);
    assert_eq!(safe_start(&history, 1), 4);
    assert_eq!(safe_start(&history, 2), 4);
    assert_eq!(safe_start(&history, 4), 4);
    assert_eq!(safe_start(&history, 5), 6);
    assert_eq!(safe_start(&history, 9), 6);
}

#[test]
fn a_merged_turn_keeps_its_origin_only_when_both_turns_share_it() {
    use crate::message::{AssistantMessage, Fingerprint, Origin};
    let anthropic = Origin::new("messages", "anthropic", "claude");
    let gemini = Origin::new("generate_content", "gemini", "gemini-pro");
    let mut next_reply = anthropic.clone();
    next_reply.response_id = Some("msg_b".to_owned());
    let mut other_context = anthropic.clone();
    other_context.context = Some(Fingerprint::of_canonical(&"tools"));
    let turn = |origin: Option<&Origin>| {
        let block = AssistantContent::text("a").with_native(serde_json::json!({"text": "a"}));
        Message::Assistant(AssistantMessage::new(vec![block]).with_origin(origin.cloned()))
    };
    let first = Some(&anthropic);
    for (origins, expected) in [
        (vec![first, first], vec![first]),
        (vec![first, Some(&next_reply)], vec![first]),
        (vec![None, None], vec![None]),
        (vec![first, Some(&gemini)], vec![first, Some(&gemini)]),
        (
            vec![first, Some(&other_context)],
            vec![first, Some(&other_context)],
        ),
        (vec![first, None], vec![first, None]),
        (vec![None, first], vec![None, first]),
        (
            vec![None, first, Some(&gemini)],
            vec![None, first, Some(&gemini)],
        ),
        (
            vec![first, None, Some(&gemini)],
            vec![first, None, Some(&gemini)],
        ),
        (
            vec![first, Some(&gemini), first],
            vec![first, Some(&gemini), first],
        ),
    ] {
        let turns: Vec<Message> = origins.iter().map(|origin| turn(*origin)).collect();
        let repaired = repair(turns.clone()).messages;
        let kept: Vec<Option<&Origin>> = repaired
            .iter()
            .filter_map(|message| match message {
                Message::Assistant(turn) => Some(turn.origin.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(kept, expected, "{turns:?}");
        assert_eq!(validate(&repaired), Ok(()), "{turns:?}");
    }

    // An empty turn has nothing to protect: it goes into its neighbour,
    // whose origin stands.
    let empty = |origin: Option<&Origin>| {
        Message::Assistant(AssistantMessage::new(Vec::new()).with_origin(origin.cloned()))
    };
    let native = turn(first);
    let foreign = turn(Some(&gemini));
    for (turns, expected) in [
        (vec![native.clone(), empty(None)], vec![first]),
        (vec![empty(None), native.clone()], vec![first]),
        (vec![native.clone(), empty(Some(&gemini))], vec![first]),
        (vec![empty(Some(&gemini)), native.clone()], vec![first]),
        (
            vec![native.clone(), empty(None), foreign.clone()],
            vec![first, Some(&gemini)],
        ),
        (
            vec![empty(None), native.clone(), foreign],
            vec![first, Some(&gemini)],
        ),
    ] {
        assert!(validate(&turns).is_err(), "{turns:?}");
        let repaired = repair(turns.clone()).messages;
        let kept: Vec<Option<&Origin>> = repaired
            .iter()
            .filter_map(|message| match message {
                Message::Assistant(turn) if !turn.content.is_empty() => Some(turn.origin.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(kept, expected, "{turns:?}");
        assert_eq!(repaired.len(), expected.len(), "{turns:?}");
        assert_eq!(validate(&repaired), Ok(()), "{turns:?}");
    }

    // Nor does a failure with nothing in it fail its neighbour, whenever
    // the merge happens.
    let failed = Message::Assistant(
        AssistantMessage::new(Vec::new()).with_stop(StopReason::Error("refused".to_owned())),
    );
    for turns in [
        vec![native.clone(), failed.clone()],
        vec![failed.clone(), native.clone()],
        vec![native.clone(), failed.clone(), turn(Some(&gemini))],
        vec![native.clone(), failed.clone(), empty(None)],
    ] {
        let repaired = repair(turns.clone()).messages;
        let Some(Message::Assistant(kept)) = repaired.first() else {
            panic!("a turn: {repaired:?}");
        };
        assert_eq!(kept.stop, None, "{turns:?}");
        assert_eq!(kept.origin.as_ref(), first, "{turns:?}");
    }
    let lone = repair(vec![empty(None), failed.clone()]).messages;
    assert_eq!(lone, vec![failed.clone()]);

    // An empty turn can't part a call from its result, or a result from
    // the rest of its batch.
    let (asked, answer) = (assistant(vec![call("c1")]), results(&["c1"]));
    let parted = vec![asked.clone(), empty(None), answer.clone()];
    assert!(validate(&parted).is_err());
    assert_eq!(repair(parted).messages, vec![asked.clone(), answer.clone()]);
    for parted in [
        vec![
            asked.clone(),
            Message::user("wait"),
            empty(None),
            answer.clone(),
        ],
        vec![
            asked.clone(),
            empty(None),
            Message::user("wait"),
            empty(None),
            answer.clone(),
        ],
    ] {
        assert!(validate(&parted).is_err(), "{parted:?}");
        let joined = vec![asked.clone(), Message::user("wait"), answer.clone()];
        assert_eq!(repair(parted).messages, repair(joined).messages);
    }
    // Where it parts nothing it stands, so a valid history stays valid.
    let standing = vec![Message::user("hi"), failed.clone()];
    assert_eq!(validate(&standing), Ok(()));
    assert_eq!(repair(standing.clone()).messages, standing);
    let batch = assistant(vec![call("c1"), call("c2")]);
    let (one, two) = (results(&["c1"]), results(&["c2"]));
    let parted = vec![batch.clone(), one.clone(), empty(None), two.clone()];
    assert!(validate(&parted).is_err());
    assert_eq!(
        repair(parted).messages,
        repair(vec![batch, one, two]).messages
    );

    // Turns with different outcomes stay apart, each with its own stop, so
    // no failure takes a good turn down or comes back as good. Those with
    // the same outcome merge, the later stop standing.
    let error = StopReason::Error("refused".to_owned());
    let aborted = StopReason::Aborted("stopped".to_owned());
    let stops = [
        None,
        Some(StopReason::Stop),
        Some(StopReason::Length),
        Some(StopReason::ToolUse),
        Some(error.clone()),
        Some(aborted.clone()),
    ];
    let ended = |stop: &Option<StopReason>| {
        Message::Assistant(
            AssistantMessage::new(vec![AssistantContent::text("a")])
                .with_origin(anthropic.clone())
                .with_stop(stop.clone()),
        )
    };
    let outcome = |stop: &Option<StopReason>| match stop {
        Some(StopReason::Error(_)) => 1,
        Some(StopReason::Aborted(_)) => 2,
        _ => 0,
    };
    for a in &stops {
        for b in &stops {
            let turns = vec![ended(a), ended(b)];
            let repaired = repair(turns.clone()).messages;
            let kept: Vec<Option<StopReason>> = repaired
                .iter()
                .filter_map(|message| match message {
                    Message::Assistant(turn) => Some(turn.stop.clone()),
                    _ => None,
                })
                .collect();
            let expected = if outcome(a) == outcome(b) {
                vec![b.clone()]
            } else {
                vec![a.clone(), b.clone()]
            };
            assert_eq!(kept, expected, "{turns:?}");
            assert_eq!(validate(&repaired), Ok(()), "{turns:?}");
        }
    }
    let of = |stop: &StopReason| ended(&Some(stop.clone()));
    for turns in [
        vec![of(&aborted), of(&error), native.clone()],
        vec![of(&error), of(&aborted), turn(Some(&gemini))],
        vec![native.clone(), of(&error), turn(Some(&gemini))],
    ] {
        assert_eq!(repair(turns.clone()).messages.len(), 3, "{turns:?}");
    }
}

/// `normalize` repairs entries as `repair` repairs messages, keeping ids
/// and leaving user entries apart: adjacent turns merge into the first's
/// id, author and made-time with the later stop standing, an entry of
/// orphan results is dropped, a made-up result goes into the next user
/// entry or, with none, a new entry by rig. Its result normalizes to
/// itself.
#[test]
fn normalize_keeps_ids_merges_turns_and_never_merges_user_entries() {
    let entry = |id: u64, message: Message| Entry {
        message: message.into(),
        author: None,
        id: Some(MessageId(id)),
        made: Some(id * 10),
        facts: Default::default(),
    };
    let first_turn = assistant(vec![AssistantContent::text("a")]);
    let Message::Assistant(second_turn) = assistant(vec![AssistantContent::text("b"), call("c1")])
    else {
        return;
    };
    let second_turn = Message::Assistant(second_turn.with_stop(StopReason::ToolUse));
    let normalized = normalize(vec![
        entry(0, Message::user("q")),
        entry(1, Message::user("more")),
        entry(2, first_turn),
        entry(3, results(&["ghost"])),
        Entry {
            author: Some(Author::plugin("continue")),
            ..entry(4, second_turn)
        },
        entry(5, Message::user("next")),
        entry(6, assistant(vec![call("c2")])),
    ]);
    assert_eq!(normalized.merged, [(MessageId(2), MessageId(4))]);
    assert_eq!(normalized.dropped, [MessageId(3)]);
    assert!(!normalized.repairs.is_empty());
    let ids: Vec<Option<MessageId>> = normalized.entries.iter().map(|entry| entry.id).collect();
    let id = |id| Some(MessageId(id));
    assert_eq!(ids, [id(0), id(1), id(2), id(5), id(6), None]);
    let merged = normalized.entries.get(2).expect("the merged turn");
    assert_eq!((merged.author.clone(), merged.made), (None, Some(20)));
    let Message::Assistant(turn) = &*merged.message else {
        panic!("a turn: {merged:?}");
    };
    assert_eq!(
        (turn.content.len(), turn.stop.clone()),
        (3, Some(StopReason::ToolUse))
    );
    let next = normalized.entries.get(3).map(|entry| &*entry.message);
    let made_up = close_pending([&tool_call("c1")]);
    let Some(Message::User { content }) = next else {
        panic!("a user entry: {next:?}");
    };
    let Message::User { content: result } = made_up else {
        panic!("results are a user message");
    };
    assert_eq!(
        content.first(),
        result.first(),
        "the made-up result goes in"
    );
    let last = normalized.entries.last().expect("an entry");
    assert_eq!(last.author, Some(Author::rig()));
    assert_eq!(last.message, close_pending([&tool_call("c2")]).into());

    let again = normalize(normalized.entries.clone());
    assert_eq!(again.entries, normalized.entries);
    assert!(again.merged.is_empty() && again.dropped.is_empty() && again.repairs.is_empty());
}

/// A system entry that arrives while calls wait goes after the entry that
/// answers them, keeping its id, as `repair` places a held system message;
/// the results keep what their calls published.
#[test]
fn normalize_puts_a_system_entry_after_the_results_it_waited_for() {
    let entry = |id: u64, message: Message| Entry {
        message: message.into(),
        author: None,
        id: Some(MessageId(id)),
        made: None,
        facts: Default::default(),
    };
    let mut context = crate::tool::ToolContext::new();
    assert!(
        context
            .insert_result(crate::tool::ExitStatus::Exited(3))
            .is_ok()
    );
    let facts: ResultFacts = [(id("c1"), context.result_context())].into_iter().collect();
    let normalized = normalize(vec![
        entry(0, assistant(vec![call("c1")])),
        entry(1, Message::system("note")),
        entry(2, results(&["c1"])).with_facts(facts.clone()),
    ]);
    let kept = normalized.entries.get(1).map(|entry| &entry.facts);
    assert_eq!(kept, Some(&facts));
    let order: Vec<(Option<MessageId>, &Message)> = normalized
        .entries
        .iter()
        .map(|entry| (entry.id, &*entry.message))
        .collect();
    let note = Message::system("note");
    let answer = results(&["c1"]);
    assert_eq!(
        order.get(1..),
        Some([(Some(MessageId(2)), &answer), (Some(MessageId(1)), &note)].as_slice())
    );
    assert!(normalized.repairs.is_empty() && normalized.merged.is_empty());
}

/// An edit that deletes the first of two calls moves the second into the
/// first's place; it keeps its own item, not the one kept at that place.
#[test]
fn a_call_moved_into_a_deleted_calls_place_keeps_its_own_item() {
    let place = |index| CallRef {
        reply: MessageId(1),
        index,
    };
    let (first, second) = (id("c1"), id("c2"));
    let items = [
        (&first, Some(place(0)), "first's"),
        (&second, Some(place(1)), "second's"),
    ];
    assert_eq!(placed(&[(place(0), &second)], &items), [Some("second's")]);
    let both = [(place(0), &first), (place(1), &second)];
    assert_eq!(placed(&both, &items), [Some("first's"), Some("second's")]);
}
