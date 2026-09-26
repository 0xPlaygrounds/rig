use proptest::prelude::*;

use super::*;
use crate::completion::{FinishReason, Usage};
use crate::error::ErrorReport;
use crate::message::{Reasoning, ReasoningContent};
use crate::streaming::UnparseableToolInput;

/// One helper call a decoder makes on the sink. Keys are small indices, so
/// generated scripts reuse them.
#[derive(Debug, Clone)]
enum Step {
    Text(String),
    TextStart(u8),
    TextEnd(u8),
    EndActiveText,
    Reasoning(String),
    ReasoningDelta(u8, String),
    ReasoningEnd {
        key: u8,
        restated: Option<String>,
        signature: Option<String>,
        wire_sent: bool,
    },
    ToolName(u8, String),
    ToolArguments(u8, String),
    ToolEnd(u8, UnparseableToolInput),
    WholeCall(u8),
    MessageId(u8),
    Unknown,
    CloseActive,
    Final(Option<FinishReason>),
    Error,
}

fn text_key(key: u8) -> BlockId {
    BlockId::wire(format!("text_{key}"))
}

fn reasoning_key(key: u8) -> BlockId {
    BlockId::wire(format!("rs_{key}"))
}

fn tool_key(key: u8) -> BlockId {
    BlockId::wire(format!("call_{key}"))
}

fn apply(out: &mut AdapterOutput, step: Step) {
    match step {
        Step::Text(text) => out.text(text),
        Step::TextStart(key) => out.text_start(text_key(key), None),
        Step::TextEnd(key) => out.text_end(text_key(key)),
        Step::EndActiveText => out.end_active_text(),
        Step::Reasoning(text) => out.reasoning(text),
        Step::ReasoningDelta(key, text) => out.reasoning_delta(&reasoning_key(key), None, text),
        Step::ReasoningEnd {
            key,
            restated,
            signature,
            wire_sent,
        } => out.reasoning_end(
            reasoning_key(key),
            restated.map(|text| Reasoning::new(&text)),
            signature,
            wire_sent,
        ),
        Step::ToolName(key, name) => out.tool_name(&tool_key(key), name),
        Step::ToolArguments(key, arguments) => out.tool_arguments(&tool_key(key), arguments),
        Step::ToolEnd(key, policy) => out.tool_end(tool_key(key), ToolCallEnd::new(policy)),
        Step::WholeCall(key) => out.tool_end(
            tool_key(key),
            ToolCallEnd::whole("lookup", serde_json::json!({"q": key})),
        ),
        Step::MessageId(key) => out.message_id(format!("msg_{key}")),
        Step::Unknown => out.unknown(UnknownPayload::new(serde_json::json!({"x": 1}))),
        Step::CloseActive => out.close_active_blocks(),
        Step::Final(reason) => out.final_record(
            StreamFinal::new("test", Usage::default(), serde_json::json!({}))
                .with_optional_finish_reason(reason),
        ),
        Step::Error => out.error(ProviderError::Provider("stream failed".to_owned())),
    }
}

type Items = Vec<Result<StreamEvent, ProviderError>>;

/// The items a sink drains for `steps`, ended with `Sink::finish` as the
/// driver ends a reply.
fn canonicalize(mut out: AdapterOutput, steps: Vec<Step>) -> Items {
    for step in steps {
        apply(&mut out, step);
    }
    Sink::<Completion>::finish(&mut out);
    out.into_items()
}

/// `items` pushed through a second sink, as a pass-through wire does.
fn recanonicalize(items: Items) -> Items {
    let mut out = AdapterOutput::new();
    for item in items {
        out.push(item);
    }
    Sink::<Completion>::finish(&mut out);
    out.into_items()
}

fn comparable(items: &Items) -> Vec<Result<StreamEvent, ErrorReport>> {
    items
        .iter()
        .map(|item| match item {
            Ok(event) => Ok(event.clone()),
            Err(error) => Err(ErrorReport::from(error)),
        })
        .collect()
}

/// Canonicalizing `steps` twice drains what canonicalizing them once does.
fn assert_idempotent(self_closing: bool, steps: Vec<Step>) {
    let out = if self_closing {
        AdapterOutput::self_closing()
    } else {
        AdapterOutput::new()
    };
    let once = canonicalize(out, steps);
    let expected = comparable(&once);
    let twice = recanonicalize(once);
    assert_eq!(comparable(&twice), expected);
}

fn fragment() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),
        "[a-z ]{1,6}",
        Just("{\"q\":".to_owned()),
        Just("1}".to_owned()),
        Just("{bad".to_owned()),
    ]
}

fn policy() -> impl Strategy<Value = UnparseableToolInput> {
    prop_oneof![
        Just(UnparseableToolInput::Drop),
        Just(UnparseableToolInput::EmptyObject),
        Just(UnparseableToolInput::Error),
        Just(UnparseableToolInput::Keep),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    let key = 0u8..3;
    prop_oneof![
        fragment().prop_map(Step::Text),
        key.clone().prop_map(Step::TextStart),
        key.clone().prop_map(Step::TextEnd),
        Just(Step::EndActiveText),
        fragment().prop_map(Step::Reasoning),
        (key.clone(), fragment()).prop_map(|(key, text)| Step::ReasoningDelta(key, text)),
        (
            key.clone(),
            proptest::option::of("[a-z]{1,4}"),
            proptest::option::of("sig[0-9]"),
            any::<bool>(),
        )
            .prop_map(|(key, restated, signature, wire_sent)| Step::ReasoningEnd {
                key,
                restated,
                signature,
                wire_sent,
            }),
        (
            key.clone(),
            prop_oneof![Just(String::new()), Just("lookup".to_owned())]
        )
            .prop_map(|(key, name)| Step::ToolName(key, name)),
        (key.clone(), fragment()).prop_map(|(key, arguments)| Step::ToolArguments(key, arguments)),
        (key.clone(), policy()).prop_map(|(key, policy)| Step::ToolEnd(key, policy)),
        key.clone().prop_map(Step::WholeCall),
        key.prop_map(Step::MessageId),
        Just(Step::Unknown),
        Just(Step::CloseActive),
        proptest::option::of(prop_oneof![
            Just(FinishReason::Stop),
            Just(FinishReason::Length),
            Just(FinishReason::ToolCalls),
        ])
        .prop_map(Step::Final),
        Just(Step::Error),
    ]
}

fn stop() -> Step {
    Step::Final(Some(FinishReason::Stop))
}

fn reasoning_end(key: u8, signature: Option<&str>, wire_sent: bool) -> Step {
    Step::ReasoningEnd {
        key,
        restated: None,
        signature: signature.map(str::to_owned),
        wire_sent,
    }
}

proptest! {
    /// Canonical input passes the sink unchanged: any helper-built sequence,
    /// canonicalized, drains the same items when canonicalized again.
    #[test]
    fn canonicalizing_is_idempotent(
        self_closing in any::<bool>(),
        steps in proptest::collection::vec(step(), 0..24),
    ) {
        assert_idempotent(self_closing, steps);
    }
}

#[test]
fn a_malformed_complete_tool_input_is_idempotent() {
    let steps = vec![
        Step::Text("calling".to_owned()),
        Step::ToolName(0, "lookup".to_owned()),
        Step::ToolArguments(0, "{bad".to_owned()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        stop(),
    ];
    let once = canonicalize(AdapterOutput::new(), steps.clone());
    assert!(
        once.iter()
            .any(|item| matches!(item, Err(ProviderError::MalformedToolInput(_)))),
        "the malformed input is an error item"
    );
    assert_idempotent(false, steps);
}

/// The minimal counterexample the property found: the error item stands in
/// place of the end, so a second sink must finish the call it reports, or
/// the stale end after it finalizes a phantom call.
#[test]
fn a_stale_end_after_a_malformed_input_stays_stale() {
    let steps = vec![
        Step::ToolArguments(0, "a".to_owned()),
        Step::ToolName(0, "lookup".to_owned()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        Step::ToolEnd(0, UnparseableToolInput::EmptyObject),
    ];
    let once = canonicalize(AdapterOutput::new(), steps.clone());
    let calls = |items: &Items| {
        items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    Ok(StreamEvent::BlockEnd {
                        block: Some(AssistantContent::ToolCall(_)),
                        ..
                    })
                )
            })
            .count()
    };
    assert_eq!(calls(&once), 0);

    // A relay carries the error as its report; the second sink reads the
    // same detail from it.
    let relayed = once
        .into_iter()
        .map(|item| item.map_err(|error| ProviderError::Relayed(Box::new(error.report()))))
        .collect::<Items>();
    assert_eq!(calls(&recanonicalize(relayed)), 0);
    assert_idempotent(false, steps);
}

#[test]
fn a_late_signature_after_a_synthesized_end_is_idempotent() {
    assert_idempotent(
        false,
        vec![
            Step::ReasoningDelta(0, "thinking".to_owned()),
            reasoning_end(0, None, false),
            Step::Text("answer".to_owned()),
            reasoning_end(0, Some("sig0"), true),
            stop(),
        ],
    );
}

#[test]
fn sibling_reasoning_under_a_finished_key_is_idempotent() {
    // The second signature cannot join the signed part, so it is a sibling
    // part under the same key.
    let steps = vec![
        Step::ReasoningDelta(0, "first".to_owned()),
        reasoning_end(0, Some("sig0"), true),
        reasoning_end(0, Some("sig1"), true),
        stop(),
    ];
    let once = canonicalize(AdapterOutput::new(), steps.clone());
    let sibling = once.iter().rposition(|item| {
        matches!(
            item,
            Ok(StreamEvent::BlockStart {
                kind: BlockKind::Reasoning { .. },
                ..
            })
        )
    });
    assert!(
        matches!(
            sibling.and_then(|start| once.get(start + 1)),
            Some(Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::Reasoning(Reasoning { content, .. })),
                ..
            })) if matches!(
                content.as_slice(),
                [ReasoningContent::Text { signature: Some(signature), .. }] if signature == "sig1"
            )
        ),
        "the sibling part begins where it ends: {once:?}"
    );
    assert_idempotent(false, steps);
}

#[test]
fn text_left_open_at_the_terminal_is_idempotent() {
    for self_closing in [false, true] {
        assert_idempotent(
            self_closing,
            vec![
                Step::TextStart(0),
                Step::Text("open".to_owned()),
                Step::Reasoning("still thinking".to_owned()),
                stop(),
            ],
        );
    }
}

#[test]
fn duplicate_terminals_are_idempotent() {
    assert_idempotent(
        false,
        vec![
            Step::Text("done".to_owned()),
            stop(),
            Step::Text("after".to_owned()),
            Step::Final(Some(FinishReason::Length)),
        ],
    );
}

#[test]
fn a_trailing_error_is_idempotent() {
    assert_idempotent(
        false,
        vec![
            Step::Text("partial".to_owned()),
            Step::ReasoningDelta(1, "half".to_owned()),
            Step::Error,
        ],
    );
}

fn raw_event() -> impl Strategy<Value = Result<StreamEvent, ErrorReport>> {
    let key = 0u8..3;
    let delta = |id: BlockId, delta: Delta| Ok(StreamEvent::BlockDelta { id, delta });
    let end = |id: BlockId, end: BlockClose| {
        Ok(StreamEvent::BlockEnd {
            id,
            end,
            block: None,
        })
    };
    prop_oneof![
        (key.clone(), fragment())
            .prop_map(move |(key, text)| delta(text_key(key), Delta::Text { text })),
        (key.clone(), fragment())
            .prop_map(move |(key, text)| delta(reasoning_key(key), Delta::Reasoning { text })),
        (key.clone(), fragment()).prop_map(move |(key, arguments)| delta(
            tool_key(key),
            Delta::ToolArguments { arguments }
        )),
        key.clone().prop_map(move |key| delta(
            tool_key(key),
            Delta::ToolName {
                name: "lookup".to_owned()
            }
        )),
        key.clone()
            .prop_map(move |key| end(text_key(key), BlockClose::Text)),
        (key.clone(), proptest::option::of("sig[0-9]"), any::<bool>()).prop_map(
            move |(key, signature, wire_sent)| end(
                reasoning_key(key),
                BlockClose::Reasoning {
                    reasoning: None,
                    signature,
                    wire_sent,
                }
            )
        ),
        (key, policy()).prop_map(move |(key, policy)| end(
            tool_key(key),
            BlockClose::ToolCall(ToolCallEnd::new(policy))
        )),
        Just(Ok(StreamEvent::Final(StreamFinal::new(
            "test",
            Usage::default(),
            serde_json::json!({})
        )))),
        Just(Ok(StreamEvent::Unknown(UnknownPayload::new(
            serde_json::json!({"x": 1})
        )))),
        Just(Err(ErrorReport::new(
            crate::error::ErrorKind::Provider,
            "relayed failure"
        ))),
    ]
}

/// What a relay yields for `items`.
fn relayed(items: Vec<Result<StreamEvent, ErrorReport>>) -> Vec<Result<StreamEvent, ErrorReport>> {
    use futures::StreamExt;

    futures::executor::block_on(
        crate::streaming::CompletionStream::relay("relay", Box::pin(futures::stream::iter(items)))
            .collect(),
    )
}

/// `items` pushed through one sink, drained after each and finished at the
/// end, as a relay canonicalizes them.
fn canonical(items: &[Result<StreamEvent, ErrorReport>]) -> Vec<Result<StreamEvent, ErrorReport>> {
    let mut out = AdapterOutput::new();
    let mut drained = Vec::new();
    for item in items {
        out.push(
            item.clone()
                .map_err(|report| ProviderError::Relayed(Box::new(report))),
        );
        drained.extend(out.drain());
    }
    Sink::<Completion>::finish(&mut out);
    drained.extend(out.drain());
    comparable(&drained)
}

proptest! {
    /// A relay is one sink over its items, finished at the end of the
    /// stream: whatever the origin sent, it yields what that sink drains.
    #[test]
    fn a_relay_canonicalizes_what_it_carries(
        items in proptest::collection::vec(raw_event(), 0..24),
    ) {
        prop_assert_eq!(relayed(items.clone()), canonical(&items));
    }

    /// A relay of canonical events yields them unchanged.
    #[test]
    fn a_relay_passes_canonical_events_unchanged(
        steps in proptest::collection::vec(step(), 0..24),
    ) {
        let items = comparable(&canonicalize(AdapterOutput::new(), steps));
        prop_assert_eq!(relayed(items.clone()), items);
    }
}

#[test]
fn a_relay_drops_a_second_terminal_and_passes_what_follows_the_first() {
    let terminal = |tokens: u64| {
        Ok(StreamEvent::Final(StreamFinal::new(
            "test",
            Usage {
                total_tokens: Some(tokens),
                ..Usage::default()
            },
            serde_json::json!({}),
        )))
    };
    let late = Ok(StreamEvent::Unknown(UnknownPayload::new(
        serde_json::json!({"late": true}),
    )));
    let relayed = relayed(vec![terminal(1), late.clone(), terminal(2)]);
    assert_eq!(relayed, vec![terminal(1), late]);
}

#[test]
fn a_truncated_relay_closes_its_open_blocks_at_its_end() {
    let text = Ok(StreamEvent::BlockDelta {
        id: text_key(0),
        delta: Delta::Text {
            text: "partial".to_owned(),
        },
    });
    let reasoning = Ok(StreamEvent::BlockDelta {
        id: reasoning_key(0),
        delta: Delta::Reasoning {
            text: "half".to_owned(),
        },
    });
    let failure = Err(ErrorReport::new(
        crate::error::ErrorKind::Provider,
        "cut off",
    ));
    let closes =
        |relayed: &[Result<StreamEvent, ErrorReport>]| -> Vec<(BlockId, AssistantContent)> {
            relayed
                .iter()
                .filter_map(|item| match item {
                    Ok(StreamEvent::BlockEnd {
                        id,
                        block: Some(block),
                        ..
                    }) => Some((id.clone(), block.clone())),
                    _ => None,
                })
                .collect()
        };

    let truncated = relayed(vec![text.clone(), reasoning.clone()]);
    assert_eq!(
        closes(&truncated),
        vec![
            (text_key(0), AssistantContent::text("partial")),
            (
                reasoning_key(0),
                AssistantContent::Reasoning(Reasoning::new("half"))
            ),
        ]
    );

    // The failure passes when it arrives; the closes follow at the end.
    let failed = relayed(vec![text, reasoning, failure.clone()]);
    assert_eq!(failed.len(), 5);
    assert_eq!(failed.get(2), Some(&failure));
    assert_eq!(closes(&failed), closes(&truncated));
}

/// A decoder that pushes a malformed-input error item by hand finishes the
/// call it reports, as the sink's own end does.
#[test]
fn an_error_item_pushed_by_hand_finishes_the_call_it_reports() {
    let mut out = AdapterOutput::new();
    out.tool_name(&tool_key(0), "lookup");
    out.tool_arguments(&tool_key(0), "{bad");
    out.error(ProviderError::MalformedToolInput(
        crate::error::MalformedToolInput {
            name: "lookup".to_owned(),
            id: crate::message::ToolCallId::from_block(&tool_key(0)),
            provider: None,
            raw: "{bad".to_owned(),
            error: "expected value".to_owned(),
        },
    ));
    out.tool_end(
        tool_key(0),
        ToolCallEnd::new(UnparseableToolInput::EmptyObject),
    );
    assert!(
        out.iter().all(|item| !matches!(
            item,
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(_)),
                ..
            })
        )),
        "the stale end finalizes no call"
    );
}
