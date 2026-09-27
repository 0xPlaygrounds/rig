use proptest::prelude::*;

use super::*;
use crate::completion::{FinishReason, Usage};
use crate::error::ErrorReport;
use crate::message::{Reasoning, ReasoningContent};
use crate::streaming::UnparseableToolInput;

/// One helper call a decoder makes on the sink. Keys are small indices, so
/// sequences reuse them.
#[derive(Debug, Clone)]
enum Step {
    Text(String),
    TextMeta(String),
    TextStart(u8),
    TextEnd(u8),
    EndActiveText,
    Reasoning(String),
    ReasoningDelta(u8, String),
    ReasoningEnd {
        key: u8,
        restatement: Option<String>,
        signature: Option<String>,
        wire_sent: bool,
    },
    ReasoningBlock(u8, String),
    CloseActiveBlocks,
    ToolName(u8, String),
    ToolArguments(u8, String),
    ToolEnd(u8, UnparseableToolInput),
    ToolEndNamed(u8, String),
    ToolWhole(u8, String),
    Image(String),
    MessageId(String),
    Unknown,
    Final(Option<FinishReason>),
    Error(String),
}

fn text_key(key: u8) -> BlockId {
    BlockId::wire(format!("msg_{key}"))
}

fn reasoning_key(key: u8) -> BlockId {
    if key.is_multiple_of(2) {
        BlockId::wire(format!("rs_{key}"))
    } else {
        MintKind::Reasoning.for_wire_index(u64::from(key))
    }
}

fn tool_key(key: u8) -> BlockId {
    BlockId::wire(format!("call_{key}"))
}

fn step() -> impl Strategy<Value = Step> {
    let word = "[a-z]{0,4}";
    let key = 0u8..3;
    // Fragments that assemble into valid JSON, and ones that do not.
    let fragment = prop_oneof![
        Just("{".to_owned()),
        Just("\"q\": 1".to_owned()),
        Just("}".to_owned()),
        Just("{\"q\": \"a\"}".to_owned()),
        Just("not json".to_owned()),
        Just("null".to_owned()),
    ];
    let policy = prop_oneof![
        Just(UnparseableToolInput::Error),
        Just(UnparseableToolInput::Drop),
        Just(UnparseableToolInput::EmptyObject),
        Just(UnparseableToolInput::Keep),
    ];
    prop_oneof![
        word.prop_map(Step::Text),
        "[a-z]{1,3}".prop_map(Step::TextMeta),
        key.clone().prop_map(Step::TextStart),
        key.clone().prop_map(Step::TextEnd),
        Just(Step::EndActiveText),
        word.prop_map(Step::Reasoning),
        (key.clone(), word).prop_map(|(key, text)| Step::ReasoningDelta(key, text)),
        (
            key.clone(),
            proptest::option::of(word),
            proptest::option::of("sig_[a-z]{1,3}"),
            any::<bool>()
        )
            .prop_map(
                |(key, restatement, signature, wire_sent)| Step::ReasoningEnd {
                    key,
                    restatement,
                    signature,
                    wire_sent,
                }
            ),
        (key.clone(), word).prop_map(|(key, text)| Step::ReasoningBlock(key, text)),
        Just(Step::CloseActiveBlocks),
        (key.clone(), "[a-z]{0,3}").prop_map(|(key, name)| Step::ToolName(key, name)),
        (key.clone(), fragment).prop_map(|(key, fragment)| Step::ToolArguments(key, fragment)),
        (key.clone(), policy).prop_map(|(key, policy)| Step::ToolEnd(key, policy)),
        (key.clone(), "[a-z]{1,3}").prop_map(|(key, name)| Step::ToolEndNamed(key, name)),
        "[a-z]{1,3}".prop_map(Step::Image),
        (key, "[a-z]{1,3}").prop_map(|(key, name)| Step::ToolWhole(key, name)),
        "[a-z]{1,3}".prop_map(Step::MessageId),
        Just(Step::Unknown),
        prop_oneof![
            Just(None),
            Just(Some(FinishReason::Stop)),
            Just(Some(FinishReason::ToolCalls))
        ]
        .prop_map(Step::Final),
        "[a-z]{1,4}".prop_map(Step::Error),
    ]
}

fn apply(out: &mut AdapterOutput, step: Step) {
    match step {
        Step::Text(text) => out.text(text),
        Step::TextMeta(value) => {
            if let Some(params) =
                crate::message::AdditionalParams::from_entries([("meta", serde_json::json!(value))])
            {
                out.text_meta(params);
            }
        }
        Step::TextStart(key) => out.text_start(text_key(key), None),
        Step::TextEnd(key) => out.text_end(text_key(key)),
        Step::EndActiveText => out.end_active_text(),
        Step::Reasoning(text) => out.reasoning(text),
        Step::ReasoningDelta(key, text) => out.reasoning_delta(&reasoning_key(key), None, text),
        Step::ReasoningEnd {
            key,
            restatement,
            signature,
            wire_sent,
        } => out.reasoning_end(
            reasoning_key(key),
            restatement.map(|text| Reasoning::new(&text)),
            signature,
            wire_sent,
        ),
        Step::ReasoningBlock(key, text) => {
            out.reasoning_block(reasoning_key(key), None, ReasoningContent::Summary(text));
        }
        Step::CloseActiveBlocks => out.close_active_blocks(),
        Step::ToolName(key, name) => out.tool_name(&tool_key(key), name),
        Step::ToolArguments(key, fragment) => out.tool_arguments(&tool_key(key), fragment),
        Step::ToolEnd(key, policy) => out.tool_end(tool_key(key), ToolCallEnd::new(policy)),
        Step::ToolEndNamed(key, name) => {
            let mut end = ToolCallEnd::new(UnparseableToolInput::Error);
            end.name = Some(name);
            out.tool_end(tool_key(key), end);
        }
        Step::Image(url) => out.content(
            &[AssistantContent::Image(crate::message::Image {
                data: crate::message::DocumentSourceKind::Url(url),
                media_type: None,
                detail: None,
                additional_params: None,
            })],
            ImagePart::Block,
        ),
        Step::ToolWhole(key, name) => out.tool_end(
            tool_key(key),
            ToolCallEnd::whole(name, serde_json::json!({"q": 1})),
        ),
        Step::MessageId(id) => out.message_id(id),
        Step::Unknown => out.unknown(serde_json::json!({"type": "extra"}).into()),
        Step::Final(reason) => out.final_record(
            StreamFinal::new("test", Usage::default(), serde_json::Value::Null)
                .with_optional_finish_reason(reason),
        ),
        Step::Error(message) => out.error(ProviderError::Provider(message)),
    }
}

/// What a sink drained, ended as the driver ends a reply, in a comparable
/// form.
fn canonical(
    items: Vec<Result<StreamEvent, ProviderError>>,
) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut out = AdapterOutput::new();
    for item in items {
        out.push(item);
    }
    out.finish();
    out.into_items()
}

fn comparable(items: &[Result<StreamEvent, ProviderError>]) -> serde_json::Value {
    serde_json::to_value(
        items
            .iter()
            .map(|item| item.as_ref().map_err(ErrorReport::from))
            .collect::<Vec<_>>(),
    )
    .unwrap_or_default()
}

/// A copy of a sink's items: events as they are, and errors either as the
/// reports a relay carries or, when not `relayed`, as they were.
fn copied(
    items: &[Result<StreamEvent, ProviderError>],
    relayed: bool,
) -> Vec<Result<StreamEvent, ProviderError>> {
    items
        .iter()
        .map(|item| match item {
            Ok(event) => Ok(event.clone()),
            Err(ProviderError::MalformedToolInput(input)) if !relayed => {
                Err(ProviderError::MalformedToolInput(input.clone()))
            }
            Err(error) => Err(ProviderError::Relayed(Box::new(ErrorReport::from(error)))),
        })
        .collect()
}

/// `steps` through a sink, plain or self-closing, ended as the driver ends
/// a reply.
fn once(self_closing: bool, steps: Vec<Step>) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut out = if self_closing {
        AdapterOutput::self_closing()
    } else {
        AdapterOutput::new()
    };
    for step in steps {
        apply(&mut out, step);
    }
    out.finish();
    out.into_items()
}

/// Canonicalizing `steps`' items again changes nothing, relayed or not, from
/// a plain or a self-closing sink. Returns the plain sink's items.
fn assert_idempotent(steps: Vec<Step>) -> Vec<Result<StreamEvent, ProviderError>> {
    for self_closing in [true, false] {
        let items = once(self_closing, steps.clone());
        for relayed in [true, false] {
            let twice = canonical(copied(&items, relayed));
            assert_eq!(
                comparable(&items),
                comparable(&twice),
                "relayed: {relayed}, self-closing: {self_closing}"
            );
        }
    }
    once(false, steps)
}

/// 2,048 cases per property, or `PROPTEST_CASES` when it is set.
fn config() -> ProptestConfig {
    if std::env::var_os("PROPTEST_CASES").is_some() {
        ProptestConfig::default()
    } else {
        ProptestConfig::with_cases(2048)
    }
}

proptest! {
    #![proptest_config(config())]

    /// Canonical events are a fixed point of the sink: pushing what a sink
    /// drained through a second sink, and ending both, yields the same items.
    /// A relay or a script that passes events through the sink again
    /// changes nothing.
    #[test]
    fn canonicalizing_twice_is_canonicalizing_once(
        steps in proptest::collection::vec(step(), 0..24),
        relayed in any::<bool>(),
        self_closing in any::<bool>(),
    ) {
        let items = once(self_closing, steps);
        let twice = canonical(copied(&items, relayed));
        prop_assert_eq!(comparable(&items), comparable(&twice));
    }
}

/// A malformed complete tool input is an error item in place of the call's
/// end. Passed through the sink again, the item still ends the call, so a
/// later call under the same key does not inherit its fragments.
#[test]
fn a_malformed_call_still_ends_when_its_error_passes_the_sink_again() {
    let items = assert_idempotent(vec![
        Step::ToolName(0, "a".into()),
        Step::ToolArguments(0, "not json".into()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        Step::ToolName(0, "b".into()),
        Step::ToolArguments(0, "{\"q\": \"a\"}".into()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
    ]);
    assert!(
        items.iter().any(|item| matches!(
            item,
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(_)),
                ..
            })
        )),
        "the second call finalizes"
    );
}

/// The same, when the call's name arrived only on its end.
#[test]
fn a_malformed_call_named_on_its_end_still_ends() {
    assert_idempotent(vec![
        Step::ToolArguments(1, "not json".into()),
        Step::ToolEndNamed(1, "a".into()),
        Step::ToolArguments(1, "{\"q\": 1}".into()),
        Step::ToolEndNamed(1, "b".into()),
    ]);
}

#[test]
fn a_late_signature_after_a_synthesized_end_is_idempotent() {
    assert_idempotent(vec![
        Step::ReasoningDelta(1, "hidden".into()),
        Step::ReasoningEnd {
            key: 1,
            restatement: None,
            signature: None,
            wire_sent: false,
        },
        Step::Text("visible".into()),
        Step::ReasoningEnd {
            key: 1,
            restatement: None,
            signature: Some("sig_a".into()),
            wire_sent: true,
        },
        Step::Final(None),
    ]);
}

#[test]
fn sibling_reasoning_under_a_finished_key_is_idempotent() {
    for signature in [None, Some("sig_b".to_owned())] {
        assert_idempotent(vec![
            Step::ReasoningEnd {
                key: 0,
                restatement: Some("a".into()),
                signature: Some("sig_a".into()),
                wire_sent: true,
            },
            Step::ReasoningEnd {
                key: 0,
                restatement: signature.is_none().then(|| "b".to_owned()),
                signature,
                wire_sent: true,
            },
            Step::Final(None),
        ]);
    }
}

#[test]
fn text_open_at_the_terminal_is_idempotent() {
    let items = assert_idempotent(vec![
        Step::TextStart(0),
        Step::Text("hi".into()),
        Step::Final(Some(FinishReason::Stop)),
    ]);
    assert!(items.iter().any(|item| matches!(
        item,
        Ok(StreamEvent::BlockEnd {
            block: Some(AssistantContent::Text(_)),
            ..
        })
    )));
}

#[test]
fn duplicate_terminals_and_a_trailing_error_are_idempotent() {
    assert_idempotent(vec![
        Step::Text("hi".into()),
        Step::Final(None),
        Step::Final(Some(FinishReason::Stop)),
        Step::Unknown,
    ]);
    assert_idempotent(vec![
        Step::Reasoning("thinking".into()),
        Step::Text("partial".into()),
        Step::Error("reset".into()),
    ]);
}

#[test]
fn images_and_text_metadata_are_idempotent() {
    assert_idempotent(vec![
        Step::TextMeta("m".into()),
        Step::Text("a".into()),
        Step::Image("u".into()),
        Step::Image("u".into()),
        Step::Final(None),
    ]);
}

fn tool_calls(items: &[Result<StreamEvent, ProviderError>]) -> usize {
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
}

/// A malformed complete tool input is an error item, and it passes a second
/// sink unchanged.
#[test]
fn a_malformed_complete_tool_input_is_idempotent() {
    let items = assert_idempotent(vec![
        Step::Text("calling".into()),
        Step::ToolName(0, "lookup".into()),
        Step::ToolArguments(0, "not json".into()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        Step::Final(Some(FinishReason::Stop)),
    ]);
    assert!(
        items
            .iter()
            .any(|item| matches!(item, Err(ProviderError::MalformedToolInput(_)))),
        "the malformed input is an error item"
    );
}

/// The error item stands in place of the call's end, so a second sink ends
/// the call it reports, and a stale end after it finalizes no phantom call.
#[test]
fn a_stale_end_after_a_malformed_input_stays_stale() {
    let items = assert_idempotent(vec![
        Step::ToolArguments(0, "a".into()),
        Step::ToolName(0, "lookup".into()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        Step::ToolEnd(0, UnparseableToolInput::EmptyObject),
    ]);
    assert_eq!(tool_calls(&items), 0);
    for relayed in [true, false] {
        assert_eq!(tool_calls(&canonical(copied(&items, relayed))), 0);
    }
}

/// An error item handed to [`AdapterOutput::error`], not pushed, ends the
/// call it reports as a pushed one does.
#[test]
fn an_error_item_passed_to_error_finishes_the_call_it_reports() {
    let first = once(
        false,
        vec![
            Step::ToolArguments(0, "a".into()),
            Step::ToolName(0, "lookup".into()),
            Step::ToolEnd(0, UnparseableToolInput::Error),
        ],
    );
    for relayed in [true, false] {
        let mut out = AdapterOutput::new();
        apply(&mut out, Step::ToolArguments(0, "a".into()));
        apply(&mut out, Step::ToolName(0, "lookup".into()));
        for item in copied(&first, relayed) {
            if let Err(error) = item {
                out.error(error);
            }
        }
        apply(
            &mut out,
            Step::ToolEnd(0, UnparseableToolInput::EmptyObject),
        );
        out.finish();
        assert_eq!(tool_calls(&out.into_items()), 0, "relayed: {relayed}");
    }
}

/// A report names its call by the exact raw input the call assembled. A call
/// that received no argument fragments is a parameterless `{}` call that
/// never reports malformed input, so a report with an empty input does not
/// end it, even when it names the call's block.
#[test]
fn a_report_of_empty_input_does_not_end_a_call_without_arguments() {
    let mut out = AdapterOutput::new();
    apply(&mut out, Step::ToolName(0, "lookup".into()));
    out.push(Err(ProviderError::MalformedToolInput(
        crate::error::MalformedToolInput {
            name: "lookup".to_owned(),
            id: crate::message::ToolCallId::from_block(&tool_key(0)),
            provider: None,
            raw: String::new(),
            error: "expected value".to_owned(),
        },
    )));
    apply(&mut out, Step::ToolEnd(0, UnparseableToolInput::Error));
    out.finish();
    assert_eq!(
        tool_calls(&out.into_items()),
        1,
        "the call finalizes as `{{}}`"
    );
}

/// Raw events a relay or a tap may receive, canonical or not: deltas, starts
/// and ends under reused keys, ends that carry their block, terminals,
/// unknown frames and failures.
fn raw_event() -> impl Strategy<Value = Result<StreamEvent, ErrorReport>> {
    let key = 0u8..3;
    let delta = |id: BlockId, delta: Delta| Ok(StreamEvent::BlockDelta { id, delta });
    let fragment = prop_oneof![
        Just(String::new()),
        "[a-z ]{1,4}",
        Just("{\"q\":".to_owned()),
        Just("1}".to_owned()),
        Just("{bad".to_owned()),
    ];
    let policy = prop_oneof![
        Just(UnparseableToolInput::Error),
        Just(UnparseableToolInput::Drop),
        Just(UnparseableToolInput::EmptyObject),
        Just(UnparseableToolInput::Keep),
    ];
    prop_oneof![
        (key.clone(), fragment.clone())
            .prop_map(move |(key, text)| delta(text_key(key), Delta::Text { text })),
        (key.clone(), fragment.clone())
            .prop_map(move |(key, text)| delta(reasoning_key(key), Delta::Reasoning { text })),
        (key.clone(), fragment).prop_map(move |(key, arguments)| delta(
            tool_key(key),
            Delta::ToolArguments { arguments }
        )),
        key.clone().prop_map(move |key| delta(
            tool_key(key),
            Delta::ToolName {
                name: "lookup".to_owned()
            }
        )),
        key.clone().prop_map(|key| Ok(StreamEvent::BlockStart {
            id: text_key(key),
            kind: BlockKind::Text {
                additional_params: None
            },
        })),
        key.clone().prop_map(|key| Ok(StreamEvent::BlockStart {
            id: reasoning_key(key),
            kind: BlockKind::Reasoning { provider_id: None },
        })),
        (key.clone(), proptest::option::of("[a-z]{1,3}")).prop_map(|(key, text)| Ok(
            StreamEvent::BlockEnd {
                id: text_key(key),
                end: BlockClose::Text,
                block: text.map(AssistantContent::text),
            }
        )),
        (
            key.clone(),
            proptest::option::of("sig_[0-9]"),
            any::<bool>(),
            proptest::option::of("[a-z]{1,3}")
        )
            .prop_map(
                |(key, signature, wire_sent, carried)| Ok(StreamEvent::BlockEnd {
                    id: reasoning_key(key),
                    end: BlockClose::Reasoning {
                        reasoning: None,
                        signature,
                        wire_sent,
                    },
                    block: carried.map(|text| AssistantContent::Reasoning(Reasoning::new(&text))),
                })
            ),
        (key.clone(), policy, any::<bool>()).prop_map(|(key, policy, carried)| Ok(
            StreamEvent::BlockEnd {
                id: tool_key(key),
                end: BlockClose::ToolCall(ToolCallEnd::new(policy)),
                block: carried.then(|| carried_tool_call(&tool_key(key))),
            }
        )),
        key.prop_map(|key| Ok(StreamEvent::BlockEnd {
            id: tool_key(key),
            end: BlockClose::ToolCall(ToolCallEnd::whole("lookup", serde_json::json!({"q": 1}))),
            block: None,
        })),
        prop_oneof![
            Just(None),
            Just(Some(FinishReason::Stop)),
            Just(Some(FinishReason::ToolCalls))
        ]
        .prop_map(|reason| Ok(StreamEvent::Final(
            StreamFinal::new("test", Usage::default(), serde_json::Value::Null)
                .with_optional_finish_reason(reason)
        ))),
        Just(Ok(StreamEvent::Unknown(UnknownPayload::new(
            serde_json::json!({"type": "extra"})
        )))),
        Just(Err(ErrorReport::new(
            crate::error::ErrorKind::Provider,
            "relayed failure"
        ))),
    ]
}

fn carried_tool_call(id: &BlockId) -> AssistantContent {
    AssistantContent::ToolCall(crate::message::ToolCall::new(
        crate::message::ToolCallId::from_block(id),
        crate::message::ToolFunction::new("carried".to_owned(), serde_json::json!({"c": 1})),
    ))
}

/// What a relay yields for `items`, and the response its fold finishes to.
fn relay(
    items: Vec<Result<StreamEvent, ErrorReport>>,
) -> (
    Vec<Result<StreamEvent, ErrorReport>>,
    crate::streaming::CompletionStream,
) {
    use futures::StreamExt;

    let mut stream =
        crate::streaming::CompletionStream::relay("relay", Box::pin(futures::stream::iter(items)));
    let yielded = futures::executor::block_on(async {
        let mut yielded = Vec::new();
        while let Some(item) = stream.next().await {
            yielded.push(item);
        }
        yielded
    });
    (yielded, stream)
}

fn reported(items: &[Result<StreamEvent, ErrorReport>]) -> serde_json::Value {
    serde_json::to_value(items).unwrap_or_default()
}

/// A relay canonicalizes as one sink does, drained after each item and
/// finished at the end of the stream. The sink and the relay share
/// `Canonical`, so this pins the relay's plumbing: its drain order, errors
/// passing as they arrive, and the closes it adds at the end.
fn sink_over(items: &[Result<StreamEvent, ErrorReport>]) -> Vec<Result<StreamEvent, ErrorReport>> {
    let mut out = AdapterOutput::new();
    let mut drained = Vec::new();
    for item in items {
        out.push(
            item.clone()
                .map_err(|report| ProviderError::Relayed(Box::new(report))),
        );
        drained.extend(out.drain());
    }
    out.finish();
    drained.extend(out.drain());
    drained
        .iter()
        .map(|item| item.as_ref().cloned().map_err(ErrorReport::from))
        .collect()
}

proptest! {
    #![proptest_config(config())]

    /// A relay yields what one sink drains for its items, drained as each
    /// arrives and finished at the end of the stream. The relay and the sink
    /// share `Canonical`, so this pins the relay's plumbing (drain order,
    /// errors passing as they arrive, the closes at the end), not the
    /// canonicalization itself.
    #[test]
    fn a_relay_canonicalizes_what_it_carries(
        items in proptest::collection::vec(raw_event(), 0..24),
    ) {
        let (yielded, _) = relay(items.clone());
        prop_assert_eq!(reported(&yielded), reported(&sink_over(&items)));
    }

    /// Canonical events, from a plain or a self-closing sink, pass a relay
    /// unchanged.
    #[test]
    fn a_relay_passes_canonical_events_unchanged(
        steps in proptest::collection::vec(step(), 0..24),
        self_closing in any::<bool>(),
    ) {
        let items: Vec<_> = once(self_closing, steps)
            .iter()
            .map(|item| item.as_ref().cloned().map_err(ErrorReport::from))
            .collect();
        let (yielded, _) = relay(items.clone());
        prop_assert_eq!(reported(&yielded), reported(&items));
    }

    /// A block an end carries under a key no event assembled survives the
    /// relay, whatever surrounds it: text, reasoning and tool calls.
    #[test]
    fn a_block_carried_only_on_its_end_survives_the_relay(
        before in proptest::collection::vec(raw_event(), 0..12),
        after in proptest::collection::vec(raw_event(), 0..12),
        kind in 0u8..3,
    ) {
        // Key 9 is outside `raw_event`'s keys, so nothing else assembles it.
        let (id, end, block) = match kind {
            0 => (text_key(9), BlockClose::Text, AssistantContent::text("carried")),
            1 => (
                reasoning_key(9),
                BlockClose::Reasoning { reasoning: None, signature: None, wire_sent: false },
                AssistantContent::Reasoning(Reasoning::new("carried")),
            ),
            _ => (
                tool_key(9),
                BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Error)),
                carried_tool_call(&tool_key(9)),
            ),
        };
        let carried = StreamEvent::BlockEnd { id: id.clone(), end, block: Some(block.clone()) };
        let clean = before.iter().all(|item| {
            matches!(item, Ok(event) if !matches!(event, StreamEvent::Final(_)))
        });
        let mut items = before;
        items.push(Ok(carried));
        items.extend(after);
        let (yielded, stream) = relay(items);
        prop_assert!(
            yielded.iter().any(|item| matches!(
                item,
                Ok(StreamEvent::BlockEnd { id: end_id, block: Some(kept), .. })
                    if *end_id == id && *kept == block
            )),
            "the carried block was dropped: {:?}",
            yielded
        );
        // Before any terminal or failure, the fold collects it too.
        if clean && yielded.iter().all(Result::is_ok) {
            prop_assert!(stream.folded().snapshot().contains(&block));
        }
    }

    /// A tap folds a stream to the response a relay of the same stream
    /// finishes to: both hold a `Canonical`, so they cannot drift apart. The
    /// tap stops at its first outcome; the relay runs to the first terminal.
    #[test]
    fn the_tap_folds_what_the_relay_folds(
        items in proptest::collection::vec(raw_event(), 0..24),
    ) {
        let mut tap = crate::serve::StreamTap::new();
        let tapped = items.iter().find_map(|item| tap.observe(item));
        let through_final = items
            .iter()
            .position(|item| matches!(item, Ok(StreamEvent::Final(_))))
            .map_or(items.len(), |position| position + 1);
        let prefix = items.get(..through_final).unwrap_or_default().to_vec();
        let (yielded, stream) = relay(prefix);
        let relay_error = yielded.iter().find_map(|item| item.as_ref().err().cloned());
        let comparable = |response: &CompletionResponse| {
            let mut value = serde_json::to_value(response).unwrap_or_default();
            if let Some(map) = value.as_object_mut() {
                map.remove("provider");
            }
            value
        };
        match (tapped, relay_error) {
            (Some(Err(tapped)), Some(relayed)) => prop_assert_eq!(tapped, relayed),
            (tapped, Some(relayed)) => {
                prop_assert!(false, "the relay failed ({:?}) and the tap gave {:?}", relayed, tapped);
            }
            (Some(Ok(crate::effect::Outcome::Completion(tapped))), None) => {
                let relayed = stream.finish();
                prop_assert!(relayed.is_ok(), "the relay did not finish: {:?}", relayed);
                if let Ok(relayed) = relayed {
                    prop_assert_eq!(comparable(&tapped), comparable(&relayed));
                }
            }
            (Some(Err(tapped)), None) => {
                let relayed = stream.finish().map_err(|error| ErrorReport::from(&error));
                prop_assert_eq!(relayed.err(), Some(tapped));
            }
            (None, None) => prop_assert!(stream.finish().is_err(), "no terminal, truncated"),
            (other, None) => prop_assert!(false, "the tap gave {:?} and the relay no error", other),
        }
    }
}

/// A second terminal is dropped, and what follows the first passes.
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
    let (yielded, _) = relay(vec![terminal(1), late.clone(), terminal(2)]);
    assert_eq!(reported(&yielded), reported(&[terminal(1), late]));
}

/// A truncated relay closes its open text and reasoning at its end, each
/// carrying its block. A failure passes when it arrives, so those closes
/// follow it.
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
        |yielded: &[Result<StreamEvent, ErrorReport>]| -> Vec<(BlockId, AssistantContent)> {
            yielded
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

    let (truncated, _) = relay(vec![text.clone(), reasoning.clone()]);
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

    let (failed, _) = relay(vec![text, reasoning, failure.clone()]);
    assert_eq!(failed.len(), 5);
    assert_eq!(
        reported(failed.get(2..3).unwrap_or_default()),
        reported(&[failure])
    );
    assert_eq!(closes(&failed), closes(&truncated));
}
