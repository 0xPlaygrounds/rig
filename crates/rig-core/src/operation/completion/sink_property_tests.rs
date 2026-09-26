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
    Sink::<Completion>::finish(&mut out);
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

/// `steps` through a sink, ended as the driver ends a reply.
fn once(steps: Vec<Step>) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut out = AdapterOutput::new();
    for step in steps {
        apply(&mut out, step);
    }
    Sink::<Completion>::finish(&mut out);
    out.into_items()
}

/// Canonicalizing `steps`' items again changes nothing, relayed or not.
fn assert_idempotent(steps: Vec<Step>) -> Vec<Result<StreamEvent, ProviderError>> {
    let items = once(steps);
    for relayed in [true, false] {
        let twice = canonical(copied(&items, relayed));
        assert_eq!(comparable(&items), comparable(&twice), "relayed: {relayed}");
    }
    items
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Canonical events are a fixed point of the sink: pushing what a sink
    /// drained through a second sink, and ending both, yields the same items.
    /// A relay or a script that passes events through the sink again
    /// changes nothing.
    #[test]
    fn canonicalizing_twice_is_canonicalizing_once(
        steps in proptest::collection::vec(step(), 0..24),
        relayed in any::<bool>(),
    ) {
        let items = once(steps);
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
