use futures::StreamExt;
use proptest::prelude::*;

use super::*;
use crate::completion::{CompletionResponse, Usage};
use crate::error::ProviderError;
use crate::message::{DocumentSourceKind, Image, Reasoning, ReasoningContent};
use crate::operation::{AdapterOutput, ImagePart};
use crate::streaming::{BlockId, CompletionStream, StreamFinal, ToolCallEnd, UnparseableToolInput};

/// A relayed stream of `items`, as the bus and a handler deliver one.
fn relayed(items: Vec<Result<StreamEvent, ProviderError>>) -> CompletionStream {
    let events = futures::stream::iter(
        items
            .into_iter()
            .map(|item| item.map_err(|error| ErrorReport::from(&error))),
    );
    CompletionStream::relay("test", Box::pin(events))
}

fn updates_of(items: Vec<Result<StreamEvent, ProviderError>>) -> Vec<Result<Update, ErrorReport>> {
    let mut stream = relayed(items);
    futures::executor::block_on(stream.updates().collect::<Vec<_>>())
}

/// The updates of `items` read in turns: for each step of `plan` (cycled),
/// `true` reads one update through a fresh [`CompletionStream::updates`]
/// call, `false` reads one event from the stream itself. Once the events
/// run out, the updates are read to their end.
fn updates_in_turns(
    items: Vec<Result<StreamEvent, ProviderError>>,
    plan: &[bool],
) -> Vec<Result<Update, ErrorReport>> {
    let mut stream = relayed(items);
    let mut updates = Vec::new();
    futures::executor::block_on(async {
        for through_updates in plan.iter().cycle() {
            if *through_updates {
                match stream.updates().next().await {
                    Some(update) => updates.push(update),
                    None => return,
                }
            } else if stream.next().await.is_none() {
                break;
            }
        }
        updates.extend(stream.updates().collect::<Vec<_>>().await);
    });
    updates
}

/// The finished text a part's deltas must concatenate to, written out here
/// rather than taken from the projection.
fn expected_text(part: &AssistantContent) -> String {
    match part {
        AssistantContent::Text(text) => text.text.clone(),
        AssistantContent::Reasoning(reasoning) => reasoning
            .content
            .iter()
            .map(|content| match content {
                ReasoningContent::Text { text, .. } => text.as_str(),
                ReasoningContent::Summary(summary) => summary.as_str(),
                _ => "",
            })
            .collect(),
        AssistantContent::ToolCall(call) => {
            serde_json::to_string(&call.function.arguments).unwrap_or_default()
        }
        AssistantContent::Image(_) => String::new(),
    }
}

fn expected_kind(part: &AssistantContent) -> PartKind {
    match part {
        AssistantContent::Text(_) => PartKind::Text,
        AssistantContent::Reasoning(_) => PartKind::Reasoning,
        AssistantContent::ToolCall(call) => PartKind::ToolCall {
            name: call.function.name.clone(),
        },
        AssistantContent::Image(_) => PartKind::Image,
    }
}

/// The index contract, against the response the stream finished to: every
/// part of `choice` starts first with its kind, its deltas concatenate to
/// its finished text, its last end is the part, and nothing names an index
/// past `choice`. Returns the response.
fn assert_contract(updates: &[Result<Update, ErrorReport>]) -> CompletionResponse {
    let done = updates
        .iter()
        .find_map(|update| match update {
            Ok(Update::Done(response)) => Some(response.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the stream finished: {updates:#?}"));
    assert!(
        matches!(updates.last(), Some(Ok(Update::Done(_)))),
        "Done comes last: {updates:#?}"
    );
    for (index, part) in done.choice.iter().enumerate() {
        let own: Vec<&Update> = updates
            .iter()
            .filter_map(|update| update.as_ref().ok())
            .filter(|update| match update {
                Update::Start { index: at, .. }
                | Update::Delta { index: at, .. }
                | Update::End { index: at, .. } => *at == index,
                _ => false,
            })
            .collect();
        assert_eq!(
            own.first(),
            Some(&&Update::Start {
                index,
                part: expected_kind(part)
            }),
            "part {index} starts first with its kind: {updates:#?}"
        );
        let starts = own
            .iter()
            .filter(|update| matches!(update, Update::Start { .. }))
            .count();
        assert_eq!(starts, 1, "part {index} starts once");
        let text: String = own
            .iter()
            .filter_map(|update| match update {
                Update::Delta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            text,
            expected_text(part),
            "part {index}'s deltas: {updates:#?}"
        );
        assert_eq!(
            own.iter().rev().find_map(|update| match update {
                Update::End { part, .. } => Some(part),
                _ => None,
            }),
            Some(part),
            "part {index}'s last end is the part: {updates:#?}"
        );
    }
    for update in updates.iter().filter_map(|update| update.as_ref().ok()) {
        if let Update::Start { index, .. }
        | Update::Delta { index, .. }
        | Update::End { index, .. } = update
        {
            assert!(
                *index < done.choice.len(),
                "{update:?} names a part of choice"
            );
        }
    }
    done
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(3),
        output_tokens: Some(5),
        ..Usage::default()
    }
}

fn terminal() -> StreamFinal {
    StreamFinal::new("test", usage(), serde_json::Value::Null)
}

fn text_key(key: u8) -> BlockId {
    BlockId::wire(format!("msg_{key}"))
}

fn reasoning_key(key: u8) -> BlockId {
    BlockId::wire(format!("rs_{key}"))
}

fn tool_key(key: u8) -> BlockId {
    BlockId::wire(format!("call_{key}"))
}

/// One part of a response, as the helper calls that stream it.
#[derive(Clone, Debug)]
enum Part {
    /// A keyed text block: explicit start, fragments, end. No fragments, or
    /// only empty ones, ends with nothing.
    Text(Vec<String>),
    /// A reasoning block: start, fragments, an end that may carry a
    /// signature and may restate the fragments whole, and a signature that
    /// may arrive after the end.
    Reasoning {
        fragments: Vec<String>,
        signature: Option<String>,
        restated: bool,
        late_signature: bool,
    },
    /// A reasoning block delivered whole.
    ReasoningWhole(String),
    /// A tool call streamed as a name and argument fragments that assemble
    /// into JSON.
    Tool {
        name: String,
        arguments: Vec<String>,
    },
    /// A tool call delivered whole on its end.
    ToolWhole(String),
    Image(String),
}

impl Part {
    /// The helper calls that stream this part as the `key`th of its kind.
    fn calls(&self, key: u8) -> Vec<Call> {
        match self {
            Part::Text(fragments) => {
                let mut calls = vec![Call::TextStart(key)];
                calls.extend(
                    fragments
                        .iter()
                        .map(|text| Call::TextDelta(key, text.clone())),
                );
                calls.push(Call::TextEnd(key));
                calls
            }
            Part::Reasoning {
                fragments,
                signature,
                restated,
                late_signature,
            } => {
                let mut calls = vec![Call::ReasoningStart(key)];
                calls.extend(
                    fragments
                        .iter()
                        .map(|text| Call::ReasoningDelta(key, text.clone())),
                );
                let restatement = restated.then(|| Reasoning::new(&fragments.concat()));
                if *late_signature {
                    calls.push(Call::ReasoningEnd(key, restatement, None));
                    calls.push(Call::ReasoningEnd(key, None, signature.clone()));
                } else {
                    calls.push(Call::ReasoningEnd(key, restatement, signature.clone()));
                }
                calls
            }
            Part::ReasoningWhole(text) => vec![Call::ReasoningWhole(key, text.clone())],
            Part::Tool { name, arguments } => {
                let mut calls = vec![Call::ToolName(key, name.clone())];
                calls.extend(
                    arguments
                        .iter()
                        .map(|fragment| Call::ToolArguments(key, fragment.clone())),
                );
                calls.push(Call::ToolEnd(key));
                calls
            }
            Part::ToolWhole(name) => vec![Call::ToolWhole(key, name.clone())],
            Part::Image(url) => vec![Call::Image(url.clone())],
        }
    }
}

/// One helper call on the completion sink.
#[derive(Clone, Debug)]
enum Call {
    TextStart(u8),
    TextDelta(u8, String),
    TextEnd(u8),
    ReasoningStart(u8),
    ReasoningDelta(u8, String),
    ReasoningEnd(u8, Option<Reasoning>, Option<String>),
    ReasoningWhole(u8, String),
    ToolName(u8, String),
    ToolArguments(u8, String),
    ToolEnd(u8),
    ToolWhole(u8, String),
    Image(String),
    Error,
}

fn apply(out: &mut AdapterOutput, call: Call) {
    match call {
        Call::TextStart(key) => out.text_start(text_key(key), None),
        Call::TextDelta(key, text) => out.push(Ok(StreamEvent::text(text_key(key), text))),
        Call::TextEnd(key) => out.text_end(text_key(key)),
        Call::ReasoningStart(key) => out.reasoning_start(&reasoning_key(key), None),
        Call::ReasoningDelta(key, text) => out.reasoning_delta(&reasoning_key(key), None, text),
        Call::ReasoningEnd(key, restatement, signature) => {
            out.reasoning_end(reasoning_key(key), restatement, signature, true);
        }
        Call::ReasoningWhole(key, text) => {
            out.reasoning_block(reasoning_key(key), None, ReasoningContent::Summary(text));
        }
        Call::ToolName(key, name) => out.tool_name(&tool_key(key), name),
        Call::ToolArguments(key, fragment) => out.tool_arguments(&tool_key(key), fragment),
        Call::ToolEnd(key) => {
            out.tool_end(tool_key(key), ToolCallEnd::new(UnparseableToolInput::Error))
        }
        Call::ToolWhole(key, name) => out.tool_end(
            tool_key(key),
            ToolCallEnd::whole(name, serde_json::json!({"q": 1})),
        ),
        Call::Image(url) => out.content(
            &[AssistantContent::Image(Image {
                data: DocumentSourceKind::Url(url),
                media_type: None,
                detail: None,
                additional_params: None,
            })],
            ImagePart::Block,
        ),
        Call::Error => out.error(ProviderError::Provider("in-band".to_owned())),
    }
}

/// `parts`, their calls interleaved in `order` (each entry picks the part
/// whose next call goes next), then the terminal record, through the sink
/// as a decoder emits them and ended as the driver ends a reply.
fn stream_of(
    parts: &[Part],
    order: &[usize],
    error_at: Option<usize>,
) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut keys = [0u8; 6];
    let mut queues: Vec<std::collections::VecDeque<Call>> = parts
        .iter()
        .map(|part| {
            let kind = match part {
                Part::Text(_) => 0,
                Part::Reasoning { .. } | Part::ReasoningWhole(_) => 1,
                Part::Tool { .. } | Part::ToolWhole(_) => 2,
                Part::Image(_) => 3,
            };
            let key = keys[kind];
            keys[kind] += 1;
            part.calls(key).into()
        })
        .collect();
    let mut out = AdapterOutput::new();
    let mut sent = 0;
    let mut pick = order.iter().copied().cycle();
    while queues.iter().any(|queue| !queue.is_empty()) {
        let open: Vec<usize> = (0..queues.len())
            .filter(|at| !queues[*at].is_empty())
            .collect();
        let at = open[pick.next().unwrap_or(0) % open.len()];
        if let Some(call) = queues[at].pop_front() {
            if error_at == Some(sent) {
                apply(&mut out, Call::Error);
            }
            apply(&mut out, call);
            sent += 1;
        }
    }
    out.final_record(terminal());
    out.finish();
    out.into_items()
}

fn fragments(text: String, cuts: Vec<usize>) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut cuts: Vec<usize> = cuts
        .into_iter()
        .map(|cut| cut % (chars.len() + 1))
        .collect();
    cuts.sort_unstable();
    let mut pieces = Vec::new();
    let mut from = 0;
    for cut in cuts.into_iter().chain([chars.len()]) {
        pieces.push(chars[from..cut.max(from)].iter().collect());
        from = cut.max(from);
    }
    pieces
}

fn part() -> impl Strategy<Value = Part> {
    let cuts = || proptest::collection::vec(0usize..16, 0..4);
    prop_oneof![
        ("[a-z ]{0,8}", cuts()).prop_map(|(text, cuts)| Part::Text(fragments(text, cuts))),
        (
            "[a-z ]{0,8}",
            cuts(),
            proptest::option::of("sig_[a-z]{1,3}"),
            any::<bool>(),
            any::<bool>()
        )
            .prop_map(|(text, cuts, signature, restated, late_signature)| {
                Part::Reasoning {
                    fragments: fragments(text, cuts),
                    signature,
                    restated,
                    late_signature,
                }
            }),
        "[a-z]{1,6}".prop_map(Part::ReasoningWhole),
        ("[a-z]{1,4}", "[a-z]{0,5}", cuts()).prop_map(|(name, value, cuts)| Part::Tool {
            name,
            arguments: fragments(format!("{{\"q\": \"{value}\", \"n\": 1}}"), cuts),
        }),
        "[a-z]{1,4}".prop_map(Part::ToolWhole),
        "[a-z]{1,4}".prop_map(Part::Image),
    ]
}

/// 2,048 cases, or `PROPTEST_CASES` when it is set.
fn config() -> ProptestConfig {
    if std::env::var_os("PROPTEST_CASES").is_some() {
        ProptestConfig::default()
    } else {
        ProptestConfig::with_cases(2048)
    }
}

proptest! {
    #![proptest_config(config())]

    /// Over canonical streams the sink's helpers build (text, reasoning,
    /// parallel tool calls with argument fragments, whole calls and images,
    /// interleaved in any order, with or without an in-band error), the
    /// updates keep the index contract against the response they end with,
    /// and that response is the one the stream folds to.
    #[test]
    fn updates_keep_the_index_contract(
        parts in proptest::collection::vec(part(), 0..7),
        order in proptest::collection::vec(0usize..7, 1..40),
        error_at in proptest::option::of(0usize..30),
        plan in proptest::collection::vec(any::<bool>(), 1..12),
    ) {
        let done = assert_contract(&updates_of(stream_of(&parts, &order, error_at)));
        // Read in turns, across calls and around events read directly, the
        // updates keep the contract and end with the same response.
        let in_turns = assert_contract(&updates_in_turns(stream_of(&parts, &order, error_at), &plan));
        prop_assert_eq!(&in_turns, &done);
        // The fold after every item, in-band errors included.
        let mut stream = relayed(stream_of(&parts, &order, error_at));
        futures::executor::block_on(async { while stream.next().await.is_some() {} });
        let folded = stream
            .finish()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        prop_assert_eq!(done, folded);
    }
}

/// Two calls started before either ends (OpenAI Chat's parallel calls), with
/// text between their fragments: the text takes the first position, since
/// it began first in `choice`, and each call starts and ends at its end.
#[test]
fn parallel_calls_with_text_between_them_keep_their_positions() {
    let parts = [
        Part::Tool {
            name: "a".into(),
            arguments: vec!["{\"q\":".into(), " 1}".into()],
        },
        Part::Tool {
            name: "b".into(),
            arguments: vec!["{\"q\":".into(), " 2}".into()],
        },
        Part::Text(vec!["hel".into(), "lo".into()]),
    ];
    // a name, b name, text start, text "hel", a args, b args, ...
    let updates = updates_of(stream_of(&parts, &[0, 1, 2, 2, 0, 1, 2, 0, 1, 2], None));
    let done = assert_contract(&updates);
    assert_eq!(done.choice.len(), 3);
    assert!(matches!(done.choice[0], AssistantContent::Text(_)));
    let first: Vec<&Update> = updates
        .iter()
        .filter_map(|u| u.as_ref().ok())
        .take(2)
        .collect();
    assert_eq!(
        first,
        [
            &Update::Start {
                index: 0,
                part: PartKind::Text
            },
            &Update::Delta {
                index: 0,
                text: "hel".into()
            }
        ],
        "text streams while the calls are open"
    );
}

/// A text block that ends empty is not in `choice`: a part after it takes
/// its place, and waits until the empty block ends.
#[test]
fn a_part_after_an_empty_text_block_waits_for_it_and_takes_its_place() {
    let items = stream_of(
        &[
            Part::Text(vec![String::new()]),
            Part::Text(vec!["b".into()]),
        ],
        &[0, 1, 1, 0, 0, 1],
        None,
    );
    let updates = updates_of(items);
    let done = assert_contract(&updates);
    assert_eq!(done.text(), "b");
}

#[test]
fn text_streams_only_the_text_deltas() {
    let items = stream_of(
        &[
            Part::Reasoning {
                fragments: vec!["think".into()],
                signature: None,
                restated: false,
                late_signature: false,
            },
            Part::Text(vec!["Par".into(), "is".into()]),
            Part::ToolWhole("lookup".into()),
        ],
        &[0],
        None,
    );
    let mut stream = relayed(items);
    let text = futures::executor::block_on(stream.text().collect::<Vec<_>>());
    let text: Vec<String> = text
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap_or_default();
    assert_eq!(text, ["Par", "is"]);
}

/// A scripted mid-stream failure: the text ended before it, the call still
/// open when it failed, then the stream's end. `partial()` holds what ended
/// and no usage, since no terminal record arrived.
#[test]
fn partial_after_a_mid_stream_failure_holds_what_ended() {
    let mut out = AdapterOutput::new();
    apply(&mut out, Call::TextStart(0));
    apply(&mut out, Call::TextDelta(0, "Paris".into()));
    apply(&mut out, Call::TextEnd(0));
    apply(&mut out, Call::ToolName(0, "lookup".into()));
    apply(&mut out, Call::ToolArguments(0, "{\"q\":".into()));
    out.error(ProviderError::Provider("connection reset".into()));
    let mut stream = relayed(out.into_items());
    let updates = futures::executor::block_on(stream.updates().collect::<Vec<_>>());
    assert!(
        updates.iter().any(
            |update| matches!(update, Err(report) if report.message.contains("connection reset"))
        ),
        "{updates:#?}"
    );
    assert!(
        !updates
            .iter()
            .any(|update| matches!(update, Ok(Update::Done(_))))
    );
    let partial = stream.partial();
    assert_eq!(partial.text(), "Paris");
    assert_eq!(partial.tool_calls().count(), 0);
    assert_eq!(partial.usage, Usage::default());
}

/// A consumer that stops after the first update: `partial()` holds what the
/// stream had delivered by then, and the rest was never read.
#[test]
fn partial_after_dropping_the_updates_early_holds_what_was_read() {
    let items = stream_of(
        &[
            Part::Text(vec!["one".into()]),
            Part::Text(vec!["two".into()]),
        ],
        &[0],
        None,
    );
    let mut stream = relayed(items);
    let first_update = {
        let mut updates = stream.updates();
        let first = futures::executor::block_on(updates.next());
        assert!(matches!(first, Some(Ok(Update::Start { index: 0, .. }))));
        first.expect("an update")
    };
    let partial = stream.partial();
    assert!(partial.choice.len() <= 1, "{partial:?}");
    assert!(!partial.text().contains("two"));
    // Reading on continues where the first call stopped: the whole
    // conversation of updates keeps the contract.
    let rest = futures::executor::block_on(stream.updates().collect::<Vec<_>>());
    let mut all = vec![first_update];
    all.extend(rest);
    assert_eq!(assert_contract(&all).text(), "onetwo");
    assert_eq!(
        all[1..3],
        [
            Ok(Update::Delta {
                index: 0,
                text: "one".into()
            }),
            Ok(Update::End {
                index: 0,
                part: AssistantContent::text("one")
            })
        ]
    );
    assert_eq!(stream.partial().text(), "onetwo");
    assert_eq!(stream.partial().usage, usage());
}

/// A stream that ends without its terminal record ends with the truncation
/// error, not `Done`.
#[test]
fn a_stream_without_a_terminal_ends_with_an_error() {
    let mut out = AdapterOutput::new();
    apply(&mut out, Call::TextStart(0));
    apply(&mut out, Call::TextDelta(0, "cut".into()));
    out.finish();
    let updates = updates_of(out.into_items());
    assert!(matches!(updates.last(), Some(Err(report)) if report.message.contains("terminal")));
    assert!(
        updates
            .iter()
            .any(|update| matches!(update, Ok(Update::End { index: 0, .. })))
    );
}
