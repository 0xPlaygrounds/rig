use super::*;

fn raw_event() -> impl Strategy<Value = Result<StreamEvent, ErrorReport>> {
    let key = 0u8..3;
    let fragment = prop_oneof![Just("{".to_owned()), Just("}".to_owned()), "[a-z]{0,4}"];
    let delta = |id, delta| Ok(StreamEvent::BlockDelta { id, delta });
    let end = |id, end| {
        Ok(StreamEvent::BlockEnd {
            id,
            end,
            block: None,
        })
    };
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
        key.clone()
            .prop_map(move |key| end(text_key(key), BlockClose::Text)),
        (key.clone(), proptest::option::of("sig[0-9]"), any::<bool>()).prop_map(
            move |(key, signature, wire_sent)| end(
                reasoning_key(key),
                BlockClose::Reasoning {
                    reasoning: None,
                    signature,
                    wire_sent
                }
            )
        ),
        (
            key,
            prop_oneof![
                Just(UnparseableToolInput::Error),
                Just(UnparseableToolInput::Drop),
                Just(UnparseableToolInput::EmptyObject),
                Just(UnparseableToolInput::Keep)
            ]
        )
            .prop_map(move |(key, policy)| end(
                tool_key(key),
                BlockClose::ToolCall(ToolCallEnd::new(policy))
            )),
        Just(Ok(StreamEvent::Final(StreamFinal::new(
            "test",
            Usage::default(),
            serde_json::json!({})
        )))),
        Just(Ok(StreamEvent::Unknown(serde_json::json!({"x": 1}).into()))),
        Just(Err(ErrorReport::new(
            crate::error::ErrorKind::Provider,
            "relayed failure"
        ))),
        (0u8..3, "[a-z]{1,5}").prop_map(|(kind, text)| authoritative_end(kind, 9, text)),
    ]
}

fn relayed(items: Vec<Result<StreamEvent, ErrorReport>>) -> Vec<Result<StreamEvent, ErrorReport>> {
    use futures::StreamExt;
    futures::executor::block_on(
        crate::streaming::CompletionStream::relay("relay", Box::pin(futures::stream::iter(items)))
            .collect(),
    )
}

fn sink_drained(
    items: &[Result<StreamEvent, ErrorReport>],
) -> Vec<Result<StreamEvent, ErrorReport>> {
    let mut out = AdapterOutput::new();
    let mut drained = Vec::new();
    for item in items {
        out.push(
            item.clone()
                .map_err(|report| ProviderError::Relayed(Box::new(report))),
        );
        drained.extend(out.drain().map(|item| item.map_err(ErrorReport::from)));
    }
    Sink::<Completion>::finish(&mut out);
    drained.extend(out.drain().map(|item| item.map_err(ErrorReport::from)));
    drained
}

fn authoritative_end(kind: u8, key: u8, text: String) -> Result<StreamEvent, ErrorReport> {
    let id = BlockId::wire(format!("authoritative_{key}"));
    let (end, block) = match kind {
        0 => (BlockClose::Text, AssistantContent::text(text)),
        1 => (
            BlockClose::Reasoning {
                reasoning: None,
                signature: None,
                wire_sent: true,
            },
            AssistantContent::Reasoning(Reasoning::new(&text)),
        ),
        _ => (
            BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Error)),
            AssistantContent::ToolCall(crate::message::ToolCall::new(
                crate::message::ToolCallId::from_block(&id),
                crate::message::ToolFunction::new(text, serde_json::json!({})),
            )),
        ),
    };
    Ok(StreamEvent::BlockEnd {
        id,
        end,
        block: Some(block),
    })
}

fn finish_relay(
    items: Vec<Result<StreamEvent, ErrorReport>>,
) -> (Vec<StreamEvent>, crate::completion::CompletionResponse) {
    use futures::StreamExt;
    futures::executor::block_on(async {
        let mut stream = crate::streaming::CompletionStream::relay(
            "relay",
            Box::pin(futures::stream::iter(items)),
        );
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.expect("valid authoritative block"));
        }
        (events, stream.finish().expect("terminal"))
    })
}

#[test]
fn a_carried_reasoning_block_gets_the_terminal_issuer() {
    let (_, response) = finish_relay(vec![
        authoritative_end(1, 0, "a".to_owned()),
        Ok(StreamEvent::Final(StreamFinal::new(
            "test",
            Usage::default(),
            serde_json::json!({}),
        ))),
    ]);
    assert_eq!(
        response.choice,
        vec![AssistantContent::Reasoning(
            Reasoning::new("a").with_provider("test")
        )]
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Both routes use Canonical. This checks relay plumbing: drain order,
    /// errors arriving immediately, and finishing at EOF, not canonicalization itself.
    #[test]
    fn a_relay_canonicalizes_what_it_carries(items in proptest::collection::vec(raw_event(), 0..24)) {
        prop_assert_eq!(relayed(items.clone()), sink_drained(&items));
    }

    #[test]
    fn a_relay_passes_canonical_events_unchanged(
        self_closing in any::<bool>(),
        steps in proptest::collection::vec(step(), 0..24),
    ) {
        let out = if self_closing { AdapterOutput::self_closing() } else { AdapterOutput::new() };
        let items: Vec<_> = once(out, steps).into_iter().map(|item| item.map_err(ErrorReport::from)).collect();
        prop_assert_eq!(relayed(items.clone()), items);
    }

    /// With no assembly, each kind's end returns no newly assembled block.
    /// Canonical must retain the carried block, and the fold must keep their order.
    #[test]
    fn authoritative_blocks_survive_a_relay_in_order(parts in proptest::collection::vec((0u8..3, "[a-z]{1,5}"), 1..20)) {
        let items: Vec<_> = parts.into_iter().enumerate().map(|(key, (kind, text))| authoritative_end(kind, u8::try_from(key).expect("small key"), text)).collect();
        let mut expected: Vec<_> = items.iter().filter_map(|item| match item { Ok(StreamEvent::BlockEnd { block, .. }) => block.clone(), _ => None }).collect();
        let mut items = items;
        items.push(Ok(StreamEvent::Final(StreamFinal::new("test", Usage::default(), serde_json::json!({})))));
        let (events, response) = finish_relay(items);
        let carried: Vec<_> = events.iter().filter_map(|event| match event { StreamEvent::BlockEnd { block, .. } => block.clone(), _ => None }).collect();
        prop_assert_eq!(carried, expected.clone());
        // Finishing attributes previously unattributed reasoning to the terminal issuer.
        for block in &mut expected {
            if let AssistantContent::Reasoning(reasoning) = block { reasoning.provider = Some("test".to_owned()); }
        }
        prop_assert_eq!(response.choice, expected);
    }
}

#[test]
fn a_stale_end_after_a_malformed_input_stays_stale() {
    let items = assert_idempotent(vec![
        Step::ToolArguments(0, "a".to_owned()),
        Step::ToolName(0, "lookup".to_owned()),
        Step::ToolEnd(0, UnparseableToolInput::Error),
        Step::ToolEnd(0, UnparseableToolInput::EmptyObject),
    ]);
    assert_eq!(tool_calls(&items), 0);
    for relayed in [true, false] {
        assert_eq!(tool_calls(&canonical(copied(&items, relayed))), 0);
    }
}

#[test]
fn a_relay_drops_a_second_terminal_and_passes_what_follows_the_first() {
    let terminal = |tokens| {
        Ok(StreamEvent::Final(StreamFinal::new(
            "test",
            Usage {
                total_tokens: Some(tokens),
                ..Usage::default()
            },
            serde_json::json!({}),
        )))
    };
    let late = Ok(StreamEvent::Unknown(
        serde_json::json!({"late": true}).into(),
    ));
    assert_eq!(
        relayed(vec![terminal(1), late.clone(), terminal(2)]),
        vec![terminal(1), late]
    );
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
    let closes = |items: &[Result<StreamEvent, ErrorReport>]| {
        items
            .iter()
            .filter_map(|item| match item {
                Ok(StreamEvent::BlockEnd {
                    id,
                    block: Some(block),
                    ..
                }) => Some((id.clone(), block.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let truncated = relayed(vec![text.clone(), reasoning.clone()]);
    assert_eq!(
        closes(&truncated),
        vec![
            (text_key(0), AssistantContent::text("partial")),
            (
                reasoning_key(0),
                AssistantContent::Reasoning(Reasoning::new("half"))
            )
        ]
    );
    let failed = relayed(vec![text, reasoning, failure.clone()]);
    assert_eq!(failed.len(), 5);
    assert_eq!(failed.get(2), Some(&failure));
    assert_eq!(closes(&failed), closes(&truncated));
}
