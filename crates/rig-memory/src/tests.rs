use super::*;
use rig_core::message::{
    AssistantContent, CallId, ToolCall, ToolFunction, ToolResult, ToolResultContent, UserContent,
};
use rig_core::transcript::validate_canonical;
use std::sync::Mutex;

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str) -> Message {
    Message::assistant(text)
}

fn tool_call_msg() -> Message {
    Message::Assistant(rig_core::message::AssistantMessage::new(vec![
        AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_1"),
            ToolFunction::new(
                rig_core::message::ToolName::new("t").expect("tool name"),
                serde_json::json!({}),
            ),
        )),
    ]))
}

fn tool_result_msg() -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: rig_core::message::CallId::from_wire("call_1"),
            name: rig_core::message::ToolName::new("t").expect("tool name"),
            content: vec![ToolResultContent::text("ok")],
        })],
    }
}

#[tokio::test]
async fn sliding_window_truncates_via_filter() {
    let mem = InMemoryConversationMemory::new()
        .with_filter(SlidingWindowMemory::last_messages(2).into_filter());

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();

    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
}

fn mixed_tool_exchange() -> Vec<Message> {
    let ids = ["call_1", "call_2"].map(CallId::from_wire);
    let calls = ids
        .iter()
        .cloned()
        .map(|id| {
            AssistantContent::ToolCall(ToolCall::new(
                id,
                ToolFunction::new(
                    rig_core::message::ToolName::new("t").expect("tool name"),
                    serde_json::json!({}),
                ),
            ))
        })
        .collect::<Vec<_>>();
    let mut results = vec![UserContent::text("Results follow:")];
    // Canonical results may be interleaved with text and answer calls in
    // a different order, but all calls must be answered in one user message.
    for id in ids.into_iter().rev() {
        results.push(UserContent::tool_result(
            id,
            rig_core::message::ToolName::new("t").expect("tool name"),
            vec![ToolResultContent::text("ok")],
        ));
        results.push(UserContent::text("Result received."));
    }
    vec![
        Message::Assistant(rig_core::message::AssistantMessage::new(calls)),
        Message::User { content: results },
    ]
}

fn window_policies(limit: usize) -> [Box<dyn MemoryPolicy>; 2] {
    [
        Box::new(SlidingWindowMemory::last_messages(limit)),
        Box::new(TokenWindowMemory::new(limit, |_: &Message| 1)),
    ]
}

fn assert_window_partition(policy: &dyn MemoryPolicy, history: &[Message], demoted_len: usize) {
    let (kept, demoted) = policy.apply_with_demoted(history.to_vec()).unwrap();
    assert_eq!(validate_canonical(&kept), Ok(()));
    assert_eq!(kept, history[demoted_len..]);
    assert_eq!(demoted, history[..demoted_len]);
    assert_eq!(policy.apply(history.to_vec()).unwrap(), kept);
    assert_eq!(demoted.into_iter().chain(kept).collect::<Vec<_>>(), history);
}

fn assert_mixed_content_cleanup(policy: &dyn MemoryPolicy) {
    let mut history = mixed_tool_exchange();
    history.extend([tool_call_msg(), tool_result_msg(), assistant("done")]);
    assert_eq!(validate_canonical(&history), Ok(()));
    // The first call is truncated; the later complete exchange must survive.
    assert_window_partition(policy, &history, 2);
}

/// Pure canonical-history transformation; no provider traffic is involved.
#[test]
fn sliding_window_demotes_mixed_content_orphan_results() {
    assert_mixed_content_cleanup(&SlidingWindowMemory::last_messages(4));
}

/// Pure canonical-history transformation; no provider traffic is involved.
#[test]
fn token_window_demotes_mixed_content_orphan_results() {
    assert_mixed_content_cleanup(&TokenWindowMemory::new(4, |_: &Message| 1));
}

#[test]
fn window_policies_demote_orphan_results_after_system_messages() {
    let mut history = mixed_tool_exchange();
    history.insert(1, Message::system("context between call and results"));
    history.extend([Message::system("keep this context"), assistant("done")]);
    assert_eq!(validate_canonical(&history), Ok(()));
    for policy in window_policies(4) {
        assert_window_partition(policy.as_ref(), &history, 3);
    }
}

#[test]
fn heuristic_counter_handles_tool_calls() {
    let counter = HeuristicTokenCounter::default();
    let cost = counter.count(&tool_call_msg());
    assert!(cost > 0);
}

#[test]
fn heuristic_counter_handles_system_messages() {
    let counter = HeuristicTokenCounter::default();
    let cost = counter.count(&Message::System {
        content: "you are helpful".into(),
    });
    assert!(cost > 0);
}

#[test]
fn heuristic_counter_clamps_invalid_bytes_per_token() {
    // Zero/NaN/negative ratios fall back to 1.0 instead of panicking.
    let counter = HeuristicTokenCounter::new(0.0, 0, 0);
    assert!(counter.count(&user("abcd")) >= 4);
    let nan = HeuristicTokenCounter::new(f32::NAN, 0, 0);
    assert!(nan.count(&user("abcd")) >= 4);
}

#[test]
fn boxed_token_counter_forwards_count() {
    let counter: Box<dyn TokenCounter> = Box::new(|_: &Message| 7);
    assert_eq!(counter.count(&user("a")), 7);
}

#[test]
fn into_filter_returns_input_on_policy_error() {
    struct FailingPolicy;
    impl MemoryPolicy for FailingPolicy {
        fn apply(&self, _: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
            Err(MemoryError::Policy("intentional failure".into()))
        }
    }

    let filter = FailingPolicy.into_filter();
    let input = vec![user("a"), assistant("b"), user("c")];
    let out = filter(input.clone());
    assert_eq!(
        out.len(),
        input.len(),
        "history must be preserved on policy error"
    );
}

#[tokio::test]
async fn policy_memory_append_and_clear_delegate_to_inner() {
    let mem = PolicyMemory::new(InMemoryConversationMemory::new(), NoopMemoryPolicy);
    mem.append(&"c".into(), vec![user("hi"), assistant("ok")])
        .await
        .unwrap();
    assert_eq!(mem.load(&"c".into()).await.unwrap().len(), 2);

    mem.clear(&"c".into()).await.unwrap();
    assert!(mem.load(&"c".into()).await.unwrap().is_empty());
}

#[test]
fn noop_policy_demotes_nothing() {
    let (kept, demoted) = NoopMemoryPolicy
        .apply_with_demoted(vec![user("a"), assistant("b")])
        .unwrap();
    assert_eq!(kept.len(), 2);
    assert!(demoted.is_empty());
}

#[test]
fn arc_memory_policy_preserves_demoted_metadata() {
    let policy: Arc<dyn MemoryPolicy> = Arc::new(SlidingWindowMemory::last_messages(1));
    let (kept, demoted) = policy
        .apply_with_demoted(vec![user("old"), assistant("new")])
        .unwrap();

    assert_eq!(kept.len(), 1);
    assert_eq!(demoted.len(), 1);
}

#[derive(Default)]
struct CountingHook {
    seen: Mutex<Vec<(String, Vec<Message>)>>,
}

impl CountingHook {
    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

impl DemotionHook for CountingHook {
    fn on_demote<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap()
                .push((conversation_id.to_string(), messages));
            Ok(())
        })
    }
}

#[tokio::test]
async fn demoting_windows_deliver_mixed_orphan_prefix_once_in_order() {
    for policy in window_policies(2) {
        let hook = Arc::new(CountingHook::default());
        let mem =
            DemotingPolicyMemory::new(InMemoryConversationMemory::new(), policy, hook.clone());
        let id: ConversationId = "mixed-results".into();
        let mut history = mixed_tool_exchange();
        history.push(assistant("done"));
        assert_eq!(validate_canonical(&history), Ok(()));
        mem.append(&id, history.clone()).await.unwrap();
        for _ in 0..2 {
            assert_eq!(mem.load(&id).await.unwrap(), history[2..]);
        }
        assert_eq!(hook.calls(), 1);
        assert_eq!(hook.seen.lock().unwrap()[0].1, history[..2]);

        // Advancing the nominal boundary onto an already-demoted result
        // must not deliver that result twice or skip the next real eviction.
        let next = [user("next"), assistant("latest")];
        mem.append(&id, vec![next[0].clone()]).await.unwrap();
        assert_eq!(
            mem.load(&id).await.unwrap(),
            vec![history[2].clone(), next[0].clone()]
        );
        assert_eq!(hook.calls(), 1);
        mem.append(&id, vec![next[1].clone()]).await.unwrap();
        for _ in 0..2 {
            assert_eq!(mem.load(&id).await.unwrap(), next);
        }
        let seen = hook.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].1, history[2..]);
        let delivered: Vec<_> = seen
            .iter()
            .flat_map(|(_, messages)| messages.clone())
            .collect();
        assert_eq!(delivered, history);
    }
}

#[derive(Default)]
struct FailingHook {
    calls: Mutex<usize>,
}

impl DemotionHook for FailingHook {
    fn on_demote<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            *self.calls.lock().unwrap() += 1;
            Err(MemoryError::backend(std::io::Error::other("hook failed")))
        })
    }
}

#[tokio::test]
async fn demoting_policy_memory_does_not_advance_watermark_on_hook_failure() {
    let hook = Arc::new(FailingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook.clone(),
    );
    mem.append(&"c".into(), vec![user("1"), assistant("2")])
        .await
        .unwrap();

    assert!(mem.load(&"c".into()).await.is_err());
    assert!(mem.load(&"c".into()).await.is_err());
    assert_eq!(*hook.calls.lock().unwrap(), 2);
}

#[tokio::test]
async fn demoting_policy_memory_skips_hook_when_nothing_evicted() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(10),
        hook.clone(),
    );

    mem.append(&"c".into(), vec![user("1"), assistant("2")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(hook.calls(), 0);
}

#[tokio::test]
async fn demoting_policy_memory_with_noop_hook_behaves_like_policy_memory() {
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        NoopDemotionHook,
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();
    assert_eq!(mem.load(&"c".into()).await.unwrap().len(), 1);
}

#[tokio::test]
async fn demoting_stale_successful_load_does_not_clear_new_reservation() {
    #[derive(Default)]
    struct IndividuallyGatedHook {
        releases: Mutex<Vec<Arc<tokio::sync::Notify>>>,
    }

    impl IndividuallyGatedHook {
        fn call_count(&self) -> usize {
            self.releases.lock().unwrap().len()
        }

        async fn wait_for_call_count(&self, expected: usize) {
            while self.call_count() < expected {
                tokio::task::yield_now().await;
            }
        }

        fn release_call(&self, index: usize) {
            let release = self.releases.lock().unwrap()[index].clone();
            release.notify_one();
        }
    }

    impl DemotionHook for IndividuallyGatedHook {
        fn on_demote<'a>(
            &'a self,
            _conversation_id: &'a ConversationId,
            _messages: Vec<Message>,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            let release = Arc::new(tokio::sync::Notify::new());
            self.releases.lock().unwrap().push(release.clone());
            Box::pin(async move {
                release.notified().await;
                Ok(())
            })
        }
    }

    let hook = Arc::new(IndividuallyGatedHook::default());
    let mem = Arc::new(DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook.clone(),
    ));

    mem.append(
        &"c".into(),
        vec![user("old 1"), assistant("old 2"), user("old 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let stale = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    hook.wait_for_call_count(1).await;

    mem.clear(&"c".into()).await.unwrap();
    mem.append(
        &"c".into(),
        vec![user("fresh 1"), assistant("fresh 2"), user("fresh 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let fresh = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    hook.wait_for_call_count(2).await;

    // Let the stale load finish successfully after the conversation id has
    // been reused. Its post-await update must not clear the fresh in-flight
    // reservation.
    hook.release_call(0);
    assert_eq!(stale.await.unwrap().unwrap().len(), 1);
    assert_eq!(hook.call_count(), 2);

    let mem_load = mem.clone();
    let mut concurrent = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    let hook_wait = hook.clone();
    let concurrent_kept = tokio::select! {
        result = &mut concurrent => result.unwrap().unwrap(),
        _ = hook_wait.wait_for_call_count(3) => {
            panic!("stale successful load must not clear the fresh in-flight reservation")
        }
    };
    assert_eq!(
        hook.call_count(),
        2,
        "stale successful load must not clear the fresh in-flight reservation"
    );

    hook.release_call(1);
    assert_eq!(fresh.await.unwrap().unwrap().len(), 1);
    assert_eq!(concurrent_kept.len(), 1);

    mem.load(&"c".into()).await.unwrap();
    assert_eq!(hook.call_count(), 2);
}

// ----------------------------------------------------------------
// CompactingMemory tests
// ----------------------------------------------------------------

// A compactor that fails the first call and succeeds afterwards, so we
// can verify failure is propagated and the watermark is not advanced.
#[derive(Default)]
struct FlakyCompactor {
    calls: std::sync::atomic::AtomicUsize,
}

impl Compactor for FlakyCompactor {
    type Artifact = TextSummary;

    fn compact<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        _carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                Err(MemoryError::Policy("flaky".into()))
            } else {
                Ok(TextSummary(format!("compacted {} messages", evicted.len())))
            }
        })
    }
}

#[tokio::test]
async fn compacting_failure_does_not_advance_watermark() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        FlakyCompactor::default(),
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();

    let err = mem.load(&"c".into()).await.unwrap_err();
    assert!(matches!(err, MemoryError::Policy(_)));

    // Retry should succeed and produce a summary.
    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
    let Message::System { content } = &loaded[0] else {
        panic!("expected summary")
    };
    assert!(content.contains("compacted"));
}

#[tokio::test]
async fn compacting_into_inner_returns_components() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        TemplateCompactor::new(),
    );
    let (_inner, _policy, _compactor) = mem.into_inner();
}

#[tokio::test]
async fn compacting_composes_with_token_window() {
    // Verify CompactingMemory is policy-agnostic: works over a
    // TokenWindowMemory just as well as a SlidingWindowMemory.
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        TokenWindowMemory::new(30, HeuristicTokenCounter::default()),
        TemplateCompactor::new(),
    );
    mem.append(
        &"c".into(),
        vec![
            user("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            assistant("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            user("cccccccccccccccccccc"),
            assistant("d"),
        ],
    )
    .await
    .unwrap();
    let loaded = mem.load(&"c".into()).await.unwrap();
    // Some prefix should have been evicted; expect a summary in front.
    assert!(loaded.len() >= 2);
    assert!(matches!(&loaded[0], Message::System { .. }));
}

#[tokio::test]
async fn template_compactor_renders_system_messages() {
    let compactor = TemplateCompactor::new();
    let evicted = vec![
        Message::System {
            content: "you are helpful".into(),
        },
        user("hi"),
        assistant("hello"),
    ];
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    let s = summary.as_str();
    assert!(s.contains("system: you are helpful"), "got: {s}");
    assert!(s.contains("user: hi"));
    assert!(s.contains("assistant: hello"));
}

#[tokio::test]
async fn template_compactor_renders_tool_call_marker() {
    let compactor = TemplateCompactor::new();
    let evicted = vec![tool_call_msg(), tool_result_msg()];
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    let s = summary.as_str();
    assert!(s.contains("[tool call: t]"), "got: {s}");
    assert!(s.contains("[tool result]"), "got: {s}");
}

/// The separator is suppressed only while the line is still empty, so an
/// empty *leading* part contributes nothing while an empty *interior* one
/// still spends its space. `Vec::join(" ")` would flatten that asymmetry;
/// this pins the rendered bytes so a future tidy-up cannot.
#[tokio::test]
async fn template_compactor_separates_parts_asymmetrically_around_empty_text() {
    let compactor = TemplateCompactor::new();
    let evicted = vec![Message::User {
        content: vec![
            UserContent::text(""),
            UserContent::text("a"),
            UserContent::text(""),
            UserContent::text("b"),
        ],
    }];
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    let rendered = summary.as_str();
    assert!(
        rendered.contains("user: a  b"),
        "leading empty part adds no separator, interior one adds its own; got: {rendered}"
    );
}

#[tokio::test]
async fn template_compactor_with_max_bytes_zero_is_unbounded() {
    let compactor = TemplateCompactor::new().with_max_bytes(0);
    let mut evicted = Vec::new();
    for i in 0..200 {
        evicted.push(user(&format!("msg {i}")));
    }
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
}

#[tokio::test]
async fn compacting_summary_stays_bounded_across_rolls() {
    // With a capped TemplateCompactor, repeated rolling must not let
    // the summary grow without bound.
    let cap = 512;
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        TemplateCompactor::new().with_max_bytes(cap),
    );
    mem.append(&"c".into(), vec![user("seed-a"), assistant("seed-b")])
        .await
        .unwrap();
    for i in 0..30 {
        mem.append(
            &"c".into(),
            vec![
                user(&format!("user line {i} ----- padding padding padding")),
                assistant(&format!("assistant line {i} ----- padding padding")),
            ],
        )
        .await
        .unwrap();
        mem.load(&"c".into()).await.unwrap();
    }
    let loaded = mem.load(&"c".into()).await.unwrap();
    let Message::System { content } = &loaded[0] else {
        panic!("expected summary");
    };
    // Allow some slack for header + marker overhead.
    let slack = "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len();
    assert!(
        content.len() <= cap + slack,
        "summary grew to {} bytes (cap {}, slack {})",
        content.len(),
        cap,
        slack,
    );
}

#[tokio::test]
async fn compacting_concurrent_with_clear_does_not_resurrect_state() {
    // A clear that lands while compaction is in flight must not be
    // overwritten by the post-await state update.
    use std::sync::atomic::{AtomicBool, Ordering};

    struct GatedCompactor {
        release: tokio::sync::Notify,
        entered: AtomicBool,
    }

    impl Compactor for GatedCompactor {
        type Artifact = TextSummary;

        fn compact<'a>(
            &'a self,
            _conversation_id: &'a ConversationId,
            _evicted: &'a [Message],
            _carry_over: Option<&'a Self::Artifact>,
        ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
            Box::pin(async move {
                self.entered.store(true, Ordering::SeqCst);
                self.release.notified().await;
                Ok(TextSummary("late summary".into()))
            })
        }
    }

    let compactor = Arc::new(GatedCompactor {
        release: tokio::sync::Notify::new(),
        entered: AtomicBool::new(false),
    });
    let mem = Arc::new(CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    ));
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();

    // Kick off a load that will block inside the compactor.
    let mem_load = mem.clone();
    let load_handle = tokio::spawn(async move { mem_load.load(&"c".into()).await });

    // Wait for the compactor to have entered.
    while !compactor.entered.load(Ordering::SeqCst) {
        tokio::task::yield_now().await;
    }

    // Clear while the compaction is in flight.
    mem.clear(&"c".into()).await.unwrap();

    // Release the compactor; it should complete and *not* resurrect
    // the cleared state.
    compactor.release.notify_one();
    let _ = load_handle.await.unwrap();

    assert_eq!(mem.tracked_conversations(), 0);
    // A subsequent load on the empty backend returns nothing.
    assert!(mem.load(&"c".into()).await.unwrap().is_empty());
}

#[tokio::test]
async fn compacting_stale_cancelled_load_does_not_clear_new_reservation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct GatedCompactor {
        release: tokio::sync::Notify,
        rendezvous: tokio::sync::Notify,
        entered: AtomicUsize,
    }

    impl Compactor for GatedCompactor {
        type Artifact = TextSummary;

        fn compact<'a>(
            &'a self,
            _conversation_id: &'a ConversationId,
            _evicted: &'a [Message],
            _carry_over: Option<&'a Self::Artifact>,
        ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
            Box::pin(async move {
                self.entered.fetch_add(1, Ordering::SeqCst);
                self.rendezvous.notify_one();
                self.release.notified().await;
                Ok(TextSummary("ran".into()))
            })
        }
    }

    let compactor = Arc::new(GatedCompactor {
        release: tokio::sync::Notify::new(),
        rendezvous: tokio::sync::Notify::new(),
        entered: AtomicUsize::new(0),
    });
    let mem = Arc::new(CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    ));

    mem.append(
        &"c".into(),
        vec![user("old 1"), assistant("old 2"), user("old 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let stale = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    compactor.rendezvous.notified().await;
    assert_eq!(compactor.entered.load(Ordering::SeqCst), 1);

    mem.clear(&"c".into()).await.unwrap();
    mem.append(
        &"c".into(),
        vec![user("fresh 1"), assistant("fresh 2"), user("fresh 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let fresh = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    compactor.rendezvous.notified().await;
    assert_eq!(compactor.entered.load(Ordering::SeqCst), 2);

    stale.abort();
    let _ = stale.await;

    let mem_load = mem.clone();
    let mut concurrent = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    let concurrent_kept = tokio::select! {
        result = &mut concurrent => result.unwrap().unwrap(),
        _ = compactor.rendezvous.notified() => {
            panic!("stale guard must not clear the fresh in-flight reservation")
        }
    };
    assert_eq!(
        compactor.entered.load(Ordering::SeqCst),
        2,
        "stale guard must not clear the fresh in-flight reservation"
    );

    compactor.release.notify_one();
    assert_eq!(fresh.await.unwrap().unwrap().len(), 2);
    assert_eq!(concurrent_kept.len(), 1);
    assert_eq!(compactor.entered.load(Ordering::SeqCst), 2);
}
