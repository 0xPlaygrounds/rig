use super::*;
use rig_core::message::{
    AssistantContent, ToolCall, ToolCallId, ToolFunction, ToolResult, ToolResultContent,
    UserContent,
};
use std::sync::Mutex;

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str) -> Message {
    Message::assistant(text)
}

fn tool_call_msg() -> Message {
    Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::new(
            ToolCallId::new_or_mint("call_1"),
            ToolFunction::new("t".into(), serde_json::json!({})),
        ))],
    }
}

fn tool_result_msg() -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_mint("call_1"),
            provider: None,
            name: "t".into(),
            content: vec![ToolResultContent::text("ok")],
        })],
    }
}

#[test]
fn noop_policy_is_identity() {
    let msgs = vec![user("a"), assistant("b")];
    let out = NoopMemoryPolicy.apply(msgs).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn sliding_window_passthrough_when_under_limit() {
    let policy = SlidingWindowMemory::last_messages(5);
    let out = policy.apply(vec![user("1"), assistant("2")]).unwrap();
    assert_eq!(out.len(), 2);
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

#[test]
fn sliding_window_drops_leading_orphan_tool_result() {
    let policy = SlidingWindowMemory::last_messages(3);
    let out = policy
        .apply(vec![
            tool_call_msg(),
            tool_result_msg(),
            user("after"),
            assistant("done"),
        ])
        .unwrap();

    assert_eq!(out.len(), 2);
    assert!(matches!(out.first(), Some(Message::User { content })
        if matches!(content.first(), Some(UserContent::Text(_)))));
}

#[test]
fn token_window_keeps_within_budget() {
    let msgs = vec![
        user("aaaa"),
        assistant("bbbb"),
        user("cccc"),
        assistant("dddd"),
    ];
    let policy = TokenWindowMemory::new(2, |_: &Message| 1);
    let out = policy.apply(msgs).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn token_window_passes_through_when_under_budget() {
    let msgs = vec![user("a"), assistant("b")];
    let policy = TokenWindowMemory::new(usize::MAX, |_: &Message| 1);
    let out = policy.apply(msgs).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn token_window_drops_leading_orphan_tool_result() {
    let policy = TokenWindowMemory::new(25, |_: &Message| 10);
    let out = policy
        .apply(vec![tool_call_msg(), tool_result_msg(), user("after")])
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(matches!(out.first(), Some(Message::User { content })
        if matches!(content.first(), Some(UserContent::Text(_)))));
}

#[test]
fn token_window_skips_message_larger_than_budget() {
    let policy = TokenWindowMemory::new(5, |_: &Message| 10);
    let out = policy.apply(vec![user("anything")]).unwrap();
    assert!(out.is_empty());
}

#[test]
fn heuristic_counter_charges_overhead_per_message() {
    let counter = HeuristicTokenCounter::default();
    let empty = counter.count(&user(""));
    assert!(
        empty >= 4,
        "default per-message overhead is at least 4 tokens"
    );
}

#[test]
fn heuristic_counter_is_monotonic_in_text_length() {
    let counter = HeuristicTokenCounter::default();
    let small = counter.count(&user("hi"));
    let big = counter.count(&user(&"x".repeat(400)));
    assert!(big > small);
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
fn heuristic_counter_drives_token_window() {
    let policy = TokenWindowMemory::new(100, HeuristicTokenCounter::default());
    let msgs = vec![user(&"a".repeat(2_000)), user("short")];
    let out = policy.apply(msgs).unwrap();
    // The huge message must be evicted; the short one retained.
    assert_eq!(out.len(), 1);
}

#[test]
fn arc_token_counter_can_drive_token_window() {
    let counter: Arc<dyn TokenCounter> = Arc::new(|_: &Message| 1);
    let policy = TokenWindowMemory::new(2, counter);
    let out = policy
        .apply(vec![user("a"), assistant("b"), user("c")])
        .unwrap();

    assert_eq!(out.len(), 2);
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
async fn policy_memory_truncates_loaded_history() {
    let mem = PolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
    );

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();

    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
}

#[tokio::test]
async fn policy_memory_propagates_policy_errors() {
    struct FailingPolicy;
    impl MemoryPolicy for FailingPolicy {
        fn apply(&self, _: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
            Err(MemoryError::Policy("intentional failure".into()))
        }
    }

    let mem = PolicyMemory::new(InMemoryConversationMemory::new(), FailingPolicy);
    mem.append(&"c".into(), vec![user("1"), assistant("2")])
        .await
        .unwrap();

    let result = mem.load(&"c".into()).await;
    assert!(matches!(result, Err(MemoryError::Policy(_))));
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
fn sliding_window_reports_demoted_prefix() {
    let policy = SlidingWindowMemory::last_messages(2);
    let (kept, demoted) = policy
        .apply_with_demoted(vec![
            user("oldest"),
            assistant("old"),
            user("recent"),
            assistant("latest"),
        ])
        .unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(demoted.len(), 2);
}

#[test]
fn token_window_reports_demoted_prefix() {
    let policy = TokenWindowMemory::new(2, |_: &Message| 1);
    let (kept, demoted) = policy
        .apply_with_demoted(vec![user("a"), assistant("b"), user("c"), assistant("d")])
        .unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(demoted.len(), 2);
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

#[test]
fn boxed_memory_policy_preserves_demoted_metadata() {
    let policy: Box<dyn MemoryPolicy> = Box::new(SlidingWindowMemory::last_messages(1));
    let (kept, demoted) = policy
        .apply_with_demoted(vec![user("old"), assistant("new")])
        .unwrap();

    assert_eq!(kept.len(), 1);
    assert_eq!(demoted.len(), 1);
}

#[test]
fn sliding_window_demotes_orphan_tool_result_with_prefix() {
    // Window keeps the last 2 messages, but the leading message of that
    // window is an orphan tool result; it must be moved into `demoted`
    // so the hook can preserve it.
    let policy = SlidingWindowMemory::last_messages(2);
    let (kept, demoted) = policy
        .apply_with_demoted(vec![
            tool_call_msg(),
            tool_result_msg(),
            user("after"),
            assistant("done"),
        ])
        .unwrap();
    assert_eq!(kept.len(), 2);
    assert!(matches!(kept.first(), Some(Message::User { content })
        if matches!(content.first(), Some(UserContent::Text(_)))));
    assert_eq!(demoted.len(), 2);
}

#[derive(Default)]
struct CountingHook {
    seen: Mutex<Vec<(String, Vec<Message>)>>,
}

impl CountingHook {
    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
    fn last_demoted_count(&self) -> usize {
        self.seen.lock().unwrap().last().map_or(0, |(_, m)| m.len())
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
async fn demoting_policy_memory_invokes_hook_on_truncation() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        hook.clone(),
    );

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();

    let kept = mem.load(&"c".into()).await.unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(hook.calls(), 1);
    assert_eq!(hook.last_demoted_count(), 2);
}

#[tokio::test]
async fn demoting_policy_memory_does_not_replay_demotions() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        hook.clone(),
    );

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();

    mem.load(&"c".into()).await.unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(hook.calls(), 1);
    assert_eq!(hook.last_demoted_count(), 2);
}

#[tokio::test]
async fn demoting_policy_memory_only_reports_newly_demoted_messages() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        hook.clone(),
    );

    mem.append(
        &"c".into(),
        vec![user("1"), assistant("2"), user("3"), assistant("4")],
    )
    .await
    .unwrap();
    mem.load(&"c".into()).await.unwrap();

    mem.append(&"c".into(), vec![user("5")]).await.unwrap();
    mem.load(&"c".into()).await.unwrap();

    assert_eq!(hook.calls(), 2);
    assert_eq!(hook.last_demoted_count(), 1);
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
async fn demoting_policy_memory_clear_resets_watermark() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook.clone(),
    );

    mem.append(&"c".into(), vec![user("1"), assistant("2")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    mem.clear(&"c".into()).await.unwrap();
    mem.append(&"c".into(), vec![user("3"), assistant("4")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();

    assert_eq!(hook.calls(), 2);
    assert_eq!(hook.last_demoted_count(), 1);
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

/// Hook that blocks until the test releases it. Used to provoke the
/// concurrent-load race against the in-flight gate.
struct GatedHook {
    calls: Arc<std::sync::atomic::AtomicUsize>,
    rendezvous: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

impl DemotionHook for GatedHook {
    fn on_demote<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        let calls = self.calls.clone();
        let rendezvous = self.rendezvous.clone();
        let release = self.release.clone();
        Box::pin(async move {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            rendezvous.notify_one();
            release.notified().await;
            Ok(())
        })
    }
}

#[tokio::test]
async fn demoting_policy_memory_serialises_concurrent_loads() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let rendezvous = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let hook = GatedHook {
        calls: calls.clone(),
        rendezvous: rendezvous.clone(),
        release: release.clone(),
    };

    let mem = Arc::new(DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook,
    ));

    mem.append(&"c".into(), vec![user("1"), assistant("2"), user("3")])
        .await
        .unwrap();

    let m1 = mem.clone();
    let first = tokio::spawn(async move { m1.load(&"c".into()).await });

    // Wait until the first load has entered the hook.
    rendezvous.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Second concurrent load on the same conversation must skip the
    // hook entirely (in-flight gate) and return the truncated view.
    let kept = mem.load(&"c".into()).await.unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "hook must not double-fire");

    // Release the first load and confirm it completes successfully.
    release.notify_one();
    let kept_first = first.await.unwrap().unwrap();
    assert_eq!(kept_first.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Subsequent loads observe the watermark and don't re-fire.
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn demoting_policy_memory_dropped_load_releases_in_flight_gate() {
    // If a `load(...)` future is dropped while awaiting the hook, the
    // in-flight gate must not leak: subsequent loads on the same
    // conversation must be able to retry demotion.
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let rendezvous = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let hook = GatedHook {
        calls: calls.clone(),
        rendezvous,
        release: release.clone(),
    };

    let mem = Arc::new(DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook,
    ));

    mem.append(&"c".into(), vec![user("1"), assistant("2"), user("3")])
        .await
        .unwrap();

    // Kick off a load that will block inside the hook, then abort it
    // while awaiting — simulating a caller-side timeout or
    // `tokio::select!` cancellation.
    let mem_load = mem.clone();
    let handle = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    while calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    handle.abort();
    let _ = handle.await;

    // The aborted future was dropped without clearing in_flight via
    // the success/error branches; the RAII guard's `Drop` should have
    // released it. A new load must therefore be able to drive a fresh
    // demotion rather than short-circuiting forever.
    let mem_load = mem.clone();
    let retry = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    for _ in 0..1_000 {
        if calls.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "retry must re-enter the hook after cancellation"
    );

    release.notify_one();
    let kept = retry.await.unwrap().unwrap();
    assert_eq!(kept.len(), 1);

    // The successful retry advances the watermark, so future loads
    // should not fire the hook again.
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn demoting_stale_cancelled_load_does_not_clear_new_reservation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let rendezvous = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let hook = GatedHook {
        calls: calls.clone(),
        rendezvous: rendezvous.clone(),
        release: release.clone(),
    };

    let mem = Arc::new(DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook,
    ));

    mem.append(
        &"c".into(),
        vec![user("old 1"), assistant("old 2"), user("old 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let stale = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    rendezvous.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    mem.clear(&"c".into()).await.unwrap();
    mem.append(
        &"c".into(),
        vec![user("fresh 1"), assistant("fresh 2"), user("fresh 3")],
    )
    .await
    .unwrap();

    let mem_load = mem.clone();
    let fresh = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    rendezvous.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    stale.abort();
    let _ = stale.await;

    let mem_load = mem.clone();
    let mut concurrent = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    let concurrent_kept = tokio::select! {
        result = &mut concurrent => result.unwrap().unwrap(),
        _ = rendezvous.notified() => {
            panic!("stale guard must not clear the fresh in-flight reservation")
        }
    };
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "stale guard must not clear the fresh in-flight reservation"
    );

    release.notify_one();
    assert_eq!(fresh.await.unwrap().unwrap().len(), 1);
    assert_eq!(concurrent_kept.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
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

#[tokio::test]
async fn forget_drops_in_process_watermark() {
    let hook = Arc::new(CountingHook::default());
    let mem = DemotingPolicyMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        hook.clone(),
    );

    mem.append(&"c".into(), vec![user("1"), assistant("2")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(mem.tracked_conversations(), 1);
    assert_eq!(hook.calls(), 1);

    // After forgetting, the next load on the same (still-populated)
    // backend re-delivers the demotion. This is the documented
    // contract: forget()/restart re-fire the hook, hooks must be
    // idempotent.
    mem.forget(&"c".into());
    assert_eq!(mem.tracked_conversations(), 0);
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(hook.calls(), 2);
}

// ----------------------------------------------------------------
// CompactingMemory tests
// ----------------------------------------------------------------

#[tokio::test]
async fn compacting_no_demotion_returns_kept_only() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(10),
        TemplateCompactor::new(),
    );

    mem.append(&"c".into(), vec![user("hi"), assistant("hello")])
        .await
        .unwrap();
    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
    // No tracking entry needed when nothing was demoted on the first load.
    // (We may have inserted a default entry; what matters is that no
    // summary message was spliced in.)
    assert!(matches!(&loaded[0], Message::User { .. }));
}

#[tokio::test]
async fn compacting_splices_summary_when_demoted() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        TemplateCompactor::new(),
    );

    mem.append(
        &"c".into(),
        vec![
            user("first"),
            assistant("second"),
            user("third"),
            assistant("fourth"),
        ],
    )
    .await
    .unwrap();

    let loaded = mem.load(&"c".into()).await.unwrap();
    // Expected shape: [summary, third, fourth]
    assert_eq!(loaded.len(), 3);
    let Message::System { content } = &loaded[0] else {
        panic!("expected summary as system message");
    };
    assert!(content.contains("[Conversation summary so far]"));
    assert!(content.contains("user: first"));
    assert!(content.contains("assistant: second"));
    // The kept window is intact.
    let Message::User { content } = &loaded[1] else {
        panic!("expected kept user message");
    };
    let Some(UserContent::Text(t)) = content.first() else {
        panic!("expected text");
    };
    assert_eq!(t.text, "third");
}

#[tokio::test]
async fn compacting_rolls_summary_forward() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        TemplateCompactor::new(),
    );

    mem.append(
        &"c".into(),
        vec![user("a"), assistant("b"), user("c"), assistant("d")],
    )
    .await
    .unwrap();

    let first = mem.load(&"c".into()).await.unwrap();
    let Message::System { content } = &first[0] else {
        panic!("summary missing");
    };
    let first_summary = content.clone();
    assert!(first_summary.contains("user: a"));
    assert!(first_summary.contains("assistant: b"));

    // Append more turns; the next load should fold the previous summary
    // into a new one that also covers the newly-evicted prefix.
    mem.append(&"c".into(), vec![user("e"), assistant("f")])
        .await
        .unwrap();
    let second = mem.load(&"c".into()).await.unwrap();
    let Message::System { content } = &second[0] else {
        panic!("summary missing");
    };
    // The new summary contains the old summary text (carry_over) plus
    // the freshly-evicted lines.
    assert!(content.contains(&first_summary));
    assert!(content.contains("user: c"));
    assert!(content.contains("assistant: d"));
}

#[tokio::test]
async fn compacting_idempotent_within_process() {
    // Loading twice with no new evictions reuses the stored summary
    // and does not re-run the compactor (we observe this via the
    // produced text: a re-run with a non-None carry_over would double
    // the header line).
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        TemplateCompactor::new(),
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();

    let first = mem.load(&"c".into()).await.unwrap();
    let second = mem.load(&"c".into()).await.unwrap();
    assert_eq!(first.len(), second.len());
    let Message::System { content: c1 } = &first[0] else {
        panic!()
    };
    let Message::System { content: c2 } = &second[0] else {
        panic!()
    };
    assert_eq!(c1, c2);
}

#[tokio::test]
async fn compacting_clear_drops_summary() {
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        TemplateCompactor::new(),
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(mem.tracked_conversations(), 1);

    mem.clear(&"c".into()).await.unwrap();
    assert_eq!(mem.tracked_conversations(), 0);
    assert!(mem.load(&"c".into()).await.unwrap().is_empty());
}

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

// A compactor that records every invocation, including the lengths of
// its `evicted` slice and whether `carry_over` was supplied.
#[derive(Default)]
struct CountingCompactor {
    log: Mutex<Vec<(usize, bool)>>,
}

impl CountingCompactor {
    fn calls(&self) -> Vec<(usize, bool)> {
        self.log.lock().unwrap().clone()
    }
}

impl Compactor for CountingCompactor {
    type Artifact = TextSummary;

    fn compact<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            self.log
                .lock()
                .unwrap()
                .push((evicted.len(), carry_over.is_some()));
            let prev = carry_over.map_or("", super::TextSummary::as_str);
            Ok(TextSummary(format!("{prev}|{}", evicted.len())))
        })
    }
}

#[tokio::test]
async fn compacting_no_demotion_does_not_invoke_compactor() {
    let compactor = Arc::new(CountingCompactor::default());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(10),
        compactor.clone(),
    );

    mem.append(&"c".into(), vec![user("a"), assistant("b")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    mem.load(&"c".into()).await.unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert!(compactor.calls().is_empty());
    // Fast path means we never installed a tracking entry either.
    assert_eq!(mem.tracked_conversations(), 0);
}

#[tokio::test]
async fn compacting_invokes_compactor_only_on_new_demotions() {
    let compactor = Arc::new(CountingCompactor::default());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        compactor.clone(),
    );

    // First eviction: 2 messages demoted.
    mem.append(
        &"c".into(),
        vec![user("a"), assistant("b"), user("c"), assistant("d")],
    )
    .await
    .unwrap();
    mem.load(&"c".into()).await.unwrap();
    // Re-load: nothing new evicted; compactor must NOT run again.
    mem.load(&"c".into()).await.unwrap();
    mem.load(&"c".into()).await.unwrap();
    let calls = compactor.calls();
    assert_eq!(
        calls.len(),
        1,
        "compactor invoked more than once: {calls:?}"
    );
    assert_eq!(calls[0], (2, false));

    // Append two more turns → another 2 demoted; compactor runs once
    // more, and this time `carry_over` must be present.
    mem.append(&"c".into(), vec![user("e"), assistant("f")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    mem.load(&"c".into()).await.unwrap();
    let calls = compactor.calls();
    assert_eq!(calls.len(), 2, "expected exactly one new call: {calls:?}");
    // Second call only compacts the *newly* evicted prefix (2 msgs)
    // with the previous summary as carry-over.
    assert_eq!(calls[1], (2, true));
}

#[tokio::test]
async fn compacting_serialises_concurrent_loads() {
    // Many concurrent loads on the same conversation must produce at
    // most ONE compactor invocation per "epoch" of new evictions.
    let compactor = Arc::new(CountingCompactor::default());
    let mem = Arc::new(CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(2),
        compactor.clone(),
    ));
    mem.append(
        &"c".into(),
        vec![user("a"), assistant("b"), user("c"), assistant("d")],
    )
    .await
    .unwrap();

    let mut handles = Vec::new();
    for _ in 0..32 {
        let mem = mem.clone();
        handles.push(tokio::spawn(async move {
            mem.load(&"c".into()).await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // Exactly one invocation: the first to acquire the lock runs the
    // compactor; the others see in_flight or the advanced watermark.
    let calls = compactor.calls();
    assert_eq!(calls.len(), 1, "expected exactly 1 call: {calls:?}");
}

#[tokio::test]
async fn compacting_clear_drops_summary_carry_over() {
    // After clear, the next load on a freshly-populated backend must
    // start compaction from scratch (carry_over=None), not roll the
    // old summary forward.
    let compactor = Arc::new(CountingCompactor::default());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(compactor.calls()[0], (2, false));

    mem.clear(&"c".into()).await.unwrap();
    assert_eq!(mem.tracked_conversations(), 0);

    mem.append(&"c".into(), vec![user("x"), assistant("y"), user("z")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    let calls = compactor.calls();
    assert_eq!(calls.len(), 2);
    // Crucial: no carry_over after clear.
    assert_eq!(calls[1], (2, false));
}

#[tokio::test]
async fn compacting_forget_drops_summary() {
    let compactor = Arc::new(CountingCompactor::default());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();
    mem.load(&"c".into()).await.unwrap();
    assert_eq!(mem.tracked_conversations(), 1);
    mem.forget(&"c".into());
    assert_eq!(mem.tracked_conversations(), 0);

    // Next load on the still-populated backend re-compacts from
    // scratch — same documented contract as DemotionHook.
    mem.load(&"c".into()).await.unwrap();
    let calls = compactor.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1], (2, false));
}

#[tokio::test]
async fn compacting_arc_compactor_works() {
    // Arc<C> forwarding impl exists on Compactor, so CompactingMemory
    // must accept it.
    let compactor: Arc<dyn Compactor<Artifact = TextSummary>> = Arc::new(TemplateCompactor::new());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor,
    );
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();
    let loaded = mem.load(&"c".into()).await.unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(matches!(&loaded[0], Message::System { .. }));
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
async fn compacting_isolates_conversations() {
    let compactor = Arc::new(CountingCompactor::default());
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    );
    mem.append(&"a".into(), vec![user("a1"), assistant("a2"), user("a3")])
        .await
        .unwrap();
    mem.append(&"b".into(), vec![user("b1"), assistant("b2"), user("b3")])
        .await
        .unwrap();

    let a = mem.load(&"a".into()).await.unwrap();
    let b = mem.load(&"b".into()).await.unwrap();
    // Each conversation gets its own summary.
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(compactor.calls().len(), 2);
    assert_eq!(mem.tracked_conversations(), 2);
}

#[tokio::test]
async fn compacting_composes_with_token_window() {
    // Verify CompactingMemory is policy-agnostic: works over a
    // TokenWindowMemory just as well as a SlidingWindowMemory.
    let mem = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        TokenWindowMemory::new(30, HeuristicTokenCounter::openai()),
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
async fn template_compactor_carry_over_threaded() {
    let compactor = TemplateCompactor::new();
    let first = compactor
        .compact(&"c".into(), &[user("hello")], None)
        .await
        .unwrap();
    assert!(!first.as_str().is_empty());

    let second = compactor
        .compact(&"c".into(), &[assistant("world")], Some(&first))
        .await
        .unwrap();
    // Carry-over text appears in the new summary.
    assert!(second.as_str().contains(first.as_str()));
    assert!(second.as_str().contains("assistant: world"));
}

#[tokio::test]
async fn template_compactor_artifact_into_message() {
    let s = TextSummary("rolled-up text".into());
    let msg: Message = s.into();
    let Message::System { content } = msg else {
        panic!("expected system message");
    };
    assert_eq!(content, "rolled-up text");
}

#[tokio::test]
async fn template_compactor_caps_summary_at_max_bytes() {
    let cap = 256;
    let compactor = TemplateCompactor::new().with_max_bytes(cap);
    // Build an evicted history large enough to exceed `cap` on its own.
    let mut evicted = Vec::new();
    for i in 0..50 {
        evicted.push(user(&format!("message number {i} with some filler")));
    }
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    assert!(
        summary.as_str().len()
            <= cap + "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len(),
        "summary len {} exceeds cap {} (plus header+marker)",
        summary.as_str().len(),
        cap,
    );
    // Header is preserved.
    assert!(
        summary
            .as_str()
            .starts_with("[Conversation summary so far]\n")
    );
    // Truncation marker is present.
    assert!(summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
    // Most recent line survives.
    assert!(summary.as_str().contains("message number 49"));
}

#[tokio::test]
async fn template_compactor_unbounded_by_default() {
    let compactor = TemplateCompactor::new();
    let mut evicted = Vec::new();
    for i in 0..200 {
        evicted.push(user(&format!("msg {i}")));
    }
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    // Without a cap, no truncation marker should appear.
    assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
    // Both ends are present.
    assert!(summary.as_str().contains("msg 0"));
    assert!(summary.as_str().contains("msg 199"));
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
async fn compacting_dropped_load_releases_in_flight_gate() {
    // If a `load(...)` future is dropped while awaiting the
    // compactor, the in-flight gate must not leak: subsequent loads
    // on the same conversation must be able to retry compaction.
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct GatedCompactor {
        release: tokio::sync::Notify,
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
                self.release.notified().await;
                Ok(TextSummary("ran".into()))
            })
        }
    }

    let compactor = Arc::new(GatedCompactor {
        release: tokio::sync::Notify::new(),
        entered: AtomicUsize::new(0),
    });
    let mem = Arc::new(CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(1),
        compactor.clone(),
    ));
    mem.append(&"c".into(), vec![user("a"), assistant("b"), user("c")])
        .await
        .unwrap();

    // Kick off a load that will block inside the compactor, then
    // abort it while awaiting — simulating a caller-side timeout
    // or `tokio::select!` cancellation.
    let mem_load = mem.clone();
    let handle = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    while compactor.entered.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    handle.abort();
    let _ = handle.await;

    // The aborted future was dropped without clearing in_flight via
    // the success/error branches; the RAII guard's `Drop` should
    // have released it. A new load must therefore be able to drive
    // a fresh compaction rather than short-circuiting forever.
    let mem_load = mem.clone();
    let retry = tokio::spawn(async move { mem_load.load(&"c".into()).await });
    // Wait for the compactor to be entered a second time. If the
    // gate had leaked, this would never happen — the load would
    // short-circuit on `in_flight = true` and return immediately.
    while compactor.entered.load(Ordering::SeqCst) < 2 {
        tokio::task::yield_now().await;
    }
    compactor.release.notify_one();
    let loaded = retry.await.unwrap().unwrap();
    assert_eq!(loaded.len(), 2);
    let Message::System { content } = &loaded[0] else {
        panic!("expected summary")
    };
    assert_eq!(content, "ran");
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

#[tokio::test]
async fn template_compactor_caps_summary_with_multiline_header() {
    // A header containing embedded newlines must not break the
    // truncation boundary calculation. The first newline in the
    // assembled buffer marks the header/body split, regardless of
    // how the caller chose to format the header.
    let cap = 256;
    let compactor = TemplateCompactor::with_header("line one\nline two").with_max_bytes(cap);
    let mut evicted = Vec::new();
    for i in 0..50 {
        evicted.push(user(&format!("message number {i} with some filler")));
    }
    let summary = compactor
        .compact(&"c".into(), &evicted, None)
        .await
        .unwrap();
    let text = summary.as_str();

    // The first line of the header is preserved as the header line.
    assert!(text.starts_with("line one\n"));
    // Truncation marker is present and the most recent line survives.
    assert!(text.contains("[\u{2026}truncated\u{2026}]"));
    assert!(text.contains("message number 49"));
    // Cap is honoured up to the header+marker overhead.
    let overhead = "line one\n".len() + "[\u{2026}truncated\u{2026}]\n".len();
    assert!(
        text.len() <= cap + overhead,
        "summary len {} exceeds cap {} plus overhead {}",
        text.len(),
        cap,
        overhead,
    );
}
