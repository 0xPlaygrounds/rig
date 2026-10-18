use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig_agent::AgentBuilder;
use rig_agent::tool::Tool;
use rig_core::conversation::pairing;
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::UserContent;
use rig_core::test_utils::{MockCompletionModel, MockFrame, MockScript, MockTurn};
use rig_core::tool::{ToolContext, ToolExecutionError};
use rig_memory::recent::LastMessages;
use tokio::sync::{Barrier, Notify};

use super::*;

/// `result`'s value, failing the test with its error.
fn ok<T>(result: Result<T, impl Debug>) -> Option<T> {
    let result = result.map_err(|error| format!("{error:?}"));
    assert_eq!(result.as_ref().err(), None);
    result.ok()
}

/// The order-lookup tool the scripted model calls.
struct LookupOrder;

impl Tool for LookupOrder {
    const NAME: &'static str = "lookup_order";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Looks an order up by id".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"order_id": {"type": "string"}}})
    }

    async fn call(&self, _: &mut ToolContext, _: Self::Args) -> Result<String, Self::Error> {
        Ok("order 42: shipped".to_owned())
    }
}

/// How one scripted append fails.
#[derive(Clone, Copy)]
enum Fault {
    /// Writes nothing and fails.
    Lost,
    /// Writes, then reports a failure (a timeout after the commit).
    Written,
}

/// [`InMemoryStore`] with faults on demand: failing loads, scripted append
/// failures, and a barrier the first `gated` loads meet at.
#[derive(Default)]
struct Faulty {
    inner: InMemoryStore,
    fail_get: AtomicBool,
    faults: Mutex<VecDeque<Fault>>,
    barrier: Option<Barrier>,
    gated: AtomicUsize,
    appends: AtomicUsize,
}

impl ThreadStore for Faulty {
    async fn get(&self, thread: &str) -> Result<Snapshot, StoreError> {
        if self.fail_get.load(Ordering::SeqCst) {
            return Err(StoreError("load refused".to_owned()));
        }
        let snapshot = self.inner.get(thread).await;
        if let Some(barrier) = &self.barrier
            && self
                .gated
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        {
            barrier.wait().await;
        }
        snapshot
    }

    async fn append(
        &self,
        thread: &str,
        expected_version: u64,
        messages: &[StoredMessage],
    ) -> Result<u64, AppendError> {
        self.appends.fetch_add(1, Ordering::SeqCst);
        let fault = self.faults.lock().map(|mut f| f.pop_front()).ok().flatten();
        match fault {
            None => self.inner.append(thread, expected_version, messages).await,
            Some(Fault::Lost) => Err(StoreError("append timed out".to_owned()).into()),
            Some(Fault::Written) => {
                self.inner
                    .append(thread, expected_version, messages)
                    .await?;
                Err(StoreError("append timed out after commit".to_owned()).into())
            }
        }
    }
}

fn agent(model: &MockCompletionModel) -> Agent {
    AgentBuilder::new(model.clone())
        .tool(LookupOrder)
        .default_max_turns(4)
        .build()
}

fn ids(messages: &[StoredMessage]) -> Vec<&str> {
    messages.iter().map(|m| m.id.as_str()).collect()
}

fn starts_with_tool_result(history: &[Arc<Message>]) -> bool {
    history.first().is_some_and(|first| {
        matches!(&**first, Message::User { content }
            if content.iter().any(|c| matches!(c, UserContent::ToolResult(_))))
    })
}

/// Two clients ask in one thread at once, both having loaded the same
/// version: the second append conflicts, reloads and lands after the first
/// exchange; both exchanges are whole, in order, once each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_asks_on_one_thread_both_land_once() {
    let model = MockCompletionModel::from_turns([MockTurn::text("A"), MockTurn::text("B")]);
    let store = Arc::new(Faulty {
        barrier: Some(Barrier::new(2)),
        gated: AtomicUsize::new(2),
        ..Faulty::default()
    });
    let shared = Arc::new(SharedThread::new(agent(&model), Arc::clone(&store)));
    let ask = |text: &'static str| {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move { shared.ask("t", text).await.map_err(|e| e.to_string()) })
    };
    let (first, second) = (ask("where is my order?"), ask("also, my address changed"));
    let replies = [first.await, second.await].map(|joined| ok(joined).and_then(ok));

    let stored = store.inner.messages("t");
    assert!(replies.iter().all(Option::is_some));
    assert_eq!(stored.len(), 4);
    assert_eq!(ids(&stored).into_iter().collect::<HashSet<_>>().len(), 4);
    assert_eq!(
        store.appends.load(Ordering::SeqCst),
        3,
        "one conflict, retried"
    );
    for reply in replies.iter().flatten() {
        let at = stored
            .iter()
            .position(|m| Some(&m.id) == reply.messages.first().map(|m| &m.id));
        let own = at.and_then(|at| stored.get(at..at + reply.messages.len()));
        assert_eq!(
            own.map(ids),
            Some(ids(&reply.messages)),
            "exchange is contiguous"
        );
        assert_eq!(
            own.and_then(<[_]>::last)
                .and_then(|m| m.message.final_answer()),
            Some(reply.text.clone())
        );
    }
}

/// Twenty threads, every store call taking 50 ms, run side by side: the
/// whole batch takes about one load and one append, not twenty of each.
#[tokio::test(start_paused = true)]
async fn twenty_threads_run_in_parallel() {
    let model = MockCompletionModel::from_turns((0..20).map(|n| MockTurn::text(format!("r{n}"))));
    let store = InMemoryStore::with_latency(Duration::from_millis(50));
    let shared = Arc::new(SharedThread::new(agent(&model), store.clone()));
    let started = tokio::time::Instant::now();
    let asks = (0..20).map(|n| {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move { shared.ask(&format!("thread-{n}"), "hi").await.is_ok() })
    });
    let done = futures::future::join_all(asks).await;
    assert!(
        done.into_iter()
            .all(|joined| joined.is_ok_and(|asked| asked))
    );
    assert!(
        started.elapsed() < Duration::from_millis(20 * 50 / 4),
        "{:?}",
        started.elapsed()
    );
    assert!((0..20).all(|n| store.messages(&format!("thread-{n}")).len() == 2));
}

/// Seeds `thread` with an exchange that called `lookup_order`, then a
/// question another client stored with no reply yet: five messages.
async fn seed_tool_thread(store: &InMemoryStore, thread: &str) -> Option<Vec<StoredMessage>> {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call(
            "call-1",
            "lookup_order",
            serde_json::json!({"order_id": "42"}),
        ),
        MockTurn::text("Order 42 has shipped."),
    ]);
    let first = SharedThread::new(agent(&model), store.clone());
    let reply = ok(first.ask(thread, "where is order 42?").await)?;
    let dangling = [StoredMessage::new(Message::user("is it insured?"))];
    ok(store.append(thread, reply.version, &dangling).await)?;
    Some(store.messages(thread))
}

/// A window of three over a thread whose cut lands between a tool call and
/// its result sends no orphaned result; the store keeps everything.
#[tokio::test]
async fn window_never_sends_an_orphaned_tool_result() {
    let store = InMemoryStore::default();
    let Some(seeded) = seed_tool_thread(&store, "t").await else {
        return;
    };
    assert_eq!(seeded.len(), 5);
    let model = MockCompletionModel::text("Yes, it is insured.");
    let shared = SharedThread::new(agent(&model), store.clone()).window(LastMessages(3));
    ok(shared.ask("t", "thanks").await);

    let requests = model.requests();
    let history = requests
        .first()
        .map(|r| r.chat_history.clone())
        .unwrap_or_default();
    assert!(!history.is_empty());
    assert!(!starts_with_tool_result(&history));
    assert!(pairing::validate(&history).is_ok());
    assert_eq!(store.messages("t").len(), 7);
}

/// A new wrapper (a fresh process) over the same store sends the whole
/// prior thread before its own prompt. The unanswered question and the
/// prompt are two user messages in a row, which the request carries as one
/// user turn holding both; the store keeps them apart.
#[tokio::test]
async fn a_fresh_wrapper_continues_the_thread() {
    let store = InMemoryStore::default();
    let Some(seeded) = seed_tool_thread(&store, "t").await else {
        return;
    };
    let model = MockCompletionModel::text("Yes, it is insured.");
    let fresh = SharedThread::new(agent(&model), store.clone());
    ok(fresh.ask("t", "hello again").await);

    let requests = model.requests();
    let history = requests
        .first()
        .map(|r| r.chat_history.clone())
        .unwrap_or_default();
    let sent: Vec<&Message> = history.iter().map(|m| &**m).collect();
    let answered: Vec<&Message> = seeded.iter().take(4).map(|m| &m.message).collect();
    assert_eq!(sent.get(..4), Some(answered.as_slice()));
    let texts = |message: &Message| match message {
        Message::User { content } => content
            .iter()
            .filter_map(|c| match c {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    assert_eq!(sent.len(), 5);
    assert_eq!(
        sent.last().map(|m| texts(m)),
        Some(vec!["is it insured?".to_owned(), "hello again".to_owned()])
    );
    let stored = store.messages("t");
    assert_eq!(stored.get(..5), Some(seeded.as_slice()));
    assert_eq!(stored.len(), 7);
}

/// A failed load never calls the model; an append failing twice (once
/// after committing) then succeeding stores each message once; a save
/// failing every attempt returns the reply, and saving it again later
/// writes it once.
#[tokio::test]
async fn store_failures_never_lose_or_duplicate() {
    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let store = Arc::new(Faulty::default());
    let shared = SharedThread::new(agent(&model), Arc::clone(&store)).save_attempts(3);

    store.fail_get.store(true, Ordering::SeqCst);
    assert!(matches!(
        shared.ask("t", "hi").await,
        Err(AskError::Load(_))
    ));
    assert_eq!(model.request_count(), 0);
    store.fail_get.store(false, Ordering::SeqCst);

    if let Ok(mut faults) = store.faults.lock() {
        faults.extend([Fault::Lost, Fault::Written]);
    }
    ok(shared.ask("t", "hi").await);
    let stored = store.inner.messages("t");
    assert_eq!(stored.len(), 2);

    if let Ok(mut faults) = store.faults.lock() {
        faults.extend([Fault::Lost; 3]);
    }
    let saved = shared.ask("t", "again").await;
    assert!(matches!(saved, Err(AskError::Save { .. })));
    let Err(AskError::Save { reply, pending, .. }) = saved else {
        return;
    };
    assert_eq!(reply, "two");
    assert_eq!(store.inner.messages("t"), stored);
    ok(shared.save("t", &pending).await);
    ok(shared.save("t", &pending).await);
    let stored = store.inner.messages("t");
    assert_eq!(stored.len(), 4);
    assert_eq!(ids(&stored).into_iter().collect::<HashSet<_>>().len(), 4);
}

/// A transport that never answers, counting entries and cancellations.
#[derive(Clone, Default)]
struct Hanging {
    entered: Arc<Notify>,
    cancelled: Arc<AtomicBool>,
}

/// Sets its flag when dropped.
struct OnDrop(Arc<AtomicBool>);

impl Drop for OnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Transport<MockScript> for Hanging {
    fn send(&self, _: rig_core::completion::CompletionRequest, _: Exchange) -> Opening<MockFrame> {
        let (entered, guard) = (
            Arc::clone(&self.entered),
            OnDrop(Arc::clone(&self.cancelled)),
        );
        Opening::new(async move {
            let _guard = guard;
            entered.notify_one();
            std::future::pending::<Result<Opened<MockFrame>, ProviderError>>().await
        })
    }
}

/// Dropping an ask while the model call is in flight cancels that call
/// and leaves the store as it was.
#[tokio::test]
async fn dropping_an_ask_cancels_and_writes_nothing() {
    let hanging = Hanging::default();
    let agent = AgentBuilder::new(Model::new(MockScript::default(), hanging.clone())).build();
    let store = InMemoryStore::default();
    let seeded = [StoredMessage::new(Message::user("earlier"))];
    ok(store.append("t", 0, &seeded).await);
    let shared = SharedThread::new(agent, store.clone());

    let finished = tokio::select! {
        _ = shared.ask("t", "hello?") => true,
        () = hanging.entered.notified() => false,
    };
    assert!(!finished);
    assert!(hanging.cancelled.load(Ordering::SeqCst));
    assert_eq!(store.messages("t"), seeded);
}
