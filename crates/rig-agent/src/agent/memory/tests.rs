use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;

use rig_core::completion::Message;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use rig_memory::{AsyncStore, Files, MemoryStore, SessionError, Store, load};

use crate::agent::stream_to_stdout;
use crate::agent::{
    Agent, AgentBuilder, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, RunStart,
    RunStartAction,
};
use crate::completion::PromptError;

/// A store reached asynchronously over a [`MemoryStore`], as one behind a
/// network is: every call waits one poll. Appends wait for good while
/// `held`, and fail while `failing`; reads fail while `unreadable`.
#[derive(Clone, Default)]
struct Remote {
    store: MemoryStore,
    held: Arc<AtomicBool>,
    failing: Arc<AtomicBool>,
    unreadable: Arc<AtomicBool>,
}

impl Remote {
    async fn call<T>(&self, call: impl FnOnce(&MemoryStore) -> io::Result<T>) -> io::Result<T> {
        let mut waited = false;
        std::future::poll_fn(|cx| {
            if std::mem::replace(&mut waited, true) {
                return Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await;
        call(&self.store)
    }
}

impl AsyncStore for Remote {
    async fn logs(&self) -> io::Result<Vec<String>> {
        self.call(Store::logs).await
    }

    async fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        if self.unreadable.load(Ordering::SeqCst) {
            return Err(io::Error::other("the database is down"));
        }
        self.call(|store| Store::read_log(store, name)).await
    }

    async fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        if self.held.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        if self.failing.load(Ordering::SeqCst) {
            return Err(io::Error::other("the database is full"));
        }
        self.call(|store| Store::append_log(store, name, bytes))
            .await
    }

    async fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.call(|store| Store::replace_log(store, name, bytes))
            .await
    }

    async fn files(&self) -> io::Result<Vec<String>> {
        self.call(Files::files).await
    }

    async fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.call(|store| Files::read_file(store, name)).await
    }

    async fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.call(|store| Files::write_file(store, name, bytes))
            .await
    }
}

fn agent(store: impl AsyncStore + Send + Sync + 'static, replies: &[&str]) -> Agent {
    let turns = replies.iter().map(|reply| MockTurn::text(*reply));
    AgentBuilder::new(MockCompletionModel::from_turns(turns))
        .memory(store)
        .build()
}

/// The texts of the messages the log `log` of `store` holds.
fn kept(store: &MemoryStore, log: &str) -> Vec<String> {
    let conversation = load(store, log).expect("the log reads");
    let text = |message: &Message| match message {
        Message::Assistant(turn) => Some(turn.text()),
        message => message.rag_text(),
    };
    let texts = conversation.messages().iter().map(|message| text(message));
    texts.map(Option::unwrap_or_default).collect()
}

#[tokio::test]
async fn a_conversation_goes_on_across_runs_and_agents() {
    for remote in [false, true] {
        let store = Remote::default();
        let model = MockCompletionModel::from_turns([
            MockTurn::text("Hi Alice!"),
            MockTurn::text("You are Alice."),
        ]);
        let first = AgentBuilder::new(model.clone());
        let first = match remote {
            true => first.memory(store.clone()),
            false => first.memory(store.store.clone()),
        };
        let first = first.conversation("alice").build();
        first.prompt("I am Alice.").await.expect("the first turn");
        first.prompt("Who am I?").await.expect("the second turn");
        let asked = model.requests();
        let second = asked.get(1).expect("a second request");
        assert_eq!(
            second.chat_history.len(),
            3,
            "the first turn and the prompt"
        );

        let reloaded = agent(store.clone(), &["Still Alice."]);
        let response = reloaded.prompt("And now?").conversation("alice").await;
        let response = response.expect("a reloaded turn");
        assert_eq!(response.messages.len(), 2, "only this run's messages");
        assert_eq!(
            kept(&store.store, "alice"),
            [
                "I am Alice.",
                "Hi Alice!",
                "Who am I?",
                "You are Alice.",
                "And now?",
                "Still Alice."
            ]
        );
    }
}

#[tokio::test]
async fn a_failed_save_fails_the_run_and_a_failed_load_calls_no_model() {
    let store = Remote::default();
    store.failing.store(true, Ordering::SeqCst);
    let failed = agent(store.clone(), &["an answer nobody kept"])
        .prompt("hello")
        .await;
    let Err(PromptError::Memory {
        error: SessionError::Io(_),
        response: Some(response),
    }) = failed
    else {
        panic!("a failed save is a failed run: {failed:?}");
    };
    assert_eq!(response.output(), "an answer nobody kept");
    assert!(kept(&store.store, "agent").is_empty());

    store.failing.store(false, Ordering::SeqCst);
    store.unreadable.store(true, Ordering::SeqCst);
    let model = MockCompletionModel::text("never asked");
    let unread = AgentBuilder::new(model.clone()).memory(store).build();
    let failed = unread.prompt("hello").await;
    assert!(
        matches!(failed, Err(PromptError::Memory { response: None, .. })),
        "{failed:?}"
    );
    assert_eq!(model.request_count(), 0);
}

/// Runs `other` to the end once the run it is a hook of has loaded its
/// conversation.
struct Overtaken(Agent);

impl AgentHook for Overtaken {
    async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let other = self.0.prompt("I was second to start.").await;
        assert!(other.is_ok(), "{other:?}");
        RunStartAction::Continue
    }
}

#[tokio::test]
async fn of_two_runs_on_one_log_the_one_that_saves_second_is_refused() {
    let store = MemoryStore::default();
    let first = agent(store.clone(), &["first to finish"]);
    let overtaken = AgentBuilder::new(MockCompletionModel::text("second to finish"))
        .memory(store.clone())
        .add_hook(Overtaken(first))
        .build();
    let refused = overtaken.prompt("I was first to start.").await;
    assert!(
        matches!(
            refused,
            Err(PromptError::Memory {
                error: SessionError::Changed { .. },
                ..
            })
        ),
        "{refused:?}"
    );
    assert_eq!(
        kept(&store, "agent"),
        ["I was second to start.", "first to finish"]
    );
}

struct StopTheTurn;

impl AgentHook for StopTheTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        ModelTurnAction::Stop("stopped".to_owned())
    }
}

#[tokio::test]
async fn a_cancelled_run_saves_nothing() {
    let store = Remote::default();
    let saved = agent(store.clone(), &["kept"]);
    saved.prompt("first").await.expect("a saved turn");

    let stopped = AgentBuilder::new(MockCompletionModel::text("never kept"))
        .memory(store.clone())
        .add_hook(StopTheTurn)
        .build();
    let cancelled = stopped.prompt("stopped").await;
    assert!(
        matches!(cancelled, Err(PromptError::Cancelled { .. })),
        "{cancelled:?}"
    );

    store.held.store(true, Ordering::SeqCst);
    let run = agent(store.clone(), &["cut short"]).prompt("dropped");
    let polled = tokio::time::timeout(std::time::Duration::from_millis(50), run).await;
    assert!(
        polled.is_err(),
        "the save waits on the held append, then is dropped"
    );
    assert_eq!(kept(&store.store, "agent"), ["first", "kept"]);
}

#[tokio::test]
async fn explicit_history_bypasses_the_store() {
    let store = MemoryStore::default();
    let agent = agent(store.clone(), &["answered"]);
    let mut history = vec![Message::user("from the caller")];
    agent.chat("hello", &mut history).await.expect("a run");
    assert_eq!(history.len(), 3);
    assert!(Store::logs(&store).expect("logs").is_empty());
}

#[tokio::test]
async fn a_streamed_run_is_kept_before_its_final_response() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("streamed"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let agent = AgentBuilder::new(model).memory(store.clone()).build();
    let mut stream = agent.prompt("go").stream();
    let response = stream_to_stdout(&mut stream).await.expect("a run");
    assert_eq!(response.output(), "streamed");
    assert_eq!(kept(&store, "agent"), ["go", "streamed"]);
}
