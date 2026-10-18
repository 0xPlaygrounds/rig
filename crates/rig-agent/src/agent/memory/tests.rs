use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;

use rig_core::completion::Message;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use rig_memory::{AsyncStore, Files, LogWriteError, MemoryStore, SessionError, Store, load};

use crate::agent::stream_to_stdout;
use crate::agent::{
    Agent, AgentBuilder, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, RunStart,
    RunStartAction,
};
use crate::completion::PromptError;

/// A store reached asynchronously over a [`MemoryStore`], as one behind a
/// network is: every call waits one poll. Appends wait for good while
/// `held`, and fail while `failing`; reads fail while `unreadable`, and
/// wait for each other in pairs on `paired`.
#[derive(Clone, Default)]
struct Remote {
    store: MemoryStore,
    held: Arc<AtomicBool>,
    failing: Arc<AtomicBool>,
    unreadable: Arc<AtomicBool>,
    paired: Option<Arc<tokio::sync::Barrier>>,
}

impl Remote {
    async fn call<T>(&self, call: impl FnOnce(&MemoryStore) -> T) -> T {
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
    async fn read_log_async(&self, name: &str) -> io::Result<Vec<u8>> {
        if self.unreadable.load(Ordering::SeqCst) {
            return Err(io::Error::other("the database is down"));
        }
        if let Some(paired) = &self.paired {
            paired.wait().await;
        }
        self.call(|store| store.read_log(name)).await
    }

    async fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        if self.held.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        if self.failing.load(Ordering::SeqCst) {
            return Err(io::Error::other("the database is full").into());
        }
        self.call(|store| store.append_log(name, len, bytes)).await
    }

    async fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        self.call(|store| store.replace_log(name, len, bytes)).await
    }

    async fn read_file_async(&self, name: &str) -> io::Result<Vec<u8>> {
        self.call(|store| store.read_file(name)).await
    }

    async fn write_file_async(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.call(|store| store.write_file(name, bytes)).await
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
    assert!(store.logs().expect("logs").is_empty());
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

/// Two runs that load and save one log at the same moments: the store keeps
/// one run's messages and refuses the other's save.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_runs_on_one_log_store_one() {
    let store = Remote {
        paired: Some(Arc::new(tokio::sync::Barrier::new(2))),
        ..Remote::default()
    };
    let agent = agent(store.clone(), &["one", "two"]);
    let runs = ["first", "second"].map(|prompt| {
        let agent = agent.clone();
        tokio::spawn(async move { agent.prompt(prompt).conversation("user-1").await })
    });
    let mut kept_run = None;
    let mut refused = 0;
    for run in runs {
        match run.await.expect("a run") {
            Ok(response) => kept_run = Some(response),
            Err(PromptError::Memory {
                error: SessionError::Changed { .. },
                response: Some(_),
            }) => refused += 1,
            Err(error) => panic!("a run failed otherwise: {error:?}"),
        }
    }
    let kept_run = kept_run.expect("one run is kept");
    assert_eq!(refused, 1);
    let texts = kept(&store.store, "user-1");
    assert_eq!(texts.len(), 2, "{texts:?}");
    assert_eq!(texts.last(), Some(&kept_run.output()));
}

#[tokio::test]
async fn an_arc_dyn_store_keeps_the_conversation() {
    let store: Arc<dyn Store> = Arc::new(MemoryStore::default());
    let agent = agent(store.clone(), &["kept"]);
    agent
        .prompt("hello")
        .conversation("user-1")
        .await
        .expect("a run");
    let conversation = load(&*store, "user-1").expect("loaded");
    assert_eq!(conversation.messages().len(), 2);
}
