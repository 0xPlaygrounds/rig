//! A session's hot paths on generated coding conversations: opening a
//! 3,000-message session, the free `load` and `save` after a turn at growing
//! sizes (in memory and in a directory), each also through `AsyncStore`, an
//! agent's load and save around a turn, a writer's turn, `Session::save` of
//! an edited session, and the token estimate a compaction policy makes on
//! every model call.
//!
//! `cargo bench -p rig-memory --bench session --features fs`. The directory
//! sessions are written under Cargo's bench scratch directory
//! (`CARGO_TARGET_TMPDIR`).

use std::fmt::Display;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use rig_core::completion::Message;
use rig_core::conversation::Conversation;
use rig_memory::{
    DirStore, MemoryStore, Session, SessionError, Store, Writer, estimate_tokens, load, load_async,
    save, save_async,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// `result`'s value, or the process ends saying what failed: a bench whose
/// input cannot be made measures nothing.
fn or_exit<T, E: Display>(result: Result<T, E>, what: &str) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("{what}: {error}");
        std::process::exit(1)
    })
}

/// The output of `work`, polled once: a local store's calls through
/// `AsyncStore` are done when made.
fn at_once<T>(work: impl Future<Output = T>) -> T {
    match std::pin::pin!(work).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(output) => output,
        Poll::Pending => {
            eprintln!("a local store's call did not finish");
            std::process::exit(1)
        }
    }
}

/// A future behind a `dyn` boundary, as an agent holds its store.
type Boxed<'a, T> = Pin<Box<dyn Future<Output = Result<T, SessionError>> + Send + 'a>>;

/// The agent's log in every session here.
const LOG: &str = "agent";

/// `turns` turns of a coding session, 4 messages each: a question, a
/// reasoning block with a tool call, its result (40 numbered lines), and the
/// answer. Every fourth call is a `write` whose content runs from 2 KB to
/// 50 KB, the rest a `read` of a path.
fn messages(turns: usize) -> Vec<Message> {
    use serde_json::json;
    (0..turns)
        .flat_map(|i| {
            let file = format!("src/module_{}.rs", i % 37);
            let lines: String = (1..=40)
                .map(|n| format!("{n:>6}\tfn handle_{i}_{n}(x: u32) -> u32 {{ x.wrapping_add({n}) }}\n"))
                .collect();
            let (name, arguments) = match i % 4 {
                0 => {
                    let size = 2_048 + (i * 7_919) % (48 * 1_024);
                    let content: String = "fn body() { work(); }\n".repeat(size / 22);
                    ("write", json!({"path": file, "content": content}))
                }
                _ => ("read", json!({"path": file})),
            };
            [
                json!({"role": "user", "content": [{"type": "text",
                    "text": format!("Turn {i}: look at {file} and tell me what `handle_{i}_*` do.")}]}),
                json!({"role": "assistant", "content": [
                    {"type": "reasoning", "text": format!("I should {name} {file} before answering.")},
                    {"type": "toolcall", "id": {"provider": format!("call-{i}")},
                     "function": {"name": name, "arguments": arguments}}]}),
                json!({"role": "user", "content": [{"type": "toolresult", "call": {"provider": format!("call-{i}")},
                    "name": name, "content": [{"type": "text", "text": lines}]}]}),
                json!({"role": "assistant", "content": [{"type": "text",
                    "text": format!("The 40 functions `handle_{i}_1` to `handle_{i}_40` in {file} each add \
                        their own number to the argument, wrapping on overflow.")}]}),
            ]
        })
        .map(|value| or_exit(serde_json::from_value(value), "a generated message"))
        .collect()
}

/// A store holding `count` messages saved to [`LOG`] in one save, and the
/// conversation as saved.
fn saved(count: usize) -> (MemoryStore, Conversation) {
    let store = MemoryStore::default();
    let mut conversation = Conversation::from_messages(messages(count / 4));
    or_exit(save(&store, LOG, &mut conversation), "saving the session");
    (store, conversation)
}

/// `Writer::open` of a 3,000-message session: reading, checking and
/// folding every log, as a host resuming a session does.
#[divan::bench(sample_count = 20)]
fn open_3000(bencher: divan::Bencher) {
    let (store, _) = saved(3_000);
    let store: Arc<dyn Store> = Arc::new(store);
    let (_, session) = or_exit(Writer::open(store.clone()), "opening the session");
    let opened = session
        .agents
        .iter()
        .map(|agent| agent.conversation.messages().len())
        .sum::<usize>();
    if opened != 3_000 {
        eprintln!("open_3000 opened {opened} messages");
        std::process::exit(1);
    }
    bencher.bench(|| Writer::open(store.clone()).map(|(_, session)| session.agents.len()));
}

/// The free `load` of a 3,000-message log, as a library user resuming a
/// conversation does, in memory and in a directory.
#[divan::bench(args = ["memory", "dir"], sample_count = 20)]
fn load_3000(bencher: divan::Bencher, store_kind: &str) {
    let store = stored(3_000, store_kind);
    let loaded = or_exit(load(&*store, LOG), "loading the session");
    if loaded.messages().len() != 3_000 {
        eprintln!("load_3000 loaded {} messages", loaded.messages().len());
        std::process::exit(1);
    }
    bencher.bench(|| load(&*store, LOG).map(|conversation| conversation.revision()));
}

/// [`load_3000`] through `AsyncStore`.
#[divan::bench(args = ["memory", "dir"], sample_count = 20)]
fn load_async_3000(bencher: divan::Bencher, store_kind: &str) {
    let store = stored(3_000, store_kind);
    let loaded = or_exit(at_once(load_async(&*store, LOG)), "loading the session");
    if loaded.messages().len() != 3_000 {
        eprintln!(
            "load_async_3000 loaded {} messages",
            loaded.messages().len()
        );
        std::process::exit(1);
    }
    bencher.bench(|| at_once(load_async(&*store, LOG)).map(|conversation| conversation.revision()));
}

/// An agent's persistence around one turn onto a 3,000-message session in
/// memory: its conversation loaded, one turn (4 messages) added and saved.
/// `sync` is the free `load` and `save`; `boxed` is `load_async` and
/// `save_async` behind the `dyn` boundary an agent holds its store
/// through. Each sample starts from a fresh copy of the session.
#[divan::bench(args = ["sync", "boxed"], sample_count = 30, sample_size = 1)]
fn persist_turn_3000(bencher: divan::Bencher, path: &str) {
    let (saved, _) = saved(3_000);
    let turn = messages(751).split_off(3_000);
    let fresh = || {
        let store = MemoryStore::default();
        copy_logs(&saved, &store);
        store
    };
    let boxed = path == "boxed";
    let persist = |store: &MemoryStore| -> Result<usize, SessionError> {
        let mut conversation = if boxed {
            let load: Boxed<'_, _> = Box::pin(load_async(store, LOG));
            at_once(load)?
        } else {
            load(store, LOG)?
        };
        conversation.extend(turn.iter().cloned());
        if boxed {
            let save: Boxed<'_, _> = Box::pin(save_async(store, LOG, &mut conversation));
            at_once(save)?;
        } else {
            save(store, LOG, &mut conversation)?;
        }
        Ok(conversation.messages().len())
    };
    let saved_turn = or_exit(persist(&fresh()), "persisting a turn");
    if saved_turn != 3_004 {
        eprintln!("persist_turn_3000 saved {saved_turn} messages");
        std::process::exit(1);
    }
    bencher
        .with_inputs(fresh)
        .bench_local_values(|store| persist(&store).map(|_| store));
}

/// A writer's turn: one more turn (4 messages) queued and flushed with one
/// append, onto a 3,000-message session, as the harness logs a turn.
#[divan::bench(args = ["memory", "dir"], sample_count = 100, sample_size = 10)]
fn writer_turn_3000(bencher: divan::Bencher, store_kind: &str) {
    let store: Arc<dyn Store> = stored(3_000, store_kind).into();
    let (writer, session) = or_exit(Writer::open(store), "opening the session");
    writer.start();
    let mut conversation = session
        .agents
        .into_iter()
        .next()
        .map(|agent| agent.conversation)
        .unwrap_or_default();
    let turn = messages(751).split_off(3_000);
    bencher.bench_local(|| {
        conversation.extend(turn.iter().cloned());
        writer.save(LOG, &mut conversation);
        writer.flush()
    });
}

/// A store of `kind` ("memory" or "dir") holding [`saved`]'s `count`
/// messages.
fn stored(count: usize, kind: &str) -> Box<dyn Store> {
    let (saved, _) = saved(count);
    let store: Box<dyn Store> = match kind {
        "memory" => Box::new(MemoryStore::default()),
        _ => {
            let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
                .join(format!("rig-memory-bench-{kind}-{count}"));
            if dir.exists() {
                or_exit(
                    std::fs::remove_dir_all(&dir),
                    "clearing the session directory",
                );
            }
            Box::new(or_exit(DirStore::open(&dir), "the session directory"))
        }
    };
    copy_logs(&saved, &*store);
    store
}

/// Copies every log of `from` into `to`, a fresh store.
fn copy_logs(from: &MemoryStore, to: &dyn Store) {
    for name in or_exit(from.logs(), "the saved logs") {
        let bytes = or_exit(from.read_log(&name), "a saved log");
        or_exit(to.replace_log(&name, 0, &bytes), "copying a log");
    }
}

/// The free `save` of one more turn (4 messages) onto a session of
/// `messages` messages, as a library user saving after every turn does, in
/// memory and in a directory. Each sample saves onto a fresh copy of the
/// session.
#[divan::bench(
    consts = [100, 1_000, 3_000],
    args = ["memory", "dir"],
    sample_count = 30,
    sample_size = 1,
)]
fn save_per_turn<const MESSAGES: usize>(bencher: divan::Bencher, store_kind: &str) {
    save_turn::<MESSAGES>(bencher, store_kind, |store, conversation| {
        save(store, LOG, conversation)
    });
}

/// [`save_per_turn`] through `AsyncStore`.
#[divan::bench(
    consts = [100, 1_000, 3_000],
    args = ["memory", "dir"],
    sample_count = 30,
    sample_size = 1,
)]
fn save_per_turn_async<const MESSAGES: usize>(bencher: divan::Bencher, store_kind: &str) {
    save_turn::<MESSAGES>(bencher, store_kind, |store, conversation| {
        at_once(save_async(store, LOG, conversation))
    });
}

/// One turn saved by `save_with` onto a fresh copy of a session of
/// `MESSAGES` messages in a store of `store_kind`, each sample.
fn save_turn<const MESSAGES: usize>(
    bencher: divan::Bencher,
    store_kind: &str,
    save_with: impl Fn(&dyn Store, &mut Conversation) -> Result<(), SessionError>,
) {
    let (saved, conversation) = saved(MESSAGES);
    let turn = messages(MESSAGES / 4 + 1).split_off(MESSAGES);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rig-memory-bench-{MESSAGES}"));
    let fresh = || -> (Box<dyn Store>, Conversation) {
        let store: Box<dyn Store> = match store_kind {
            "memory" => Box::new(MemoryStore::default()),
            _ => {
                if dir.exists() {
                    or_exit(
                        std::fs::remove_dir_all(&dir),
                        "clearing the session directory",
                    );
                }
                Box::new(or_exit(DirStore::open(&dir), "the session directory"))
            }
        };
        copy_logs(&saved, &*store);
        let mut conversation = conversation.clone();
        conversation.extend(turn.iter().cloned());
        (store, conversation)
    };
    let (store, mut check) = fresh();
    or_exit(save_with(&*store, &mut check), "saving a turn");
    let reloaded = or_exit(load(&*store, LOG), "reloading the session");
    if reloaded.messages().len() != MESSAGES + 4 {
        eprintln!("save_per_turn saved {} messages", reloaded.messages().len());
        std::process::exit(1);
    }
    bencher
        .with_inputs(fresh)
        .bench_local_values(|(store, mut conversation)| {
            save_with(&*store, &mut conversation).map(|()| store)
        });
}

/// `Session::save` of one more turn onto a 3,000-message session read
/// whole, as an offline edit (an importer, an edit of a stopped session)
/// saves. Each sample saves onto a fresh copy of the session.
#[divan::bench(sample_count = 20, sample_size = 1)]
fn session_save_3000(bencher: divan::Bencher) {
    let (saved, _) = saved(3_000);
    let turn = messages(751).split_off(3_000);
    let fresh = || {
        let store = MemoryStore::default();
        copy_logs(&saved, &store);
        let mut session = or_exit(Session::read(&store), "reading the session");
        if let Some(agent) = session.agent_mut(LOG) {
            agent.conversation.extend(turn.iter().cloned());
        }
        (store, session)
    };
    let (store, mut check) = fresh();
    or_exit(check.save(&store), "saving the session");
    let reloaded = or_exit(load(&store, LOG), "reloading the session");
    if reloaded.messages().len() != 3_004 {
        eprintln!(
            "session_save_3000 saved {} messages",
            reloaded.messages().len()
        );
        std::process::exit(1);
    }
    bencher
        .with_inputs(fresh)
        .bench_local_values(|(store, mut session)| session.save(&store).map(|()| store));
}

/// The default token estimate of a 3,000-message history whose tool calls
/// carry 2 KB to 50 KB of arguments.
#[divan::bench(sample_count = 30)]
fn estimate_tokens_3000(bencher: divan::Bencher) {
    let history = messages(750);
    bencher.bench(|| estimate_tokens(&history));
}
