//! A session's hot paths on generated coding conversations: opening a
//! 3,000-message session, the free `save` after a turn at growing sizes (in
//! memory and in a directory), `Session::save` of an edited session, and
//! the token estimate a compaction policy makes on every model call.
//!
//! `cargo bench -p rig-memory --bench session --features fs`. The directory
//! sessions are written under Cargo's bench scratch directory
//! (`CARGO_TARGET_TMPDIR`).

use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::Conversation;
use rig_memory::{DirStore, MemoryStore, Session, Store, Writer, estimate_tokens, load, save};

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

/// Copies every log of `from` into `to`, replacing what `to` held.
fn copy_logs(from: &MemoryStore, to: &dyn Store) {
    for name in or_exit(from.logs(), "the saved logs") {
        let bytes = or_exit(from.read_log(&name), "a saved log");
        or_exit(to.replace_log(&name, &bytes), "copying a log");
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
    or_exit(save(&*store, LOG, &mut check), "saving a turn");
    let reloaded = or_exit(load(&*store, LOG), "reloading the session");
    if reloaded.messages().len() != MESSAGES + 4 {
        eprintln!("save_per_turn saved {} messages", reloaded.messages().len());
        std::process::exit(1);
    }
    bencher
        .with_inputs(fresh)
        .bench_local_values(|(store, mut conversation)| {
            save(&*store, LOG, &mut conversation).map(|()| store)
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
