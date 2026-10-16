//! The thin client `rig --attach` runs: it connects to a session's socket,
//! prints what arrives as plain lines and sends what is typed. The app only
//! hears how it ended, to exit.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use serde_json::{Value, json};

use crate::socket_attach::server::sockets;
use crate::socket_attach::{ATTACH_SESSION, ENDED};

/// How the client ended.
enum Ending {
    /// Ctrl-D: the session runs on.
    Detached,
    /// The session ended, or the connection broke.
    Ended,
}

/// Connects at startup, and exits when the client ends.
pub(crate) fn build(app: &mut App) {
    let home = RigHome::of(app.world());
    let session = app
        .world()
        .get_resource::<Invoked>()
        .and_then(|invoked| invoked.flag_value(ATTACH_SESSION).map(str::to_owned));
    let socket = find(home.root(), session.as_deref());
    app.add_systems(Startup, move |wake: Res<Wake>, mut commands: Commands| {
        let started = socket.clone().and_then(|path| {
            attach(&path).map_err(|why| format!("cannot attach to {}: {why}", path.display()))
        });
        match started {
            Ok(endings) => {
                let pool = IoTaskPool::get_or_init(TaskPool::default);
                commands.spawn(Running::stream(pool, &wake, endings));
            }
            Err(why) => {
                eprintln!("rig: {why}");
                commands.trigger(Exit(ExitReason::Failed(1)));
            }
        }
    })
    .add_observer(ended);
}

/// Exits once the client ended.
fn ended(_: On<Insert<Done<Ending>>>, mut commands: Commands) {
    commands.trigger(Exit(ExitReason::Quit));
}

/// The socket of `session`, else of the newest session that listens.
fn find(home: &Path, session: Option<&str>) -> Result<PathBuf, String> {
    let dir = sockets(home);
    if let Some(session) = session {
        let path = dir.join(format!("{session}.sock"));
        return if path.exists() {
            Ok(path)
        } else {
            Err(format!(
                "session {session} does not listen ({})",
                path.display()
            ))
        };
    }
    let entries = std::fs::read_dir(&dir).map_err(|_| "no session listens".to_owned())?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "sock")
        })
        .filter(|path| UnixStream::connect(path).is_ok())
        .max_by_key(|path| path.metadata().and_then(|meta| meta.modified()).ok())
        .ok_or_else(|| "no session listens".to_owned())
}

/// A picker or question open in the session, as this client saw it.
struct Open {
    id: u64,
    items: usize,
    free_text: bool,
}

/// Starts the threads that print what arrives and send what is typed;
/// each says how the client ended.
fn attach(path: &Path) -> io::Result<async_channel::Receiver<Ending>> {
    let stream = UnixStream::connect(path)?;
    let (endings, received) = async_channel::unbounded();
    let open = Arc::new(Mutex::new(Vec::<Open>::new()));
    let (reader, shown, ended) = (stream.try_clone()?, Arc::clone(&open), endings.clone());
    std::thread::Builder::new()
        .name("rig-attach-print".to_owned())
        .spawn(move || {
            let mut printer = Printer::default();
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                let event: Value = serde_json::from_str(&line).unwrap_or_default();
                let mut open = shown.lock().unwrap_or_else(PoisonError::into_inner);
                printer.show(&event, &mut open);
            }
            printer.end_line();
            ended.send_blocking(Ending::Ended).ok();
        })?;
    let mut writer = stream;
    std::thread::Builder::new()
        .name("rig-attach-type".to_owned())
        .spawn(move || {
            for typed in io::stdin().lock().lines().map_while(Result::ok) {
                let line = {
                    let open = open.lock().unwrap_or_else(PoisonError::into_inner);
                    answer(&typed, open.first())
                };
                if writeln!(writer, "{line}").is_err() {
                    return;
                }
            }
            endings.send_blocking(Ending::Detached).ok();
        })?;
    Ok(received)
}

/// The line to send for what the user typed: the answer to the oldest open
/// picker (an item's number, text where it takes text, or an empty line to
/// close it), else a message.
fn answer(typed: &str, open: Option<&Open>) -> Value {
    let Some(open) = open else {
        return json!({ "text": typed });
    };
    let item = typed
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|number| number.checked_sub(1))
        .filter(|&index| index < open.items);
    match item {
        Some(index) => json!({ "pick": open.id, "choice": index }),
        None if typed.trim().is_empty() => json!({ "pick": open.id, "choice": null }),
        None if open.free_text => json!({ "pick": open.id, "choice": typed }),
        None => json!({ "text": typed }),
    }
}

/// Prints events as plain lines; streamed text runs on until the next
/// event.
#[derive(Default)]
struct Printer {
    in_text: bool,
}

impl Printer {
    fn end_line(&mut self) {
        if std::mem::take(&mut self.in_text) {
            println!();
        }
    }

    fn line(&mut self, text: &str) {
        self.end_line();
        println!("{text}");
    }

    fn show(&mut self, event: &Value, open: &mut Vec<Open>) {
        let text = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or_default();
        let agent = text("agent");
        match text("type") {
            "backlog" => {
                let items = event.get("items").and_then(Value::as_array);
                for item in items.into_iter().flatten() {
                    let role = item.get("role").and_then(Value::as_str).unwrap_or_default();
                    let said = item.get("text").and_then(Value::as_str).unwrap_or_default();
                    self.line(&format!("{role}: {said}"));
                }
                self.line(&format!("-- attached to {agent}; Ctrl-D detaches --"));
            }
            "delivery" => self.line(&format!("{agent} < {}", text("text"))),
            "text" => {
                if !self.in_text {
                    print!("{agent} > ");
                    self.in_text = true;
                }
                print!("{}", text("text"));
                io::stdout().flush().ok();
            }
            "reasoning" => {}
            "tool_start" => {
                let arguments = event.get("arguments").cloned().unwrap_or_default();
                self.line(&format!("{agent}: {} {arguments}", text("name")));
            }
            "tool_end" => {
                let first = text("output").lines().next().unwrap_or_default();
                self.line(&format!("{agent}: {} -> {first}", text("name")));
            }
            "notice" => self.line(&format!("[{}] {}", text("level"), text("text"))),
            "picker" => {
                let id = event.get("id").and_then(Value::as_u64).unwrap_or_default();
                let items = event.get("items").and_then(Value::as_array);
                let labels: Vec<String> = items
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .map(|(index, label)| {
                        format!("{}) {}", index + 1, label.as_str().unwrap_or_default())
                    })
                    .collect();
                let free_text = event
                    .get("free_text")
                    .and_then(Value::as_bool)
                    .unwrap_or_default();
                self.line(&format!("? {} {}", text("title"), text("body")));
                self.line(&format!(
                    "  {}  (a number{}, or an empty line to close)",
                    labels.join("  "),
                    if free_text { " or text" } else { "" }
                ));
                open.retain(|seen| seen.id != id);
                open.push(Open {
                    id,
                    items: labels.len(),
                    free_text,
                });
            }
            "closed" => {
                let id = event.get("id").and_then(Value::as_u64);
                let before = open.len();
                open.retain(|seen| Some(seen.id) != id);
                if open.len() < before {
                    self.line("  (answered)");
                }
            }
            "turn_end" => self.line(&format!("{agent}: turn {}", text("outcome"))),
            "error" => self.line(&format!("! {}", text("text"))),
            "ended" => self.line(ENDED),
            _ => {}
        }
    }
}
