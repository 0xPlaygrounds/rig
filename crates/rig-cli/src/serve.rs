//! The verbs about the server rather than a session: `rig serve`, `rig
//! server status|stop`, `rig connect --stdio`, and `rig attach`, which
//! starts the terminal view (this binary in its `view` role).

use std::io::{BufRead, Write};
use std::process::Command;
use std::time::{Duration, Instant};

use rig_client::{Autostart, Client, Options, Target};
use rig_harness_protocol::{Home, env};
use rig_protocol::Exit;
use serde_json::{Value, json};

use crate::args::Args;
use crate::out::{self, Failure};
use crate::verbs::Ctx;

/// Replaces this process with this binary in `role`, with `vars` set.
fn become_role(role: &str, vars: &[(&str, &str)], verb: &str) -> Result<u8, Failure> {
    let exe = std::env::current_exe().map_err(|error| {
        Failure::new(
            Exit::Failed,
            format!("could not find the rig binary: {error}"),
            format!("rig help {verb}"),
        )
    })?;
    let mut command = Command::new(exe);
    command.env(env::ROLE, role).envs(vars.iter().copied());
    run(command, verb)
}

#[cfg(unix)]
fn run(mut command: Command, verb: &str) -> Result<u8, Failure> {
    use std::os::unix::process::CommandExt;
    let error = command.exec();
    Err(Failure::new(
        Exit::Failed,
        format!("could not start rig {verb}: {error}"),
        format!("rig help {verb}"),
    ))
}

#[cfg(not(unix))]
fn run(mut command: Command, verb: &str) -> Result<u8, Failure> {
    let status = command.status().map_err(|error| {
        Failure::new(
            Exit::Failed,
            format!("could not start rig {verb}: {error}"),
            format!("rig help {verb}"),
        )
    })?;
    Ok(status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .unwrap_or(1))
}

/// `rig attach ID [--takeover]`: the terminal view on that session (by id
/// or name), as its driver.
pub fn attach(args: &Args) -> Result<u8, Failure> {
    let session = args.session("attach")?;
    let ctx = Ctx::new(args)?;
    let id = crate::verbs::resolve(&ctx.connect()?, session);
    let how = if args.has("takeover") {
        "takeover"
    } else {
        "driver"
    };
    become_role(
        "view",
        &[(env::SESSION, id.0.as_str()), (env::ATTACH, how)],
        "attach",
    )
}

/// `rig serve`: `rigd` in the foreground.
pub fn serve() -> Result<u8, Failure> {
    become_role("server", &[], "serve")
}

fn local() -> Target {
    Target::Local {
        rig_home: Home::from_env().root().to_path_buf(),
    }
}

/// A connection to a running local server, never starting one.
fn running() -> Option<Client> {
    let options = Options {
        autostart: Autostart::Off,
        reconnect: false,
        give_up: Duration::from_millis(500),
        ..Options::default()
    };
    Client::connect_with(&local(), "rig-cli", options).ok()
}

/// `rig server status|stop`.
pub fn server(args: &Args) -> Result<u8, Failure> {
    let json = args.has("json");
    match args.words.first().map(String::as_str) {
        Some("status") => {
            let hello = running().and_then(|client| client.hello());
            let Some(hello) = hello else {
                if json {
                    out::line(&out::object(
                        "server",
                        json!({"status": "stopped", "next": "rig agents"}),
                    ));
                } else {
                    eprintln!(
                        "rig: no rig server runs; the next command starts one\nnext: rig agents"
                    );
                }
                return Ok(Exit::Unavailable.code());
            };
            if json {
                out::line(&out::object(
                    "server",
                    json!({
                        "status": "running",
                        "pid": hello.server.pid,
                        "version": hello.server.version,
                    }),
                ));
            } else {
                out::text(&format!(
                    "running: pid {}, rig {}",
                    hello.server.pid, hello.server.version
                ));
            }
            Ok(0)
        }
        Some("stop") => {
            let Some(pid) = running()
                .and_then(|client| client.hello())
                .map(|hello| hello.server.pid)
            else {
                if json {
                    out::line(&out::object("server", json!({"status": "stopped"})));
                } else {
                    out::text("no rig server runs");
                }
                return Ok(0);
            };
            terminate(pid)?;
            let deadline = Instant::now() + Duration::from_secs(10);
            while running().is_some() {
                if Instant::now() > deadline {
                    return Err(Failure::new(
                        Exit::TimedOut,
                        format!("the rig server (pid {pid}) is still running"),
                        format!("kill {pid}"),
                    ));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            if json {
                out::line(&out::object(
                    "server",
                    json!({"status": "stopped", "pid": pid}),
                ));
            } else {
                out::text(&format!("stopped the rig server (pid {pid})"));
            }
            Ok(0)
        }
        _ => Err(Failure::usage(
            "rig server needs `status` or `stop`",
            "server",
        )),
    }
}

/// Asks process `pid` to exit (SIGTERM; on Windows, taskkill).
fn terminate(pid: u32) -> Result<(), Failure> {
    let pid_text = pid.to_string();
    let mut command = if cfg!(windows) {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid_text]);
        command
    } else {
        let mut command = Command::new("kill");
        command.args(["-TERM", &pid_text]);
        command
    };
    let status = command.status().map_err(|error| {
        Failure::new(
            Exit::Failed,
            format!("could not signal the rig server: {error}"),
            format!("kill {pid}"),
        )
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(Failure::new(
            Exit::Failed,
            format!("could not stop the rig server (pid {pid})"),
            format!("kill {pid}"),
        ))
    }
}

/// The hello line with the local token put in, when `line` is a
/// `rig/hello` request whose token is missing or empty (a remote client
/// has none: its transport, ssh or koh, authenticated it). Any other line
/// is `None`, and passes as it is.
pub fn with_token(line: &[u8], token: &str) -> Option<Vec<u8>> {
    let mut frame: Value = serde_json::from_slice(line).ok()?;
    if frame.get("method").and_then(Value::as_str) != Some("rig/hello") {
        return None;
    }
    let params = frame.get_mut("params")?.as_object_mut()?;
    let empty = params
        .get("token")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty);
    if !empty {
        return None;
    }
    params.insert("token".to_owned(), token.into());
    let mut text = serde_json::to_vec(&frame).ok()?;
    text.push(b'\n');
    Some(text)
}

/// `rig connect --stdio`: the frames on stdin go to the local server
/// (started if needed), its frames to stdout. The first `rig/hello`
/// gets this host's token, since only someone who could run this command
/// as this user could read it.
pub fn connect(args: &Args) -> Result<u8, Failure> {
    if !args.has("stdio") {
        return Err(Failure::usage("rig connect needs --stdio", "connect"));
    }
    bridge()
}

#[cfg(unix)]
fn bridge() -> Result<u8, Failure> {
    let home = Home::from_env();
    let unavailable = |error: std::io::Error| {
        Failure::new(
            Exit::Unavailable,
            format!("could not reach the rig server: {error}"),
            "RIG_VERBOSE=1 rig server status",
        )
    };
    // A client starts the server if none runs; the bridge's own
    // connection follows.
    let ctx = Ctx {
        target: local(),
        json: false,
    };
    let starter = ctx.connect()?;
    let token = rig_protocol::read_token(home.root()).map_err(unavailable)?;
    let conn = rig_protocol::Conn::connect(&rig_protocol::Endpoint::for_home(home.root()))
        .map_err(unavailable)?;
    drop(starter);
    let (mut from_server, mut to_server) = conn.split().map_err(unavailable)?;
    std::thread::spawn(move || {
        let mut stdout = std::io::stdout().lock();
        let mut line = Vec::new();
        loop {
            line.clear();
            match from_server.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if stdout
                        .write_all(&line)
                        .and_then(|()| stdout.flush())
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
        // The server hung up: so does the bridge.
        std::process::exit(0);
    });
    let mut stdin = std::io::stdin().lock();
    let mut line = Vec::new();
    let mut said_hello = false;
    loop {
        line.clear();
        match stdin.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let rewritten = (!said_hello).then(|| with_token(&line, &token)).flatten();
        if rewritten.is_some() {
            said_hello = true;
        }
        let sent = to_server
            .write_all(rewritten.as_deref().unwrap_or(&line))
            .and_then(|()| to_server.flush());
        if sent.is_err() {
            break;
        }
    }
    Ok(0)
}

#[cfg(not(unix))]
fn bridge() -> Result<u8, Failure> {
    Err(Failure::new(
        Exit::Unavailable,
        "rig connect --stdio needs the local socket, which this platform's build lacks yet",
        "rig help-agents remote",
    ))
}
