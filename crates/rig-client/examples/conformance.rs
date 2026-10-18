//! The conformance adapter for rig-client: reads the runner's requests on
//! stdin, answers on stdout (see `tests/conformance/run`).
//!
//! ```sh
//! cargo build -p rig-client --example conformance
//! tests/conformance/run target/debug/examples/conformance
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Duration;

use rig_client::{Autostart, Client, ClientError, Events, Options, Target};
use rig_protocol::{RpcError, Seq, SessionId, codes};
use serde_json::{Value, json};

#[derive(Default)]
struct Adapter {
    client: Option<Client>,
    streams: HashMap<String, Events>,
}

fn error(error: ClientError) -> Value {
    match error {
        ClientError::Rpc(RpcError {
            code,
            message,
            data,
        }) => json!({"code": code, "message": message, "data": data}),
        ClientError::Version { update, supported } => json!({
            "code": codes::VERSION,
            "message": "no common version",
            "data": {"update": update, "supported": supported},
        }),
        other => json!({"code": -1, "message": other.to_string(), "data": {"next": other.next()}}),
    }
}

fn usage(message: &str) -> ClientError {
    ClientError::Rpc(RpcError::new(codes::INVALID_PARAMS, message))
}

impl Adapter {
    fn client(&self) -> Result<&Client, ClientError> {
        self.client.as_ref().ok_or_else(|| usage("connect first"))
    }

    fn handle(&mut self, request: &Value) -> Result<Value, ClientError> {
        let text = |key: &str| request.get(key).and_then(Value::as_str).unwrap_or_default();
        match text("op") {
            "connect" => {
                let home = std::env::var_os("RIG_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_default();
                let options = Options {
                    token: request
                        .get("token")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    // The runner owns the server: restarting it is its job.
                    autostart: Autostart::Off,
                    give_up: Duration::from_secs(60),
                    ..Options::default()
                };
                let client = Client::connect_with(
                    &Target::Local { rig_home: home },
                    "conformance",
                    options,
                )?;
                let hello = serde_json::to_value(client.hello()).unwrap_or_default();
                self.client = Some(client);
                Ok(hello)
            }
            "call" => {
                let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
                self.client()?.call_raw(text("method"), params)
            }
            "subscribe" => {
                let session = SessionId(text("session").to_owned());
                let after = Seq(request.get("after").and_then(Value::as_u64).unwrap_or(0));
                let follow = request
                    .get("follow")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let events = self.client()?.subscribe_with(&session, after, follow)?;
                self.streams.insert(text("name").to_owned(), events);
                Ok(json!({}))
            }
            "next" => {
                let until = request.get("until").and_then(Value::as_str);
                let count = request.get("count").and_then(Value::as_u64);
                let events = self
                    .streams
                    .get_mut(text("name"))
                    .ok_or_else(|| usage("no such stream"))?;
                let mut out = Vec::new();
                for event in events.by_ref() {
                    let event = event?;
                    let done = until == Some(event.kind.as_str());
                    out.push(serde_json::to_value(&event).unwrap_or_default());
                    if done || count == Some(out.len() as u64) {
                        break;
                    }
                }
                Ok(Value::Array(out))
            }
            other => Err(usage(&format!("no op {other}"))),
        }
    }
}

fn main() {
    let mut adapter = Adapter::default();
    let stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(request) => match adapter.handle(&request) {
                Ok(value) => json!({"ok": value}),
                Err(e) => json!({"error": error(e)}),
            },
            Err(e) => json!({"error": {"code": codes::PARSE, "message": e.to_string()}}),
        };
        let mut out = stdout.lock();
        if writeln!(out, "{answer}")
            .and_then(|()| out.flush())
            .is_err()
        {
            break;
        }
    }
}
