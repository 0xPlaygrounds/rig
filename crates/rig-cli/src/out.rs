//! What a verb prints: on success its text or one JSON object on stdout;
//! on failure `rig: <what happened>` and `next: <command>` on stderr (and
//! with `--json` the error object on stdout too).

use std::io::Write;

use rig_client::ClientError;
use rig_protocol::Exit;
use serde_json::{Map, Value, json};

/// Why a verb failed, the exit code it ends with, and the command that
/// fixes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub exit: Exit,
    pub message: String,
    pub next: String,
    /// The wire's error code, when the server refused.
    pub code: Option<i32>,
    /// Extra fields for the JSON object, such as `sessionId` or `status`.
    pub extra: Map<String, Value>,
}

impl Failure {
    pub fn new(exit: Exit, message: impl Into<String>, next: impl Into<String>) -> Failure {
        Failure {
            exit,
            message: message.into(),
            next: next.into(),
            code: None,
            extra: Map::new(),
        }
    }

    /// A command line the verb refuses: exit 2, next `rig help VERB`.
    pub fn usage(message: impl Into<String>, verb: &str) -> Failure {
        Failure::new(Exit::Usage, message, format!("rig help {verb}"))
    }

    /// A client error; `next` is the fix when the error names none.
    pub fn client(error: ClientError, next: impl Into<String>) -> Failure {
        let exit = error.exit();
        let named = error.next().map(str::to_owned);
        let (message, code) = match &error {
            ClientError::Rpc(rpc) => (rpc.message.clone(), Some(rpc.code)),
            ClientError::Unavailable { .. } => ("the rig server is unavailable".to_owned(), None),
            other => (other.to_string(), None),
        };
        let next = named
            .filter(|next| !next.is_empty())
            .unwrap_or_else(|| match &error {
                ClientError::Unavailable { .. } | ClientError::Io(_) => {
                    "RIG_VERBOSE=1 rig server status".to_owned()
                }
                ClientError::Version { .. } => "rig build && rig server stop".to_owned(),
                ClientError::Rpc(_) => next.into(),
            });
        Failure {
            exit,
            message,
            next,
            code,
            extra: Map::new(),
        }
    }

    /// The same failure with another `next:`.
    pub fn next_is(mut self, next: &str) -> Failure {
        next.clone_into(&mut self.next);
        self
    }

    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Failure {
        self.extra.insert(key.to_owned(), value.into());
        self
    }

    /// Prints the failure and returns its exit code.
    pub fn report(&self, json: bool) -> u8 {
        if json {
            let mut object = json!({
                "rig": 1,
                "kind": "error",
                "message": self.message,
                "next": self.next,
                "exit": self.exit.code(),
            });
            if let (Some(code), Some(map)) = (self.code, object.as_object_mut()) {
                map.insert("code".to_owned(), code.into());
            }
            if let Some(map) = object.as_object_mut() {
                map.extend(self.extra.clone());
            }
            line(&object);
        }
        eprintln!("rig: {}\nnext: {}", self.message, self.next);
        self.exit.code()
    }
}

/// Prints `value` as one line of JSON on stdout.
pub fn line(value: &Value) {
    let mut stdout = std::io::stdout().lock();
    // A closed stdout (a reader gone) is nobody's to tell.
    writeln!(stdout, "{value}").ok();
    stdout.flush().ok();
}

/// Prints `text` on stdout, ending it with a newline.
pub fn text(text: &str) {
    let mut stdout = std::io::stdout().lock();
    if text.ends_with('\n') {
        write!(stdout, "{text}").ok();
    } else {
        writeln!(stdout, "{text}").ok();
    }
    stdout.flush().ok();
}

/// A JSON object of `kind` with `"rig":1`, extended by `fields`.
pub fn object(kind: &str, fields: Value) -> Value {
    let mut object = Map::new();
    object.insert("rig".to_owned(), 1.into());
    object.insert("kind".to_owned(), kind.into());
    if let Value::Object(fields) = fields {
        object.extend(fields);
    }
    Value::Object(object)
}
