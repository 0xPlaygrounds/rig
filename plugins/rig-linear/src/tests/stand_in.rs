//! A stand-in for Linear's API on a local port: it keeps every request it
//! gets and answers each as Linear would, unless a test scripts otherwise.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

/// The session the stand-in creates.
pub(crate) const SESSION_ID: &str = "session-1";

/// Its link.
pub(crate) const SESSION_URL: &str = "https://linear.example/acme/issue/ENG-1/agent-session-1";

/// One request as the stand-in got it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Request {
    /// The path, such as `/graphql`.
    pub(crate) path: String,
    /// The `authorization` header, if any.
    pub(crate) authorization: Option<String>,
    /// The `content-type` header, if any.
    pub(crate) content_type: Option<String>,
    /// The body.
    pub(crate) body: String,
}

impl Request {
    /// The body as JSON, or `null`.
    pub(crate) fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_default()
    }

    /// The GraphQL query's text.
    pub(crate) fn query(&self) -> String {
        self.json()
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }
}

/// A status, extra headers and a body.
pub(crate) type Reply = (u16, Vec<(&'static str, String)>, String);

/// What answers each request: the request and how many came before it.
type Answer = dyn Fn(&Request, usize) -> Option<Reply> + Send + Sync;

/// The running stand-in.
pub(crate) struct StandIn {
    /// Its GraphQL endpoint.
    pub(crate) url: String,
    seen: Arc<Mutex<Vec<Request>>>,
}

impl StandIn {
    /// A stand-in answering as Linear would.
    pub(crate) fn start() -> Self {
        Self::scripted(|_, _| None)
    }

    /// A stand-in where `script` answers what it wants to, and Linear's
    /// answers the rest.
    pub(crate) fn scripted(
        script: impl Fn(&Request, usize) -> Option<Reply> + Send + Sync + 'static,
    ) -> Self {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
            return Self {
                url: "http://127.0.0.1:9/graphql".to_owned(),
                seen,
            };
        };
        let port = listener.local_addr().map(|addr| addr.port()).unwrap_or(9);
        let answer: Arc<Answer> = Arc::new(script);
        let kept = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (answer, kept) = (Arc::clone(&answer), Arc::clone(&kept));
                std::thread::spawn(move || serve(stream, &*answer, &kept));
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}/graphql"),
            seen,
        }
    }

    /// Every request so far, in order.
    pub(crate) fn requests(&self) -> Vec<Request> {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The bodies of every request so far, as JSON.
    pub(crate) fn bodies(&self) -> Vec<Value> {
        self.requests().iter().map(Request::json).collect()
    }
}

/// Answers the requests of one connection until it closes.
fn serve(stream: TcpStream, answer: &Answer, seen: &Mutex<Vec<Request>>) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    while let Some(request) = read(&mut reader) {
        let index = {
            let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
            seen.push(request.clone());
            seen.len() - 1
        };
        let (status, headers, body) = answer(&request, index).unwrap_or_else(|| linear(&request));
        let mut head = format!(
            "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n",
            if status == 200 { "OK" } else { "Error" },
            body.len()
        );
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        if writer
            .write_all(head.as_bytes())
            .and_then(|()| writer.write_all(body.as_bytes()))
            .and_then(|()| writer.flush())
            .is_err()
        {
            return;
        }
    }
}

/// The next request on the connection, or `None` once it closed.
fn read(reader: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut line = String::new();
    reader.read_line(&mut line).ok().filter(|read| *read > 0)?;
    let path = line.split_whitespace().nth(1)?.to_owned();
    let mut request = Request {
        path,
        ..Request::default()
    };
    let mut length = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).ok().filter(|read| *read > 0)?;
        let header = line.trim_end();
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            continue;
        };
        let value = value.trim().to_owned();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.parse().unwrap_or(0),
            "authorization" => request.authorization = Some(value),
            "content-type" => request.content_type = Some(value),
            _ => {}
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    request.body = String::from_utf8_lossy(&body).into_owned();
    Some(request)
}

/// What Linear answers `request`.
pub(crate) fn linear(request: &Request) -> Reply {
    if request.path == "/oauth/token" {
        let token = json!({ "access_token": "app-token", "token_type": "Bearer" });
        return (200, Vec::new(), token.to_string());
    }
    let query = request.query();
    let data = if query.contains("issue(id:") {
        json!({ "issue": {
            "id": "issue-uuid",
            "identifier": "ENG-1",
            "url": "https://linear.example/acme/issue/ENG-1",
        } })
    } else if query.contains("agentSessionCreateOnIssue") {
        json!({ "agentSessionCreateOnIssue": {
            "success": true,
            "agentSession": { "id": SESSION_ID, "url": SESSION_URL },
        } })
    } else if query.contains("agentSession(id:") {
        json!({ "agentSession": {
            "id": SESSION_ID,
            "url": SESSION_URL,
            "issue": { "identifier": "ENG-1" },
        } })
    } else if query.contains("agentSessionUpdate") {
        json!({ "agentSessionUpdate": { "success": true } })
    } else if query.contains("agentActivityCreate") {
        json!({ "agentActivityCreate": { "success": true } })
    } else {
        return (
            400,
            Vec::new(),
            json!({ "errors": [{ "message": "unknown query" }] }).to_string(),
        );
    };
    (200, Vec::new(), json!({ "data": data }).to_string())
}
