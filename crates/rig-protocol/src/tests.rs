use std::path::PathBuf;

use serde_json::json;

use crate::event::{Event, Notice};
use crate::{
    DirGrant, EventRecord, Frame, FrameError, Grant, MAX_FRAME, Request, RequestId, Rule, Seq,
    SessionId, Side, Wider, codes, read_frame, write_frame,
};

fn frames(input: &[u8]) -> Vec<Result<Option<Frame>, FrameError>> {
    let mut reader = input;
    let mut line = Vec::new();
    let mut out = Vec::new();
    loop {
        let next = read_frame(&mut reader, &mut line);
        let end = matches!(next, Ok(None));
        out.push(next);
        if end {
            return out;
        }
    }
}

#[test]
fn a_written_frame_reads_back() {
    let request = Request::new(RequestId::Num(7), "rig/session/list", json!({"tree": true}));
    let mut wire = Vec::new();
    write_frame(&mut wire, &Frame::Request(request)).ok();
    assert_eq!(wire.last(), Some(&b'\n'));
    let mut line = Vec::new();
    let read = read_frame(&mut wire.as_slice(), &mut line);
    assert!(
        matches!(&read, Ok(Some(Frame::Request(r))) if r.id == RequestId::Num(7) && r.params == json!({"tree": true})),
        "{read:?}"
    );
}

#[test]
fn responses_and_notifications_are_told_apart() {
    let input = concat!(
        r#"{"jsonrpc":"2.0","rig":1,"id":"a","result":{"ok":true}}"#,
        "\r\n\n",
        r#"{"jsonrpc":"2.0","rig":1,"id":2,"error":{"code":-32012,"message":"no such session"}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","rig":1,"method":"rig/event","params":{}}"#,
    );
    let read = frames(input.as_bytes());
    assert!(matches!(read.first(), Some(Ok(Some(Frame::Response(_))))));
    assert!(matches!(read.get(1), Some(Ok(Some(Frame::Response(_))))));
    assert!(matches!(
        read.get(2),
        Some(Ok(Some(Frame::Notification(_))))
    ));
    assert!(matches!(read.get(3), Some(Ok(None))));
}

#[test]
fn bad_lines_are_refused_and_the_next_line_still_reads() {
    let good = r#"{"jsonrpc":"2.0","rig":1,"method":"m","params":null}"#;
    let mut input = Vec::new();
    input.extend(std::iter::repeat_n(b'x', MAX_FRAME + 10));
    input.push(b'\n');
    input.extend(b"[1,2]\n");
    input.extend(b"\xff\xfe\n");
    input.extend(br#"{"jsonrpc":"2.0","id":1,"method":"m"}"#);
    input.push(b'\n');
    input.extend(br#"{"jsonrpc":"2.0","rig":2,"id":1,"method":"m"}"#);
    input.push(b'\n');
    input.extend(good.as_bytes());
    let read = frames(&input);
    assert!(matches!(read.first(), Some(Err(FrameError::TooLarge))));
    assert!(matches!(read.get(1), Some(Err(FrameError::Batch))));
    assert!(matches!(read.get(2), Some(Err(FrameError::Utf8))));
    assert!(matches!(
        read.get(3),
        Some(Err(FrameError::Version {
            got: None,
            id: Some(RequestId::Num(1))
        }))
    ));
    let refusal = match read.get(4) {
        Some(Err(newer)) => newer.rpc(),
        _ => None,
    };
    assert_eq!(refusal.as_ref().map(|e| e.code), Some(codes::VERSION));
    assert_eq!(
        refusal.and_then(|e| e.data).and_then(|d| d.update),
        Some(Side::Server)
    );
    assert!(matches!(
        read.get(5),
        Some(Ok(Some(Frame::Notification(_))))
    ));
}

fn dir(path: &str, write: bool) -> DirGrant {
    DirGrant {
        path: PathBuf::from(path),
        write,
    }
}

#[test]
fn a_child_grant_is_the_intersection_or_names_what_is_wider() {
    let parent = Grant {
        tools: Some(vec!["read".into(), "shell".into()]),
        dirs: Some(vec![dir("/w", true), dir("/docs", false)]),
        net: false,
        children: Some(4),
        approve: vec![
            Rule::ShellPrefix {
                prefix: "cargo".into(),
            },
            Rule::Deny { name: "rm".into() },
        ],
    };
    let child = parent.narrow(&Grant {
        tools: Some(vec!["read".into()]),
        dirs: Some(vec![dir("/w/sub", true), dir("/docs/a", false)]),
        approve: vec![Rule::ShellPrefix {
            prefix: "cargo test".into(),
        }],
        ..Grant::default()
    });
    assert_eq!(
        child,
        Ok(Grant {
            tools: Some(vec!["read".into()]),
            dirs: Some(vec![dir("/w/sub", true), dir("/docs/a", false)]),
            net: false,
            children: Some(4),
            approve: vec![
                Rule::Deny { name: "rm".into() },
                Rule::ShellPrefix {
                    prefix: "cargo test".into()
                },
            ],
        })
    );
    let wider = |requested: Grant| parent.narrow(&requested).err();
    assert_eq!(
        wider(Grant {
            tools: Some(vec!["edit".into()]),
            ..Grant::default()
        }),
        Some(Wider::Tools(vec!["edit".into()]))
    );
    assert_eq!(
        wider(Grant {
            dirs: Some(vec![dir("/etc", false)]),
            ..Grant::default()
        }),
        Some(Wider::Dir(PathBuf::from("/etc")))
    );
    assert_eq!(
        wider(Grant {
            dirs: Some(vec![dir("/docs/a", true)]),
            ..Grant::default()
        }),
        Some(Wider::Write(PathBuf::from("/docs")))
    );
    assert_eq!(
        wider(Grant {
            net: true,
            ..Grant::default()
        }),
        Some(Wider::Net)
    );
    assert_eq!(
        wider(Grant {
            children: Some(9),
            ..Grant::default()
        }),
        Some(Wider::Children {
            requested: 9,
            limit: 4
        })
    );
    assert_eq!(
        wider(Grant {
            approve: vec![Rule::Tool {
                name: "shell".into()
            }],
            ..Grant::default()
        }),
        Some(Wider::Rule(Rule::Tool {
            name: "shell".into()
        }))
    );
}

#[test]
fn built_in_events_round_trip_and_unknown_kinds_stay_records() {
    let notice = Event::Notice(Notice {
        level: "info".into(),
        text: "hi".into(),
    });
    let record = EventRecord {
        session: SessionId("s-1".into()),
        seq: Seq(3),
        at_ms: 0,
        kind: notice.kind().into(),
        data: notice.encode(),
        source: None,
    };
    assert_eq!(Event::decode(&record), Some(notice));
    let plugin = EventRecord {
        kind: "x/todo/added".into(),
        ..record
    };
    assert_eq!(Event::decode(&plugin), None);
}

// The version fixtures: v1 frames round-trip byte for byte, through the
// frame and through each method's typed params and result; v0 and v2
// frames are refused naming the side to update.

mod fixtures {
    use serde_json::Value;

    use crate::event::Event;
    use crate::methods::{self, VersionRange};
    use crate::upgrade::negotiate;
    use crate::{
        EventRecord, Frame, FrameError, Method, Outcome, RequestId, Side, codes, read_frame,
        write_frame,
    };

    fn lines(name: &str) -> Vec<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        std::fs::read_to_string(&path)
            .map(|text| text.lines().map(str::to_owned).collect())
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn frame(line: &str) -> Frame {
        let mut line_buf = Vec::new();
        let frame = read_frame(&mut line.as_bytes(), &mut line_buf)
            .unwrap_or_else(|error| panic!("{line}: {error}"))
            .unwrap_or_else(|| panic!("{line}: empty"));
        let mut wire = Vec::new();
        write_frame(&mut wire, &frame).expect("write");
        assert_eq!(
            String::from_utf8_lossy(&wire),
            format!("{line}\n"),
            "the frame round-trips"
        );
        frame
    }

    /// `value` read as `T` and written back, byte for byte.
    fn typed<T: serde::Serialize + serde::de::DeserializeOwned>(
        value: &Value,
    ) -> Result<(), String> {
        let typed: T = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        let back = serde_json::to_string(&typed).map_err(|e| e.to_string())?;
        let want = serde_json::to_string(value).map_err(|e| e.to_string())?;
        if back == want {
            Ok(())
        } else {
            Err(format!("typed:   {back}\nfixture: {want}"))
        }
    }

    macro_rules! by_method {
        ($method:expr, $value:expr, $part:ident, [$($m:ty),* $(,)?]) => {{
            let method: &str = $method;
            $(if method == <$m as Method>::NAME {
                typed::<<$m as Method>::$part>($value)
            } else)* {
                Err(format!("no method {method}"))
            }
        }};
    }

    fn check(method: &str, value: &Value, params: bool) -> Result<(), String> {
        use methods::*;
        match params {
            true => by_method!(
                method,
                value,
                Params,
                [
                    Hello,
                    Start,
                    methods::Send,
                    Wait,
                    Answer,
                    Stop,
                    List,
                    methods::Read,
                    Subscribe,
                    Unsubscribe,
                    ReadEvents,
                    Attach,
                    Detach,
                    Reload,
                    WorkerHello,
                    WorkerEvents,
                    WorkerApplied,
                    WorkerQuiescent,
                    WorkerReply,
                ]
            ),
            false => by_method!(
                method,
                value,
                Result,
                [
                    Hello,
                    Start,
                    methods::Send,
                    Wait,
                    Answer,
                    Stop,
                    List,
                    methods::Read,
                    Subscribe,
                    Unsubscribe,
                    ReadEvents,
                    Attach,
                    Detach,
                    Reload,
                    WorkerHello,
                    WorkerEvents,
                    WorkerApplied,
                    WorkerQuiescent,
                    WorkerReply,
                ]
            ),
        }
    }

    #[test]
    fn v1_requests_and_responses_round_trip_through_their_types() {
        let mut methods = std::collections::HashMap::new();
        for line in lines("v1/requests.jsonl") {
            let Frame::Request(request) = frame(&line) else {
                panic!("not a request: {line}");
            };
            check(&request.method, &request.params, true)
                .unwrap_or_else(|e| panic!("{} params:\n{e}", request.method));
            methods.insert(request.id, request.method);
        }
        assert_eq!(methods.len(), 23);
        let mut errors = 0;
        for line in lines("v1/responses.jsonl") {
            let Frame::Response(response) = frame(&line) else {
                panic!("not a response: {line}");
            };
            match (&response.outcome, methods.get(&response.id)) {
                (Outcome::Result(result), Some(method)) => {
                    check(method, result, false).unwrap_or_else(|e| panic!("{method} result:\n{e}"))
                }
                (Outcome::Error(_), _) => errors += 1,
                (Outcome::Result(_), None) => panic!("no request for {line}"),
            }
        }
        assert_eq!(errors, 3);
    }

    #[test]
    fn v1_events_round_trip_and_every_built_in_kind_is_covered() {
        let mut kinds = std::collections::BTreeSet::new();
        for line in lines("v1/events.jsonl") {
            let value: Value = serde_json::from_str(&line).expect("json");
            typed::<EventRecord>(&value).unwrap_or_else(|e| panic!("record:\n{e}"));
            let record: EventRecord = serde_json::from_value(value).expect("record");
            match Event::decode(&record) {
                Some(event) => {
                    assert_eq!(event.kind(), record.kind);
                    assert_eq!(
                        serde_json::to_string(&event.encode()).ok(),
                        serde_json::to_string(&record.data).ok(),
                        "{} re-encodes",
                        record.kind
                    );
                    kinds.insert(event.kind());
                }
                None => assert!(record.kind.starts_with("x/"), "{} must decode", record.kind),
            }
        }
        let all: std::collections::BTreeSet<_> = Event::KINDS.iter().copied().collect();
        assert_eq!(kinds, all, "a fixture for every built-in kind");
    }

    fn refusal(line: &str) -> (Side, Option<RequestId>) {
        let mut buf = Vec::new();
        let error = read_frame(&mut line.as_bytes(), &mut buf).err();
        let Some(error @ FrameError::Version { .. }) = error else {
            panic!("{line} must be refused: {error:?}");
        };
        let FrameError::Version { id, .. } = &error else {
            unreachable!()
        };
        let rpc = error.rpc().expect("an answer");
        assert_eq!(rpc.code, codes::VERSION);
        let data = rpc.data.expect("data");
        assert_eq!(data.supported, vec![1]);
        assert!(data.next.is_some(), "it says what to do");
        (data.update.expect("a side"), id.clone())
    }

    #[test]
    fn older_frames_are_refused_naming_the_client_and_newer_the_server() {
        for line in lines("refuse/v0.jsonl") {
            assert_eq!(refusal(&line).0, Side::Client, "{line}");
        }
        let v2 = lines("refuse/v2.jsonl");
        for line in &v2 {
            assert_eq!(refusal(line).0, Side::Server, "{line}");
        }
        // The refusal answers the request it refuses.
        assert_eq!(
            v2.first().map(|l| refusal(l).1),
            Some(Some(RequestId::Num(1)))
        );
    }

    #[test]
    fn hello_settles_on_the_highest_shared_version_or_names_who_updates() {
        let side = |min, max| {
            negotiate(VersionRange { min, max })
                .err()
                .and_then(|e| e.data)
                .and_then(|d| d.update)
        };
        assert_eq!(negotiate(VersionRange { min: 1, max: 3 }).ok(), Some(1));
        assert_eq!(side(2, 3), Some(Side::Server));
        assert_eq!(side(0, 0), Some(Side::Client));
    }
}
