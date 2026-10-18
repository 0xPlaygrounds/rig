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
    };
    assert_eq!(Event::decode(&record), Some(notice));
    let plugin = EventRecord {
        kind: "x/todo/added".into(),
        ..record
    };
    assert_eq!(Event::decode(&plugin), None);
}
