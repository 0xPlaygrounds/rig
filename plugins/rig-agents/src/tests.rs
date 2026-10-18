//! The tools against an in-memory server behind [`Channel`]: it answers
//! `rig/session/*` as `rigd` does for a parent's worker and keeps each
//! child's journal, which a test writes the child's events into.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use rig_harness::prelude::*;
use rig_harness_test_support::{
    MemoryStore, ToolOutputs, app_on, first_agent, log_tool_outputs, run_until, start_call,
};
use rig_protocol::event::{
    AskKind, AskOpened, AskOption, CommandAccepted, Event, Message, SessionStarted, TurnEnded,
    TurnStatus,
};
use rig_protocol::methods::{self, SessionInfo};
use rig_protocol::{
    AgentId, Answerers, AskId, BuildId, ClientId, CommandSeq, EventRecord, Key, Method, RpcError,
    Seq, SessionId, codes,
};
use serde_json::{Value, json};

use super::*;

#[derive(Default)]
struct Server {
    journals: BTreeMap<String, Vec<EventRecord>>,
    parents: BTreeMap<String, String>,
    /// Every call, as method and params.
    calls: Vec<(String, Value)>,
}

#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Server>>);

impl Server {
    fn push(&mut self, session: &str, event: Event) -> Seq {
        let journal = self.journals.entry(session.to_owned()).or_default();
        let seq = Seq(journal.len() as u64 + 1);
        journal.push(EventRecord {
            session: SessionId(session.to_owned()),
            seq,
            at_ms: 0,
            kind: event.kind().to_owned(),
            data: event.encode(),
            source: None,
        });
        seq
    }

    fn accept(&mut self, session: &str, method: &str) -> CommandSeq {
        let next = self.journals.get(session).map_or(0, |j| {
            j.iter().filter(|r| r.kind == "command.accepted").count() as u64
        }) + 1;
        self.push(
            session,
            Event::CommandAccepted(CommandAccepted {
                command_seq: CommandSeq(next),
                key: Key("k".into()),
                method: method.to_owned(),
                by: ClientId("parent".into()),
                text: None,
                command: None,
            }),
        );
        CommandSeq(next)
    }
}

fn parse<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    serde_json::from_value(params).map_err(|e| RpcError::new(codes::INVALID_PARAMS, e.to_string()))
}

fn json_of<T: serde::Serialize>(value: T) -> Result<Value, RpcError> {
    serde_json::to_value(value).map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))
}

impl Channel for Fake {
    fn call_raw(&self, method: &str, params: Value) -> Result<Value, RpcError> {
        let mut server = self
            .0
            .lock()
            .map_err(|_| RpcError::new(codes::INTERNAL, "poisoned"))?;
        server.calls.push((method.to_owned(), params.clone()));
        match method {
            m if m == methods::Start::NAME => {
                let start: methods::Start = parse(params)?;
                let id = format!("s-{}", server.parents.len() + 1);
                server.parents.insert(id.clone(), "parent".into());
                server.push(
                    &id,
                    Event::SessionStarted(SessionStarted {
                        cwd: start.cwd,
                        name: start.name,
                        host: None,
                        parent: Some(SessionId("parent".into())),
                        build: BuildId("b".into()),
                        grant: start.grant,
                    }),
                );
                let command = server.accept(&id, methods::Start::NAME);
                json_of(methods::Started {
                    session_id: SessionId(id),
                    seq: Seq(1),
                    command_seq: Some(command),
                })
            }
            m if m == methods::Send::NAME => {
                let send: methods::Send = parse(params)?;
                let command = server.accept(&send.session_id.0, methods::Send::NAME);
                json_of(methods::Sent {
                    command_seq: command,
                    disposition: methods::Disposition::Steered,
                })
            }
            m if m == methods::Answer::NAME => json_of(methods::Resolved { resolved: true }),
            m if m == methods::Stop::NAME => {
                let stop: methods::Stop = parse(params)?;
                json_of(methods::Stopping {
                    stopping: vec![stop.session_id],
                })
            }
            m if m == methods::List::NAME => {
                let infos: Vec<SessionInfo> = server
                    .parents
                    .iter()
                    .map(|(id, parent)| SessionInfo {
                        session_id: SessionId(id.clone()),
                        name: None,
                        cwd: "/w".into(),
                        host: None,
                        state: rig_protocol::event::State::Running,
                        parent: Some(SessionId(parent.clone())),
                        children: Vec::new(),
                        last_seq: Seq(0),
                        usage: Default::default(),
                        worker: true,
                    })
                    .collect();
                json_of(infos)
            }
            other => Err(RpcError::new(codes::METHOD_NOT_FOUND, other)),
        }
    }

    fn subscribe(&self, session: &SessionId, after: Seq) -> Result<Box<dyn Feed>, RpcError> {
        Ok(Box::new(FakeFeed {
            server: self.clone(),
            session: session.0.clone(),
            next: after.0 as usize,
        }))
    }
}

struct FakeFeed {
    server: Fake,
    session: String,
    next: usize,
}

impl Feed for FakeFeed {
    fn try_next(&mut self) -> Option<Result<EventRecord, RpcError>> {
        let server = self.server.0.lock().ok()?;
        let record = server.journals.get(&self.session)?.get(self.next)?.clone();
        self.next += 1;
        Some(Ok(record))
    }
}

struct Test {
    app: App,
    wakes: std::sync::mpsc::Receiver<()>,
    agent: Entity,
    fake: Fake,
    calls: u32,
}

impl Test {
    fn new() -> Test {
        let store = MemoryStore::default();
        let (mut app, wakes) = app_on(&store);
        let fake = Fake::default();
        app.init_resource::<ToolOutputs>()
            .add_observer(log_tool_outputs)
            .add_plugins((
                AgentPlugin,
                SessionLogPlugin,
                rig_harness::clock::ClockPlugin,
                AgentsPlugin,
            ))
            .insert_resource(link(
                Arc::new(fake.clone()),
                SessionId("parent".into()),
                "/work".into(),
                None,
            ));
        app.update();
        let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
        Test {
            app,
            wakes,
            agent,
            fake,
            calls: 0,
        }
    }

    /// Starts a call of `tool`; its id, for [`Test::output`].
    fn call(&mut self, tool: &str, args: Value) -> String {
        self.calls += 1;
        let id = format!("c{}", self.calls);
        start_call(self.app.world_mut(), self.agent, (tool, &id), args);
        self.app.update();
        id
    }

    fn output(&self, id: &str) -> Option<String> {
        self.app
            .world()
            .resource::<ToolOutputs>()
            .0
            .get(id)
            .cloned()
    }

    /// Runs until call `id` answered, and returns its text.
    fn answered(&mut self, id: &str) -> String {
        let id = id.to_owned();
        run_until(&mut self.app, &self.wakes, |w| {
            w.resource::<ToolOutputs>().0.contains_key(&id)
        });
        self.output(&id).unwrap_or_default()
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
        }
    }

    fn child(&self, n: usize) -> String {
        format!("s-{n}")
    }

    fn with(&self, f: impl FnOnce(&mut Server)) {
        if let Ok(mut server) = self.fake.0.lock() {
            f(&mut server);
        }
    }

    fn ends(&self, session: &str, through: u64, answer: &str) {
        self.with(|s| {
            s.push(
                session,
                Event::TurnEnded(TurnEnded {
                    agent_id: AgentId("a".into()),
                    turn_id: "t".into(),
                    status: TurnStatus::Answered,
                    answer: Some(answer.into()),
                    error: None,
                    through: CommandSeq(through),
                }),
            );
        });
    }
}

#[test]
fn a_parent_steers_and_waits_on_three_helpers_and_gets_each_report_once() {
    let mut t = Test::new();
    for name in ["a", "b", "c"] {
        let started = t.call(
            START_AGENT,
            json!({ "task": format!("do {name}"), "name": name }),
        );
        let text = t.answered(&started);
        assert!(text.contains("Started helper"), "{text}");
    }
    // Each starts in the parent's folder, with what it was handed.
    let starts: Vec<Value> = t
        .fake
        .0
        .lock()
        .map(|s| {
            s.calls
                .iter()
                .filter(|(m, _)| m == methods::Start::NAME)
                .map(|(_, p)| p.clone())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(starts.len(), 3);
    assert!(
        starts.iter().all(|p| p.get("cwd") == Some(&json!("/work"))
            && p.get("prompt").is_some_and(Value::is_string))
    );

    // A correction to b, mid-task.
    let steer = t.call(
        MESSAGE_AGENT,
        json!({ "id": "b", "text": "use the other parser" }),
    );
    assert!(t.answered(&steer).contains("before its next model call"));

    let wait = t.call(WAIT_AGENT, json!({ "ids": ["a", "b", "c"] }));
    // Progress, and b's end of the turn before it read the correction,
    // never settle the wait.
    t.with(|s| {
        s.push(
            "s-1",
            Event::Message(Message {
                agent_id: AgentId("a".into()),
                message_id: "m".into(),
                role: "assistant".into(),
                text: "halfway there".into(),
            }),
        );
    });
    t.ends(&t.child(2), 1, "b without the correction");
    t.frames(5);
    assert_eq!(t.output(&wait), None, "progress settled the wait");

    t.ends(&t.child(1), 1, "a's report");
    let first = t.answered(&wait);
    assert!(
        first.starts_with("`a` (s-1) answered:\na's report"),
        "{first}"
    );

    t.ends(&t.child(2), 2, "b's report");
    let second = t.call(WAIT_AGENT, json!({ "ids": ["a", "b", "c"] }));
    let second = t.answered(&second);
    assert!(
        second.contains("`b` (s-2) answered:\nb's report"),
        "{second}"
    );

    t.ends(&t.child(3), 1, "c's report");
    let third = t.call(WAIT_AGENT, json!({ "ids": ["s-1", "s-2", "s-3"] }));
    assert!(t.answered(&third).contains("c's report"));

    // Every report was given once: a wait now returns at once.
    let again = t.call(WAIT_AGENT, json!({ "ids": ["a", "b", "c"] }));
    let again = t.answered(&again);
    assert!(again.starts_with("Nothing to wait for"), "{again}");
}

#[test]
fn a_helpers_ask_settles_a_wait_and_is_answered_by_its_option() {
    let mut t = Test::new();
    let started = t.call(
        START_AGENT,
        json!({ "task": "run the tests", "name": "tests" }),
    );
    t.answered(&started);
    let wait = t.call(WAIT_AGENT, json!({ "ids": ["tests"] }));
    t.with(|s| {
        s.push(
            "s-1",
            Event::AskOpened(AskOpened {
                ask_id: AskId("a-1".into()),
                agent_id: AgentId("x".into()),
                agent_name: "tests".into(),
                kind: AskKind::Approval,
                title: "Run `cargo test`?".into(),
                body: String::new(),
                options: vec![
                    AskOption {
                        option_id: "yes".into(),
                        label: "Allow".into(),
                        kind: "allow_once".into(),
                    },
                    AskOption {
                        option_id: "no".into(),
                        label: "Deny".into(),
                        kind: "reject_once".into(),
                    },
                ],
                deadline_ms: 0,
                answerers: Answerers::Parent,
            }),
        );
    });
    let asked = t.answered(&wait);
    assert!(asked.contains("needs approval (ask `a-1`)"), "{asked}");

    let answer = t.call(
        ANSWER_AGENT,
        json!({ "id": "tests", "ask": "a-1", "allow": true }),
    );
    assert!(t.answered(&answer).starts_with("Answered"));
    let sent = t.fake.0.lock().ok().and_then(|s| {
        s.calls
            .iter()
            .find(|(m, _)| m == methods::Answer::NAME)
            .map(|(_, p)| p.clone())
    });
    assert_eq!(
        sent.map(|p| p.get("optionId").cloned()),
        Some(Some(json!("yes")))
    );
}

#[test]
fn a_start_before_the_link_is_made_waits_for_it() {
    let mut t = Test::new();
    run_until(&mut t.app, &t.wakes, |w| {
        w.iter_entities().all(|e| !e.contains::<super::Recovery>())
    });
    // A fresh worker: its link to rigd is still being made.
    t.app.world_mut().remove_resource::<AgentsLink>();
    t.app.world_mut().insert_resource(LinkPending);
    let started = t.call(START_AGENT, json!({ "task": "early", "name": "early" }));
    t.frames(3);
    assert_eq!(
        t.output(&started),
        None,
        "answered before the link: {:?}",
        t.output(&started)
    );
    t.app.world_mut().remove_resource::<LinkPending>();
    t.app.world_mut().insert_resource(link(
        Arc::new(t.fake.clone()),
        SessionId("parent".into()),
        "/work".into(),
        None,
    ));
    let text = t.answered(&started);
    assert!(text.contains("Started helper `early`"), "{text}");
}

#[test]
fn a_tool_on_a_name_that_is_no_helper_says_which_there_are() {
    let mut t = Test::new();
    let started = t.call(START_AGENT, json!({ "task": "x", "name": "one" }));
    t.answered(&started);
    let wait = t.call(WAIT_AGENT, json!({ "ids": ["two"] }));
    let refused = t.answered(&wait);
    assert!(
        refused.contains("`two` is not one of your helpers") && refused.contains("`one` (s-1)"),
        "{refused}"
    );
}

#[test]
fn the_fux_view_is_one_argument_per_word() {
    let args = view::fux_args("/bin/rig", "fix; rm -rf ~", &SessionId("s-1".into()));
    assert_eq!(
        args,
        [
            "opts",
            "str",
            "fix; rm -rf ~",
            "name",
            "str",
            "/bin/rig",
            "program",
            "str",
            "attach",
            "arg",
            "str",
            "s-1",
            "arg",
            "run"
        ]
    );
}
