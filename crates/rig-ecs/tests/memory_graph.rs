//! Memory is the graph; retrieval attaches (CONTRACT §11–§12): the
//! conversation is loaded before the first turn and appended at the
//! settle, retrieval runs before every fold, and the required row names
//! both.
//!
//! | claim | test |
//! |---|---|
//! | an agent that remembers loads before its first turn; the loaded messages are `Remembered` utterances before the prompt; the settle appends what the run said, in order | `memory_loads_before_the_first_turn_and_appends_at_the_settle` |
//! | a run given history loads nothing and appends nothing; the row still names the memory | `memory_is_bypassed_by_history` |
//! | a failed load fails the run `Memory`, with no completion | `memory_a_failed_load_fails_the_run` |
//! | a second run on the agent loads what the first appended | `memory_a_second_run_loads_what_the_first_appended` |
//! | retrieval: the query is the prompt, results are documents after the static ones; retrieved tools are advertised first and a `Retrievable` grant never otherwise; a document entity is reused by id | `memory_retrieval_attaches_documents_and_tools` |

use crate::run_support;

use rig_core::serve::Dispatch;
use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use rig_cassette::ecs::identity::required_row;
use rig_core::{
    effect::{
        EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, MemoryOp,
        MemoryOutcome, Outcome, RetrieveQuery, RetrievedDocuments,
    },
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message},
    serve::Serve,
};
use rig_ecs::{
    agent::{
        Context, Conversation, DocumentId, DocumentText, Failed, Failure, Grant, MessageParts,
        Remembered, Remembers, Retrievable, Retrieval, RetrievalKind, Retrieves, RunResult,
        Settled,
    },
    bus::{EffectOutcome, PendingEffect},
    systems::RunCommands,
};
use run_support::*;

const MODEL: &str = "t/model:default";
const MEMORY: &str = "t/memory";
const INDEX: &str = "t/retrieve:context#0";
const TOOLS: &str = "t/retrieve:tools#0";
const ADD: &str = "t/tool:add#0";
const SUB: &str = "t/tool:subtract#1";

/// A store: one conversation, every op logged.
struct Store {
    messages: Arc<Mutex<Vec<Message>>>,
    ops: Arc<Mutex<Vec<String>>>,
    refuse_load: bool,
}

impl Serve for Store {
    type Family = rig_core::effect::family::Memory;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MEMORY),
            family: FamilyDescriptor::Memory {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let EffectKind::Memory { op } = kind else {
            return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                "not memory",
            )));
        };
        let outcome = match op {
            MemoryOp::Load { conversation } => {
                self.ops
                    .lock()
                    .unwrap()
                    .push(format!("load:{conversation}"));
                if self.refuse_load {
                    Err(ErrorReport::new(ErrorKind::MemoryBackend, "refused"))
                } else {
                    Ok(MemoryOutcome::Loaded {
                        messages: self.messages.lock().unwrap().clone(),
                    })
                }
            }
            MemoryOp::Append {
                conversation,
                messages,
            } => {
                self.ops
                    .lock()
                    .unwrap()
                    .push(format!("append:{conversation}:{}", messages.len()));
                self.messages.lock().unwrap().extend(messages);
                Ok(MemoryOutcome::Appended)
            }
            MemoryOp::Clear { conversation } => {
                self.ops
                    .lock()
                    .unwrap()
                    .push(format!("clear:{conversation}"));
                self.messages.lock().unwrap().clear();
                Ok(MemoryOutcome::Cleared)
            }
        };
        rig_core::serve::Reply::Outcome(outcome.map(Outcome::Memory))
    }
}

/// An index answering every query with fixed results, logging the queries.
struct Index {
    key: &'static str,
    queries: Arc<Mutex<Vec<(String, u64)>>>,
    documents: Vec<(f64, String, serde_json::Value)>,
    ids: Vec<(f64, String)>,
}

impl Serve for Index {
    type Family = rig_core::effect::family::Retrieve;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.key),
            family: FamilyDescriptor::Retrieve {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let EffectKind::Retrieve { query } = kind else {
            return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                "not a retrieval",
            )));
        };
        let answer = match query {
            RetrieveQuery::TopN { req } => {
                self.queries
                    .lock()
                    .unwrap()
                    .push((req.query().to_owned(), req.samples()));
                RetrievedDocuments::Scored(self.documents.clone())
            }
            RetrieveQuery::TopNIds { req } => {
                self.queries
                    .lock()
                    .unwrap()
                    .push((req.query().to_owned(), req.samples()));
                RetrievedDocuments::Ids(self.ids.clone())
            }
        };
        rig_core::serve::Reply::Outcome(Ok(Outcome::Documents(answer)))
    }
}

type Shared<T> = Arc<Mutex<Vec<T>>>;

fn store(app: &mut bevy_app::App, refuse_load: bool) -> (Entity, Shared<Message>, Shared<String>) {
    let messages = Arc::new(Mutex::new(Vec::new()));
    let ops = Arc::new(Mutex::new(Vec::new()));
    let entity = register(
        app,
        MEMORY,
        Store {
            messages: Arc::clone(&messages),
            ops: Arc::clone(&ops),
            refuse_load,
        },
    );
    (entity, messages, ops)
}

/// Tick until `run` ended and nothing is pending without an outcome.
fn ran(app: &mut bevy_app::App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        (world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some())
            && world
                .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
                .iter(world)
                .count()
                == 0
    });
}

fn utterances(world: &mut World, run: Entity) -> Vec<(bool, MessageParts)> {
    utterances_of(world, run)
        .into_iter()
        .map(|entity| {
            (
                world.get::<Remembered>(entity).is_some(),
                rig_ecs::agent::content::parts::read_message(world, entity)
                    .expect("valid memory graph"),
            )
        })
        .collect()
}

#[test]
fn memory_loads_before_the_first_turn_and_appends_at_the_settle() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "Ada");
    let (memory, messages, ops) = store(&mut app, false);
    messages.lock().unwrap().extend([
        Message::user("My name is Ada."),
        Message::assistant("Noted."),
    ]);
    app.world_mut()
        .entity_mut(agent)
        .insert((Remembers(memory), Conversation("c1".to_owned())));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "What is my name?", false, None);
    ran(&mut app, run, "the run");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("Ada")
    );
    // The model saw the loaded history, then the prompt.
    let requests = requests.lock().unwrap();
    assert_eq!(
        texts(&requests[0]),
        vec![
            "system:You are terse.",
            "user:My name is Ada.",
            "assistant:Noted.",
            "user:What is my name?"
        ]
    );
    // The graph: two remembered utterances before the prompt and the answer.
    let graph = utterances(app.world_mut(), run);
    assert_eq!(
        graph
            .iter()
            .map(|(remembered, _)| *remembered)
            .collect::<Vec<_>>(),
        vec![true, true, false, false]
    );
    // The store: a load, then an append of what the run said (two).
    assert_eq!(&*ops.lock().unwrap(), &["load:c1", "append:c1:2"]);
    assert_eq!(messages.lock().unwrap().len(), 4);
    // The row names the memory.
    assert_eq!(
        required_row(app.world_mut(), agent).get(&HandlerKey::from(MEMORY)),
        Some(&EffectFamily::Memory)
    );
}

#[test]
fn memory_is_bypassed_by_history() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "ok");
    let (memory, _, ops) = store(&mut app, false);
    app.world_mut()
        .entity_mut(agent)
        .insert((Remembers(memory), Conversation("c1".to_owned())));
    let history = vec![MessageParts::from_message(&Message::user("earlier")).unwrap()];
    let run = app
        .world_mut()
        .spawn_run(agent, &history, "now", false, None);
    ran(&mut app, run, "the run");
    assert!(ops.lock().unwrap().is_empty(), "{:?}", ops.lock().unwrap());
    assert_eq!(
        texts(&requests.lock().unwrap()[0]),
        vec!["system:You are terse.", "user:earlier", "user:now"]
    );
    assert_eq!(
        required_row(app.world_mut(), agent).get(&HandlerKey::from(MEMORY)),
        Some(&EffectFamily::Memory)
    );
}

#[test]
fn memory_a_failed_load_fails_the_run() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "never");
    let (memory, _, ops) = store(&mut app, true);
    app.world_mut()
        .entity_mut(agent)
        .insert((Remembers(memory), Conversation("c1".to_owned())));
    let run = app.world_mut().spawn_run(agent, &[], "go", false, None);
    ran(&mut app, run, "the run");
    match app.world().get::<Failed>(run) {
        Some(Failed(Failure::Memory(report))) => assert_eq!(report.kind, ErrorKind::MemoryBackend),
        other => panic!("the run fails at the load: {other:?}"),
    }
    assert!(requests.lock().unwrap().is_empty(), "no completion");
    assert_eq!(&*ops.lock().unwrap(), &["load:c1"]);
}

#[test]
fn memory_a_second_run_loads_what_the_first_appended() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "ok");
    let (memory, _, ops) = store(&mut app, false);
    app.world_mut()
        .entity_mut(agent)
        .insert((Remembers(memory), Conversation("c1".to_owned())));
    let first = app.world_mut().spawn_run(agent, &[], "one", false, None);
    ran(&mut app, first, "the first run");
    let second = app.world_mut().spawn_run(agent, &[], "two", false, None);
    ran(&mut app, second, "the second run");
    assert_eq!(
        &*ops.lock().unwrap(),
        &["load:c1", "append:c1:2", "load:c1", "append:c1:2"]
    );
    assert_eq!(
        texts(&requests.lock().unwrap()[1]),
        vec![
            "system:You are terse.",
            "user:one",
            "assistant:ok",
            "user:two"
        ]
    );
    let graph = utterances(app.world_mut(), second);
    assert_eq!(
        graph
            .iter()
            .map(|(remembered, _)| *remembered)
            .collect::<Vec<_>>(),
        vec![true, true, false, false]
    );
}

#[test]
fn memory_retrieval_attaches_documents_and_tools() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![
            vec![call("c1", "subtract", serde_json::json!({"x": 3, "y": 1}))],
            vec![AssistantContent::text("2")],
        ],
    );
    let queries = Arc::new(Mutex::new(Vec::new()));
    let index = register(
        &mut app,
        INDEX,
        Index {
            key: INDEX,
            queries: Arc::clone(&queries),
            documents: vec![
                (0.9, "d1".to_owned(), serde_json::json!("a glarb")),
                (0.5, "d2".to_owned(), serde_json::json!({"k": 1})),
            ],
            ids: Vec::new(),
        },
    );
    let tool_index = register(
        &mut app,
        TOOLS,
        Index {
            key: TOOLS,
            queries: Arc::clone(&queries),
            documents: Vec::new(),
            ids: vec![(0.7, "subtract".to_owned())],
        },
    );
    let add = register(&mut app, ADD, Adder::new(ADD));
    let subtract = register(&mut app, SUB, Subtractor);
    let world = app.world_mut();
    world.entity_mut(agent).insert(rig_ecs::agent::MaxTurns(2));
    let static_document = world
        .spawn((
            DocumentId("s0".to_owned()),
            DocumentText("static".to_owned()),
        ))
        .id();
    world.spawn((Context(static_document), ChildOf(agent)));
    world.spawn((Grant(add), ChildOf(agent)));
    world.spawn((Grant(subtract), Retrievable, ChildOf(agent)));
    world.spawn((
        Retrieves(index),
        Retrieval {
            samples: 2,
            what: RetrievalKind::Documents,
        },
        ChildOf(agent),
    ));
    world.spawn((
        Retrieves(tool_index),
        Retrieval {
            samples: 1,
            what: RetrievalKind::Tools,
        },
        ChildOf(agent),
    ));
    let run = world.spawn_run(agent, &[], "subtract one from three", false, None);
    ran(&mut app, run, "the run");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("2")
    );
    // Both indexes, before every turn, with the prompt and the samples: the
    // two effects of a turn are dispatched together, so within a turn the
    // order is the pool's — sorted per turn.
    let queries = queries.lock().unwrap();
    let mut per_turn: Vec<Vec<(String, u64)>> = queries.chunks(2).map(<[_]>::to_vec).collect();
    for turn in &mut per_turn {
        turn.sort();
    }
    let both = vec![
        ("subtract one from three".to_owned(), 1),
        ("subtract one from three".to_owned(), 2),
    ];
    assert_eq!(per_turn, vec![both.clone(), both]);
    let requests = requests.lock().unwrap();
    for request in requests.iter() {
        // The static document, then the results in order; the string value
        // keeps its quotes.
        assert_eq!(
            request
                .documents
                .iter()
                .map(|d| (d.id.as_str(), d.text.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("s0", "static"),
                ("d1", "\"a glarb\""),
                ("d2", "{\n  \"k\": 1\n}")
            ]
        );
        // The retrieved tool first, then the static grant.
        assert_eq!(
            request
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec!["subtract", "add"]
        );
    }
    // One document entity per id, reused on the second turn.
    let ids: Vec<String> = app
        .world_mut()
        .query::<&DocumentId>()
        .iter(app.world())
        .map(|id| id.0.clone())
        .collect();
    assert_eq!(ids.len(), 3, "{ids:?}");
    // The row: the indexes as `retrieve`, the retrievable tool as a tool.
    let row = required_row(app.world_mut(), agent);
    assert_eq!(
        row.get(&HandlerKey::from(INDEX)),
        Some(&EffectFamily::Retrieve)
    );
    assert_eq!(
        row.get(&HandlerKey::from(TOOLS)),
        Some(&EffectFamily::Retrieve)
    );
    assert_eq!(row.get(&HandlerKey::from(SUB)), Some(&EffectFamily::Tool));
}

/// A tool that subtracts `y` from `x`.
struct Subtractor;

impl Serve for Subtractor {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(SUB),
            family: FamilyDescriptor::Tool {
                name: "subtract".to_owned(),
                description: "subtracts y from x".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let EffectKind::ToolCall { args, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                "not a call",
            )));
        };
        let args: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
        let value = args["x"].as_i64().unwrap_or(0) - args["y"].as_i64().unwrap_or(0);
        rig_core::serve::Reply::Outcome(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::json(
                serde_json::json!(value),
            )),
        }))
    }
}

#[test]
fn unavailable_retrieval_preserves_static_context_and_tool_grants() {
    let mut app = app();
    let (agent, requests) = capturing_agent(&mut app, MODEL, MODEL, "ok");
    let add = register(&mut app, ADD, Adder::new(ADD));
    let subtract = register(&mut app, SUB, Subtractor);
    let unavailable = app.world_mut().spawn_empty().id();
    let world = app.world_mut();
    let document = world
        .spawn((
            DocumentId("static-id".into()),
            DocumentText("static context".into()),
        ))
        .id();
    world.spawn((Context(document), ChildOf(agent)));
    world.spawn((Grant(add), ChildOf(agent)));
    world.spawn((Grant(subtract), Retrievable, ChildOf(agent)));
    world.spawn((
        Retrieves(unavailable),
        Retrieval {
            samples: 1,
            what: RetrievalKind::Documents,
        },
        ChildOf(agent),
    ));
    let run = world.spawn_run(agent, &[], "hello", false, None);
    ran(&mut app, run, "run without an available retrieval handler");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .documents
            .iter()
            .map(|document| (document.id.as_str(), document.text.as_str()))
            .collect::<Vec<_>>(),
        vec![("static-id", "static context")]
    );
    assert_eq!(
        requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["add"]
    );
}
