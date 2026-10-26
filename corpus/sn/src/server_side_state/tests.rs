use std::collections::{HashMap, VecDeque};
use std::fs;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use corpus_base::session_replay::{
    Replay, ReplayFlags, ReplaySource, ReplayStatus, SessionReplayPlugin,
};
use rig_basics::subagents::SubagentsPlugin;
use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
use rig_core::completion::{ModelRef, Usage};
use rig_core::effect::{FamilyDescriptor, model_key};
use rig_core::error::ErrorKind;
use rig_core::message::{Origin as Made, Reasoning, ToolName};
use rig_core::providers::registry::ConnectError;
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// What the subagent is asked; the server answers it apart from the script.
const SUB_PROMPT: &str = "Count the chairs.";

/// One request the server got.
#[derive(Clone, Debug)]
struct Got {
    previous: Option<String>,
    items: Vec<Message>,
    answered: bool,
}

/// The server's memory: each response's conversation by id, what it got,
/// and the replies it gives in order.
#[derive(Default)]
struct Books {
    next: u32,
    kept: HashMap<String, Vec<Message>>,
    got: Vec<Got>,
    replies: VecDeque<Vec<AssistantContent>>,
}

/// A scripted stateful server: it assigns ids, remembers conversations by
/// id, and records what each request contained.
#[derive(Clone, Default)]
struct Server(Arc<Mutex<Books>>);

impl Server {
    fn books(&self) -> MutexGuard<'_, Books> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn script(&self, replies: impl IntoIterator<Item = Vec<AssistantContent>>) {
        self.books().replies.extend(replies);
    }

    fn got(&self) -> Vec<Got> {
        self.books().got.clone()
    }

    /// Answers a request as the server would.
    fn answer(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<CompletionResponse, ErrorReport> {
        let mut books = self.books();
        let previous = previous_of(request).map(str::to_owned);
        let items: Vec<Message> = body(request).into_iter().cloned().collect();
        let known = match &previous {
            None => Some(Vec::new()),
            Some(id) => books.kept.get(id).cloned(),
        };
        books.got.push(Got {
            previous,
            items: items.clone(),
            answered: known.is_some(),
        });
        let Some(mut conversation) = known else {
            let mut report =
                ErrorReport::new(ErrorKind::ProviderResponse, "Previous response not found.");
            report.code = Some(NOT_FOUND.to_owned());
            return Err(report);
        };
        let asked_sub = items.len() == 1
            && items.iter().any(|message| {
                matches!(message, Message::User { content } if content.iter().any(|part| {
                    matches!(part, UserContent::Text(text) if text.text.contains(SUB_PROMPT))
                }))
            });
        let choice = if asked_sub {
            vec![AssistantContent::text("Four chairs.")]
        } else {
            books
                .replies
                .pop_front()
                .ok_or_else(|| ErrorReport::new(ErrorKind::Provider, "no reply scripted"))?
        };
        books.next += 1;
        let id = format!("resp_{}", books.next);
        let mut origin = Made::new("test.stateful", "stateful", model);
        origin.response_id = Some(id.clone());
        let response =
            CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
        conversation.extend(items);
        conversation.extend(response.message());
        books.kept.insert(id, conversation);
        Ok(response)
    }
}

/// A model of the server.
struct ServerModel {
    server: Server,
    model: String,
}

impl Serve for ServerModel {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: model_key(&self.model),
            family: FamilyDescriptor::Completion {
                model: ModelRef::from(self.model.as_str()),
                capabilities: Default::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _: Dispatch) -> Reply {
        let EffectKind::Completion { request, stream } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                "completions only",
            )));
        };
        let answer = Reply::Outcome(
            self.server
                .answer(&request, &self.model)
                .map(|response| Answer::Completion(response.into())),
        );
        // Streamed as asked, its failure as the stream's first item.
        if stream {
            Reply::Stream(answer.into_stream())
        } else {
            answer
        }
    }
}

/// The catalog with the server's models `stateful/a` and `stateful/b`.
fn models(server: &Server) -> (Models, Option<ProviderId>) {
    let mut connector = Connector::new(Catalog::builtin().clone());
    let Ok(vendor) = ProviderId::declare("stateful") else {
        return (Models(connector), None);
    };
    for name in ["a", "b"] {
        connector
            .catalog_mut()
            .insert(ModelSpec::new(vendor, name).with_tools(true));
    }
    let server = server.clone();
    connector.provide(
        vendor,
        move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
            let model = ServerModel {
                server: server.clone(),
                model: spec.reference(),
            };
            Ok::<_, ConnectError>(ErasedHandler::new(model))
        },
    );
    (Models(connector), Some(vendor))
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct LookArgs {}

/// `look` sees a room.
fn look(called: On<ToolCalled<LookArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("look"),
            ToolResult::success("a room".into()),
        );
    }
}

/// The style the system prompt asks for once the test changed it.
const STYLE: &str = "Be very brief.";

/// The plugin, the kernel, subagents and the session service on session
/// `id` of `home`, with a system prompt asking for `style`, after its
/// first frame.
fn start(home: &Home, id: &SessionId, server: &Server, style: &str) -> (App, Receiver<()>) {
    let (models, vendor) = models(server);
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(models)
        .insert_resource(StatefulVendors(vendor.into_iter().collect()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            SubagentsPlugin,
            ServerSideStatePlugin,
        ))
        .add_open_tool("look", "Look around.", ToolOptions::default(), look)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<EffectRecorded>(&mut app);
    app.world_mut()
        .spawn(PromptSection::new(50, "style", style));
    app.finish();
    app.update();
    (app, wakes)
}

/// The user types `text` to `agent`; runs until `turns` more turns ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    let ended = app.world().resource::<Ended>().0.len() + turns;
    let typed = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= ended
    });
}

/// Chooses `model` for `agent`.
fn choose(app: &mut App, agent: Entity, model: &str) {
    let model = model.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.update();
}

/// What the server got last.
fn last(server: &Server) -> Option<Got> {
    server.got().last().cloned()
}

/// A call of `tool` with `args`.
fn tool_call(id: &str, tool: &str, args: serde_json::Value) -> Option<AssistantContent> {
    let name = ToolName::new(tool).ok()?;
    Some(AssistantContent::tool_call(id, name, args))
}

/// The text of a user message, or the tool result it carries.
fn user_text(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };
    content.iter().find_map(|part| match part {
        UserContent::Text(text) => Some(text.text.clone()),
        UserContent::ToolResult(result) => Some(format!("result {}", result.name)),
        _ => None,
    })
}

/// What each item of a request was, as text.
fn texts(got: &Got) -> Vec<String> {
    got.items
        .iter()
        .map(|message| user_text(message).unwrap_or_else(|| "assistant".to_owned()))
        .collect()
}

/// The agent no other agent spawned.
fn parent(app: &mut App) -> Entity {
    let world = app.world_mut();
    let mut agents = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    agents.iter(world).next().unwrap_or(Entity::PLACEHOLDER)
}

/// The agent's chain.
fn chain(app: &App, agent: Entity) -> ServerChain {
    app.world()
        .get::<ServerChain>(agent)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn sends_only_the_new_part_while_the_server_holds_the_rest() {
    let home = scratch_home("sn-server-side-state");
    let id = SessionId::generate();
    let server = Server::default();
    let (mut app, wakes) = start(&home, &id, &server, "Be brief.");
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    choose(&mut app, agent, "stateful/a");

    // A turn with one tool call; the first reply reasons, opaquely.
    let encrypted = AssistantContent::Reasoning(Reasoning {
        redacted: true,
        ..Reasoning::new("")
    })
    .with_native(json!({ "type": "reasoning", "encrypted_content": "opaque" }));
    let mut first = vec![encrypted];
    first.extend(tool_call("c1", "look", json!({})));
    server.script([first, vec![AssistantContent::text("A room.")]]);
    say(&mut app, &wakes, agent, "Where am I?", 1);
    let got = server.got();
    assert_eq!(got.len(), 2, "{got:#?}");
    assert_eq!(
        got.first().map(|got| (got.previous.clone(), texts(got))),
        Some((None, vec!["Where am I?".to_owned()]))
    );
    assert_eq!(
        got.get(1).map(|got| (got.previous.clone(), texts(got))),
        Some((Some("resp_1".to_owned()), vec!["result look".to_owned()]))
    );

    // The next message goes alone, with the latest id.
    server.script([vec![AssistantContent::text("Still a room.")]]);
    say(&mut app, &wakes, agent, "And now?", 1);
    let got = last(&server).map(|got| (got.previous.clone(), texts(&got)));
    assert_eq!(
        got,
        Some((Some("resp_2".to_owned()), vec!["And now?".to_owned()]))
    );
    // The effect log keeps the logical request, the whole conversation.
    let records = &app.world().resource::<Collected<EffectRecorded>>().0;
    let logged = records
        .iter()
        .rev()
        .find_map(|EffectRecorded(record)| match &record.kind {
            EffectKind::Completion { request, .. } => Some(body(request).len()),
            _ => None,
        });
    assert_eq!(logged, Some(5));

    // The server forgets: the incremental attempt is refused, the full one
    // in the same call answers, with the opaque reasoning sent back, and
    // the user sees no error.
    server.books().kept.clear();
    server.script([vec![AssistantContent::text("Yes.")]]);
    say(&mut app, &wakes, agent, "Sure?", 1);
    let got = server.got();
    let tail: Vec<(Option<String>, bool)> = got
        .iter()
        .rev()
        .take(2)
        .rev()
        .map(|got| (got.previous.clone(), got.answered))
        .collect();
    assert_eq!(tail, [(Some("resp_3".to_owned()), false), (None, true)]);
    let full = got.last().map(|got| got.items.clone()).unwrap_or_default();
    assert_eq!(full.len(), 7, "{full:#?}");
    let opaque = full.iter().any(|message| {
        matches!(message, Message::Assistant(reply) if reply.content.iter().any(|block| {
            matches!(block, AssistantContent::Reasoning(_)) && block.native_item().is_some()
        }))
    });
    assert!(opaque, "the encrypted reasoning goes back in full");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.last(), Some(TurnOutcome::Answered(_))),
        "{ended:?}"
    );
    let errors = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        errors
            .iter()
            .all(|notice| notice.level != NoticeLevel::Error),
        "{errors:?}"
    );

    // A changed system prompt, then another model and back: in full.
    let world = app.world_mut();
    for mut section in world.query::<&mut PromptSection>().iter_mut(world) {
        if section.tag == "style" {
            section.text = STYLE.to_owned();
        }
    }
    for (model, why) in [
        (None, "system prompt changed"),
        (Some("stateful/b"), "model changed"),
        (Some("stateful/a"), "model changed"),
    ] {
        if let Some(model) = model {
            choose(&mut app, agent, model);
        }
        server.script([vec![AssistantContent::text("Fine.")]]);
        say(&mut app, &wakes, agent, "Go on.", 1);
        assert_eq!(last(&server).and_then(|got| got.previous), None, "{why}");
        assert_eq!(chain(&app, agent).last_full.as_deref(), Some(why));
    }
    server.script([vec![AssistantContent::text("Fine.")]]);
    say(&mut app, &wakes, agent, "Go on.", 1);
    assert!(last(&server).is_some_and(|got| got.previous.is_some()));

    // `/state` for the agent.
    app.world_mut().trigger(RunCommand::new(agent, "state"));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said = notices
        .last()
        .map(|notice| notice.text.clone())
        .unwrap_or_default();
    assert!(
        said.ends_with("incremental: 3, full: 5 (last: model changed)"),
        "{said}"
    );

    // A `task` subagent starts its own chain, with no parent id; the
    // parent's chain goes on.
    let task = json!({ "description": "Chairs", "prompt": SUB_PROMPT });
    server.script([
        tool_call("t1", "task", task).into_iter().collect(),
        vec![AssistantContent::text("Waiting.")],
        vec![AssistantContent::text("Four, then.")],
        vec![AssistantContent::text("Noted.")],
    ]);
    // The report comes as soon as it is sent: in the parent's turn when
    // that still runs, else in a turn of its own; four requests either way.
    let sent = server.got().len() + 4;
    let asked = Deliver::new(agent, "How many chairs?", DeliveryMode::Queue);
    app.world_mut().trigger(asked);
    run_until(&mut app, &wakes, |world| {
        let idle = world
            .try_query::<&ActiveTurn>()
            .is_none_or(|mut turns| turns.iter(world).next().is_none());
        idle && server.got().len() >= sent
    });
    let got = server.got();
    let sub = got
        .iter()
        .find(|got| texts(got).iter().any(|text| text.contains(SUB_PROMPT)));
    assert!(sub.is_some_and(|sub| sub.previous.is_none()), "{got:#?}");
    assert!(
        last(&server).is_some_and(|got| got.previous.is_some()),
        "{got:#?}"
    );

    let original = chain(&app, agent).link.map(|link| link.response_id);
    drop(app);

    // Quit and resume the original: it goes on incrementally from its id.
    let (mut app, wakes) = start(&home, &id, &server, STYLE);
    let agent = parent(&mut app);
    server.script([vec![AssistantContent::text("Back.")]]);
    say(&mut app, &wakes, agent, "Back again.", 1);
    let got = last(&server).map(|got| (got.previous.clone(), texts(&got)));
    assert!(original.is_some());
    assert_eq!(got, Some((original, vec!["Back again.".to_owned()])));
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}

/// The plugin, the kernel and the effect log on session `id` of `home`,
/// every agent on `stateful/a`, before its first frame.
fn logged(home: &Home, id: &SessionId, server: &Server) -> (App, Receiver<()>) {
    let (models, vendor) = models(server);
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(models)
        .insert_resource(StatefulVendors(vendor.into_iter().collect()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EffectLogPlugin,
            ServerSideStatePlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(|agent: On<Add<Agent>>, mut commands: Commands| {
            let model = "stateful/a".to_owned();
            commands.trigger(SetModel {
                entity: agent.entity,
                model,
            });
        });
    (app, wakes)
}

/// The session recorded in `dir`, read as the app's stores read it.
fn record_of(dir: &std::path::Path) -> std::io::Result<ReplaySource> {
    let store = StoreOpener::default().open(&SessionDir::at(dir))?;
    ReplaySource::from_store(&*store, ReplayFlags::default())
}

/// The replay answers outside the chain: it sees the whole request it
/// recorded, not the part the chain would upload, so a session whose
/// second request went incrementally replays without the server.
#[test]
fn a_chained_session_replays_from_its_record() {
    let home = scratch_home("sn-server-state-replay");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded, replayed) = (id("1-1"), id("1-2"));
    let server = Server::default();
    let (mut app, wakes) = logged(&home, &recorded, &server);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    server.script([
        vec![AssistantContent::text("A room.")],
        vec![AssistantContent::text("Still a room.")],
    ]);
    say(&mut app, &wakes, agent, "Where am I?", 1);
    say(&mut app, &wakes, agent, "And now?", 1);
    let previous = last(&server).and_then(|got| got.previous);
    assert_eq!(previous.as_deref(), Some("resp_1"), "{:#?}", server.got());
    // The effect log is written at the end of the frame.
    app.update();
    drop(app);

    let dir = home.session(&recorded).path().to_owned();
    let source = record_of(&dir);
    assert!(source.is_ok(), "{:?}", source.err());
    let idle = Server::default();
    let (mut app, wakes) = logged(&home, &replayed, &idle);
    if let Ok(source) = source {
        app.insert_resource(source);
    }
    app.add_plugins(SessionReplayPlugin);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        world.get_resource::<Replay>().is_some_and(|replay| {
            matches!(
                replay.status(),
                ReplayStatus::Finished | ReplayStatus::Stopped
            )
        })
    });
    let replay = app.world().get_resource::<Replay>();
    let outcome = replay.map(|replay| (replay.status(), replay.divergences(), replay.answered()));
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 0))));
    assert!(idle.got().is_empty(), "{:#?}", idle.got());
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ServerSideStatePlugin);
}
