use bevy_ecs::prelude::*;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_ecs::session_log::ConversationChanged;
use rig_harness_test_support::{app_on, connect, reply, run_until};
use rig_memory::MemoryStore;
use rig_protocol::event::{self, Event};
use rig_protocol::{CommandSeq, EventRecord, Seq, SessionId};

use super::mirror::{self, Mirror};
use super::*;

/// What a session's journal would hold for the kernel app it records, in
/// the order the app made it: each turn started and ended, each saved
/// conversation change, as the worker exports them.
#[derive(Resource, Default)]
struct Journal(Vec<Event>);

fn record_started(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut journal: ResMut<Journal>,
) {
    let Some(id) = turns
        .get(added.entity)
        .ok()
        .and_then(|of| ids.get(of.0).ok())
    else {
        return;
    };
    journal.0.push(Event::TurnStarted(event::TurnStarted {
        agent_id: rig_protocol::AgentId(id.0.clone()),
        turn_id: added.entity.to_string(),
    }));
}

fn record_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    ids: Query<&AgentId>,
    mut journal: ResMut<Journal>,
) {
    let Ok((outcome, agent)) = turns.get(settled.entity) else {
        return;
    };
    let Ok(id) = ids.get(agent.0) else {
        return;
    };
    let status = match outcome {
        TurnOutcome::Answered(_) => event::TurnStatus::Answered,
        TurnOutcome::Failed(_) => event::TurnStatus::Failed,
        TurnOutcome::Stopped(_) => event::TurnStatus::Stopped,
    };
    journal.0.push(Event::TurnEnded(event::TurnEnded {
        agent_id: rig_protocol::AgentId(id.0.clone()),
        turn_id: settled.entity.to_string(),
        status,
        answer: None,
        error: None,
        through: CommandSeq(1),
    }));
}

fn record_changes(
    mut changes: MessageReader<ConversationChanged>,
    ids: Query<&AgentId>,
    mut journal: ResMut<Journal>,
    mut index: Local<u64>,
) {
    for changed in changes.read() {
        let Ok(id) = ids.get(changed.agent) else {
            continue;
        };
        *index += 1;
        journal
            .0
            .push(Event::ConversationChange(event::ConversationChange {
                agent_id: rig_protocol::AgentId(id.0.clone()),
                log: id.0.clone(),
                index: *index,
                record: serde_json::to_value(&changed.change).unwrap_or_default(),
            }));
    }
}

/// The journal's records, numbered from 1.
fn records(events: &[Event]) -> Vec<EventRecord> {
    let numbered = events.iter().zip(1..);
    numbered
        .map(|(event, seq)| EventRecord {
            session: SessionId("s-1".into()),
            seq: Seq(seq),
            at_ms: 0,
            source: None,
            kind: event.kind().to_owned(),
            data: event.encode(),
        })
        .collect()
}

/// A view's world: the mirror and the messages it writes, no kernel.
fn mirror_world() -> World {
    let mut app = App::new();
    app.add_plugins(MirrorPlugin);
    std::mem::take(app.world_mut())
}

fn conversation_of(world: &mut World, agent: Entity) -> serde_json::Value {
    let conversation = world.get::<Conversation>(agent);
    serde_json::to_value(conversation.map(|c| &c.0)).unwrap_or_default()
}

#[test]
fn the_mirror_holds_the_conversation_the_kernel_holds_and_the_turn_while_it_runs() {
    let model = MockCompletionModel::from_stream_turns([reply("hello there"), reply("bye")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins(AgentPlugin)
        .init_resource::<Journal>()
        .add_observer(record_started)
        .add_observer(record_ended)
        .add_systems(Last, record_changes.after(rig_ecs::WriteLogs));
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let agent = app.world_mut().spawn((Agent, connection)).id();
    for text in ["hi", "and again"] {
        let world = app.world_mut();
        let ended = world.resource::<Journal>().0.len();
        world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            let journal = &world.resource::<Journal>().0;
            journal
                .iter()
                .skip(ended)
                .any(|event| matches!(event, Event::TurnEnded(_)))
        });
        // The turn's last records are saved the frame it ends.
        app.update();
    }
    let id = app.world().get::<AgentId>(agent).cloned();
    let kernel = conversation_of(app.world_mut(), agent);
    let events = std::mem::take(&mut app.world_mut().resource_mut::<Journal>().0);
    let records = records(&events);

    let mut view = mirror_world();
    let started = records
        .iter()
        .position(|record| record.kind == "turn.ended");
    let (first, rest) = records.split_at(started.unwrap_or(records.len()));
    mirror::apply(&mut view, first.to_vec());
    let mirrored = id
        .as_ref()
        .and_then(|id| view.resource::<Mirror>().agent(&id.0));
    assert!(mirrored.is_some(), "no agent mirrored");
    let Some(mirrored) = mirrored else {
        return;
    };
    assert!(
        view.get::<ActiveTurn>(mirrored).is_some(),
        "the first turn runs"
    );
    // Applied again, as after a resubscribe: nothing changes twice.
    mirror::apply(&mut view, records.clone());
    mirror::apply(&mut view, rest.to_vec());
    assert!(
        view.get::<ActiveTurn>(mirrored).is_none(),
        "both turns ended"
    );
    assert_eq!(conversation_of(&mut view, mirrored), kernel);
}

#[test]
fn a_message_for_a_mirrored_agent_goes_to_the_session_and_quit_stays_in_the_view() {
    let mut app = App::new();
    app.add_plugins(MirrorPlugin);
    let (link, mut client, _inbound, outbound) = ViewLink::detached();
    client.session = Some(SessionId("s-1".into()));
    app.insert_resource(link).insert_resource(client);
    app.update();
    let record = EventRecord {
        session: SessionId("s-1".into()),
        seq: Seq(1),
        at_ms: 0,
        source: None,
        kind: "turn.started".into(),
        data: serde_json::json!({ "agentId": "a-1", "turnId": "t-1" }),
    };
    mirror::apply(app.world_mut(), [record]);
    let agent = app.world().resource::<Mirror>().agent("a-1");
    assert!(agent.is_some(), "no agent mirrored");
    let Some(agent) = agent else {
        return;
    };
    let world = app.world_mut();
    world.trigger(Deliver::new(agent, "fix it", DeliveryMode::Queue));
    world.trigger(StopTurn::new(agent));
    world.trigger(Deliver::new(agent, "/model gpt", DeliveryMode::Steer));
    world.trigger(Deliver::new(agent, "/quit", DeliveryMode::Steer));
    app.update();
    let sent: Vec<String> = outbound
        .try_iter()
        .map(|request| match request {
            Outbound::Send(send) => format!("{:?} {}", send.mode, send.text.unwrap_or_default()),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        sent,
        [
            "Some(Queue) fix it",
            "Stop(Turn)",
            "Some(Steer) /model gpt",
            "Detach"
        ]
    );
    let exits = app.world().resource::<Messages<AppExit>>();
    assert!(!exits.is_empty(), "/quit leaves the view");
}

#[test]
fn an_agents_status_items_reach_the_mirrored_agent_and_the_app() {
    use crate::status::{AppStatus, Side, StatusEvent, StatusItem, StatusItems, Tone};
    let mut items = StatusItems::default();
    items.show(StatusItem::new("model", Side::Left, 10, 9).says("gpt-5", Tone::Bold));
    let agent = StatusEvent {
        agent_id: Some("a-1".into()),
        turn: false,
        items: items.clone(),
    };
    let app = StatusEvent {
        agent_id: None,
        turn: false,
        items: items.clone(),
    };
    let records = [agent, app]
        .into_iter()
        .zip(1..)
        .map(|(event, seq)| EventRecord {
            session: SessionId("s-1".into()),
            seq: Seq(seq),
            at_ms: 0,
            source: None,
            kind: StatusEvent::KIND.into(),
            data: serde_json::to_value(&event).unwrap_or_default(),
        });
    let mut view = mirror_world();
    mirror::apply(&mut view, records);
    let mirrored = view.resource::<Mirror>().agent("a-1");
    let shown = mirrored.and_then(|agent| view.get::<StatusItems>(agent));
    assert_eq!(shown, Some(&items));
    assert_eq!(view.resource::<AppStatus>().0, items);
}
