use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    ToolOutputs, app_on, calls, log_tool_outputs, reply, run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

const TEAM: &str = r#"
[member.backend]
role = "Owns the server and its database schema."
model = "test/backend"
tools = ["write"]
scope = "server/**"

[member.frontend]
role = "Owns the web client."
model = "test/frontend"
"#;

/// When each member's model calls started and settled, in order.
#[derive(Resource, Default)]
struct Timeline(Vec<String>);

fn started(
    started: On<Add<Started>>,
    calls: Query<&ForAgent, With<ModelRequest>>,
    members: Query<&TeamMember>,
    mut timeline: ResMut<Timeline>,
) {
    if let Ok(ForAgent(agent)) = calls.get(started.entity)
        && let Ok(TeamMember(name)) = members.get(*agent)
    {
        timeline.0.push(format!("{name} started"));
    }
}

fn settled(
    settled: On<Add<Settled>>,
    calls: Query<&ForAgent, With<ModelRequest>>,
    members: Query<&TeamMember>,
    mut timeline: ResMut<Timeline>,
) {
    if let Ok(ForAgent(agent)) = calls.get(settled.entity)
        && let Ok(TeamMember(name)) = members.get(*agent)
    {
        timeline.0.push(format!("{name} settled"));
    }
}

/// The lead's model, the members' and the write tool, on `store`, in
/// `project`, after the first frame; and the lead, on its model, working
/// in `project`, as its members do (`ServicesPlugin` passes it on).
fn start(
    store: &MemoryStore,
    project: &Path,
    [lead, backend, frontend]: [&MockCompletionModel; 3],
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(ProjectRoot(project.to_owned()))
        .insert_resource(RigHome(Home::at(project.join("home"))))
        .insert_resource(scripted_models([
            ("test/lead", lead),
            ("test/backend", backend),
            ("test/frontend", frontend),
        ]))
        .add_plugins((ServicesPlugin, AgentPlugin, SessionLogPlugin))
        .add_plugins((WriteTool, StandingTeamPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Timeline>()
        .add_observer(started)
        .add_observer(settled);
    app.finish();
    app.update();
    let mut leads = app
        .world_mut()
        .query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let lead = leads
        .iter(app.world())
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    app.world_mut()
        .entity_mut(lead)
        .insert(WorkDir(project.to_owned()));
    let model = "test/lead".to_owned();
    app.world_mut().trigger(SetModel {
        entity: lead,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, lead)
}

/// Sends the lead `text` and runs until its model was asked `requests`
/// times in all and its turn ended.
fn ask_lead(
    (app, wakes, lead): (&mut App, &Receiver<()>, Entity),
    model: &MockCompletionModel,
    text: &str,
    requests: usize,
) {
    app.world_mut()
        .trigger(Deliver::new(lead, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        model.requests().len() >= requests && world.get::<ActiveTurn>(lead).is_none()
    });
}

/// Each request `model` was sent, as JSON text.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let text = |request| serde_json::to_string(request).unwrap_or_default();
    requests.iter().map(text).collect()
}

fn delegate(id: &str, member: &str, task: &str) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, DELEGATE, json!({ "member": member, "task": task }))
}

#[test]
fn members_work_side_by_side_keep_their_memory_and_their_scope_across_a_resume() {
    let project = scratch_home("standing-team-project").root().to_owned();
    std::fs::create_dir_all(&project).ok();
    std::fs::write(project.join("team.toml"), TEAM).ok();
    let lead = MockCompletionModel::from_stream_turns([
        vec![
            delegate("d1", "backend", "Add a last_login column"),
            delegate("d2", "frontend", "Show last_login on the profile"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Both done."),
        // Two to one member: one after the other, in order.
        vec![
            delegate("d3", "backend", "Follow-up: put it in web/x.js"),
            delegate("d4", "backend", "Then document the column"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Backend may not."),
    ]);
    let backend = MockCompletionModel::from_stream_turns([
        reply("Added users.last_login (timestamp)."),
        vec![
            MockStreamEvent::tool_call(
                "w0",
                "write",
                json!({ "path": "server/x.rs", "content": "x" }),
            ),
            MockStreamEvent::tool_call(
                "w1",
                "write",
                json!({ "path": "web/x.js", "content": "x" }),
            ),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("web/ is not mine."),
        reply("Documented last_login."),
    ]);
    let frontend = MockCompletionModel::from_stream_turns([reply("The profile shows last_login.")]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &project, [&lead, &backend, &frontend]);

    ask_lead(
        (&mut app, &wakes, agent),
        &lead,
        "Add last_login end to end.",
        2,
    );
    let timeline = app.world().resource::<Timeline>().0.clone();
    let first_settle = timeline.iter().position(|step| step.ends_with("settled"));
    let started_before = |name: &str| {
        let start = timeline
            .iter()
            .position(|step| *step == format!("{name} started"));
        start
            .zip(first_settle)
            .is_some_and(|(start, settle)| start < settle)
    };
    assert!(
        started_before("backend") && started_before("frontend"),
        "{timeline:?}"
    );
    let to_lead = sent(&lead);
    let second = to_lead.get(1).cloned().unwrap_or_default();
    assert!(second.contains("Added users.last_login"), "{second}");
    assert!(second.contains("The profile shows last_login."), "{second}");

    ask_lead(
        (&mut app, &wakes, agent),
        &lead,
        "Also put it in the client.",
        4,
    );
    let to_backend = sent(&backend);
    let follow_up = to_backend.get(1).cloned().unwrap_or_default();
    for remembered in [
        "Add a last_login column",
        "Added users.last_login",
        "Follow-up",
    ] {
        assert!(
            follow_up.contains(remembered),
            "{remembered} in {follow_up}"
        );
    }
    let then = to_backend.get(3).cloned().unwrap_or_default();
    assert!(
        then.contains("web/ is not mine.") && then.contains("Then document"),
        "{then}"
    );
    let to_lead = sent(&lead);
    let fourth = to_lead.get(3).cloned().unwrap_or_default();
    let (first, second) = (fourth.find("web/ is not mine."), fourth.find("Documented"));
    assert!(first.zip(second).is_some_and(|(a, b)| a < b), "{fourth}");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let write = outputs.get("w1").cloned().unwrap_or_default();
    assert!(
        write.contains("may only write under `server/**`"),
        "{write}"
    );
    assert!(!project.join("web/x.js").exists());
    let inside = outputs.get("w0").cloned().unwrap_or_default();
    assert!(project.join("server/x.rs").exists(), "{inside}");

    // A new app on the same session: the members come back with their
    // conversations, and the next assignment continues them.
    drop(app);
    let lead = MockCompletionModel::from_stream_turns([
        calls(
            "d4",
            DELEGATE,
            json!({ "member": "backend", "task": "Rename it to last_seen" }),
        ),
        reply("Renamed."),
    ]);
    let backend = MockCompletionModel::from_stream_turns([reply("Renamed to last_seen.")]);
    let frontend = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut resumed, wakes, agent) = start(&store, &project, [&lead, &backend, &frontend]);
    ask_lead(
        (&mut resumed, &wakes, agent),
        &lead,
        "Rename last_login.",
        2,
    );
    let to_backend = sent(&backend);
    let after_resume = to_backend.first().cloned().unwrap_or_default();
    for remembered in [
        "Add a last_login column",
        "Follow-up",
        "Rename it to last_seen",
    ] {
        assert!(
            after_resume.contains(remembered),
            "{remembered} in {after_resume}"
        );
    }
    let to_lead = sent(&lead);
    let last = to_lead.get(1).cloned().unwrap_or_default();
    assert!(last.contains("Renamed to last_seen."), "{last}");
}

#[test]
fn a_member_hired_in_a_limited_run_keeps_the_leads_tools() {
    let project = scratch_home("standing-team-limited").root().to_owned();
    std::fs::create_dir_all(&project).ok();
    std::fs::write(project.join("team.toml"), TEAM).ok();
    let lead = MockCompletionModel::from_stream_turns([
        vec![
            delegate("d1", "frontend", "Look at the client"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Done."),
    ]);
    let frontend = MockCompletionModel::from_stream_turns([reply("Looked.")]);
    let backend = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &project, [&lead, &backend, &frontend]);
    app.insert_resource(RunTools::new([DELEGATE]));
    ask_lead((&mut app, &wakes, agent), &lead, "Have a look.", 2);
    let world = app.world_mut();
    let mut members = world.query::<(&TeamMember, &ToolAccess)>();
    let hired = members
        .iter(world)
        .find(|(member, _)| member.0 == "frontend");
    // Out of this run only: a later run gives it the lead's `write`.
    assert!(hired.is_some_and(|(_, access)| access.keeps("write") && !access.allows("write")));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(StandingTeamPlugin);
}
