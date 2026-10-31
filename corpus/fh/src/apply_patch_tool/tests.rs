use std::fs;
use std::path::Path;

use rig_basics::coding_tools::{EditTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// The names of the tools the model's `n`-th request offered.
fn offered(model: &MockCompletionModel, n: usize) -> Vec<String> {
    let requests = model.requests();
    let tools = requests.get(n).map(|request| request.tools.clone());
    let names = tools.unwrap_or_default().into_iter();
    names.map(|tool| tool.name.as_str().to_owned()).collect()
}

fn text(dir: &Path, file: &str) -> Option<String> {
    fs::read_to_string(dir.join(file)).ok()
}

const PATCH: &str = "*** Begin Patch
*** Add File: new.txt
+fresh
*** Update File: a.txt
@@
 one
-two
+TWO
 three
*** Delete File: old.txt
*** End Patch";

/// `b.txt`'s hunk applies; `a.txt`'s second hunk's context is not there.
const BROKEN: &str = "*** Begin Patch
*** Update File: b.txt
@@
-bee
+BEE
*** Update File: a.txt
@@
 one
-TWO
+2
@@
 missing line
-three
+3
*** End Patch";

/// Applies, but a gate refuses the `write` of `secret.txt`, and another
/// asks the user about `asked.txt`'s.
const GUARDED: &str = "*** Begin Patch
*** Update File: a.txt
@@
-TWO
+2
*** Add File: secret.txt
+key
*** Add File: asked.txt
+maybe
*** End Patch";

/// Applies, but `slow.txt`'s write waits a frame for an approval.
const SLOW: &str = "*** Begin Patch
*** Add File: slow.txt
+slow
*** Update File: c.txt
@@
-sea
+SEA
*** End Patch";

/// Adds a file and deletes another, from an agent that may not `delete`.
const FORBIDDEN: &str = "*** Begin Patch
*** Add File: d.txt
+dee
*** Delete File: b.txt
*** End Patch";

/// Whether a call writes a file named `name`, by its footprint.
fn writes(footprint: &Footprint, name: &str) -> bool {
    let Touches::Writes(Scope::Paths(paths)) = &footprint.0 else {
        return false;
    };
    paths.iter().any(|path| path.ends_with(name))
}

/// Another plugin's rule on any file tool: never `secret.txt`.
fn no_secrets(mut calls: Query<(&Footprint, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (footprint, mut decision) in &mut calls {
        if writes(footprint, "secret.txt") {
            decision.decide(Decider::plugin("test"), Outcome::refused("no secrets"));
        }
    }
}

/// On the hold of an approval of `slow.txt`'s write.
#[derive(Component)]
struct Approval;

/// On a call asked about.
#[derive(Component)]
struct Asked;

/// Another plugin's approval of `slow.txt`: held until the next frame;
/// and a question to the user about `asked.txt`, which nobody answers.
fn ask_for_slow(
    calls: Query<(Entity, &Footprint, &Decision), (ReadyToStart, ToolCalls, Without<Asked>)>,
    mut commands: Commands,
) {
    for (call, footprint, decision) in &calls {
        if !decision.is_decided() && writes(footprint, "slow.txt") {
            commands.entity(call).insert(Asked);
            commands.spawn((Approval, HoldOf(call)));
        }
        if !decision.is_decided() && writes(footprint, "asked.txt") {
            commands.entity(call).insert(Asked);
            commands.spawn((Ask::yes_no("Write asked.txt?"), HoldOf(call)));
        }
    }
}

/// Whether `c.txt`'s write had started when `slow.txt`'s was approved.
#[derive(Resource, Default)]
struct AtApproval(Option<bool>);

/// Approves `slow.txt`'s write, noting whether `c.txt`'s started first.
fn approve(
    approvals: Query<Entity, With<Approval>>,
    calls: Query<(&Footprint, Has<Started>), ToolCalls>,
    mut seen: ResMut<AtApproval>,
    mut commands: Commands,
) {
    for approval in &approvals {
        let started = calls
            .iter()
            .any(|(footprint, started)| started && writes(footprint, "c.txt"));
        seen.0 = Some(started);
        commands.entity(approval).despawn();
    }
}

#[test]
fn a_patch_model_gets_apply_patch_and_patches_apply_whole_or_not_at_all() {
    let home = scratch_home("apply-patch-tool");
    let dir = home.root().join("project");
    fs::create_dir_all(&dir).ok();
    fs::write(home.root().join("patch.toml"), "models = [\"gpt-5*\"]").ok();
    fs::write(dir.join("a.txt"), "one\ntwo\nthree\n").ok();
    fs::write(dir.join("old.txt"), "gone\n").ok();
    fs::write(dir.join("b.txt"), "bee\n").ok();
    fs::write(dir.join("c.txt"), "sea\n").ok();

    let gpt = MockCompletionModel::from_stream_turns([
        calls("p1", APPLY_PATCH, json!({ "input": PATCH })),
        calls("p2", APPLY_PATCH, json!({ "input": BROKEN })),
        calls("p3", APPLY_PATCH, json!({ "input": GUARDED })),
        calls("p4", APPLY_PATCH, json!({ "input": SLOW })),
        reply("Done."),
        calls("p5", APPLY_PATCH, json!({ "input": FORBIDDEN })),
        reply("Done."),
    ]);
    let claude = MockCompletionModel::from_stream_turns([reply("Hello.")]);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([
        ("test/gpt-5-test", &gpt),
        ("test/claude-test", &claude),
    ]))
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        EditTool,
        WriteTool,
        ApplyPatchPlugin,
    ))
    .add_tool_per_call(rig_tools::Delete::default(), ToolOptions::default())
    .add_systems(
        AgentPass,
        (no_secrets, ask_for_slow).chain().in_set(Stage::Gate),
    )
    .add_systems(Update, approve.before(AgentLoop))
    .init_resource::<AtApproval>()
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let world = app.world_mut();
    world.entity_mut(agent).insert(WorkDir(dir.clone()));
    let model = "test/gpt-5-test".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Patch it.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    // Once a patch was refused, nobody is asked about its other files.
    let world = app.world_mut();
    let mut open = world.query_filtered::<(), (With<Ask>, Without<Outcome>)>();
    assert_eq!(open.iter(world).count(), 0, "a question stays open");

    let first = offered(&gpt, 0);
    assert!(first.iter().any(|name| name == APPLY_PATCH), "{first:?}");
    assert!(
        !first.iter().any(|name| name == "edit" || name == "write"),
        "{first:?}"
    );

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let applied = outputs.get("p1").cloned().unwrap_or_default();
    for line in ["A new.txt", "M a.txt", "D old.txt"] {
        assert!(applied.contains(line), "{applied}");
    }
    assert_eq!(text(&dir, "new.txt").as_deref(), Some("fresh\n"));
    assert_eq!(text(&dir, "old.txt"), None);

    let refused = outputs.get("p2").cloned().unwrap_or_default();
    assert!(refused.contains("a.txt, hunk 2"), "{refused}");
    assert!(refused.contains("missing line"), "{refused}");
    assert_eq!(text(&dir, "a.txt").as_deref(), Some("one\nTWO\nthree\n"));
    assert_eq!(text(&dir, "b.txt").as_deref(), Some("bee\n"));

    let guarded = outputs.get("p3").cloned().unwrap_or_default();
    assert!(guarded.contains("no secrets"), "{guarded}");
    assert_eq!(text(&dir, "a.txt").as_deref(), Some("one\nTWO\nthree\n"));
    assert_eq!(text(&dir, "secret.txt"), None);

    let slow = outputs.get("p4").cloned().unwrap_or_default();
    assert!(
        slow.contains("A slow.txt") && slow.contains("M c.txt"),
        "{slow}"
    );
    assert_eq!(
        app.world().resource::<AtApproval>().0,
        Some(false),
        "c.txt's write waits for slow.txt's approval"
    );
    assert_eq!(text(&dir, "c.txt").as_deref(), Some("SEA\n"));

    let world = app.world_mut();
    let model = "test/claude-test".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Hi.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 1
    });
    let after = offered(&claude, 0);
    assert!(after.iter().any(|name| name == "edit"), "{after:?}");
    assert!(after.iter().any(|name| name == "write"), "{after:?}");
    assert!(!after.iter().any(|name| name == APPLY_PATCH), "{after:?}");

    // The kernel refuses the `delete` this agent may not make before the
    // gates, so the patch's `write` is refused with it.
    let world = app.world_mut();
    let tools = [APPLY_PATCH, "edit", "write"].map(str::to_owned);
    world
        .entity_mut(agent)
        .insert(ToolAccess::Only(tools.into()));
    let model = "test/gpt-5-test".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Patch again.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 2
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let forbidden = outputs.get("p5").cloned().unwrap_or_default();
    assert!(forbidden.contains("not applied"), "{forbidden}");
    assert!(forbidden.contains("delete b.txt"), "{forbidden}");
    assert_eq!(text(&dir, "d.txt"), None);
    assert_eq!(text(&dir, "b.txt").as_deref(), Some("bee\n"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ApplyPatchPlugin);
}

/// With the plugins it is often run with: early tool starts and permission
/// modes, which it orders itself against, and the other tools.
#[test]
fn stages_are_ordered_with_the_plugins_it_runs_with() {
    rig_defaults_test_support::assert_stages_ordered((
        ApplyPatchPlugin,
        corpus_shared::permission_modes::PermissionModesPlugin,
        crate::early_tool_start::EarlyToolStartPlugin,
        crate::transcript_search::TranscriptSearchPlugin,
        crate::encrypted_sessions::EncryptedSessionsPlugin,
        crate::remote_workspace::RemoteWorkspacePlugin,
    ));
}
