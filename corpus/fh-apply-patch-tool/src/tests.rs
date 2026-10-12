use std::fs;
use std::path::Path;

use rig_coding_tools::{EditTool, WriteTool};
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

/// Applies, but a gate refuses the `write` of `secret.txt`.
const GUARDED: &str = "*** Begin Patch
*** Update File: a.txt
@@
-TWO
+2
*** Add File: secret.txt
+key
*** End Patch";

/// Another plugin's rule on `write`: never `secret.txt`.
fn no_secrets(mut calls: Query<(&ToolRequest, &mut Decision), (Undecided, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments.get("path");
        let path = path.and_then(serde_json::Value::as_str).unwrap_or_default();
        if request.call.function.name.as_str() == "write" && path.ends_with("secret.txt") {
            decision.decide(Decider::plugin("test"), Outcome::refused("no secrets"));
        }
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

    let gpt = MockCompletionModel::from_stream_turns([
        calls("p1", APPLY_PATCH, json!({ "input": PATCH })),
        calls("p2", APPLY_PATCH, json!({ "input": BROKEN })),
        calls("p3", APPLY_PATCH, json!({ "input": GUARDED })),
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
        JournalPlugin,
        EditTool,
        WriteTool,
        ApplyPatchPlugin,
    ))
    .add_systems(AgentPass, no_secrets.in_set(Stage::Gate))
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
    });
    world.trigger(Deliver::new(agent, "Patch it.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

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

    let world = app.world_mut();
    let model = "test/claude-test".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
    });
    world.trigger(Deliver::new(agent, "Hi.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 1
    });
    let after = offered(&claude, 0);
    assert!(after.iter().any(|name| name == "edit"), "{after:?}");
    assert!(after.iter().any(|name| name == "write"), "{after:?}");
    assert!(!after.iter().any(|name| name == APPLY_PATCH), "{after:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ApplyPatchPlugin);
}
