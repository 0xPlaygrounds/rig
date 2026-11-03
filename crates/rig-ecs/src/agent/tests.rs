use bevy_ecs::system::RunSystemOnce;
use serde_json::json;

use super::*;
use crate::tools::keep_within_parents;

/// `names` as owned strings.
fn owned(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[test]
fn a_grant_gives_only_tools_the_granting_access_has() {
    let parent = ToolAccess::all().narrow(["read", "shell"]);
    let registered = ["edit", "read", "shell"];
    let shell = parent.grant(&owned(&["shell"]), registered);
    assert_eq!(
        shell.ok().as_ref().and_then(ToolAccess::names),
        Some(&owned(&["shell"])[..])
    );
    let refused = parent.grant(&owned(&["read", "edit"]), registered);
    let refused = refused.err().map(|refused| refused.to_string());
    assert_eq!(
        refused.as_deref(),
        Some("`edit` is not one of your tools (read, shell)")
    );
    // A name no tool has is refused from every tool too.
    assert!(
        ToolAccess::all()
            .grant(&owned(&["bogus"]), registered)
            .is_err()
    );
}

#[test]
fn narrowing_never_widens_an_access() {
    let read = ToolAccess::all().narrow(["read"]);
    assert_eq!(read.narrow(["read", "shell"]), read);
    assert!(!read.within(&ToolAccess::all()).allows("shell"));
    assert_eq!(ToolAccess::all().within(&read), read);
    let none = read.narrow(["shell"]);
    assert_eq!(none.names(), Some(&[][..]));
}

#[test]
fn an_access_saves_as_it_did_before_its_names_were_private() {
    let saved = [
        json!("All"),
        json!({ "Only": ["read", "shell"] }),
        json!({ "Only": [] }),
    ];
    for saved in saved {
        let access: Option<ToolAccess> = serde_json::from_value(saved.clone()).ok();
        let again = access.and_then(|access| serde_json::to_value(access).ok());
        assert_eq!(again, Some(saved));
    }
}

#[test]
fn an_agent_is_kept_within_every_agent_above_it() {
    let mut world = World::new();
    world.init_resource::<Messages<Notice>>();
    let parent = world
        .spawn(ToolAccess::all().narrow(["read", "run_steel"]))
        .id();
    let child = (world.spawn((ToolAccess::all().narrow(["run_steel"]), SpawnedBy(parent)))).id();
    // As a session saved before the check restores it: wider than its parent.
    let leaf = world
        .spawn((
            ToolAccess::all().narrow(["shell", "run_steel"]),
            SpawnedBy(child),
        ))
        .id();
    let free = world.spawn((ToolAccess::all(), SpawnedBy(child))).id();
    world.run_system_once(keep_within_parents).ok();
    let names = |world: &World, agent| {
        let access = world.get::<ToolAccess>(agent);
        access.and_then(ToolAccess::names).map(<[String]>::to_vec)
    };
    assert_eq!(names(&world, leaf), Some(owned(&["run_steel"])));
    assert_eq!(names(&world, free), Some(owned(&["run_steel"])));
    assert_eq!(names(&world, parent), Some(owned(&["read", "run_steel"])));
}

#[test]
fn the_runs_tools_limit_an_access_and_are_never_saved() {
    let run = RunTools::new(["read", "edit"]);
    let access = ToolAccess::all()
        .narrow(["read", "shell"])
        .for_run(Some(&run));
    assert!(access.allows("read") && !access.allows("shell") && access.keeps("shell"));
    // A child may be given what its parent keeps, and uses it within the
    // run's tools too.
    let child = access.grant(&owned(&["shell"]), ["read", "shell"]);
    assert!(child.is_ok_and(|child| child.keeps("shell") && !child.allows("shell")));
    let free = ToolAccess::all().for_run(Some(&run));
    assert!(!free.within(&ToolAccess::all()).allows("shell"));
    assert!(!ToolAccess::all().within(&free).allows("shell"));
    let saved = serde_json::to_value(&access).ok();
    assert_eq!(saved, Some(json!({ "Only": ["read", "shell"] })));
}

#[test]
fn an_agent_is_kept_within_the_runs_tools_however_it_was_made() {
    let mut world = World::new();
    world.init_resource::<Messages<Notice>>();
    world.insert_resource(RunTools::new(["read", "task"]));
    let parent = world.spawn(ToolAccess::all()).id();
    // Made afresh, not from its parent's access, as a plugin may.
    let child = world.spawn((ToolAccess::all(), SpawnedBy(parent))).id();
    world.run_system_once(keep_within_parents).ok();
    for agent in [parent, child] {
        let access = world.get::<ToolAccess>(agent);
        assert!(access.is_some_and(|access| access.allows("task") && !access.allows("shell")));
        let saved = access.and_then(|access| serde_json::to_value(access).ok());
        assert_eq!(saved, Some(json!("All")));
    }
}

/// An agent kept within its parent loses tools with a warning naming them;
/// one kept within the run's tools, which the user gave, gets none.
#[test]
fn an_agent_that_loses_tools_to_its_parent_is_told_which() {
    let mut world = World::new();
    world.init_resource::<Messages<Notice>>();
    world.insert_resource(RunTools::new(["read", "shell", "write"]));
    for name in ["read", "shell", "write", "edit"] {
        if let Ok(name) = rig_core::message::ToolName::new(name) {
            let def = rig_core::completion::ToolDefinition::new(name, "", json!({}));
            world.spawn(crate::tools::ToolDef(std::sync::Arc::new(def)));
        }
    }
    let parent = world.spawn(ToolAccess::all().narrow(["read"])).id();
    let child = world.spawn((ToolAccess::all(), SpawnedBy(parent))).id();
    world.run_system_once(keep_within_parents).ok();
    let said: Vec<(Option<Entity>, String)> = world
        .resource_mut::<Messages<Notice>>()
        .drain()
        .map(|notice| (notice.agent, notice.text))
        .collect();
    assert_eq!(
        said,
        [(
            Some(child),
            "Lost `shell`, `write`: an agent above it does not have them.".to_owned()
        )]
    );
}

/// The run's tools as the run starts narrow every agent without a word, as
/// the user gave them; a change to them after the start is said.
#[test]
fn the_runs_tools_changed_after_the_start_are_said() {
    let mut world = World::new();
    world.init_resource::<Messages<Notice>>();
    for name in ["read", "shell"] {
        if let Ok(name) = rig_core::message::ToolName::new(name) {
            let def = rig_core::completion::ToolDefinition::new(name, "", json!({}));
            world.spawn(crate::tools::ToolDef(std::sync::Arc::new(def)));
        }
    }
    let agent = world.spawn(ToolAccess::all()).id();
    world.insert_resource(RunTools::new(["read", "shell"]));
    let keep = world.register_system(keep_within_parents);
    world.run_system(keep).ok();
    world.flush();
    *world.resource_mut::<RunTools>() = RunTools::new(["read"]);
    world.run_system(keep).ok();
    let said: Vec<(Option<Entity>, String)> = world
        .resource_mut::<Messages<Notice>>()
        .drain()
        .map(|notice| (notice.agent, notice.text))
        .collect();
    assert_eq!(
        said,
        [(
            Some(agent),
            "Lost `shell`: this run's tools leave it out.".to_owned()
        )]
    );
}
