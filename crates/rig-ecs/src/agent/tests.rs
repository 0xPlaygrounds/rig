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
    let parent = ToolAccess::All.narrow(["read", "shell"]);
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
        ToolAccess::All
            .grant(&owned(&["bogus"]), registered)
            .is_err()
    );
}

#[test]
fn narrowing_never_widens_an_access() {
    let read = ToolAccess::All.narrow(["read"]);
    assert_eq!(read.narrow(["read", "shell"]), read);
    assert!(!read.within(&ToolAccess::All).allows("shell"));
    assert_eq!(ToolAccess::All.within(&read), read);
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
    let parent = world
        .spawn(ToolAccess::All.narrow(["read", "run_steel"]))
        .id();
    let child = (world.spawn((ToolAccess::All.narrow(["run_steel"]), SpawnedBy(parent)))).id();
    // As a session saved before the check restores it: wider than its parent.
    let leaf = world
        .spawn((
            ToolAccess::All.narrow(["shell", "run_steel"]),
            SpawnedBy(child),
        ))
        .id();
    let free = world.spawn((ToolAccess::All, SpawnedBy(child))).id();
    world.run_system_once(keep_within_parents).ok();
    let names = |world: &World, agent| {
        let access = world.get::<ToolAccess>(agent);
        access.and_then(ToolAccess::names).map(<[String]>::to_vec)
    };
    assert_eq!(names(&world, leaf), Some(owned(&["run_steel"])));
    assert_eq!(names(&world, free), Some(owned(&["run_steel"])));
    assert_eq!(names(&world, parent), Some(owned(&["read", "run_steel"])));
}
