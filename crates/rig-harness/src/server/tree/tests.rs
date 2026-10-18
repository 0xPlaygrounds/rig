use std::path::PathBuf;

use bevy_ecs::prelude::*;
use rig_protocol::methods::{Budget, Start};
use rig_protocol::{BuildId, DirGrant, Grant, Key, SessionId, codes};

use super::*;
use crate::server::{Session, SessionState};

pub(crate) fn start(cwd: &str, grant: Option<Grant>) -> Start {
    Start {
        cwd: PathBuf::from(cwd),
        prompt: Some("go".into()),
        name: None,
        host: None,
        build: None,
        grant,
        policy: None,
        budget: None,
        context: None,
        unload: None,
        view: false,
        key: Key("k".into()),
    }
}

/// Spawns a session placed under `parent` through `admit_child`, as `rigd`'s
/// start does.
pub(crate) fn spawn(
    world: &mut World,
    id: &str,
    parent: Option<Entity>,
    req: &Start,
) -> Result<Entity, rig_protocol::RpcError> {
    let grant = match parent {
        Some(parent) => Some(admit_child(world, parent, req)?),
        None => req.grant.clone(),
    };
    let session = world
        .spawn(Session {
            id: SessionId(id.into()),
            cwd: req.cwd.clone(),
            host: None,
            build: BuildId("b".into()),
            name: None,
        })
        .id();
    place(world, session, parent, req, grant);
    Ok(session)
}

/// The session, or a placeholder the test's next assertion fails on.
pub(crate) fn ok(spawned: Result<Entity, rig_protocol::RpcError>) -> Entity {
    spawned.unwrap_or(Entity::PLACEHOLDER)
}

fn world() -> World {
    let mut world = World::new();
    world.add_observer(add_usage);
    world
}

fn dirs(path: &str, write: bool) -> Grant {
    Grant {
        dirs: Some(vec![DirGrant {
            path: path.into(),
            write,
        }]),
        net: true,
        ..Grant::default()
    }
}

#[test]
fn a_child_never_gets_more_than_its_parent() {
    let mut world = world();
    let root = ok(spawn(
        &mut world,
        "r",
        None,
        &start("/w", Some(dirs("/w", false))),
    ));
    let child = ok(spawn(
        &mut world,
        "c",
        Some(root),
        &start("/w/src", Some(dirs("/w/src", false))),
    ));
    assert_eq!(
        world.get::<SessionGrant>(child).and_then(|g| g
            .0
            .dirs
            .as_ref()?
            .first()
            .map(|d| d.path.clone())),
        Some(PathBuf::from("/w/src"))
    );

    // Writing where the parent only reads, a folder outside it, network,
    // and a cwd outside the grant: each is refused as wider.
    for (cwd, grant) in [
        ("/w", dirs("/w", true)),
        ("/etc", dirs("/etc", false)),
        ("/w", dirs("/w/src", false)),
    ] {
        let grandchild = spawn(&mut world, "g", Some(child), &start(cwd, Some(grant)));
        assert_eq!(
            grandchild.err().map(|e| e.code),
            Some(codes::GRANT_WIDER),
            "{cwd}"
        );
    }
    let no_net_root = ok(spawn(
        &mut world,
        "n",
        None,
        &start(
            "/w",
            Some(Grant {
                net: false,
                ..Grant::default()
            }),
        ),
    ));
    let net = Grant {
        net: true,
        ..Grant::default()
    };
    assert_eq!(
        spawn(&mut world, "x", Some(no_net_root), &start("/w", Some(net)))
            .err()
            .map(|e| e.code),
        Some(codes::GRANT_WIDER)
    );
    // A start without a grant inherits the parent's.
    let inherits = ok(spawn(&mut world, "i", Some(child), &start("/w/src", None)));
    assert_eq!(
        world.get::<SessionGrant>(inherits).map(|g| &g.0),
        world.get::<SessionGrant>(child).map(|g| &g.0)
    );
    assert_eq!(
        policy_of(&world, inherits).answerers,
        rig_protocol::Answerers::Parent
    );
}

#[test]
fn a_tree_is_at_most_four_children_deep() {
    let mut world = world();
    let mut at = ok(spawn(&mut world, "0", None, &start("/w", None)));
    for depth in 1..=MAX_DEPTH {
        at = ok(spawn(
            &mut world,
            &depth.to_string(),
            Some(at),
            &start("/w", None),
        ));
    }
    let deeper = spawn(&mut world, "5", Some(at), &start("/w", None));
    assert_eq!(deeper.err().map(|e| e.code), Some(codes::LIMIT));
}

#[test]
fn the_ninth_running_child_waits_for_a_slot() {
    let mut world = world();
    let root = ok(spawn(&mut world, "r", None, &start("/w", None)));
    let mut children = Vec::new();
    for n in 0..9 {
        let child = ok(spawn(
            &mut world,
            &format!("c{n}"),
            Some(root),
            &start("/w", None),
        ));
        if admit_run(&mut world, child) {
            world.entity_mut(child).insert(SessionState::Running);
        }
        children.push(child);
    }
    let ninth = children.last().copied().unwrap_or(Entity::PLACEHOLDER);
    assert!(world.get::<Queued>(ninth).is_some(), "the ninth is queued");
    assert!(
        children
            .iter()
            .take(8)
            .all(|&c| world.get::<Queued>(c).is_none())
    );
    assert!(next_queued(&mut world).is_empty(), "no slot yet");

    // One answers and goes idle: the ninth starts.
    world
        .entity_mut(children.first().copied().unwrap_or(Entity::PLACEHOLDER))
        .insert(SessionState::Idle);
    assert_eq!(next_queued(&mut world), vec![ninth]);
    assert_eq!(
        world.get::<SessionState>(ninth),
        Some(&SessionState::Running)
    );
}

#[test]
fn a_parents_own_children_limit_caps_its_children() {
    let mut world = world();
    let root = ok(spawn(
        &mut world,
        "r",
        None,
        &start(
            "/w",
            Some(Grant {
                children: Some(1),
                net: true,
                ..Grant::default()
            }),
        ),
    ));
    let a = ok(spawn(
        &mut world,
        "a",
        Some(root),
        &start(
            "/w",
            Some(Grant {
                children: Some(1),
                net: true,
                ..Grant::default()
            }),
        ),
    ));
    assert!(admit_run(&mut world, a));
    world.entity_mut(a).insert(SessionState::Running);
    let b = ok(spawn(&mut world, "b", Some(root), &start("/w", None)));
    assert!(!admit_run(&mut world, b));
    let wider = Grant {
        children: Some(2),
        net: true,
        ..Grant::default()
    };
    assert_eq!(
        spawn(&mut world, "c", Some(root), &start("/w", Some(wider)))
            .err()
            .map(|e| e.code),
        Some(codes::GRANT_WIDER)
    );
}

#[test]
fn stopping_a_tree_stops_its_leaves_first() {
    let mut world = world();
    let root = ok(spawn(&mut world, "r", None, &start("/w", None)));
    let a = ok(spawn(&mut world, "a", Some(root), &start("/w", None)));
    let a1 = ok(spawn(&mut world, "a1", Some(a), &start("/w", None)));
    let b = ok(spawn(&mut world, "b", Some(root), &start("/w", None)));
    world
        .entity_mut(b)
        .insert(Queued(std::time::Instant::now()));
    let ids: Vec<String> = stop_tree(&mut world, root)
        .into_iter()
        .map(|s| s.0)
        .collect();
    assert_eq!(ids, ["a1", "a", "b", "r"]);
    assert!(
        world.get::<Queued>(b).is_none(),
        "a queued child leaves the queue"
    );

    // Only leaves go out first; a parent waits for its children.
    let mut first = ready_to_stop(&mut world);
    first.sort();
    let mut leaves = vec![a1, b];
    leaves.sort();
    assert_eq!(first, leaves);
    assert!(ready_to_stop(&mut world).is_empty());
    for s in [a1, b] {
        world.entity_mut(s).insert(SessionState::Stopped);
    }
    assert_eq!(ready_to_stop(&mut world), vec![a]);
    world.entity_mut(a).insert(SessionState::Stopped);
    assert_eq!(ready_to_stop(&mut world), vec![root]);
    // The tree's entities stay, with their history.
    assert!(world.get_entity(a1).is_ok());
}

#[test]
fn a_trees_usage_spends_its_roots_budget() {
    let mut world = world();
    let mut req = start("/w", None);
    req.budget = Some(Budget {
        tokens: Some(1_000),
        cost_usd: None,
    });
    let root = ok(spawn(&mut world, "r", None, &req));
    let child = ok(spawn(&mut world, "c", Some(root), &start("/w", None)));
    let grandchild = ok(spawn(&mut world, "g", Some(child), &start("/w", None)));
    world.entity_mut(grandchild).insert(SessionState::Running);

    world.trigger(Usage {
        session: grandchild,
        tokens: 600,
        cost_usd: 0.5,
    });
    world.flush();
    assert_eq!(world.get::<Spent>(grandchild).map(|x| x.tokens), Some(600));
    assert_eq!(
        world.get::<Spent>(child).map(|x| x.tokens),
        Some(0),
        "a session's own spend is its own"
    );
    assert_eq!(
        world.get::<TreeBudget>(root).map(|x| x.spent_tokens),
        Some(600)
    );
    assert!(exhausted(&world, grandchild).is_none());

    world.trigger(Usage {
        session: child,
        tokens: 400,
        cost_usd: 0.0,
    });
    world.flush();
    assert_eq!(exhausted(&world, grandchild), Some(root));
    assert_eq!(running_under(&world, root), vec![grandchild]);
}
