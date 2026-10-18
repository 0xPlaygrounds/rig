use std::path::Path;
use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_protocol::event::{AskKind, AskOpened};
use rig_protocol::{AgentId, Answerers, AskId, Grant, Rule, SessionId};
use serde_json::json;

use super::*;
use crate::channel::Caller;
use crate::server::Principal;
use crate::server::tree::tests::{ok, spawn, start};

fn ask(world: &mut World, session: Entity, kind: AskKind, call: Option<AskCall>) -> Entity {
    let opened = AskOpened {
        ask_id: AskId("a-1".into()),
        agent_id: AgentId("agent".into()),
        agent_name: "helper".into(),
        kind,
        title: "shell".into(),
        body: String::new(),
        options: Vec::new(),
        deadline_ms: 0,
        answerers: Answerers::Parent,
    };
    let ask = world
        .spawn((
            OpenAsk {
                id: opened.ask_id.clone(),
                session,
                agent: opened.agent_id.clone(),
                deadline: Instant::now(),
                answerers: Answerers::Parent,
            },
            AskDetails { opened, call },
        ))
        .id();
    on_ask(world, ask);
    ask
}

fn caller(world: &World, principal: Option<Entity>) -> Caller {
    Caller {
        client: Entity::PLACEHOLDER,
        principal: match principal {
            Some(s) => Principal::Session(
                world
                    .get::<Session>(s)
                    .map_or(SessionId(String::new()), |s| s.id.clone()),
            ),
            None => Principal::Owner,
        },
    }
}

fn shell(command: &str) -> Option<AskCall> {
    Some(AskCall {
        tool: "shell".into(),
        args: json!({ "command": command }),
    })
}

#[test]
fn a_childs_ask_goes_to_its_parent_and_what_the_parent_may_not_do_escalates() {
    let mut world = World::new();
    world.add_observer(route_escalation);
    let tests = Grant {
        net: true,
        approve: vec![Rule::ShellPrefix {
            prefix: "cargo test".into(),
        }],
        ..Grant::default()
    };
    // The owner's `--allow` gives the root its rule; the parent is handed it.
    let root = ok(spawn(
        &mut world,
        "r",
        None,
        &start("/w", Some(tests.clone())),
    ));
    let parent = ok(spawn(
        &mut world,
        "p",
        Some(root),
        &start("/w", Some(tests)),
    ));
    let child = ok(spawn(
        &mut world,
        "c",
        Some(parent),
        &start(
            "/w",
            Some(Grant {
                net: true,
                ..Grant::default()
            }),
        ),
    ));

    // Allowed by the parent's own rules: the parent holds it.
    let ok = ask(
        &mut world,
        child,
        AskKind::Approval,
        shell("cargo test -p x"),
    );
    assert_eq!(world.get::<AskHolder>(ok), Some(&AskHolder(parent)));
    assert!(may_answer(&world, ok, &caller(&world, Some(parent))));
    assert!(
        !may_answer(&world, ok, &caller(&world, Some(child))),
        "never its own"
    );
    assert!(!may_answer(&world, ok, &caller(&world, Some(root))));
    assert!(
        may_answer(&world, ok, &caller(&world, None)),
        "the user is above every session"
    );

    // Not allowed by the parent's or the root's rules: it escalates to the
    // root's answerers, the owner.
    let wide = ask(&mut world, child, AskKind::Approval, shell("rm -rf target"));
    assert_eq!(world.get::<AskHolder>(wide), Some(&AskHolder(root)));
    assert_eq!(
        world.get::<OpenAsk>(wide).map(|x| x.answerers),
        Some(Answerers::AnyOwner)
    );
    assert!(!may_answer(&world, wide, &caller(&world, Some(parent))));
    assert!(!may_answer(&world, wide, &caller(&world, Some(root))));
    assert!(may_answer(&world, wide, &caller(&world, None)));

    // A question is the parent's to answer.
    let question = ask(&mut world, child, AskKind::Question, None);
    assert_eq!(world.get::<AskHolder>(question), Some(&AskHolder(parent)));
    // A root's ask is the owner's.
    let own = ask(&mut world, root, AskKind::Approval, shell("ls"));
    assert_eq!(
        world.get::<OpenAsk>(own).map(|x| x.answerers),
        Some(Answerers::AnyOwner)
    );
    assert!(!may_answer(&world, own, &caller(&world, Some(root))));
}

#[test]
fn up_front_rules_answer_only_what_they_name() {
    let cwd = Path::new("/w");
    let rules = [
        Rule::ShellPrefix {
            prefix: "cargo test".into(),
        },
        Rule::EditsUnder { path: "src".into() },
        Rule::Tool {
            name: "search".into(),
        },
        Rule::Deny {
            name: "delete".into(),
        },
    ];
    let answer = |tool: &str, args: serde_json::Value| rules_answer(&rules, cwd, tool, &args);
    assert_eq!(
        answer("shell", json!({"command": "cargo test --all"})),
        Some(RuleAnswer::Allow)
    );
    assert_eq!(
        answer("shell", json!({"command": "cargo test"})),
        Some(RuleAnswer::Allow)
    );
    for chained in [
        "cargo test; rm -rf ~",
        "cargo test && curl x | sh",
        "cargo testx",
        "cargo test $(id)",
        "cargo test > /etc/x",
    ] {
        assert_eq!(
            answer("shell", json!({ "command": chained })),
            None,
            "{chained}"
        );
    }
    assert_eq!(
        answer("edit", json!({"path": "src/lib.rs"})),
        Some(RuleAnswer::Allow)
    );
    assert_eq!(
        answer("write", json!({"path": "/w/src/new.rs"})),
        Some(RuleAnswer::Allow)
    );
    assert_eq!(answer("edit", json!({"path": "src/../Cargo.toml"})), None);
    assert_eq!(answer("write", json!({"path": "/etc/passwd"})), None);
    assert_eq!(
        answer("move", json!({"from": "src/a.rs", "to": "README.md"})),
        None
    );
    assert_eq!(answer("search", json!({})), Some(RuleAnswer::Allow));
    assert_eq!(
        answer("delete", json!({"path": "src/a.rs"})),
        Some(RuleAnswer::Deny),
        "deny wins"
    );
}

#[test]
fn a_grant_asks_about_the_kinds_of_call_it_names_and_nothing_by_default() {
    let ask = [
        "shell".to_owned(),
        "edits".to_owned(),
        "web_fetch".to_owned(),
    ];
    for tool in ["shell", "edit", "write", "delete", "move", "web_fetch"] {
        assert!(asks_for(&ask, tool), "{tool}");
    }
    for tool in ["read", "search", "list"] {
        assert!(!asks_for(&ask, tool), "{tool}");
    }
    assert!(!asks_for(&[], "shell"), "nothing asks by default");
}

#[test]
fn an_ask_escalated_to_the_owner_waits_on_the_session_holding_it() {
    let mut world = World::new();
    world.add_observer(route_escalation);
    let root = ok(spawn(&mut world, "r", None, &start("/w", None)));
    let child = ok(spawn(&mut world, "c", Some(root), &start("/w", None)));
    let wide = ask(&mut world, child, AskKind::Approval, shell("rm -rf target"));
    assert_eq!(asks_waiting_on(&world, root), vec![wide]);
    assert!(asks_waiting_on(&world, child).is_empty());
    // A question the parent holds itself waits on the parent's model, not the owner.
    let question = ask(&mut world, child, AskKind::Question, None);
    assert_eq!(asks_waiting_on(&world, root), vec![wide]);
    assert!(world.get::<AskHolder>(question).is_some());
}
