use super::*;
use serde_json::json;

#[test]
fn only_delivery_boundaries_are_excluded() {
    let log = json!({
        "header": {
            "deliveries": [{"batch": 1, "id": 0, "kind": {"delivery": "stream", "items": 1}}],
            "programs": {"run/0": {"required": {}, "policy": 7}},
            "stream_errors": {},
            "delivery_limitations": [],
            "handlers": [],
            "signature": {},
            "hooks": [],
            "required": {}
        },
        "records": [{"id": 0, "scope": "run/0", "parent": null, "events": [], "tool_output": null}]
    });
    let mut differently_scheduled = log.clone();
    differently_scheduled["header"]["deliveries"] = json!([]);
    assert_eq!(
        without_delivery_boundaries(log.clone()),
        without_delivery_boundaries(differently_scheduled)
    );
    for pointer in [
        "/header/programs/run~10/policy",
        "/header/stream_errors",
        "/header/delivery_limitations",
        "/header/handlers",
        "/header/signature",
        "/header/hooks",
        "/header/required",
        "/records/0/id",
        "/records/0/scope",
        "/records/0/parent",
        "/records/0/events",
        "/records/0/tool_output",
    ] {
        let mut changed = log.clone();
        *changed.pointer_mut(pointer).expect("test field") = json!("changed");
        assert_ne!(
            without_delivery_boundaries(log.clone()),
            without_delivery_boundaries(changed),
            "{pointer}"
        );
    }
}

#[tokio::test]
async fn capture_scopes_are_independent() {
    capture_world_programs(async {
        PROGRAMS.with(|programs| {
            programs.borrow_mut().insert(
                "outer".into(),
                (ServingPolicy::default(), Checkpoint::default()),
            );
        });
        capture_world_programs(async {
            assert!(PROGRAMS.with(|programs| programs.borrow().is_empty()));
        })
        .await;
        assert!(PROGRAMS.with(|programs| programs.borrow().contains_key("outer")));
    })
    .await;
    assert!(PROGRAMS.try_with(|_| ()).is_err());
}
