use super::*;

fn operation(action: &str, key: &str, value: Option<i64>) -> Operation {
    Operation {
        action: action.into(),
        key: key.into(),
        value,
    }
}

#[test]
#[should_panic(expected = "validate before the first answer")]
fn an_unvalidated_intermediate_answer_is_rejected() {
    let mut state = State::new(Task::Inventory);
    for (key, value) in state.expected() {
        state
            .execute(operation("apply", &key, Some(value)))
            .expect("inventory update");
    }
    state.assert_initial_inventory();
}

#[test]
fn validation_depends_on_state_and_latest_revision() {
    let mut state = State::new(Task::Repair);
    assert!(state.execute(operation("check", "all", None)).is_err());
    assert_eq!(
        state
            .execute(operation("check", "all", None))
            .expect("retry")["passed"],
        false
    );
    for (key, value) in state.expected() {
        state
            .execute(operation("apply", &key, Some(value)))
            .expect("edit");
    }
    assert_eq!(
        state
            .execute(operation("check", "all", None))
            .expect("validate")["passed"],
        true
    );
    let verified = state.verified_revision;
    state
        .execute(operation("apply", "days_per_week", Some(8)))
        .expect("incorrect edit");
    assert_ne!(verified, Some(state.revision));
    assert_eq!(
        state
            .execute(operation("check", "all", None))
            .expect("validate")["passed"],
        false
    );
}

#[test]
fn reconciliation_uses_confirmed_credit_and_every_invoice() {
    let state = State::new(Task::Reconcile);
    for n in 1..=6 {
        let page = state.read(&format!("page/{n}")).expect("page");
        let total: i64 = page["records"]
            .as_array()
            .expect("invoices")
            .iter()
            .map(|r| {
                r["quantity"].as_i64().expect("quantity") * r["unit_price"].as_i64().expect("price")
            })
            .sum();
        let credit = state.read(&format!("credit/{n}")).expect("credit");
        let reconciled = total - credit["credit"].as_i64().expect("confirmed");
        assert_eq!(
            state.expected().get(&format!("page/{n}")),
            Some(&reconciled)
        );
        assert_ne!(credit["credit"], credit["supersedes"]["credit"]);
    }
}

#[test]
fn follow_up_requires_verified_initial_inventory_and_invalidates_it() {
    let mut state = State::new(Task::Inventory);
    assert!(state.execute(operation("revise", "all", None)).is_err());
    for (key, value) in state.expected() {
        state
            .execute(operation("apply", &key, Some(value)))
            .expect("apply");
    }
    state
        .execute(operation("check", "all", None))
        .expect("check");
    state
        .execute(operation("revise", "all", None))
        .expect("follow up");
    assert_ne!(state.verified_revision, Some(state.revision));
    assert_eq!(state.expected().get("depot/1"), Some(&6));
    assert_eq!(
        state
            .execute(operation("check", "all", None))
            .expect("check")["passed"],
        false
    );
}

#[test]
fn unknown_keys_and_operation_limit_are_real_errors() {
    let mut state = State::new(Task::Reconcile);
    assert!(state.execute(operation("read", "page/7", None)).is_err());
    assert!(
        state
            .execute(operation("apply", "unknown", Some(4)))
            .is_err()
    );
    for _ in 2..60 {
        state
            .execute(operation("read", "index", None))
            .expect("within budget");
    }
    assert!(state.execute(operation("read", "index", None)).is_err());
    assert!(state.values.is_empty());
}
