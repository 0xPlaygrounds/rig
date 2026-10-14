use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};
use serde_json::json;

use super::*;

/// Each settled tool call's result: whether it succeeded, and its text.
#[derive(Resource, Default)]
struct Results(HashMap<String, (bool, String)>);

fn log_results(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome)>,
    mut log: ResMut<Results>,
) {
    if let Ok((request, outcome)) = calls.get(settled.entity) {
        let result = outcome.tool_result();
        let text = result.output().render();
        log.0
            .insert(request.call.id.to_string(), (result.is_success(), text));
    }
}

/// A project with `dev.db` (`users`, 500 rows, and `orders`) named `dev`
/// in its `.rig/db.toml`, with a row limit of 50.
fn project() -> Option<PathBuf> {
    let dir = scratch_home("corpus-oc-sql-query-tool")
        .root()
        .to_path_buf();
    std::fs::create_dir_all(dir.join(".rig")).ok()?;
    let toml =
        "[[database]]\nname = \"dev\"\nkind = \"sqlite\"\npath = \"dev.db\"\nrow_limit = 50\n";
    std::fs::write(dir.join(CONFIG), toml).ok()?;
    let db = Connection::open(dir.join("dev.db")).ok()?;
    db.execute_batch(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, deleted INTEGER DEFAULT 0);
         CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), total REAL);
         CREATE INDEX orders_by_user ON orders(user_id);
         WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 500)
         INSERT INTO users (id, name) SELECT i, 'user ' || i FROM n;
         INSERT INTO orders (user_id, total) VALUES (42, 9.5);",
    )
    .ok()?;
    Some(dir)
}

fn users(dir: &Path) -> Option<i64> {
    let db = Connection::open(dir.join("dev.db")).ok()?;
    db.query_row("SELECT count(*) FROM users", [], |row| row.get(0))
        .ok()
}

#[test]
fn the_agent_reads_but_cannot_change_the_database() {
    let dir = project();
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let query = |id, db, sql| calls(id, "db_query", json!({ "db": db, "sql": sql }));
    let model = MockCompletionModel::from_stream_turns([
        calls("schema", "db_schema", json!({ "db": "dev" })),
        calls(
            "table",
            "db_schema",
            json!({ "db": "dev", "table": "orders" }),
        ),
        query("select", "dev", "select * from users order by id"),
        query("delete", "dev", "delete from users"),
        query("two", "dev", "select 1; drop table users"),
        query("prod", "prod", "select 1"),
        reply("Done."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, SqlQueryToolPlugin))
        .init_resource::<Results>()
        .add_observer(log_results)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent with the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    app.world_mut().trigger(Deliver::new(
        agent,
        "Why is user 42's cart empty?",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let results = &app.world().resource::<Results>().0;
    let get = |id: &str| results.get(id).cloned().unwrap_or_default();

    let (ok, schema) = get("schema");
    assert!(
        ok && schema.contains("users") && schema.contains("orders") && schema.contains("500"),
        "{schema}"
    );
    let (ok, table) = get("table");
    assert!(
        ok && table.contains("user_id")
            && table.contains("users(id)")
            && table.contains("orders_by_user"),
        "{table}"
    );

    let (ok, rows) = get("select");
    let table_rows = rows.lines().filter(|line| line.starts_with("| ")).count();
    assert!(ok && rows.contains("500 rows (showing first 50)"), "{rows}");
    assert_eq!(table_rows, 51, "a header and 50 rows: {rows}");

    let (ok, delete) = get("delete");
    assert!(!ok && delete.contains("only read statements"), "{delete}");
    let (ok, two) = get("two");
    assert!(!ok && two.contains("one statement"), "{two}");
    assert_eq!(users(&dir), Some(500), "users is still there, every row");

    let (ok, prod) = get("prod");
    assert!(
        !ok && prod.contains("unknown database `prod`") && prod.contains("dev"),
        "{prod}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SqlQueryToolPlugin);
}
