//! OC-05: read-only SQL tools. `<project>/.rig/db.toml` names the
//! development databases, and two tools let the agent look at them:
//! `db_schema` lists their tables with row counts or one table's columns,
//! keys and indexes, and `db_query` runs one read statement and answers a
//! Markdown table.
//!
//! ```toml
//! [[database]]
//! name = "dev"
//! kind = "sqlite"
//! path = "dev.db"   # relative to the project
//! row_limit = 50    # 200 by default
//! ```
//!
//! Nothing can change data: the connection is opened read-only and
//! `query_only`, and a statement runs only when it is one statement that
//! reads (`SELECT`, `WITH`, `VALUES`, `EXPLAIN`, a `PRAGMA` without `=`)
//! and SQLite says it is read-only. Each query stops after
//! [`QUERY_TIMEOUT`], and as soon as its call is cancelled, such as when
//! the turn is interrupted.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use rusqlite::types::ValueRef;
use rusqlite::{Batch, Connection, ErrorCode, OpenFlags, Row, Statement};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// Where the databases are named, relative to the project (or the calling
/// agent's `WorkDir`).
pub const CONFIG: &str = ".rig/db.toml";

/// How many rows a query shows when its database names no `row_limit`.
pub const DEFAULT_ROW_LIMIT: usize = 200;

/// How long one query may run.
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

/// How many characters of a cell are shown.
pub const MAX_CELL_CHARS: usize = 200;

/// Adds the `db_schema` and `db_query` tools.
#[derive(Default)]
pub struct SqlQueryToolPlugin;

impl Plugin for SqlQueryToolPlugin {
    fn build(&self, app: &mut App) {
        let options = || ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_tool_per_call(DbSchema(At::default()), options(), |place| {
            DbSchema(At::from(place))
        })
        .add_tool_per_call(DbQuery(At::default()), options(), |place| {
            DbQuery(At::from(place))
        });
    }
}

/// The `db.toml` file.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    /// The databases, by name.
    #[serde(default)]
    database: Vec<Database>,
}

/// One named database.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Database {
    name: String,
    /// `sqlite`, the only kind this plugin opens.
    kind: String,
    /// A sqlite file, relative to the project.
    path: Option<String>,
    /// A connection URL, for kinds other than sqlite.
    url: Option<String>,
    row_limit: Option<usize>,
}

/// Where a call works: the app's workspace and the calling agent's
/// directory.
#[derive(Clone)]
struct At {
    workspace: Arc<dyn Workspace>,
    base: PathBuf,
}

impl Default for At {
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: PathBuf::new(),
        }
    }
}

impl From<CallPlace> for At {
    fn from(place: CallPlace) -> Self {
        Self {
            workspace: place.workspace,
            base: place.base,
        }
    }
}

impl At {
    /// Opens the database named `name` read-only, its queries stopped
    /// once `cancelled` or past [`QUERY_TIMEOUT`]; with its row limit.
    fn open(&self, name: &str, cancelled: &Arc<AtomicBool>) -> Result<Open, ToolExecutionError> {
        if let Some(host) = self.workspace.host() {
            return Err(ToolExecutionError::invalid_args(format!(
                "databases open on this machine, and the workspace is on {host}"
            )));
        }
        let text = self
            .workspace
            .read_text(&self.base.join(CONFIG))
            .map_err(|error| {
                ToolExecutionError::not_found(format!(
                    "no databases configured ({CONFIG}): {}",
                    error.message()
                ))
            })?;
        let config: Config = toml::from_str(&text)
            .map_err(|error| ToolExecutionError::invalid_args(format!("{CONFIG}: {error}")))?;
        let Some(database) = config.database.iter().find(|it| it.name == name) else {
            let names: Vec<&str> = config.database.iter().map(|it| it.name.as_str()).collect();
            return Err(ToolExecutionError::not_found(format!(
                "unknown database `{name}`; configured: {}",
                names.join(", ")
            )));
        };
        if database.kind != "sqlite" {
            return Err(ToolExecutionError::invalid_args(format!(
                "`{}` is a {} database{}; this tool opens sqlite only",
                database.name,
                database.kind,
                database
                    .url
                    .as_ref()
                    .map(|url| format!(" at {url}"))
                    .unwrap_or_default()
            )));
        }
        let Some(path) = &database.path else {
            return Err(ToolExecutionError::invalid_args(format!(
                "{CONFIG}: `{}` names no sqlite `path`",
                database.name
            )));
        };
        let path = self.workspace.place(Some(&self.base), false, path)?;
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let connection = Connection::open_with_flags(&path, flags).map_err(|error| {
            ToolExecutionError::not_found(format!("{}: {error}", path.display()))
        })?;
        connection
            .pragma_update(None, "query_only", true)
            .map_err(sql_error)?;
        let deadline = Instant::now() + QUERY_TIMEOUT;
        let stop = cancelled.clone();
        connection.progress_handler(
            1000,
            Some(move || stop.load(Ordering::Relaxed) || Instant::now() >= deadline),
        );
        let limit = database.row_limit.unwrap_or(DEFAULT_ROW_LIMIT);
        Ok(Open {
            connection,
            limit,
            deadline,
        })
    }

    /// Opens `db` off the async thread and runs `work` on it, stopping it
    /// when the call is cancelled (its future dropped).
    async fn run(
        &self,
        db: String,
        work: impl FnOnce(&Open) -> Result<String, ToolExecutionError> + Send + 'static,
    ) -> Result<String, ToolExecutionError> {
        let at = self.clone();
        let cancel = CancelOnDrop::default();
        let cancelled = cancel.0.clone();
        let answer = blocking(move || {
            let open = at.open(&db, &cancelled)?;
            work(&open).map_err(|error| open.stopped(error))
        });
        answer.await
    }
}

/// An open database.
struct Open {
    connection: Connection,
    limit: usize,
    deadline: Instant,
}

impl Open {
    /// `error`, or a timeout or cancellation when the query was stopped.
    fn stopped(&self, error: ToolExecutionError) -> ToolExecutionError {
        if error.code() != Some(INTERRUPTED) {
            error
        } else if Instant::now() >= self.deadline {
            ToolExecutionError::timeout(format!(
                "the query ran longer than {} s",
                QUERY_TIMEOUT.as_secs()
            ))
        } else {
            ToolExecutionError::cancelled("the query was cancelled")
        }
    }
}

/// Raises its flag when dropped: a call's future dropped stops its query.
#[derive(Default)]
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The `db_schema` tool.
struct DbSchema(At);

/// `db_schema`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SchemaArgs {
    /// The database's name in db.toml.
    db: String,
    /// One table, for its columns, keys and indexes; every table with its
    /// row count when left out.
    table: Option<String>,
}

impl Tool for DbSchema {
    const NAME: &'static str = "db_schema";
    type Args = SchemaArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "List a development database's tables with their row counts, or one table's columns, types, keys and indexes. Call it before `db_query`.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<SchemaArgs>()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: SchemaArgs,
    ) -> Result<String, ToolExecutionError> {
        self.0
            .run(args.db, move |open| match &args.table {
                None => tables(&open.connection),
                Some(table) => table_schema(&open.connection, table),
            })
            .await
    }
}

/// The `db_query` tool.
struct DbQuery(At);

/// `db_query`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct QueryArgs {
    /// The database's name in db.toml.
    db: String,
    /// One read statement: SELECT, WITH, VALUES, EXPLAIN or a reading
    /// PRAGMA.
    sql: String,
}

impl Tool for DbQuery {
    const NAME: &'static str = "db_query";
    type Args = QueryArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run one read-only SQL statement on a development database and see its rows as a Markdown table. Use it to read the development databases directly.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<QueryArgs>()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: QueryArgs,
    ) -> Result<String, ToolExecutionError> {
        self.0
            .run(args.db, move |open| query(open, &args.sql))
            .await
    }
}

/// Runs `sql`, refused unless it is one statement that reads.
fn query(open: &Open, sql: &str) -> Result<String, ToolExecutionError> {
    let mut batch = Batch::new(&open.connection, sql);
    let Some(mut statement) = batch.next().map_err(sql_error)? else {
        return Err(ToolExecutionError::invalid_args("no SQL statement given"));
    };
    if !matches!(batch.next(), Ok(None)) {
        return Err(ToolExecutionError::refused(
            "refused: one statement per call; nothing ran",
        ));
    }
    if !reads(sql, &statement) {
        return Err(ToolExecutionError::refused(
            "refused: only read statements run (SELECT, WITH, VALUES, EXPLAIN, a PRAGMA that reads); nothing ran",
        ));
    }
    let names: Vec<String> = statement
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut rows = statement.query([]).map_err(sql_error)?;
    let (mut shown, mut total) = (Vec::new(), 0_usize);
    while let Some(row) = rows.next().map_err(sql_error)? {
        total += 1;
        if shown.len() < open.limit {
            shown.push(cells(row, names.len())?);
        }
    }
    let mut out = markdown(&names, &shown);
    if total > shown.len() {
        out.push_str(&format!("\n{total} rows (showing first {})\n", shown.len()));
    } else {
        out.push_str(&format!("\n{total} rows\n"));
    }
    Ok(out)
}

/// Whether `statement`, prepared from `sql`, only reads: by its first
/// keyword, and by SQLite.
fn reads(sql: &str, statement: &Statement<'_>) -> bool {
    let sql = skip_comments(sql).trim_start_matches(['(', ' ', '\t', '\n', '\r']);
    let keyword: String = sql
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_lowercase();
    let allowed = match keyword.as_str() {
        "select" | "with" | "values" | "explain" => true,
        "pragma" => !sql.contains('='),
        _ => false,
    };
    allowed && statement.readonly()
}

/// `sql` after its leading whitespace and comments.
fn skip_comments(mut sql: &str) -> &str {
    loop {
        sql = sql.trim_start();
        if let Some(rest) = sql.strip_prefix("--") {
            sql = rest.split_once('\n').map_or("", |(_, after)| after);
        } else if let Some(rest) = sql.strip_prefix("/*") {
            sql = rest.split_once("*/").map_or("", |(_, after)| after);
        } else {
            return sql;
        }
    }
}

/// Every table with its row count.
fn tables(connection: &Connection) -> Result<String, ToolExecutionError> {
    let names = table_names(connection)?;
    let mut rows = Vec::new();
    for name in names {
        let count: i64 = connection
            .query_row(
                &format!("SELECT count(*) FROM {}", quoted(&name)),
                [],
                |row| row.get(0),
            )
            .map_err(sql_error)?;
        rows.push(vec![name, count.to_string()]);
    }
    Ok(markdown(&["table".to_owned(), "rows".to_owned()], &rows))
}

/// The tables' names, in order.
fn table_names(connection: &Connection) -> Result<Vec<String>, ToolExecutionError> {
    let mut statement = connection
        .prepare(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .map_err(sql_error)?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(sql_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sql_error)?;
    Ok(names)
}

/// `table`'s columns, foreign keys and indexes.
fn table_schema(connection: &Connection, table: &str) -> Result<String, ToolExecutionError> {
    let names = table_names(connection)?;
    if !names.iter().any(|name| name == table) {
        return Err(ToolExecutionError::not_found(format!(
            "no table `{table}`; tables: {}",
            names.join(", ")
        )));
    }
    let columns = rows_of(
        connection,
        "SELECT name, type, CASE WHEN \"notnull\" THEN 'not null' ELSE '' END, \
         coalesce(dflt_value, ''), CASE WHEN pk > 0 THEN 'pk ' || pk ELSE '' END \
         FROM pragma_table_info(?1) ORDER BY cid",
        table,
    )?;
    let keys = rows_of(
        connection,
        "SELECT \"from\", \"table\" || '(' || coalesce(\"to\", '') || ')', on_delete \
         FROM pragma_foreign_key_list(?1) ORDER BY id, seq",
        table,
    )?;
    let indexes = rows_of(
        connection,
        "SELECT l.name, CASE WHEN l.\"unique\" THEN 'unique' ELSE '' END, \
         (SELECT group_concat(i.name, ', ') FROM pragma_index_info(l.name) AS i) \
         FROM pragma_index_list(?1) AS l ORDER BY l.name",
        table,
    )?;
    let header = |names: &[&str]| {
        names
            .iter()
            .map(|&name| name.to_owned())
            .collect::<Vec<_>>()
    };
    let mut out = format!("## {table}\n\n");
    out.push_str(&markdown(
        &header(&["column", "type", "null", "default", "key"]),
        &columns,
    ));
    if !keys.is_empty() {
        out.push_str("\nForeign keys:\n\n");
        out.push_str(&markdown(
            &header(&["column", "references", "on delete"]),
            &keys,
        ));
    }
    if !indexes.is_empty() {
        out.push_str("\nIndexes:\n\n");
        out.push_str(&markdown(
            &header(&["index", "unique", "columns"]),
            &indexes,
        ));
    }
    Ok(out)
}

/// The rows `sql` answers for `table`, as cells.
fn rows_of(
    connection: &Connection,
    sql: &str,
    table: &str,
) -> Result<Vec<Vec<String>>, ToolExecutionError> {
    let mut statement = connection.prepare(sql).map_err(sql_error)?;
    let width = statement.column_count();
    let mut rows = statement.query([table]).map_err(sql_error)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(sql_error)? {
        out.push(cells(row, width)?);
    }
    Ok(out)
}

/// The first `width` cells of `row`, as shown.
fn cells(row: &Row<'_>, width: usize) -> Result<Vec<String>, ToolExecutionError> {
    (0..width)
        .map(|index| row.get_ref(index).map(cell).map_err(sql_error))
        .collect()
}

/// A value as a table cell: cut at [`MAX_CELL_CHARS`], a blob by its size.
fn cell(value: ValueRef<'_>) -> String {
    let text = match value {
        ValueRef::Null => return "NULL".to_owned(),
        ValueRef::Integer(it) => return it.to_string(),
        ValueRef::Real(it) => return it.to_string(),
        ValueRef::Blob(bytes) => return format!("<blob {} bytes>", bytes.len()),
        ValueRef::Text(bytes) => String::from_utf8_lossy(bytes),
    };
    let mut chars = text.chars();
    let mut shown: String = chars.by_ref().take(MAX_CELL_CHARS).collect();
    if chars.next().is_some() {
        shown.push('…');
    }
    shown
}

/// `rows` under `header`, as a Markdown table.
fn markdown(header: &[String], rows: &[Vec<String>]) -> String {
    let line = |cells: &[String]| {
        let cells: Vec<String> = cells
            .iter()
            .map(|cell| cell.replace('|', "\\|").replace(['\n', '\r'], " "))
            .collect();
        format!("| {} |\n", cells.join(" | "))
    };
    let mut out = line(header);
    out.push_str(&format!("|{}\n", "---|".repeat(header.len())));
    for row in rows {
        out.push_str(&line(row));
    }
    out
}

/// `name` as a quoted SQL identifier.
fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The code of an error from a query SQLite interrupted.
const INTERRUPTED: &str = "interrupted";

/// A SQLite error as the tool's; an interrupted query is told apart by
/// [`Open::stopped`].
fn sql_error(error: rusqlite::Error) -> ToolExecutionError {
    if error.sqlite_error_code() == Some(ErrorCode::OperationInterrupted) {
        return ToolExecutionError::other("the query was interrupted").with_code(INTERRUPTED);
    }
    ToolExecutionError::invalid_args(format!("SQL error: {error}"))
}
