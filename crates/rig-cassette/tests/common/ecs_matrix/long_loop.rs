//! The long tool loop: one deterministic, in-memory repository toolset and
//! the programs that drive a model through at least six dependent tool
//! turns over it, shared by both interpreters on five provider columns.
//!
//! The repository ([`RepoHandle`]) is five small ASCII files and four
//! tools — `list_files`, `read_file`, `write_file`, `run_tests` — whose
//! outputs are pure functions of the tree's state. Two files are wrong
//! (`src/lib.rs`'s `days_in_year`, `src/calendar.rs`'s `days_in_week`);
//! `run_tests` stops at the first failing test and reports only that
//! assertion, so a model learns of the second bug only after fixing the
//! first, and reports `PASS` only when both are right. Every invocation is
//! logged and exposed for assertions ([`Invocation`]). Two bugs in two
//! files make a batched loop (gpt-4.1-mini reads all three files in one
//! turn under [`LOOP_PROMPT`]) still at least seven tool turns long.
//!
//! The tree is host state, deliberately outside the saved scene (as the
//! checkpoint row's parallel gates are): a fresh world restored from a
//! cut rebinds host tools over the *same* tree, so the tail of a resumed
//! run sees the writes its head committed. One tree lives per cell name
//! for the duration of a test ([`lease`]); a second test of the same cell
//! in the same process waits for it.
//!
//! Rows and their disposition (`tests/README.md`, the family's grid):
//!
//! | Row | Cell | Disposition |
//! |---|---|---|
//! | 1 | [`LONG_UNARY`], [`LONG_STREAMED`] | recorded |
//! | 2 | [`PARALLEL_CALLS`] | recorded (a live batch of four reads; one missing path) |
//! | 3 | [`BIG_RESULT`] | recorded (`run_tests` emits a 48 KiB report) |
//! | 4 | [`TOOL_ERROR_MIDWAY`] | recorded (`run_tests` errs once, the model re-runs) |
//! | 4 | [`INVALID_ARGS_MIDWAY`] | scripted: a live model cannot be steered into malformed arguments; the row-1 unary recording is served by the sequenced transport with turn k's arguments rewritten ([`UnaryShape::with_invalid_args`]) |
//! | 4 | [`PROVIDER_FAULT_MIDWAY`] | scripted, world-only: the row-1 unary recording with a retryable status inserted before turn k; rig-agent has no retry budget (CONTRACT §5) |
//! | 5 | [`MAX_TURNS_MIDWAY`], [`OUTPUT_CAP_MIDWAY`] | recorded |
//! | 6–8 | row 1 through `resume_after` cuts, the usage table, `despawn_run` | reuse |
#![allow(dead_code, reason = "long-loop cells run on five provider columns")]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock, Mutex};

use rig_agent::completion::Usage;

use rig_core::effect::EffectFamily;

use rig_core::effect::EffectKind;

use rig_core::effect::EffectRecord;

use rig_core::effect::Outcome;

use rig_cassette::effect_log::EffectLog;

use rig_core::error::ErrorKind;

use rig_core::message::AssistantContent;

use rig_core::message::Message;

use rig_core::message::UserContent;

use rig_core::tool::Tool;

use rig_core::tool::ToolContext;

use rig_core::tool::ToolErrorKind;

use rig_core::tool::ToolExecutionError;

use serde::{Deserialize, Serialize};

use super::cells::{CELL, Cell, ThinkingWire, ToolKind};
use super::corpus::{Ending, Program};

// -- programs -----------------------------------------------------------------

pub(crate) const PREAMBLE: &str = "You are a careful software engineer working in a small Rust repository through tools. Follow the requested tool protocol exactly: one tool call per model turn unless the protocol says otherwise. Never invent a tool result and never repeat a tool result's text. Your final answer must be exactly the single word done, given only after run_tests reports PASS.";
pub(crate) const LOOP_PROMPT: &str = "The crate's test suite fails. Make it pass. Protocol: (1) call list_files; (2) call read_file on README.md; (3) call read_file on tests/basic.rs; (4) call read_file on src/lib.rs; (5) call run_tests to see the failing assertion; (6) call write_file on the one file the failing assertion points at, with that file's complete corrected contents, changing only the returned value; (7) call run_tests again. Repeat steps 6 and 7 until run_tests reports PASS, then answer with the single word done.";
/// Row 2: four reads in one batch, one of them on a path that does not
/// exist, so one result in the batch is an error; the batch reads every
/// source file up front, so the two fixes need no further reads.
pub(crate) const PARALLEL_PROMPT: &str = "The crate's test suite fails. Make it pass. Protocol: first, in one single model turn, call read_file four times in parallel on tests/basic.rs, src/lib.rs, src/calendar.rs and docs/DESIGN.md (docs/DESIGN.md may be missing; if read_file reports an error for it, ignore it and never retry it). Then call run_tests. Then call write_file on the one file the failing assertion points at, with that file's complete corrected contents, changing only the returned value. Then call run_tests again. Repeat the write and run_tests until run_tests reports PASS, then answer with the single word done.";
/// Row 3: the same protocol; `run_tests` appends a 48 KiB verbose log.
pub(crate) const BIG_PROMPT: &str = "The crate's test suite fails. Make it pass. run_tests appends a long verbose runner log to its report; never repeat, quote or summarize that log. Protocol: (1) call list_files; (2) call read_file on README.md; (3) call read_file on tests/basic.rs; (4) call read_file on src/lib.rs; (5) call run_tests to see the failing assertion; (6) call write_file on the one file the failing assertion points at, with that file's complete corrected contents, changing only the returned value; (7) call run_tests again. Repeat steps 6 and 7 until run_tests reports PASS, then answer with the single word done.";
/// Row 4: `run_tests` fails transiently on its first call; the protocol
/// asks for a plain re-run.
pub(crate) const TOOL_ERROR_PROMPT: &str = "The crate's test suite fails. Make it pass. Protocol: (1) call list_files; (2) call read_file on README.md; (3) call read_file on tests/basic.rs; (4) call read_file on src/lib.rs; (5) call run_tests to see the failing assertion; if run_tests reports a transient runner failure, call run_tests again before doing anything else; (6) call write_file on the one file the failing assertion points at, with that file's complete corrected contents, changing only the returned value; (7) call run_tests again. Repeat steps 6 and 7 until run_tests reports PASS, then answer with the single word done.";

/// A long loop is at least this many tool turns (rows 1, 3, 4): even a
/// loop that batches the three reads into one turn has, with two bugs,
/// list, reads, run, write, run, write, run = 7.
pub(crate) const MIN_TOOL_TURNS: usize = 6;
/// Row 5's model-call budget: the protocol's third call is still a read.
pub(crate) const MAX_TURNS_BUDGET: usize = 3;
/// Row 5's output cap: enough for a `list_files`/`read_file` call (~10–17
/// output tokens on every column), not for the `write_file` call that
/// carries the whole corrected file (~52 tokens on gpt-4.1-mini; the first
/// round's 64 never cut it).
pub(crate) const OUTPUT_CAP_TOKENS: u64 = 32;
/// Row 4's scripted faults land at this tool turn (1-based): after the
/// first two turns, with history behind them.
pub(crate) const FAULT_TURN: usize = 3;

const BASE: Program = Program {
    preamble: Some(PREAMBLE),
    prompt: LOOP_PROMPT,
    temperature: Some(0.0),
    max_tokens: Some(2048),
    max_turns: Some(12),
    tool_concurrency: Some(3),
    ..Program::DEFAULT
};

/// The repository toolset, in registration order.
pub(crate) const REPO_TOOLS: &[ToolKind] = &[
    ToolKind::RepoListFiles,
    ToolKind::RepoReadFile,
    ToolKind::RepoWriteFile,
    ToolKind::RepoRunTests,
];

pub(crate) const LONG_UNARY: Cell = Cell {
    name: "long_loop_long_unary",
    program: BASE,
    tools: REPO_TOOLS,
    events: true,
    ..CELL
};
pub(crate) const LONG_STREAMED: Cell = Cell {
    name: "long_loop_long_streamed",
    program: Program {
        streamed: true,
        ..BASE
    },
    ..LONG_UNARY
};
pub(crate) const PARALLEL_CALLS: Cell = Cell {
    name: "long_loop_parallel_calls",
    program: Program {
        prompt: PARALLEL_PROMPT,
        ..BASE
    },
    ..LONG_UNARY
};
pub(crate) const BIG_RESULT: Cell = Cell {
    name: "long_loop_big_result",
    program: Program {
        prompt: BIG_PROMPT,
        ..BASE
    },
    ..LONG_UNARY
};
pub(crate) const TOOL_ERROR_MIDWAY: Cell = Cell {
    name: "long_loop_tool_error_midway",
    program: Program {
        prompt: TOOL_ERROR_PROMPT,
        ..BASE
    },
    ..LONG_UNARY
};
/// Scripted (see the module doc): the row-1 unary recording served by the
/// sequenced transport, the [`FAULT_TURN`]th tool call's arguments
/// rewritten to the wrong type. The adapter answers an `invalid_args`
/// error result, the tool never runs, the loop goes on.
pub(crate) const INVALID_ARGS_MIDWAY: Cell = Cell {
    name: "long_loop_invalid_args_midway",
    ..LONG_UNARY
};
/// Scripted, world-only (see the module doc): a retryable status before
/// the [`FAULT_TURN`]th completion, re-issued under the default budget with
/// the same history, the loop going on over the same run.
pub(crate) const PROVIDER_FAULT_MIDWAY: Cell = Cell {
    name: "long_loop_provider_fault_midway",
    ..LONG_UNARY
};
pub(crate) const MAX_TURNS_MIDWAY: Cell = Cell {
    name: "long_loop_max_turns_midway",
    program: Program {
        max_turns: Some(MAX_TURNS_BUDGET),
        ending: Ending::MaxTurns,
        ..BASE
    },
    ..LONG_UNARY
};
/// Row 5's output-cap cell with the ending the column's wire gives a
/// length-cut turn under `max_tokens: 32`. How a cut turn ends, and at
/// which turn the cap bites, are per-wire facts of the model and rig-core's
/// decoders (round 3 recordings), so each column names its own:
///
/// * [`Ending::Failed`]`(Response)`: the run fails at the capped
///   completion. OpenAI Chat and DeepSeek: the shared chat decoder
///   (`deserialize_choices_dropping_incomplete_tool_calls`) drops a
///   length-cut tool call whose arguments do not parse, the turn carries
///   neither text nor a call, and rig-agent refuses it ("produced no
///   answer and stopped with finish_reason=Length") after 1–2 tool turns.
///   OpenAI Responses: the same, at its own cut. Gemini: HTTP 200 with
///   `finishReason: "MALFORMED_FUNCTION_CALL"` and no content on the very
///   first request (a 10-token `list_files` call does not fit), which
///   rig-core decodes as a response error — the record itself is `Err`.
/// * [`Ending::Answer`]: Anthropic opens with a text preamble, the cap
///   cuts it at turn 1 (`stop_reason: max_tokens`), and rig settles the
///   run `Ok(text)` with a `Length` finish; nothing dispatched, no fix.
pub(crate) const fn output_cap_cell(ending: Ending) -> Cell {
    Cell {
        name: "long_loop_output_cap_midway",
        program: Program {
            max_tokens: Some(OUTPUT_CAP_TOKENS),
            ending,
            ..BASE
        },
        provider_retries: Some(0),
        ..LONG_UNARY
    }
}
/// [`output_cap_cell`] on the wires whose cut turn fails the run (OpenAI
/// Chat, OpenAI Responses, Gemini, DeepSeek).
pub(crate) const OUTPUT_CAP_MIDWAY: Cell = output_cap_cell(Ending::Failed(ErrorKind::Response));
/// [`output_cap_cell`] on Anthropic, whose cut text preamble is the answer
/// under a `Length` finish.
pub(crate) const OUTPUT_CAP_MIDWAY_LENGTH_ANSWER: Cell = output_cap_cell(Ending::Answer);

/// The cells whose cassette is a live recording of this family.
pub(crate) const RECORDED: &[&Cell] = &[
    &LONG_UNARY,
    &LONG_STREAMED,
    &PARALLEL_CALLS,
    &BIG_RESULT,
    &TOOL_ERROR_MIDWAY,
    &MAX_TURNS_MIDWAY,
    &OUTPUT_CAP_MIDWAY,
];

pub(crate) fn is_long_loop(cell: &Cell) -> bool {
    cell.name.starts_with("long_loop_")
}

/// The provider directory and cassette scenario a long-loop cell's live
/// recording lives under on `thinking`'s wire (`crates/rig-cassette/fixtures/cassettes/<provider>/
/// long_loop_matrix[_chat|_responses]/<cell>.yaml`). The per-wire file
/// spells the same literal at its wrapper call site for the census.
pub(crate) fn scenario(thinking: ThinkingWire, cell: &Cell) -> (&'static str, String) {
    let suffix = cell
        .name
        .strip_prefix("long_loop_")
        .expect("a long-loop cell");
    match thinking {
        ThinkingWire::Anthropic => ("anthropic", format!("long_loop_matrix/{suffix}")),
        ThinkingWire::OpenAiChat => ("openai", format!("long_loop_matrix_chat/{suffix}")),
        ThinkingWire::OpenAiResponses => ("openai", format!("long_loop_matrix_responses/{suffix}")),
        ThinkingWire::Gemini => ("gemini", format!("long_loop_matrix/{suffix}")),
        ThinkingWire::DeepSeek => ("deepseek", format!("long_loop_matrix/{suffix}")),
        other => panic!("no long-loop column on the {other:?} wire"),
    }
}

/// The tool turns of `cell`'s live recording on `thinking`'s wire: one
/// completion per recorded interaction, the last of which answers.
pub(crate) fn recorded_tool_turns(thinking: ThinkingWire, cell: &Cell) -> usize {
    let (provider, scenario) = scenario(thinking, cell);
    let interactions = crate::cassettes::recorded_statuses_and_bodies(provider, &scenario).len();
    assert!(
        interactions > MIN_TOOL_TURNS,
        "{}: the recording holds {interactions} interactions; a long loop has at least {} tool turns and an answer",
        cell.name,
        MIN_TOOL_TURNS
    );
    interactions - 1
}

// -- the repository -----------------------------------------------------------

/// The committed fixture: five small ASCII files. `src/lib.rs` and
/// `src/calendar.rs` each return one wrong value; `tests/basic.rs` says
/// which ones are right, one test per function, in [`FIXES`]' order.
pub(crate) const FIXTURE: [(&str, &str); 5] = [
    (
        "Cargo.toml",
        "[package]\nname = \"repo_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    ),
    (
        "README.md",
        "# repo_fixture\n\nA tiny library crate with two functions, `days_in_year` (src/lib.rs) and\n`days_in_week` (src/calendar.rs), and one test for each.\nRun the test suite with the run_tests tool; the runner stops at the first\nfailing test.\n",
    ),
    (
        "src/calendar.rs",
        "/// The number of days in a week.\npub fn days_in_week() -> u32 {\n    8\n}\n",
    ),
    (
        "src/lib.rs",
        "pub mod calendar;\n\n/// The number of days in a common (non-leap) year.\npub fn days_in_year() -> u32 {\n    360\n}\n",
    ),
    (
        "tests/basic.rs",
        "use repo_fixture::calendar::days_in_week;\nuse repo_fixture::days_in_year;\n\n#[test]\nfn a_common_year_has_365_days() {\n    assert_eq!(days_in_year(), 365);\n}\n\n#[test]\nfn a_week_has_7_days() {\n    assert_eq!(days_in_week(), 7);\n}\n",
    ),
];

/// One of the fixture's bugs: the file and function that return the wrong
/// value, the test that asserts on it, and the value it expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fix {
    pub(crate) path: &'static str,
    pub(crate) function: &'static str,
    pub(crate) test: &'static str,
    /// The `tests/basic.rs` line the assertion is on.
    pub(crate) line: usize,
    pub(crate) value: u64,
}

/// The two bugs, in the order the runner's tests run: the first is
/// reported until it is fixed, then the second.
pub(crate) const FIXES: [Fix; 2] = [
    Fix {
        path: "src/lib.rs",
        function: "days_in_year",
        test: "a_common_year_has_365_days",
        line: 6,
        value: 365,
    },
    Fix {
        path: "src/calendar.rs",
        function: "days_in_week",
        test: "a_week_has_7_days",
        line: 11,
        value: 7,
    },
];
/// The first bug, the one every loop hits first (and the cut rows' tests).
pub(crate) const FIRST_FIX: Fix = FIXES[0];
pub(crate) const MISSING_PATH: &str = "docs/DESIGN.md";
/// `run_tests`'s transient failure (row 4), an `Other` tool error.
pub(crate) const TRANSIENT_RUNNER_FAILURE: &str = "transient runner failure: the test runner crashed before running any test; call run_tests again";
pub(crate) const PASS_LINE: &str = "run_tests: PASS";
pub(crate) const FAIL_LINE: &str = "run_tests: FAIL";
const BIG_REPORT_HEADER: &str = "\n--- verbose runner log (49152 bytes) ---\n";

/// One tool invocation, as the tree saw it: the tool, its parsed
/// arguments, and what it answered (the error's message for `Err`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Invocation {
    pub(crate) tool: &'static str,
    pub(crate) args: serde_json::Value,
    pub(crate) output: Result<String, String>,
}

/// What a cell switches on the tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RepoConfig {
    /// `run_tests` appends the 48 KiB filler (row 3).
    big_report: bool,
    /// `run_tests` errs on this 1-based call (row 4).
    transient_failure_at: Option<usize>,
}

impl RepoConfig {
    fn of(cell: &Cell) -> Self {
        Self {
            big_report: cell.name == BIG_RESULT.name,
            transient_failure_at: (cell.name == TOOL_ERROR_MIDWAY.name).then_some(1),
        }
    }
}

struct Repo {
    files: BTreeMap<String, String>,
    config: RepoConfig,
    invocations: Vec<Invocation>,
    run_tests_calls: usize,
}

/// The tree the four tools share, and the assertions read.
#[derive(Clone)]
pub(crate) struct RepoHandle(Arc<Mutex<Repo>>);

impl RepoHandle {
    fn fresh(config: RepoConfig) -> Self {
        Self(Arc::new(Mutex::new(Repo {
            files: FIXTURE
                .iter()
                .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
                .collect(),
            config,
            invocations: Vec::new(),
            run_tests_calls: 0,
        })))
    }

    fn with<T>(&self, f: impl FnOnce(&mut Repo) -> T) -> T {
        let mut repo = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut repo)
    }

    pub(crate) fn invocations(&self) -> Vec<Invocation> {
        self.with(|repo| repo.invocations.clone())
    }

    pub(crate) fn file(&self, path: &str) -> Option<String> {
        self.with(|repo| repo.files.get(path).cloned())
    }

    /// The last `run_tests` verdict: `Some(true)` for PASS.
    pub(crate) fn last_verdict(&self) -> Option<bool> {
        self.with(|repo| {
            repo.invocations
                .iter()
                .rev()
                .find(|invocation| invocation.tool == RunTests::NAME)
                .and_then(|invocation| invocation.output.as_ref().ok())
                .map(|report| report.starts_with(PASS_LINE))
        })
    }

    fn record(
        &self,
        tool: &'static str,
        args: serde_json::Value,
        output: Result<String, ToolExecutionError>,
    ) -> Result<String, ToolExecutionError> {
        self.with(|repo| {
            repo.invocations.push(Invocation {
                tool,
                args,
                output: match &output {
                    Ok(text) => Ok(text.clone()),
                    Err(error) => Err(error.message().to_owned()),
                },
            });
        });
        output
    }
}

/// The value `fix.function` returns: the first integer literal after the
/// function's opening brace in `fix.path`.
pub(crate) fn returned_value(files: &BTreeMap<String, String>, fix: &Fix) -> Result<u64, String> {
    let Fix { path, function, .. } = fix;
    let source = files
        .get(*path)
        .ok_or_else(|| format!("error[E0583]: file not found for module: {path} is missing"))?;
    let at = source
        .find(&format!("fn {function}"))
        .ok_or_else(|| format!("error[E0425]: cannot find function `{function}` in `{path}`"))?;
    let body = source[at..]
        .split_once('{')
        .map(|(_, body)| body)
        .ok_or_else(|| format!("error: expected `{{` after `fn {function}`"))?;
    let digits: String = body
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .map_err(|_| format!("error[E0308]: mismatched types: `{function}` returns no integer"))
}

/// The report `run_tests` answers for a tree: a pure function of it. The
/// runner stops at the first failing test, so the report names one
/// assertion at a time; `PASS` only once every [`FIXES`] value is right.
pub(crate) fn test_report(files: &BTreeMap<String, String>) -> (bool, String) {
    let compile_error = || {
        let lib = files.get(FIRST_FIX.path)?;
        (!lib.contains("mod calendar")).then(|| {
            "error[E0432]: unresolved import `repo_fixture::calendar`: could not find `calendar` in `repo_fixture` (src/lib.rs declares no `pub mod calendar;`)".to_owned()
        })
    };
    let mut values = Vec::with_capacity(FIXES.len());
    for fix in &FIXES {
        match returned_value(files, fix) {
            Ok(value) => values.push(value),
            Err(compile_error) => {
                return (
                    false,
                    format!(
                        "{FAIL_LINE}\n{compile_error}\n\nerror: could not compile `repo_fixture` (lib)\n"
                    ),
                );
            }
        }
    }
    if let Some(compile_error) = compile_error() {
        return (
            false,
            format!(
                "{FAIL_LINE}\n{compile_error}\n\nerror: could not compile `repo_fixture` (lib)\n"
            ),
        );
    }
    let total = FIXES.len();
    let mut lines = format!("running {total} tests\n");
    for (n, (fix, value)) in FIXES.iter().zip(&values).enumerate() {
        if *value == fix.value {
            lines.push_str(&format!("test {} ... ok\n", fix.test));
            continue;
        }
        let not_run = total - n - 1;
        lines.push_str(&format!(
            "test {test} ... FAILED\n(the runner stops at the first failing test; {not_run} later test(s) did not run)\n\nfailures:\n\n---- {test} stdout ----\nthread '{test}' panicked at tests/basic.rs:{line}:5:\nassertion `left == right` failed\n  left: {value}\n right: {expected}\nnote: `{function}` is defined in {path}\n\ntest result: FAILED. {n} passed; 1 failed; 0 ignored; {not_run} not run\n",
            test = fix.test,
            line = fix.line,
            expected = fix.value,
            function = fix.function,
            path = fix.path,
        ));
        return (false, format!("{FAIL_LINE}\n{lines}"));
    }
    lines.push_str(&format!(
        "\ntest result: ok. {total} passed; 0 failed; 0 ignored\n"
    ));
    (true, format!("{PASS_LINE}\n{lines}"))
}

/// Row 3's report: the ordinary one and the checkpoint row's filler, one
/// representation on every wire.
pub(crate) fn big_report(report: &str) -> String {
    let mut text = report.to_owned();
    text.push_str(BIG_REPORT_HEADER);
    text.push_str(&super::checkpoint::large_result());
    text
}

/// No arguments; unknown fields are refused so that the scripted
/// invalid-args rewrite (`{"path": 42}`) is `invalid_args` on every tool,
/// including `run_tests` when a batched read turn makes it the
/// [`FAULT_TURN`]th call.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoArgs {}
#[derive(Deserialize, Serialize)]
pub(crate) struct PathArgs {
    pub(crate) path: String,
}
#[derive(Deserialize, Serialize)]
pub(crate) struct WriteArgs {
    pub(crate) path: String,
    pub(crate) content: String,
}

fn no_args_schema() -> serde_json::Value {
    serde_json::json!({"type":"object","properties":{},"additionalProperties":false})
}

pub(crate) struct ListFiles(pub RepoHandle);
impl Tool for ListFiles {
    const NAME: &'static str = "list_files";
    type Error = ToolExecutionError;
    type Args = NoArgs;
    type Output = String;
    fn description(&self) -> String {
        "List every file in the repository, one path per line.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        no_args_schema()
    }
    async fn call(&self, _: &mut ToolContext, _: NoArgs) -> Result<String, Self::Error> {
        let listing = self.0.with(|repo| {
            repo.files
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join("\n")
        });
        self.0
            .record(Self::NAME, serde_json::json!({}), Ok(listing))
    }
}

pub(crate) struct ReadFile(pub RepoHandle);
impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Error = ToolExecutionError;
    type Args = PathArgs;
    type Output = String;
    fn description(&self) -> String {
        "Read one repository file and return its complete text. Errors when the path does not exist.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"path":{"type":"string","description":"The file's path relative to the repository root, e.g. src/lib.rs."}},"required":["path"],"additionalProperties":false})
    }
    async fn call(&self, _: &mut ToolContext, args: PathArgs) -> Result<String, Self::Error> {
        let output = self
            .0
            .with(|repo| repo.files.get(&args.path).cloned())
            .ok_or_else(|| ToolExecutionError::not_found(format!("no such file: {}", args.path)));
        self.0
            .record(Self::NAME, serde_json::json!({"path": args.path}), output)
    }
}

pub(crate) struct WriteFile(pub RepoHandle);
impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolExecutionError;
    type Args = WriteArgs;
    type Output = String;
    fn description(&self) -> String {
        "Replace one repository file's complete text (or create it). Always send the whole file."
            .into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"path":{"type":"string","description":"The file's path relative to the repository root."},"content":{"type":"string","description":"The complete new text of the file."}},"required":["path","content"],"additionalProperties":false})
    }
    async fn call(&self, _: &mut ToolContext, args: WriteArgs) -> Result<String, Self::Error> {
        let bytes = args.content.len();
        self.0.with(|repo| {
            repo.files.insert(args.path.clone(), args.content.clone());
        });
        self.0.record(
            Self::NAME,
            serde_json::json!({"path": args.path, "content": args.content}),
            Ok(format!("wrote {bytes} bytes to {}", args.path)),
        )
    }
}

pub(crate) struct RunTests(pub RepoHandle);
impl Tool for RunTests {
    const NAME: &'static str = "run_tests";
    type Error = ToolExecutionError;
    type Args = NoArgs;
    type Output = String;
    fn description(&self) -> String {
        "Run the crate's test suite and return the runner's report. The first line is run_tests: PASS or run_tests: FAIL. The runner stops at the first failing test and reports only that assertion; later tests run only once it passes.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        no_args_schema()
    }
    async fn call(&self, _: &mut ToolContext, _: NoArgs) -> Result<String, Self::Error> {
        let output = self.0.with(|repo| {
            repo.run_tests_calls += 1;
            if repo.config.transient_failure_at == Some(repo.run_tests_calls) {
                return Err(ToolExecutionError::other(TRANSIENT_RUNNER_FAILURE));
            }
            let (_, report) = test_report(&repo.files);
            Ok(if repo.config.big_report {
                big_report(&report)
            } else {
                report
            })
        });
        self.0.record(Self::NAME, serde_json::json!({}), output)
    }
}

// -- the lease: one tree per cell, per test ------------------------------------

struct Slot {
    lease: Arc<tokio::sync::Mutex<()>>,
    repo: Option<RepoHandle>,
}

static SLOTS: LazyLock<Mutex<HashMap<&'static str, Slot>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The tree a test holds for its cell: fresh at acquisition, shared by
/// every tool bound under the cell's name until the lease drops (the head
/// world's and the restored world's alike).
pub(crate) struct Lease {
    pub(crate) repo: RepoHandle,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

pub(crate) async fn lease(cell: &Cell) -> Lease {
    let mutex = {
        let mut slots = SLOTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slots
            .entry(cell.name)
            .or_insert_with(|| Slot {
                lease: Arc::new(tokio::sync::Mutex::new(())),
                repo: None,
            })
            .lease
            .clone()
    };
    let guard = mutex.lock_owned().await;
    let repo = RepoHandle::fresh(RepoConfig::of(cell));
    SLOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_mut(cell.name)
        .expect("the slot was created above")
        .repo = Some(repo.clone());
    Lease {
        repo,
        _guard: guard,
    }
}

/// The leased tree of `cell`: the tools bind over it (`agent::typed_tool`,
/// `world::open_inner`).
pub(crate) fn repo(cell: &Cell) -> RepoHandle {
    SLOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(cell.name)
        .and_then(|slot| slot.repo.clone())
        .unwrap_or_else(|| panic!("{}: no leased repository; run the cell through long_loop::run_agent or long_loop_world::run_world", cell.name))
}

// -- the record ---------------------------------------------------------------

/// One model turn of the record: its completion and the tool records the
/// interpreter dispatched for it.
pub(crate) struct Turn<'a> {
    pub(crate) completion: &'a EffectRecord,
    pub(crate) tools: Vec<&'a EffectRecord>,
}

pub(crate) fn turns(log: &EffectLog) -> Vec<Turn<'_>> {
    let mut turns: Vec<Turn<'_>> = Vec::new();
    for record in &log.records {
        match record.kind.family() {
            EffectFamily::Completion => turns.push(Turn {
                completion: record,
                tools: Vec::new(),
            }),
            EffectFamily::Tool => turns
                .last_mut()
                .expect("a tool record follows its completion")
                .tools
                .push(record),
            other => panic!("a long-loop record is a completion or a tool, not {other:?}"),
        }
    }
    turns
}

/// The calls a completion record's response asked for, in order.
pub(crate) fn requested_calls(record: &EffectRecord) -> Vec<(String, serde_json::Value)> {
    match &record.outcome {
        Ok(Outcome::Completion(response)) => response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => {
                    Some((call.function.name.clone(), call.function.arguments.clone()))
                }
                _ => None,
            })
            .collect(),
        Ok(other) => panic!("a completion record answers a completion, not {other:?}"),
        Err(_) => Vec::new(),
    }
}

/// The ids of the calls a completion record's response asked for, in
/// call order: the `i`-th id belongs to the `i`-th record of `turn.tools`
/// (`assert_log` step 1 pins the two orders together).
pub(crate) fn requested_call_ids(record: &EffectRecord) -> Vec<&rig_core::message::ToolCallId> {
    match &record.outcome {
        Ok(Outcome::Completion(response)) => response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(&call.id),
                _ => None,
            })
            .collect(),
        Ok(other) => panic!("a completion record answers a completion, not {other:?}"),
        Err(_) => Vec::new(),
    }
}

pub(crate) fn dispatched_call(record: &EffectRecord) -> (String, serde_json::Value) {
    match &record.kind {
        EffectKind::ToolCall { name, args } => (
            name.clone(),
            serde_json::from_str(args).expect("dispatched tool arguments are JSON"),
        ),
        other => panic!("a tool record, not {other:?}"),
    }
}

pub(crate) fn dispatched_result(record: &EffectRecord) -> &rig_core::tool::ToolResult {
    match &record.outcome {
        Ok(Outcome::ToolResult { result }) => result,
        other => panic!("the tool outcome is published: {other:?}"),
    }
}

pub(crate) fn request_history(record: &EffectRecord) -> &[Message] {
    match &record.kind {
        EffectKind::Completion { request, .. } => &request.chat_history,
        other => panic!("a completion record, not {other:?}"),
    }
}

fn usage(record: &EffectRecord) -> Option<&Usage> {
    match &record.outcome {
        Ok(Outcome::Completion(response)) => Some(&response.usage),
        _ => None,
    }
}

/// The wire's own report of a completion's prompt-side usage, read off the
/// record's raw payload: `(input, cached_input, cache_creation)` in the
/// dialect's own fields, `None` where the dialect has no such field or the
/// wire omitted it. Usage is what the adapter parsed *from* this payload,
/// so the two must agree on every record.
pub(crate) fn raw_usage(
    thinking: ThinkingWire,
    record: &EffectRecord,
) -> (Option<u64>, Option<u64>, Option<u64>) {
    let Ok(Outcome::Completion(response)) = &record.outcome else {
        panic!("a completion record")
    };
    let raw = &response.raw;
    let count = |value: &serde_json::Value| value.as_u64();
    match thinking {
        ThinkingWire::Anthropic => {
            let usage = &raw["usage"];
            assert!(
                usage.is_object(),
                "the Anthropic record keeps its usage: {raw}"
            );
            (
                count(&usage["input_tokens"]),
                count(&usage["cache_read_input_tokens"]),
                count(&usage["cache_creation_input_tokens"]),
            )
        }
        ThinkingWire::OpenAiChat => {
            let usage = &raw["usage"];
            assert!(usage.is_object(), "the Chat record keeps its usage: {raw}");
            (
                count(&usage["prompt_tokens"]),
                count(&usage["prompt_tokens_details"]["cached_tokens"]),
                None,
            )
        }
        ThinkingWire::DeepSeek => {
            let usage = &raw["usage"];
            assert!(
                usage.is_object(),
                "the DeepSeek record keeps its usage: {raw}"
            );
            (
                count(&usage["prompt_tokens"]),
                count(&usage["prompt_cache_hit_tokens"]),
                None,
            )
        }
        ThinkingWire::OpenAiResponses => {
            let usage = &raw["usage"];
            assert!(
                usage.is_object(),
                "the Responses record keeps its usage: {raw}"
            );
            (
                count(&usage["input_tokens"]),
                count(&usage["input_tokens_details"]["cached_tokens"]),
                None,
            )
        }
        ThinkingWire::Gemini => {
            let usage = if raw["usageMetadata"].is_object() {
                &raw["usageMetadata"]
            } else {
                &raw["usage_metadata"]
            };
            assert!(
                usage.is_object(),
                "the Gemini record keeps its usage: {raw}"
            );
            (
                count(&usage["promptTokenCount"]),
                count(&usage["cachedContentTokenCount"]),
                None,
            )
        }
        other => panic!("no long-loop column on the {other:?} wire"),
    }
}

/// The wire's cache accounting, for the usage table (row 7):
/// `cache_conformance::CacheAccounting` by dialect.
fn prompt_tokens(thinking: ThinkingWire, usage: &Usage) -> u64 {
    let input = usage.input_tokens.unwrap_or(0);
    match thinking {
        // Anthropic reports reads and writes *beside* `input_tokens`.
        ThinkingWire::Anthropic => {
            input
                + usage.cached_input_tokens.unwrap_or(0)
                + usage.cache_creation_input_tokens.unwrap_or(0)
        }
        _ => input,
    }
}

/// The shape a cell's record must have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// The protocol to `done`: at least [`MIN_TOOL_TURNS`] tool turns (a
    /// batched read turn is legal; call/dispatch correlation is what is
    /// pinned).
    Long,
    /// Row 2: the first turn is a batch of four reads, one an error.
    Parallel,
    /// Row 4: a tool answered `invalid_args` at [`FAULT_TURN`].
    InvalidArgs,
    /// Row 4: a provider error record before [`FAULT_TURN`]'s completion.
    ProviderFault,
    /// Row 5: exactly [`MAX_TURNS_BUDGET`] completions, no answer.
    MaxTurns,
    /// Row 5: the last completion is the cut, possibly the first; nothing
    /// ran after it and the tree is untouched; the ending is the column's
    /// ([`output_cap_cell`]), the cut's finish reason recorded.
    OutputCap,
}

impl Shape {
    fn of(cell: &Cell) -> Self {
        match cell.name {
            "long_loop_long_unary"
            | "long_loop_long_streamed"
            | "long_loop_big_result"
            | "long_loop_tool_error_midway" => Self::Long,
            "long_loop_parallel_calls" => Self::Parallel,
            "long_loop_invalid_args_midway" => Self::InvalidArgs,
            "long_loop_provider_fault_midway" => Self::ProviderFault,
            "long_loop_max_turns_midway" => Self::MaxTurns,
            "long_loop_output_cap_midway" => Self::OutputCap,
            other => panic!("not a long-loop cell: {other}"),
        }
    }
}

/// Assert the actual loop: turn count, call/result correlation, history
/// growth, usage, the tree's invocations and final state, the answer.
/// Producer/native effect-log parity is the caller's independent gate.
pub(crate) fn assert_log(cell: &Cell, thinking: ThinkingWire, log: &EffectLog) {
    let shape = Shape::of(cell);
    let repo = repo(cell);
    let turns = turns(log);
    assert!(!turns.is_empty(), "{}: at least one completion", cell.name);

    // (1) every call is dispatched, in order, within its turn.
    let mut dispatched: Vec<(String, serde_json::Value)> = Vec::new();
    let mut provider_errors = Vec::new();
    for (n, turn) in turns.iter().enumerate() {
        let requested = requested_calls(turn.completion);
        if turn.completion.outcome.is_err() {
            provider_errors.push(n);
            assert!(
                turn.tools.is_empty(),
                "{}: a failed completion dispatches nothing",
                cell.name
            );
            continue;
        }
        let calls: Vec<_> = turn
            .tools
            .iter()
            .map(|record| dispatched_call(record))
            .collect();
        assert_eq!(
            calls, requested,
            "{}: turn {n} dispatches exactly the calls the model made, in call order",
            cell.name
        );
        dispatched.extend(calls);
    }
    let tool_turns = turns.iter().filter(|turn| !turn.tools.is_empty()).count();

    // (2) the history grows by one assistant and one tool-result utterance
    // per tool turn, and every request carries the whole prior history
    // behind the preamble (every wire folds it in as `chat_history[0]`, a
    // `Message::System`) and the prompt.
    let mut previous: Option<&[Message]> = None;
    let mut tool_turns_before = 0;
    let mut last_tool_turn: Option<&Turn<'_>> = None;
    for (n, turn) in turns.iter().enumerate() {
        let history = request_history(turn.completion);
        assert_eq!(
            history.len(),
            2 + 2 * tool_turns_before,
            "{}: request {n} carries the preamble, the prompt and two utterances per completed tool turn",
            cell.name
        );
        assert!(
            matches!(&history[0], Message::System { content } if content == PREAMBLE),
            "{}: request {n}'s first message is the preamble",
            cell.name
        );
        assert!(
            matches!(&history[1], Message::User { content } if content.iter().any(|part| matches!(part, UserContent::Text(text) if text.text == cell.program.prompt))),
            "{}: request {n}'s second message is the prompt",
            cell.name
        );
        if let Some(previous) = previous {
            assert_eq!(
                serde_json::to_value(&history[..previous.len()]).expect("history serializes"),
                serde_json::to_value(previous).expect("history serializes"),
                "{}: request {n} extends request {}'s history",
                cell.name,
                n - 1
            );
            if history.len() > previous.len() {
                let assistant = &history[previous.len()];
                let results = &history[previous.len() + 1];
                assert!(
                    matches!(assistant, Message::Assistant { content, .. } if content.iter().any(|part| matches!(part, AssistantContent::ToolCall(_)))),
                    "{}: the turn's assistant utterance carries its calls",
                    cell.name
                );
                assert!(
                    matches!(results, Message::User { content } if content.iter().all(|part| matches!(part, UserContent::ToolResult(_))) && !content.is_empty()),
                    "{}: the turn's tool-result utterance carries only results",
                    cell.name
                );
                // The utterance *is* the turn's executed results: one part
                // per tool record, in call order (a parallel batch completes
                // in scheduling order, but is replayed in call order),
                // each answering its record's call id with the record's
                // committed output byte for byte — row 3's requirement
                // that the 48 KiB result reaches the next request intact.
                let source = last_tool_turn.expect("the history grew after a completed tool turn");
                let parts: Vec<&rig_core::message::ToolResult> = match results {
                    Message::User { content } => content
                        .iter()
                        .filter_map(|part| match part {
                            UserContent::ToolResult(result) => Some(result),
                            _ => None,
                        })
                        .collect(),
                    other => panic!("{}: a tool-result utterance, not {other:?}", cell.name),
                };
                let ids = requested_call_ids(source.completion);
                assert_eq!(
                    parts.len(),
                    source.tools.len(),
                    "{}: request {n}'s tool-result utterance carries one result per executed call",
                    cell.name
                );
                for (i, ((part, record), id)) in
                    parts.iter().zip(&source.tools).zip(&ids).enumerate()
                {
                    let (name, _) = dispatched_call(record);
                    assert_eq!(
                        &part.call, *id,
                        "{}: request {n}'s result {i} ({name}) answers the turn's {i}th call, in call order",
                        cell.name
                    );
                    let replayed = rig_core::tool::ToolOutput::content(part.content.clone())
                        .expect("a tool-result part carries content")
                        .render();
                    assert_eq!(
                        replayed,
                        dispatched_result(record).output().render(),
                        "{}: request {n}'s result {i} ({name}) is the committed result, byte for byte",
                        cell.name
                    );
                }
            }
        }
        previous = Some(history);
        if !turn.tools.is_empty() && turn.completion.outcome.is_ok() {
            tool_turns_before += 1;
            last_tool_turn = Some(turn);
        }
    }

    // (3) usage: every answered completion's usage is the wire's own report
    // of it (the oracle the cache rules below stand on); the prompt side
    // never shrinks (row 7's table, printed and kept as evidence).
    let mut table = Vec::new();
    let mut last_prompt = 0;
    for (n, turn) in turns.iter().enumerate() {
        let Some(usage) = usage(turn.completion) else {
            continue;
        };
        let (input, cached, created) = raw_usage(thinking, turn.completion);
        assert_eq!(
            (
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens
            ),
            (input, cached, created),
            "{}: turn {n}'s usage is the wire's report",
            cell.name
        );
        let prompt = prompt_tokens(thinking, usage);
        assert!(prompt > 0, "{}: turn {n} billed prompt tokens", cell.name);
        assert!(
            prompt >= last_prompt,
            "{}: turn {n}'s prompt ({prompt}) is not smaller than the previous ({last_prompt}): the history only appends",
            cell.name
        );
        last_prompt = prompt;
        table.push(UsageRow {
            turn: n,
            prompt,
            input: usage.input_tokens.unwrap_or(0),
            cached_input: usage.cached_input_tokens.unwrap_or(0),
            cache_creation: usage.cache_creation_input_tokens.unwrap_or(0),
            output: usage.output_tokens.unwrap_or(0),
        });
    }
    eprintln!(
        "LONG_LOOP_USAGE {}",
        serde_json::json!({"cell": cell.name, "wire": format!("{thinking:?}"), "turns": table})
    );
    write_usage_evidence(cell, thinking, &table);
    assert_cache_table(cell, thinking, &table);

    // (4) the tree saw exactly the dispatched calls, and answered them
    // with what the record holds.
    let invocations = repo.invocations();
    let executed: Vec<_> = turns
        .iter()
        .flat_map(|turn| turn.tools.iter().copied())
        .filter(|record| !dispatched_result(record).is_error_kind(ToolErrorKind::InvalidArgs))
        .collect();
    assert_eq!(
        invocations.len(),
        executed.len(),
        "{}: the tree ran every dispatched call once and nothing else: {invocations:?}",
        cell.name
    );
    // Turn order is pinned; within a turn, a batch under concurrency
    // reaches the tree in scheduling order, so each turn's calls are
    // compared as a set.
    let mut seen: Vec<(String, serde_json::Value)> = invocations
        .iter()
        .map(|invocation| (invocation.tool.to_owned(), invocation.args.clone()))
        .collect();
    let mut expected: Vec<_> = executed
        .iter()
        .map(|record| dispatched_call(record))
        .collect();
    let mut start = 0;
    for turn in &turns {
        let batch = turn
            .tools
            .iter()
            .filter(|record| !dispatched_result(record).is_error_kind(ToolErrorKind::InvalidArgs))
            .count();
        let end = (start + batch).min(seen.len());
        let key =
            |call: &(String, serde_json::Value)| serde_json::to_string(call).expect("serializes");
        seen[start..end].sort_by_key(key);
        expected[start..end].sort_by_key(key);
        start = end;
    }
    assert_eq!(seen, expected, "{}: the tree's invocations", cell.name);
    // Each record consumes the first still-unmatched invocation with its
    // tool and arguments: a repeated `run_tests` with identical arguments
    // has a different output each time, and a parallel batch reaches the
    // tree in scheduling order.
    let mut consumed = vec![false; invocations.len()];
    for record in &executed {
        let (name, args) = dispatched_call(record);
        let result = dispatched_result(record);
        let index = invocations
            .iter()
            .enumerate()
            .position(|(i, invocation)| {
                !consumed[i] && invocation.tool == name && invocation.args == args
            })
            .expect("matched above");
        consumed[index] = true;
        let invocation = &invocations[index];
        match &invocation.output {
            Ok(output) => {
                assert!(
                    !result.is_error(),
                    "{}: {name} succeeded: {result:?}",
                    cell.name
                );
                assert_eq!(
                    result.output().render(),
                    *output,
                    "{}: {name}'s committed result is the tree's answer, byte for byte",
                    cell.name
                );
            }
            Err(message) => {
                assert!(
                    result.is_error(),
                    "{}: {name} failed: {result:?}",
                    cell.name
                );
                assert_eq!(
                    result.error().map(ToolExecutionError::message),
                    Some(message.as_str()),
                    "{}: {name}'s error is the tree's",
                    cell.name
                );
            }
        }
    }

    // (5) the shape. The answer is read only where the program ends in one:
    // `golden_answer` expects the log's last completion to be `Ok`, which a
    // `Failed(..)` ending does not promise (Gemini's cut record is `Err`).
    let answer = || super::corpus::golden_answer(log);
    let ends_done = || {
        let answer = answer();
        assert!(
            answer.trim().to_ascii_lowercase().contains("done"),
            "{}: the model answers done, not {answer:?}",
            cell.name
        );
        assert_eq!(
            repo.last_verdict(),
            Some(true),
            "{}: run_tests last reported PASS",
            cell.name
        );
        let files = repo.with(|repo| repo.files.clone());
        for fix in &FIXES {
            assert_eq!(
                returned_value(&files, fix),
                Ok(fix.value),
                "{}: {} in {} is fixed",
                cell.name,
                fix.function,
                fix.path
            );
        }
        let (passes, _) = test_report(&files);
        assert!(passes, "{}: the tree holds both fixes", cell.name);
        assert!(
            turns.last().expect("a turn").tools.is_empty(),
            "{}: the answering turn calls nothing",
            cell.name
        );
    };
    match shape {
        Shape::Long => {
            assert!(
                tool_turns >= MIN_TOOL_TURNS,
                "{}: a long loop has at least {MIN_TOOL_TURNS} tool turns, not {tool_turns}",
                cell.name
            );
            assert!(provider_errors.is_empty());
            if cell.name == TOOL_ERROR_MIDWAY.name {
                let failed: Vec<_> = invocations
                    .iter()
                    .filter(|invocation| invocation.output.is_err())
                    .collect();
                assert_eq!(failed.len(), 1, "{}: run_tests erred once", cell.name);
                assert_eq!(
                    failed[0].output.as_ref().err().map(String::as_str),
                    Some(TRANSIENT_RUNNER_FAILURE)
                );
            } else {
                assert!(
                    invocations
                        .iter()
                        .all(|invocation| invocation.output.is_ok()),
                    "{}: no tool erred: {invocations:?}",
                    cell.name
                );
            }
            if cell.name == BIG_RESULT.name {
                let reports: Vec<_> = invocations
                    .iter()
                    .filter(|invocation| invocation.tool == RunTests::NAME)
                    .collect();
                assert!(!reports.is_empty());
                for report in reports {
                    let text = report.output.as_ref().expect("run_tests answered");
                    assert!(
                        text.len() > 49152 && text.ends_with(&super::checkpoint::large_result()),
                        "{}: the report carries the whole 48 KiB filler ({} bytes)",
                        cell.name,
                        text.len()
                    );
                }
            }
            ends_done();
        }
        Shape::Parallel => {
            let batch = &turns[0];
            assert_eq!(
                batch.tools.len(),
                4,
                "{}: the first turn reads four files at once",
                cell.name
            );
            let reads: Vec<_> = batch
                .tools
                .iter()
                .map(|record| dispatched_call(record))
                .collect();
            assert!(reads.iter().all(|(name, _)| name == ReadFile::NAME));
            let missing: Vec<_> = batch
                .tools
                .iter()
                .filter(|record| dispatched_result(record).is_error())
                .collect();
            assert_eq!(missing.len(), 1, "{}: one read fails", cell.name);
            assert_eq!(
                dispatched_call(missing[0]).1,
                serde_json::json!({"path": MISSING_PATH}),
                "{}: the missing path's read is the error",
                cell.name
            );
            assert!(dispatched_result(missing[0]).is_error_kind(ToolErrorKind::NotFound));
            assert!(
                turns[1..].iter().all(|turn| turn.tools.len() <= 1),
                "{}: only the first turn batches",
                cell.name
            );
            ends_done();
        }
        Shape::InvalidArgs => {
            let invalid: Vec<_> = turns
                .iter()
                .enumerate()
                .filter(|(_, turn)| {
                    turn.tools.iter().any(|record| {
                        dispatched_result(record).is_error_kind(ToolErrorKind::InvalidArgs)
                    })
                })
                .map(|(n, _)| n)
                .collect();
            assert_eq!(
                invalid,
                [FAULT_TURN - 1],
                "{}: the {FAULT_TURN}th call's arguments were refused, nothing else",
                cell.name
            );
            assert!(
                turns.len() > FAULT_TURN,
                "{}: the loop went on after the refused call",
                cell.name
            );
            assert!(provider_errors.is_empty());
        }
        Shape::ProviderFault => {
            assert_eq!(
                provider_errors,
                [FAULT_TURN - 1],
                "{}: one provider error record, before the {FAULT_TURN}th completion",
                cell.name
            );
            let failed = turns[FAULT_TURN - 1].completion;
            let retried = turns
                .get(FAULT_TURN)
                .expect("the completion is re-issued")
                .completion;
            let report = failed.outcome.as_ref().expect_err("the provider's error");
            assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
            assert!(report.retryable, "a retryable status: {report:?}");
            assert!(retried.outcome.is_ok(), "the re-issued completion answers");
            assert_eq!(
                serde_json::to_value(request_history(failed)).expect("serializes"),
                serde_json::to_value(request_history(retried)).expect("serializes"),
                "{}: the retry carries the same whole history",
                cell.name
            );
            assert!(
                tool_turns >= MIN_TOOL_TURNS,
                "{}: the loop went on to its end: {tool_turns} tool turns",
                cell.name
            );
            ends_done();
        }
        Shape::MaxTurns => {
            assert_eq!(
                turns.len(),
                MAX_TURNS_BUDGET,
                "{}: exactly the budget's completions",
                cell.name
            );
            assert!(provider_errors.is_empty());
            assert!(
                turns
                    .iter()
                    .all(|turn| !requested_calls(turn.completion).is_empty()),
                "{}: the model was still calling tools when the budget ran out",
                cell.name
            );
            assert_ne!(
                repo.last_verdict(),
                Some(true),
                "{}: the fix never landed within the budget",
                cell.name
            );
        }
        Shape::OutputCap => {
            // The cut is the last record; it may be the very first one
            // (Gemini and Anthropic, round 3), so no minimum number of tool
            // turns is asked of this row. Nothing runs after the cut and the
            // tree is exactly the fixture: no write ever executed.
            let cut = turns.len() - 1;
            let last = turns.last().expect("a turn");
            assert!(
                last.tools.is_empty(),
                "{}: nothing is dispatched for the capped completion",
                cell.name
            );
            assert_ne!(
                repo.last_verdict(),
                Some(true),
                "{}: the fix never landed under the cap",
                cell.name
            );
            assert!(
                !invocations
                    .iter()
                    .any(|invocation| invocation.tool == WriteFile::NAME),
                "{}: no write executed under the cap: {invocations:?}",
                cell.name
            );
            let files = repo.with(|repo| repo.files.clone());
            let fixture: BTreeMap<String, String> = FIXTURE
                .iter()
                .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
                .collect();
            assert_eq!(files, fixture, "{}: the tree is unchanged", cell.name);
            let (passes, _) = test_report(&files);
            assert!(!passes, "{}: the tree still fails", cell.name);
            let evidence = match cell.program.ending {
                Ending::Failed(ErrorKind::Response) => {
                    // The run failed with kind `Response` (the producer's
                    // `expect_ending` / the world's `assert_ending` pin the
                    // run's own error). The cut record is either the wire's
                    // response error (Gemini: HTTP 200, `finishReason:
                    // "MALFORMED_FUNCTION_CALL"`, no content — rig-core's
                    // `function_call_finish_reason_error` makes the record
                    // `Err(kind: response)`), or an `Ok` completion that
                    // decoded to neither text nor a whole tool call under a
                    // truncating finish (OpenAI Chat and DeepSeek: the chat
                    // decoder drops the cut call; OpenAI Responses), which
                    // rig-agent then refuses as "produced no answer and
                    // stopped with finish_reason=Length" after the record.
                    // The finish reason is recorded, not pinned: which one a
                    // wire reports, and at which turn the cap bites (turn 1
                    // on Gemini), is the wire's fact.
                    match &last.completion.outcome {
                        Err(report) => {
                            assert_eq!(
                                report.kind,
                                ErrorKind::Response,
                                "{}: the cut record is the wire's response error: {report:?}",
                                cell.name
                            );
                            assert_eq!(
                                provider_errors,
                                [cut],
                                "{}: the cut record is the only failed completion",
                                cell.name
                            );
                            serde_json::json!({
                                "cut_turn": cut,
                                "outcome": "err",
                                "report": serde_json::to_value(report).expect("the report serializes"),
                            })
                        }
                        Ok(Outcome::Completion(response)) => {
                            assert!(
                                provider_errors.is_empty(),
                                "{}: no completion failed: {provider_errors:?}",
                                cell.name
                            );
                            assert!(
                                response.choice.iter().all(|part| match part {
                                    AssistantContent::ToolCall(_) => false,
                                    AssistantContent::Text(text) => text.text.is_empty(),
                                    _ => true,
                                }),
                                "{}: the cut turn carries neither text nor a whole call (rig-agent fails a run that answers nothing under a truncating finish): {:?}",
                                cell.name,
                                response.choice
                            );
                            serde_json::json!({
                                "cut_turn": cut,
                                "outcome": "ok",
                                "finish_reason": response.finish_reason().map(|reason| format!("{reason:?}")),
                                "raw_finish": raw_finish_reason(thinking, &response.raw),
                            })
                        }
                        Ok(other) => panic!("{}: a completion record, not {other:?}", cell.name),
                    }
                }
                Ending::Answer => {
                    // Anthropic: a text preamble under `max_tokens: 32`, cut
                    // at turn 1 (`stop_reason: max_tokens`, round 3); rig
                    // settles the run `Ok` with a `Length` finish, the text
                    // its answer.
                    assert!(provider_errors.is_empty());
                    let Ok(Outcome::Completion(response)) = &last.completion.outcome else {
                        panic!("{}: the capped completion answered", cell.name)
                    };
                    assert_eq!(
                        response.finish_reason(),
                        Some(rig_agent::completion::FinishReason::Length),
                        "{}: the last completion was cut by the cap",
                        cell.name
                    );
                    assert!(
                        response
                            .choice
                            .iter()
                            .all(|part| !matches!(part, AssistantContent::ToolCall(_))),
                        "{}: the cut call was dropped, not dispatched",
                        cell.name
                    );
                    assert_ne!(
                        answer().trim().to_ascii_lowercase(),
                        "done",
                        "{}: the cut turn's text is not the protocol's answer",
                        cell.name
                    );
                    serde_json::json!({
                        "cut_turn": cut,
                        "outcome": "ok",
                        "finish_reason": "Length",
                        "raw_finish": raw_finish_reason(thinking, &response.raw),
                    })
                }
                other => panic!("{}: not an output-cap ending: {other:?}", cell.name),
            };
            write_output_cap_evidence(cell, thinking, &evidence);
        }
    }

    if cell.program.streamed {
        assert_stream_delivery(cell, &turns);
    }
}

/// One answered completion's usage, as row 7's table holds it: `prompt` is
/// the wire's prompt-token denominator ([`prompt_tokens`]), the rest the
/// adapter's `Usage` (asserted equal to the wire's raw report in
/// `assert_log` step 3, which is what makes these rows an oracle).
#[derive(Clone, Copy, Debug, Serialize)]
struct UsageRow {
    turn: usize,
    prompt: u64,
    input: u64,
    cached_input: u64,
    cache_creation: u64,
    output: u64,
}

/// Any prompt above this many tokens is large enough for every automatic
/// prefix cache on the columns (OpenAI's documented minimum is 1,024;
/// Gemini 2.5 Flash's implicit cache engages from 1,024): a loop that
/// grows past it and never reads a single cached token is a rewritten
/// prefix, not provider variance.
const AUTOMATIC_CACHE_PROMPT_TOKENS: u64 = 2048;

/// Row 7: the usage table's per-wire cache facts.
///
/// The conformance harness's ratio rule (`assert_agent_growth_still_hits`,
/// "once ≥ 80% hit, every later turn ≥ 80%") is arithmetically wrong for
/// this loop and is not used here: after a 48 KiB `run_tests` report is
/// appended, the next request's cacheable prefix is at most the previous
/// prompt, ~51–67% of the new one (round 3: OpenAI Chat turn 9 read 16,256
/// of 31,854 = 51.0%, Responses 16,256/31,833 = 51.1%, Gemini's last turn
/// 68,824/103,022 = 66.8%, DeepSeek r10 32,512/48,044); DeepSeek and Gemini
/// account hits at coarse granularity (78.7–79.9% on `tool_error_midway`
/// across every attempt, deterministic per history); and the Gemini and
/// OpenAI caches are best-effort (a 0 between two 1,024-token hits on
/// Responses `long_streamed`, hits vanishing and returning on Gemini).
/// What every wire does promise, and what is pinned:
///
/// * every wire: `cached_input ≤ prompt` on every turn, the usage table's
///   rows being the adapter's `Usage` already asserted equal to the wire's
///   raw report (step 3) — the oracle;
/// * Anthropic: this program sets no `cache_control`, so cache reads and
///   cache creation are exactly 0 on every turn (round 3: "cached/creation
///   counts are 0 on every request" across all 57 recorded requests);
/// * DeepSeek: a documented deterministic prefix cache in 64-token blocks —
///   `prompt_cache_hit_tokens` is non-decreasing across turns once it is
///   positive (equality allowed: 1536, 1536, 1664, 1792 on
///   `tool_error_midway`);
///   turn 1 is 0 or a hit left by an earlier cell on the same prefix
///   (round 3: 512 on every first request), so nothing is asserted of it;
/// * OpenAI Chat, OpenAI Responses, Gemini: best-effort automatic caching —
///   once any prompt exceeds [`AUTOMATIC_CACHE_PROMPT_TOKENS`], at least one
///   turn read a cached prefix (`cached_input > 0`); nothing about
///   monotonicity or ratios (Gemini `long_streamed` hit on turns {2, 5, 9,
///   10, 11}, {4, 8, 9}, {8, 11} on three otherwise token-identical
///   attempts). A loop that never grows past the threshold is printed, not
///   asserted (`max_turns_midway`, the capped cells).
fn assert_cache_table(cell: &Cell, thinking: ThinkingWire, rows: &[UsageRow]) {
    for row in rows {
        assert!(
            row.cached_input <= row.prompt,
            "{}: turn {} read {} cached tokens of a {}-token prompt",
            cell.name,
            row.turn,
            row.cached_input,
            row.prompt
        );
    }
    match thinking {
        ThinkingWire::Anthropic => {
            for row in rows {
                assert_eq!(
                    (row.cached_input, row.cache_creation),
                    (0, 0),
                    "{}: turn {} — the program sets no cache_control, so Anthropic neither reads nor creates a cache",
                    cell.name,
                    row.turn
                );
            }
        }
        ThinkingWire::DeepSeek => {
            let mut engaged: Option<(usize, u64)> = None;
            for row in rows {
                if let Some((since, floor)) = engaged {
                    assert!(
                        row.cached_input >= floor,
                        "{}: turn {} read {} cached tokens, fewer than turn {}'s {}; DeepSeek's prefix cache is deterministic and the history only appends",
                        cell.name,
                        row.turn,
                        row.cached_input,
                        since,
                        floor
                    );
                }
                if row.cached_input > 0 {
                    engaged = Some((row.turn, row.cached_input));
                }
            }
        }
        ThinkingWire::OpenAiChat | ThinkingWire::OpenAiResponses | ThinkingWire::Gemini => {
            let largest = rows.iter().map(|row| row.prompt).max().unwrap_or(0);
            if largest > AUTOMATIC_CACHE_PROMPT_TOKENS {
                assert!(
                    rows.iter().any(|row| row.cached_input > 0),
                    "{}: the prompt grew to {largest} tokens and no turn read a cached prefix; the wire's automatic cache never engaged on a history that only appends: {rows:?}",
                    cell.name
                );
            } else {
                eprintln!(
                    "LONG_LOOP_CACHE {}: the prompt never exceeded {AUTOMATIC_CACHE_PROMPT_TOKENS} tokens ({largest}); the best-effort cache is not asked for a hit",
                    cell.name
                );
            }
        }
        other => panic!("no long-loop column on the {other:?} wire"),
    }
}

/// The wire's own finish reason of an answered completion, read off the
/// record's raw payload for the output-cap evidence; `null` where the
/// dialect keeps it elsewhere.
fn raw_finish_reason(thinking: ThinkingWire, raw: &serde_json::Value) -> serde_json::Value {
    match thinking {
        ThinkingWire::Anthropic => raw["stop_reason"].clone(),
        ThinkingWire::OpenAiChat | ThinkingWire::DeepSeek => {
            raw["choices"][0]["finish_reason"].clone()
        }
        ThinkingWire::OpenAiResponses => serde_json::json!({
            "status": raw["status"],
            "incomplete_details": raw["incomplete_details"],
        }),
        ThinkingWire::Gemini => raw["candidates"][0]["finishReason"].clone(),
        other => panic!("no long-loop column on the {other:?} wire"),
    }
}

/// The streamed cells: one terminal per completion, every dispatched call
/// delivered as a completed tool block with actual delta or block-start
/// delivery, the answer's text streamed.
fn assert_stream_delivery(cell: &Cell, turns: &[Turn<'_>]) {
    use rig_core::streaming::BlockKind;

    use rig_core::streaming::Delta;

    use rig_core::streaming::StreamEvent;

    for (n, turn) in turns.iter().enumerate() {
        assert!(matches!(
            turn.completion.kind,
            EffectKind::Completion { stream: true, .. }
        ));
        let events = turn
            .completion
            .events
            .as_ref()
            .expect("actual provider stream delivery is retained");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Final(_)))
                .count(),
            1,
            "{}: one actual terminal for completion {n}",
            cell.name
        );
        let delivered: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::BlockEnd {
                    id,
                    block: Some(AssistantContent::ToolCall(_)),
                    ..
                } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(
            delivered.len(),
            turn.tools.len(),
            "{}: completion {n} delivered one completed tool block per dispatched call",
            cell.name
        );
        for id in &delivered {
            assert!(events.iter().any(|event| matches!(event,
                StreamEvent::BlockStart { id: started, kind: BlockKind::ToolCall }
                | StreamEvent::BlockDelta { id: started, delta: Delta::ToolName { .. } | Delta::ToolArguments { .. } }
                if started == *id
            )), "{}: the completed call has actual matching tool-block or delta delivery", cell.name);
        }
        if turn.tools.is_empty() && turn.completion.outcome.is_ok() {
            assert!(events.iter().any(|event| matches!(event, StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if !text.is_empty())), "{}: actual answer text was streamed", cell.name);
        }
    }
}

/// The transcript both interpreters hold after settlement (the producer's
/// `PromptResponse::messages`, the world's utterances): the prompt, two
/// utterances per completed tool turn, and the answer where there is one.
/// Unlike a request's `chat_history` (`assert_log` step 2), neither
/// transcript holds the preamble: it is the agent's, not an utterance.
pub(crate) fn assert_transcript(cell: &Cell, log: &EffectLog, history: &[Message]) {
    let turns = turns(log);
    let tool_turns = turns
        .iter()
        .filter(|turn| !turn.tools.is_empty() && turn.completion.outcome.is_ok())
        .count();
    let answered = cell.program.ending == Ending::Answer;
    let expected = 1 + 2 * tool_turns + usize::from(answered);
    let roles: Vec<_> = history
        .iter()
        .map(|message| match message {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
        })
        .collect();
    assert_eq!(
        history.len(),
        expected,
        "{}: the transcript is the prompt, two utterances per tool turn and the answer: {roles:?}",
        cell.name
    );
    for (n, role) in roles.iter().enumerate() {
        let expected = if n % 2 == 0 { "user" } else { "assistant" };
        assert_eq!(*role, expected, "{}: utterance {n}'s role", cell.name);
    }
}

// -- evidence -----------------------------------------------------------------

fn attempt_directory() -> Option<std::path::PathBuf> {
    std::env::var_os("RIG_LONG_LOOP_ATTEMPT_DIR")
        .or_else(|| std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR"))
        .map(std::path::PathBuf::from)
}

/// Retain scrubbed records before semantic assertions, including rejected
/// live shapes (`RIG_LONG_LOOP_ATTEMPT_DIR`, or the checkpoint row's
/// `RIG_CHECKPOINT_ATTEMPT_DIR`).
pub(crate) fn write_attempt(cell: &Cell, log: &EffectLog) {
    let Some(directory) = attempt_directory() else {
        return;
    };
    let path = directory.join(format!("{}.effects.json", cell.name));
    std::fs::create_dir_all(&directory).expect("create evidence directory");
    let value = crate::cassettes::scrub_artifact(
        &serde_json::to_value(log).expect("effect log serializes"),
    );
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&value).expect("scrubbed log serializes"),
    )
    .expect("save attempt log");
    let invocations: Vec<_> = repo(cell)
        .invocations()
        .iter()
        .map(|invocation| {
            serde_json::json!({
                "tool": invocation.tool,
                "args_bytes": serde_json::to_string(&invocation.args).expect("serializes").len(),
                "output": match &invocation.output {
                    Ok(text) => serde_json::json!({"ok_bytes": text.len()}),
                    Err(message) => serde_json::json!({"error": message}),
                },
            })
        })
        .collect();
    std::fs::write(
        directory.join(format!("{}.invocations.json", cell.name)),
        serde_json::to_string_pretty(&invocations).expect("serializes"),
    )
    .expect("save invocation log");
    eprintln!("LONG_LOOP_ATTEMPT effects={}", path.display());
}

/// Row 7's usage table beside the attempt's records
/// (`<cell>.usage.json`), the rows `assert_cache_table` judged.
fn write_usage_evidence(cell: &Cell, thinking: ThinkingWire, rows: &[UsageRow]) {
    let Some(directory) = attempt_directory() else {
        return;
    };
    std::fs::create_dir_all(&directory).expect("create evidence directory");
    let value = serde_json::json!({
        "cell": cell.name, "wire": format!("{thinking:?}"), "turns": rows,
    });
    std::fs::write(
        directory.join(format!("{}.usage.json", cell.name)),
        serde_json::to_string_pretty(&value).expect("serializes"),
    )
    .expect("save usage table");
}

/// Row 5's cut record beside the attempt's records
/// (`<cell>.output_cap.json`): which turn the cap bit at and the wire's
/// own finish reason or error — recorded, not asserted.
fn write_output_cap_evidence(cell: &Cell, thinking: ThinkingWire, cut: &serde_json::Value) {
    let value = serde_json::json!({
        "cell": cell.name, "wire": format!("{thinking:?}"), "cut": cut,
    });
    eprintln!("LONG_LOOP_OUTPUT_CAP {value}");
    let Some(directory) = attempt_directory() else {
        return;
    };
    std::fs::create_dir_all(&directory).expect("create evidence directory");
    std::fs::write(
        directory.join(format!("{}.output_cap.json", cell.name)),
        serde_json::to_string_pretty(&value).expect("serializes"),
    )
    .expect("save output-cap evidence");
}

/// Retain the JSON scene and effect head used at a fresh-world cut, with
/// sizes and hashes of the originals and the scrubbed external copies.
pub(crate) fn write_cut_evidence(cell: &Cell, cut: usize, encoded_scene: &str, encoded_head: &str) {
    use sha2::{Digest, Sha256};
    let Some(directory) = attempt_directory() else {
        return;
    };
    std::fs::create_dir_all(&directory).expect("create cut evidence directory");
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let mut artifacts = Vec::new();
    for (kind, source) in [("scene", encoded_scene), ("head", encoded_head)] {
        let value: serde_json::Value = serde_json::from_str(source).expect("actual cut JSON");
        let scrubbed = crate::cassettes::scrub_artifact(&value);
        let bytes = serde_json::to_vec_pretty(&scrubbed).expect("scrubbed cut JSON");
        let filename = format!("{}-cut-{cut}.{kind}.json", cell.name);
        let path = directory.join(&filename);
        let text = std::str::from_utf8(&bytes).expect("JSON is UTF-8");
        assert!(
            crate::cassettes::artifact_safety_failures(&path, text).is_empty(),
            "external cut evidence contains no sensitive data"
        );
        std::fs::write(&path, &bytes).expect("write cut artifact");
        artifacts.push(serde_json::json!({
            "kind": kind, "file": filename,
            "source_bytes": source.len(), "source_sha256": hash(source.as_bytes()),
            "artifact_bytes": bytes.len(), "artifact_sha256": hash(&bytes),
            "scrubbed_value_changed": scrubbed != value,
        }));
    }
    let metadata = serde_json::json!({
        "cell": cell.name, "cut_tool_turns": cut, "streamed": cell.program.streamed,
        "test_thread": std::thread::current().name(),
        "tree_invocations_at_cut": repo(cell).invocations().len(),
        "restoration_input": "original encoded scene and head; external copies are scrubbed; the repository tree is host state rebound to the fresh world",
        "artifacts": artifacts,
    });
    let path = directory.join(format!("{}-cut-{cut}.metadata.json", cell.name));
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&metadata).expect("cut metadata JSON"),
    )
    .expect("write cut metadata");
    eprintln!("LONG_LOOP_CUT_EVIDENCE {}", path.display());
}

/// The fresh world's restore time beside its cut, once measured.
pub(crate) fn write_restore_timing(
    cell: &Cell,
    cut: usize,
    scene_bytes: usize,
    head_bytes: usize,
    restore_us: u128,
) {
    let line = serde_json::json!({
        "cell": cell.name, "cut": cut,
        "scene_bytes": scene_bytes, "head_bytes": head_bytes, "restore_us": restore_us,
    });
    eprintln!("LONG_LOOP_SCENE {line}");
    let Some(directory) = attempt_directory() else {
        return;
    };
    std::fs::create_dir_all(&directory).expect("create cut evidence directory");
    std::fs::write(
        directory.join(format!("{}-cut-{cut}.restore.json", cell.name)),
        serde_json::to_vec_pretty(&line).expect("serializes"),
    )
    .expect("write restore timing");
}

// -- the producer -------------------------------------------------------------

/// The first strict replay validates the program and saves complete
/// evidence before goldens exist (`LONG_LOOP_AUDIT_REPLAY`); it does not
/// claim golden parity. Otherwise the ordinary producer golden callback.
pub(crate) async fn run_agent<M: rig_agent::completion::CompletionModel + Clone + 'static>(
    wire: &super::Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let audit = std::env::var_os("LONG_LOOP_AUDIT_REPLAY").is_some();
    if audit {
        assert_eq!(
            std::env::var("RIG_PROVIDER_TEST_MODE").as_deref(),
            Ok("replay")
        );
        assert!(
            std::env::var_os("RIG_REGENERATE_GOLDEN").is_none(),
            "initial audit never generates goldens"
        );
        assert!(
            attempt_directory().is_some(),
            "initial replay must save evidence"
        );
    }
    let lease = lease(cell).await;
    let log = super::agent::run_agent(wire, cell, |log| {
        if audit {
            eprintln!(
                "LONG_LOOP_AUDIT_REPLAY {}: strict transport/program validation only; no golden parity claimed",
                cell.name
            );
        } else {
            golden(log);
        }
    })
    .await;
    drop(lease);
    log
}

/// A scripted cell: the rig-agent runner over one sequenced transport and
/// the world over another built the same way, each asserted against the
/// cell over its own fresh tree.
pub(crate) async fn run_scripted<M: rig_agent::completion::CompletionModel + Clone + 'static>(
    cell: &Cell,
    wire: impl Fn() -> super::Wire<M>,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let lease = lease(cell).await;
    super::agent::run_agent(&wire(), cell, |_| {}).await;
    drop(lease);
    super::long_loop_world::run_world(&wire(), cell, golden).await
}

// -- the scripted rows: the row-1 recording on the sequenced transport --------

/// The unary reply shape of a dialect, for the frame surgery the scripted
/// rows do on this wire's own row-1 recording (the streamed rows' analogue
/// is `stream_faults::SseShape`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnaryShape {
    /// Anthropic Messages: `content[].{type:"tool_use", input}`.
    Anthropic,
    /// OpenAI Chat Completions and every OpenAI-compatible wire (DeepSeek):
    /// `choices[0].message.tool_calls[].function.arguments`, a JSON string.
    Chat,
    /// OpenAI Responses: `output[].{type:"function_call", arguments}`, a
    /// JSON string.
    Responses,
    /// Gemini REST: `candidates[0].content.parts[].functionCall.args`.
    Gemini,
}

impl UnaryShape {
    pub(crate) fn of(thinking: ThinkingWire) -> Self {
        match thinking {
            ThinkingWire::Anthropic => Self::Anthropic,
            ThinkingWire::OpenAiChat | ThinkingWire::DeepSeek => Self::Chat,
            ThinkingWire::OpenAiResponses => Self::Responses,
            ThinkingWire::Gemini => Self::Gemini,
            other => panic!("no long-loop column on the {other:?} wire"),
        }
    }

    /// The recorded reply with its first tool call's arguments replaced by
    /// [`INVALID_ARGS`]: a `path` of the wrong type, which no repository
    /// tool's arguments accept.
    pub(crate) fn with_invalid_args(self, body: &str) -> String {
        let mut reply: serde_json::Value =
            serde_json::from_str(body).expect("a recorded JSON reply");
        let invalid = || serde_json::json!({"path": 42});
        let replaced = match self {
            Self::Anthropic => reply["content"]
                .as_array_mut()
                .expect("content")
                .iter_mut()
                .find(|part| part["type"] == "tool_use")
                .map(|part| part["input"] = invalid()),
            Self::Chat => reply["choices"][0]["message"]["tool_calls"]
                .as_array_mut()
                .expect("tool_calls")
                .first_mut()
                .map(|call| {
                    call["function"]["arguments"] = serde_json::Value::String(invalid().to_string())
                }),
            Self::Responses => reply["output"]
                .as_array_mut()
                .expect("output")
                .iter_mut()
                .find(|item| item["type"] == "function_call")
                .map(|item| item["arguments"] = serde_json::Value::String(invalid().to_string())),
            Self::Gemini => reply["candidates"][0]["content"]["parts"]
                .as_array_mut()
                .expect("parts")
                .iter_mut()
                .find(|part| part["functionCall"].is_object())
                .map(|part| part["functionCall"]["args"] = invalid()),
        };
        assert!(
            replaced.is_some(),
            "the recorded reply carries a tool call: {body}"
        );
        reply.to_string()
    }
}

/// The recorded unary replies of `cell`'s row-1 recording on `thinking`'s
/// wire, as the sequenced transport serves them, with the scripted row's
/// fault applied: turn [`FAULT_TURN`]'s arguments rewritten
/// (`INVALID_ARGS_MIDWAY`), or `fault_reply` inserted before turn
/// [`FAULT_TURN`]'s reply (`PROVIDER_FAULT_MIDWAY`).
pub(crate) fn scripted_replies(
    thinking: ThinkingWire,
    cell: &Cell,
    fault_reply: Option<rig_agent::test_utils::MockHttpResponse>,
) -> Vec<rig_agent::test_utils::MockHttpResponse> {
    use rig_agent::test_utils::MockHttpResponse;

    let (provider, scenario) = scenario(thinking, &LONG_UNARY);
    let recorded = crate::cassettes::recorded_statuses_and_bodies(provider, &scenario);
    assert!(
        recorded.len() > FAULT_TURN,
        "{}: the row-1 recording has a turn {FAULT_TURN} to fault",
        cell.name
    );
    let shape = UnaryShape::of(thinking);
    let mut replies = Vec::new();
    for (n, (status, body)) in recorded.into_iter().enumerate() {
        assert_eq!(status, 200, "{}: the row-1 recording succeeded", cell.name);
        if n + 1 == FAULT_TURN {
            match (cell.name, &fault_reply) {
                (name, None) if name == INVALID_ARGS_MIDWAY.name => {
                    replies.push(MockHttpResponse::success(shape.with_invalid_args(&body)));
                    continue;
                }
                (name, Some(reply)) if name == PROVIDER_FAULT_MIDWAY.name => {
                    replies.push(reply.clone());
                }
                (name, reply) => panic!(
                    "{name}: not a scripted long-loop cell, or the wrong fault reply: {reply:?}"
                ),
            }
        }
        replies.push(MockHttpResponse::success(body));
    }
    replies
}

// -- the negative matcher probe ------------------------------------------------

/// The last tool result's distinctive text, carried by the answering turn's
/// request: the passing report's summary line.
const PASS_MARKER: &str = "2 passed; 0 failed; 0 ignored";

/// Matcher sensitivity, deliberately separate from native continuation: send
/// the actual recorded request bodies to the real replay server, alter only
/// the last byte of the passing report inside the request that first carries
/// it, and prove the strict matcher refuses it while every original request
/// still matches. Positive native cells independently construct and send
/// their own requests.
pub(crate) async fn assert_request_body_rejected(provider: &'static str, scenario: &'static str) {
    use futures::FutureExt;
    use std::panic::AssertUnwindSafe;

    assert_eq!(
        crate::cassettes::CassetteMode::current(),
        crate::cassettes::CassetteMode::Replay,
        "negative matcher tests never record"
    );
    let path = crate::cassettes::cassette_path(provider, scenario);
    let yaml = std::fs::read_to_string(&path).expect("the live recording exists");
    let interactions: Vec<serde_json::Value> = serde_yaml::Deserializer::from_str(&yaml)
        .map(|document| serde_json::Value::deserialize(document).expect("recorded YAML document"))
        .collect();
    assert!(
        interactions.len() > MIN_TOOL_TURNS,
        "the complete recorded long loop: {} interactions",
        interactions.len()
    );
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    assert_eq!(bodies.len(), interactions.len());
    let target = bodies
        .iter()
        .position(|(request, _)| request.contains(PASS_MARKER))
        .expect("a request carries the passing report");
    assert!(target > 0, "the first request carries no tool result");
    let original = &bodies[target].0;
    assert_eq!(
        original.matches(PASS_MARKER).count(),
        1,
        "the request carries the passing report exactly once"
    );
    let start = original.find(PASS_MARKER).expect("the marker");
    let at = start + PASS_MARKER.len() - 1;
    let mut changed = original.as_bytes().to_vec();
    assert_eq!(changed[at], b'd');
    changed[at] = b'X';
    let changed = String::from_utf8(changed).expect("one ASCII substitution");
    assert_eq!(changed.len(), original.len());
    assert_eq!(
        changed
            .bytes()
            .zip(original.bytes())
            .filter(|(left, right)| left != right)
            .count(),
        1,
        "only the last result byte changed"
    );
    serde_json::from_str::<serde_json::Value>(&changed).expect("mutation preserves JSON syntax");

    let cassette = crate::cassettes::ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        provider,
        scenario,
        "https://long-loop.invalid",
    )
    .await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("local HTTP client");
    async fn post(
        client: &reqwest::Client,
        provider: &str,
        base: &str,
        interaction: &serde_json::Value,
        body: &str,
    ) -> (u16, String) {
        let request = &interaction["when"];
        assert_eq!(request["method"], "POST");
        let mut url = reqwest::Url::parse(base).expect("local replay URL");
        assert_eq!(
            url.host_str(),
            Some("127.0.0.1"),
            "only the real local replay server receives the probe"
        );
        url.set_path(request["path"].as_str().expect("recorded request path"));
        for query in request["query_param"]
            .as_array()
            .expect("recorded query parameters")
        {
            url.query_pairs_mut().append_pair(
                query["name"].as_str().expect("query name"),
                query["value"].as_str().expect("query value"),
            );
        }
        // Recordings intentionally omit credentials; satisfy the real replay
        // policy with fixed dummy headers, never credentials from the host.
        let mut builder = match provider {
            "anthropic" => client
                .post(url)
                .header("x-api-key", "long-loop-replay-only"),
            "openai" | "deepseek" => client
                .post(url)
                .header("authorization", "Bearer long-loop-replay-only"),
            "gemini" => client.post(url),
            other => panic!("not a long-loop provider: {other}"),
        };
        for header in request["header"].as_array().expect("recorded headers") {
            builder = builder.header(
                header["name"].as_str().expect("header name"),
                header["value"].as_str().expect("header value"),
            );
        }
        let response = builder
            .body(body.to_owned())
            .send()
            .await
            .expect("local replay response");
        let status = response.status().as_u16();
        let body = response.text().await.expect("consume local response body");
        (status, body)
    }
    let base = cassette.base_url();
    for (interaction, body) in interactions.iter().zip(&bodies).take(target) {
        let sent = post(&client, provider, &base, interaction, &body.0).await;
        assert_eq!(
            u64::from(sent.0),
            interaction["then"]["status"]
                .as_u64()
                .expect("recorded status"),
            "unmodified earlier request matches: {}",
            sent.1
        );
    }
    let rejected = post(&client, provider, &base, &interactions[target], &changed).await;
    assert_eq!(
        rejected.0, 404,
        "last-byte result mutation is rejected by real strict matching: {}",
        rejected.1
    );
    let diagnostic: serde_json::Value =
        serde_json::from_str(&rejected.1).expect("real matcher diagnostic");
    let candidate = &diagnostic["candidates"][target];
    assert_eq!(
        candidate["body_matches"], false,
        "the body specifically fails matching"
    );
    for field in [
        "method_matches",
        "path_matches",
        "query_matches",
        "headers_match",
        "required_headers_match",
    ] {
        assert_eq!(
            candidate[field], true,
            "all other request attributes match: {field}"
        );
    }
    for (interaction, body) in interactions.iter().zip(&bodies).skip(target) {
        let sent = post(&client, provider, &base, interaction, &body.0).await;
        assert_eq!(
            u64::from(sent.0),
            interaction["then"]["status"]
                .as_u64()
                .expect("recorded status"),
            "original request still matches; the miss consumed nothing: {}",
            sent.1
        );
    }
    let failed = AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await
        .expect_err("finish remembers the rejected mutation");
    let message = failed
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            failed
                .downcast_ref::<&str>()
                .map(|value| (*value).to_owned())
        })
        .expect("matcher failure message");
    assert!(
        message.contains("received unexpected replay request(s)"),
        "specific replay mismatch: {message}"
    );
    assert!(
        !message.contains("left unused interactions"),
        "all original requests were consumed: {message}"
    );
    eprintln!(
        "LONG_LOOP_NEGATIVE provider={provider} scenario={scenario} request_index={target} body_offset={at} changed_bytes=1 status=404 original_requests_consumed={}",
        interactions.len()
    );
}

/// Mutate the last tool-result continuation of the streamed loop.
pub(crate) async fn assert_stream_request_rejected(provider: &'static str, scenario: &'static str) {
    assert_request_body_rejected(provider, scenario).await;
}

/// The toolset's pure functions, runnable before any recording exists.
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> BTreeMap<String, String> {
        FIXTURE
            .iter()
            .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
            .collect()
    }

    #[test]
    fn the_fixture_fails_until_both_fixes_land() {
        let [first, second] = FIXES;
        let mut files = fixture();
        // The first bug alone is reported; the second test never ran.
        let (passes, report) = test_report(&files);
        assert!(!passes);
        assert!(report.starts_with(FAIL_LINE), "{report}");
        assert!(report.contains("left: 360"), "{report}");
        assert!(report.contains("tests/basic.rs:6:5"), "{report}");
        assert!(
            report.contains("note: `days_in_year` is defined in src/lib.rs"),
            "{report}"
        );
        assert!(
            report.contains("0 passed; 1 failed; 0 ignored; 1 not run"),
            "{report}"
        );
        assert!(!report.contains(second.test), "{report}");
        assert!(!report.contains("left: 8"), "{report}");
        // Fixing the second first changes nothing the model can see.
        let calendar_fixed =
            "/// The number of days in a week.\npub fn days_in_week() -> u32 {\n    7\n}\n";
        let mut early = files.clone();
        early.insert(second.path.to_owned(), calendar_fixed.to_owned());
        assert_eq!(test_report(&early), test_report(&files));
        // The first fix reveals the second bug.
        files.insert(
            first.path.to_owned(),
            "pub mod calendar;\n\npub fn days_in_year() -> u32 {\n    return 365;\n}\n".to_owned(),
        );
        let (passes, report) = test_report(&files);
        assert!(!passes);
        assert!(report.starts_with(FAIL_LINE), "{report}");
        assert!(
            report.contains("test a_common_year_has_365_days ... ok"),
            "{report}"
        );
        assert!(report.contains("left: 8"), "{report}");
        assert!(report.contains("tests/basic.rs:11:5"), "{report}");
        assert!(
            report.contains("note: `days_in_week` is defined in src/calendar.rs"),
            "{report}"
        );
        assert!(
            report.contains("1 passed; 1 failed; 0 ignored; 0 not run"),
            "{report}"
        );
        assert!(!report.contains(PASS_MARKER));
        // Both fixed: PASS, and the marker the negative probe mutates.
        files.insert(second.path.to_owned(), calendar_fixed.to_owned());
        let (passes, report) = test_report(&files);
        assert!(passes, "{report}");
        assert!(report.starts_with(PASS_LINE));
        assert_eq!(report.matches(PASS_MARKER).count(), 1, "{report}");
        assert!(
            report.ends_with("d\n"),
            "the probe flips the marker's last byte: {report}"
        );
        // The report is a pure function of the tree.
        assert_eq!(test_report(&files), test_report(&files.clone()));
        // Dropping `pub mod calendar;` from src/lib.rs is a compile error.
        let mut no_mod = files.clone();
        no_mod.insert(
            first.path.to_owned(),
            "pub fn days_in_year() -> u32 {\n    365\n}\n".to_owned(),
        );
        let (passes, report) = test_report(&no_mod);
        assert!(!passes);
        assert!(report.contains("E0432"), "{report}");
        // A missing file is a compile error.
        for fix in &FIXES {
            let mut missing = files.clone();
            missing.remove(fix.path);
            let (passes, report) = test_report(&missing);
            assert!(!passes);
            assert!(report.contains("missing"), "{report}");
            assert!(report.contains("could not compile"), "{report}");
        }
    }

    #[test]
    fn the_big_report_carries_the_checkpoint_filler_whole() {
        let (_, report) = test_report(&fixture());
        let big = big_report(&report);
        assert!(big.starts_with(&report));
        assert!(big.ends_with(&super::super::checkpoint::large_result()));
        assert_eq!(big.len(), report.len() + BIG_REPORT_HEADER.len() + 49152);
    }

    #[test]
    fn invalid_args_rewrite_each_dialects_first_call() {
        let cases = [
            (
                UnaryShape::Anthropic,
                r#"{"content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"src/lib.rs"}}]}"#,
                "/content/1/input",
                false,
            ),
            (
                UnaryShape::Chat,
                r#"{"choices":[{"message":{"tool_calls":[{"id":"call_1","function":{"name":"read_file","arguments":"{\"path\":\"src/lib.rs\"}"}}]}}]}"#,
                "/choices/0/message/tool_calls/0/function/arguments",
                true,
            ),
            (
                UnaryShape::Responses,
                r#"{"output":[{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"src/lib.rs\"}"}]}"#,
                "/output/0/arguments",
                true,
            ),
            (
                UnaryShape::Gemini,
                r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"src/lib.rs"}}}]}}]}"#,
                "/candidates/0/content/parts/0/functionCall/args",
                false,
            ),
        ];
        for (shape, body, pointer, stringified) in cases {
            let rewritten: serde_json::Value =
                serde_json::from_str(&shape.with_invalid_args(body)).expect("JSON");
            let args = rewritten.pointer(pointer).expect("the call's arguments");
            let args = if stringified {
                serde_json::from_str(args.as_str().expect("a JSON string")).expect("JSON")
            } else {
                args.clone()
            };
            assert_eq!(args, serde_json::json!({"path": 42}), "{shape:?}");
            assert!(
                serde_json::from_value::<PathArgs>(args.clone()).is_err(),
                "{shape:?}: no repository tool accepts the rewritten arguments"
            );
            assert!(
                serde_json::from_value::<WriteArgs>(args.clone()).is_err(),
                "{shape:?}: write_file refuses the rewritten arguments"
            );
            assert!(
                serde_json::from_value::<NoArgs>(args).is_err(),
                "{shape:?}: list_files/run_tests refuse the rewritten arguments too (a batched loop's {FAULT_TURN}th call is run_tests)"
            );
        }
    }

    #[tokio::test]
    async fn the_tools_log_every_invocation_over_one_tree() {
        let handle = RepoHandle::fresh(RepoConfig {
            big_report: false,
            transient_failure_at: Some(1),
        });
        let mut context = ToolContext::default();
        let listing = ListFiles(handle.clone())
            .call(&mut context, NoArgs {})
            .await
            .expect("listed");
        assert_eq!(
            listing,
            "Cargo.toml\nREADME.md\nsrc/calendar.rs\nsrc/lib.rs\ntests/basic.rs"
        );
        let missing = ReadFile(handle.clone())
            .call(
                &mut context,
                PathArgs {
                    path: MISSING_PATH.to_owned(),
                },
            )
            .await
            .expect_err("no such file");
        assert_eq!(missing.message(), format!("no such file: {MISSING_PATH}"));
        let transient = RunTests(handle.clone())
            .call(&mut context, NoArgs {})
            .await
            .expect_err("the first run errs");
        assert_eq!(transient.message(), TRANSIENT_RUNNER_FAILURE);
        assert_eq!(handle.last_verdict(), None);
        let report = RunTests(handle.clone())
            .call(&mut context, NoArgs {})
            .await
            .expect("the second run reports");
        assert!(report.starts_with(FAIL_LINE));
        assert_eq!(handle.last_verdict(), Some(false));
        WriteFile(handle.clone())
            .call(
                &mut context,
                WriteArgs {
                    path: FIRST_FIX.path.to_owned(),
                    content: "pub mod calendar;\npub fn days_in_year() -> u32 {\n    365\n}\n"
                        .to_owned(),
                },
            )
            .await
            .expect("written");
        let report = RunTests(handle.clone())
            .call(&mut context, NoArgs {})
            .await
            .expect("the third run reports the second bug");
        assert!(report.starts_with(FAIL_LINE));
        assert!(report.contains(FIXES[1].test), "{report}");
        assert_eq!(handle.last_verdict(), Some(false));
        WriteFile(handle.clone())
            .call(
                &mut context,
                WriteArgs {
                    path: FIXES[1].path.to_owned(),
                    content: "pub fn days_in_week() -> u32 {\n    7\n}\n".to_owned(),
                },
            )
            .await
            .expect("written");
        let report = RunTests(handle.clone())
            .call(&mut context, NoArgs {})
            .await
            .expect("the fourth run reports");
        assert!(report.starts_with(PASS_LINE));
        assert_eq!(handle.last_verdict(), Some(true));
        let invocations = handle.invocations();
        assert_eq!(
            invocations
                .iter()
                .map(|invocation| (invocation.tool, invocation.output.is_ok()))
                .collect::<Vec<_>>(),
            [
                ("list_files", true),
                ("read_file", false),
                ("run_tests", false),
                ("run_tests", true),
                ("write_file", true),
                ("run_tests", true),
                ("write_file", true),
                ("run_tests", true),
            ]
        );
    }
}
