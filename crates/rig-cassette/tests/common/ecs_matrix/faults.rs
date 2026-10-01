//! The grid's failure rows as data: every way a run ends badly, as a
//! [`Cell`] over the same `Program` table the happy paths use, plus the
//! fault the cell drives ([`Fault`]) and the scene it saves ([`Scene`]).
//! The per-wire file specialises a cell's fault with the wire's own facts
//! (the recorded status, the body's code) by struct update and serves a
//! recorded fault from its cassette. A scripted fault runs here, over the
//! sequenced transport and labelled frames the wire's [`Scripted`] names.

use rig_core::effect::EffectFamily;

use rig_core::effect::EffectFamily::Completion;

use rig_core::effect::EffectFamily::Memory as Mem;

use rig_core::effect::EffectFamily::Tool;

use rig_core::error::ErrorKind;

use super::Wire;
use super::cells::{
    BASIC_PREAMBLE, BASIC_PROMPT, CELL, Cell, ENDINGS_TOOL_OUTCOME_CANCELLED, Memory, READY_PROMPT,
    TOOLS_PREAMBLE, TWO_TOOL_STREAM_PREAMBLE, ToolKind,
};
use super::corpus::{CONVERSATION, Ending, Program};
use super::extra::{Cut, cancel_at};
use super::world::{run_scripted, run_world};
use crate::goldens::capture_world_programs;
use rig_cassette::effect_log::EffectLog;
use rig_core::http_client::DynHttpClient;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_test_support::stream_faults::{
    SseShape, recorded_sse_frames, scripted, sse_bytes, status_reply,
};

/// The fault a cell drives, and what the world must say about it beyond
/// the record (the record itself is the oracle's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The wire refuses the request before any frame: a recorded 4xx. The
    /// report carries the recorded status, the status table's verdict and
    /// the body's own code (`None` where the envelope is prose).
    Setup {
        status: u16,
        code: Option<&'static str>,
    },
    /// A retryable status served by the sequenced unary transport: the
    /// row-1 reply with its status rewritten. `retry_after` says the reply
    /// carries a `Retry-After` header the report must keep.
    Status {
        status: u16,
        code: Option<&'static str>,
        retry_after: bool,
    },
    /// The stream ends after text, before its terminal record.
    TruncatedAfterText,
    /// The stream ends after a complete tool call, before its terminal
    /// record: the tool never runs.
    TruncatedAfterToolCall,
    /// An error frame after text, with the facts the frame names as the
    /// funnel reports them (`SseShape::error_{code,message,status}`).
    ErrorAfterText {
        code: Option<&'static str>,
        message: &'static str,
        status: Option<u16>,
    },
    /// A provider refusal, in the wire's own shape: the ending says what
    /// the wire makes of it (the refusal text is the answer on OpenAI; the
    /// run fails `Provider` on Gemini).
    Refusal,
    /// The provider filtered the turn (`FinishReason::ContentFilter`): with
    /// text, the text is the answer and the reason is on the record; with
    /// none, the run fails as an answerless truncation.
    Filtered { with_text: bool },
    /// A granted tool answers `status: error`; the run answers around it.
    ToolError,
    /// The second of two calls in one turn errs.
    BatchSecondFails,
    /// `Cancelled` inserted while the tool child is in flight (the tool is
    /// parked until the driver releases it): the tool's record holds its
    /// real answer, nothing is committed, `despawn_run` is `InFlight`
    /// until the tool lands.
    StopWhileToolRuns,
    /// The store refuses the load: the run fails before any completion.
    FailingLoad,
}

/// The scene a cell saves (CONTRACT §13), beside its ordinary cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scene {
    None,
    /// Saved after `Failed`: the failure, the report and the history load
    /// in a fresh world and stay failed.
    AfterFailure,
    /// Saved while the completion stream is unfinished with observed
    /// progress: the load is refused before anything is spawned.
    MidStream,
    /// Saved while the tool is in flight with no stream progress: the load
    /// restarts the intent under its saved id and finishes to the golden.
    WithToolInFlight,
}

pub(crate) const BROKEN_ADD_PROMPT: &str = "Use the add tool to add 17 and 25. If the tool reports an error, do not call it again: reply with the error message it gave.";
pub(crate) const BROKEN_ORCHARD_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
If a tool reports an error, do not call it again. \
After both tools have answered, respond in one short sentence that includes the harbor label and the orchard tool's error text.";
/// The setup cells' request: the #2490 stream-fault cells' own.
pub(crate) const SETUP_PROMPT: &str = "Say hi.";
pub(crate) const SETUP_MAX_TOKENS: u64 = 16;
/// The orchard tool's error (`FailingOrchard`).
pub(crate) const BROKEN_ORCHARD: &str = "the orchard lookup is broken";

const C: &[EffectFamily] = &[Completion];
const CT: &[EffectFamily] = &[Completion, Tool];
const CTC: &[EffectFamily] = &[Completion, Tool, Completion];
const CTTC: &[EffectFamily] = &[Completion, Tool, Tool, Completion];
const CCCC: &[EffectFamily] = &[Completion, Completion, Completion, Completion];

const SETUP: Program = Program {
    // No system instruction at all: strict matching distinguishes it from
    // the corpus default's empty one.
    preamble: None,
    prompt: SETUP_PROMPT,
    max_tokens: Some(SETUP_MAX_TOKENS),
    ending: Ending::ProviderError,
    ..Program::DEFAULT
};
const TEXT_STREAM: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    streamed: true,
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: BROKEN_ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

// -- row 1: a setup failure ---------------------------------------------------

/// The wire refuses the request, unary. The per-wire file names the
/// recorded status and code; the run is saved after it failed and loaded.
pub(crate) const SETUP_UNARY: Cell = Cell {
    name: "fault_setup_unary",
    program: SETUP,
    families: C,
    fault: Some(Fault::Setup {
        status: 0,
        code: None,
    }),
    scene: Scene::AfterFailure,
    ..CELL
};
pub(crate) const SETUP_STREAMED: Cell = Cell {
    name: "fault_setup_streamed",
    program: Program {
        streamed: true,
        ..SETUP
    },
    ..SETUP_UNARY
};

// -- row 2: a retryable status ------------------------------------------------

/// A 429 with `Retry-After`, the retry budget spent to zero: the run fails
/// at once with the report a host needs to retry (parity with rig-agent,
/// which has no budget).
pub(crate) const STATUS_429: Cell = Cell {
    name: "fault_status_429",
    program: SETUP,
    families: C,
    fault: Some(Fault::Status {
        status: 429,
        code: None,
        retry_after: true,
    }),
    provider_retries: Some(0),
    ..CELL
};
pub(crate) const STATUS_503: Cell = Cell {
    name: "fault_status_503",
    fault: Some(Fault::Status {
        status: 503,
        code: None,
        retry_after: false,
    }),
    ..STATUS_429
};
/// The same 503 under the default budget (CONTRACT §5): re-issued three
/// times, four records, `Failed(Provider)` with the last. World-only:
/// rig-agent has no budget.
pub(crate) const STATUS_503_RETRIED: Cell = Cell {
    name: "fault_status_503_retried",
    families: CCCC,
    provider_retries: None,
    ..STATUS_503
};

// -- rows 3, 4, 5: the stream ends badly --------------------------------------

/// EOF after text: a truncation (`Failed(Response)`), the prefix kept,
/// nothing committed; a scene saved mid-stream is refused. The retry
/// budget is spent to zero: a truncation is not retried (its report is
/// `Response`, not retryable), and a wire that classified the EOF as a
/// transient transport failure would then surface it as the ending, not
/// as three re-issued requests.
pub(crate) const TRUNCATED_AFTER_TEXT: Cell = Cell {
    name: "fault_truncated_after_text",
    program: Program {
        ending: Ending::Failed(ErrorKind::Response),
        ..TEXT_STREAM
    },
    events: true,
    families: C,
    fault: Some(Fault::TruncatedAfterText),
    scene: Scene::MidStream,
    provider_retries: Some(0),
    ..CELL
};
/// EOF after a complete tool call: the tool never runs.
pub(crate) const TRUNCATED_AFTER_TOOL_CALL: Cell = Cell {
    name: "fault_truncated_after_tool_call",
    program: Program {
        prompt: super::cells::ADD_PROMPT,
        streamed: true,
        ending: Ending::Failed(ErrorKind::Response),
        ..TOOLS
    },
    tools: &[ToolKind::Adder],
    events: true,
    families: C,
    fault: Some(Fault::TruncatedAfterToolCall),
    provider_retries: Some(0),
    ..CELL
};
/// An error frame after text: the provider's error, the prefix kept, the
/// error item at its position. The per-wire file names the frame's facts.
/// The budget is spent to zero for the wires whose envelope is retryable.
pub(crate) const ERROR_AFTER_TEXT: Cell = Cell {
    name: "fault_error_after_text",
    program: Program {
        ending: Ending::ProviderError,
        ..TEXT_STREAM
    },
    events: true,
    families: C,
    fault: Some(Fault::ErrorAfterText {
        code: None,
        message: "",
        status: None,
    }),
    provider_retries: Some(0),
    ..CELL
};

// -- row 6: a refusal ---------------------------------------------------------

/// The wire's refusal in its own shape: OpenAI streams it as the answer
/// (`refusal` deltas); Gemini refuses the prompt outright
/// (`promptFeedback`), which the per-wire file names as `Failed(Provider)`.
pub(crate) const REFUSAL: Cell = Cell {
    name: "fault_refusal",
    program: TEXT_STREAM,
    events: true,
    families: C,
    fault: Some(Fault::Refusal),
    ..CELL
};
/// The wire filters the turn after some text (`finish_reason:
/// content_filter`, Gemini's `SAFETY`): the text is the answer, the reason
/// on the record.
pub(crate) const FILTERED_WITH_TEXT: Cell = Cell {
    name: "fault_filtered_with_text",
    fault: Some(Fault::Filtered { with_text: true }),
    ..REFUSAL
};
/// The wire filters the whole turn: no text, `content_filter`. rig-agent
/// fails the run (the model produced no answer); the world must agree.
pub(crate) const FILTERED_EMPTY: Cell = Cell {
    name: "fault_filtered_empty",
    program: Program {
        ending: Ending::Failed(ErrorKind::Response),
        ..TEXT_STREAM
    },
    fault: Some(Fault::Filtered { with_text: false }),
    ..REFUSAL
};

// -- row 7: a tool that errs --------------------------------------------------

pub(crate) const TOOL_ERROR: Cell = Cell {
    name: "fault_tool_error",
    program: TOOLS,
    tools: &[ToolKind::BrokenAdder],
    families: CTC,
    fault: Some(Fault::ToolError),
    ..CELL
};
pub(crate) const TOOL_ERROR_STREAMED: Cell = Cell {
    name: "fault_tool_error_streamed",
    program: Program {
        streamed: true,
        ..TOOLS
    },
    events: true,
    ..TOOL_ERROR
};

// -- row 8: a batch with one failure ------------------------------------------

const TWO_TOOLS: Program = Program {
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: BROKEN_ORCHARD_PROMPT,
    max_turns: Some(8),
    streamed: true,
    tool_concurrency: Some(1),
    ..Program::DEFAULT
};
/// Two calls in one turn, the second tool errs, concurrency one: both
/// parts in call order, the failing part the error's output.
pub(crate) const BATCH_SECOND_FAILS: Cell = Cell {
    name: "fault_batch_second_fails",
    program: TWO_TOOLS,
    tools: &[ToolKind::Alpha, ToolKind::BrokenBeta],
    events: true,
    families: CTTC,
    fault: Some(Fault::BatchSecondFails),
    ..CELL
};
/// The same over the same recording, concurrency two.
pub(crate) const BATCH_SECOND_FAILS_CONCURRENT: Cell = Cell {
    name: "fault_batch_second_fails_concurrent",
    program: Program {
        tool_concurrency: Some(2),
        ..TWO_TOOLS
    },
    ..BATCH_SECOND_FAILS
};

// -- row 9: a stop while a tool is running ------------------------------------

/// `endings_tool_outcome_cancelled`'s program over its own recording, the
/// adder parked: `Cancelled` lands while the tool is in flight. The record
/// is the golden's (`[Completion, Tool]`, the real answer); `despawn_run`
/// waits for the tool.
pub(crate) const STOP_WHILE_TOOL_RUNS: Cell = Cell {
    name: "fault_stop_while_tool_runs",
    program: ENDINGS_TOOL_OUTCOME_CANCELLED.program,
    tools: &[ToolKind::Adder],
    families: CT,
    fault: Some(Fault::StopWhileToolRuns),
    ..CELL
};

// -- row 10: a failing store --------------------------------------------------

const MEMORY: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: READY_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    conversation: Some(CONVERSATION),
    ending: Ending::MemoryError,
    ..Program::DEFAULT
};
/// The load fails before any completion: the memory record first and
/// alone, `Failed(Memory)`.
pub(crate) const FAILING_LOAD: Cell = Cell {
    name: "fault_failing_load",
    program: MEMORY,
    memory: Memory::FailingLoad,
    families: &[Mem],
    fault: Some(Fault::FailingLoad),
    ..CELL
};
pub(crate) const FAILING_LOAD_STREAMED: Cell = Cell {
    name: "fault_failing_load_streamed",
    program: Program {
        streamed: true,
        ..MEMORY
    },
    ..FAILING_LOAD
};

// -- row 13: a run saved with its tool in flight ------------------------------

/// `resume_tool_turn`'s program over its own recording, the adder parked:
/// the scene saved while the tool is in flight restarts the intent under
/// its saved id in a fresh world and finishes to the golden.
pub(crate) const SCENE_TOOL_IN_FLIGHT: Cell = Cell {
    name: "fault_scene_tool_in_flight",
    program: super::cells::RESUME_TOOL_TURN.program,
    tools: &[ToolKind::Adder],
    families: CTC,
    scene: Scene::WithToolInFlight,
    ..CELL
};

/// A wire's scripted rows: the faults served by the sequenced transport
/// over labelled frames cut or rewritten from the wire's own recordings,
/// each run by the rig-agent runner and the world over the same frames.
/// Every row takes the caller's literal world golden.
pub(crate) struct Scripted<M> {
    /// The provider directory the frames and the status reply are read from.
    pub(crate) provider: &'static str,
    pub(crate) shape: SseShape,
    /// The streamed text answer the text rows cut, on the wire's own model.
    pub(crate) text_stream: &'static str,
    /// The streamed tool call the tool rows cut, on the wire's own model.
    pub(crate) tool_stream: &'static str,
    /// The recorded setup failure the status rows rewrite, and its body's code.
    pub(crate) setup_reply: &'static str,
    pub(crate) code: Option<&'static str>,
    /// The wire over a scripted transport. The client's key must never reach
    /// a recording or a trace.
    pub(crate) wire: fn(DynHttpClient) -> Wire<M>,
}

impl<W, T> Scripted<rig::driver::Model<W, T>>
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    /// The frames of `scenario`'s first interaction.
    pub(crate) fn recorded(&self, scenario: &str) -> Vec<String> {
        recorded_sse_frames(self.provider, scenario, 0)
    }

    /// The wire over a transport that answers one streaming request with
    /// `frames`, then EOF.
    pub(crate) fn stream(&self, frames: &[String]) -> Wire<rig::driver::Model<W, T>> {
        (self.wire)(DynHttpClient::new(scripted(vec![sse_bytes(frames)])))
    }

    /// The wire over a transport that answers each unary request with the
    /// next of `replies`.
    fn unary(&self, replies: Vec<MockHttpResponse>) -> Wire<rig::driver::Model<W, T>> {
        (self.wire)(DynHttpClient::new(SequencedHttpClient::new(replies)))
    }

    fn status_cell(&self, status: u16, retry_after: bool, cell: Cell) -> Cell {
        Cell {
            fault: Some(Fault::Status {
                status,
                code: self.code,
                retry_after,
            }),
            ..cell
        }
    }

    fn reply(&self, status: u16, retry_after: bool) -> MockHttpResponse {
        status_reply(self.provider, self.setup_reply, status, retry_after)
    }

    async fn stream_row(&self, cell: &Cell, frames: Vec<String>, golden: impl FnOnce(&EffectLog)) {
        capture_world_programs(run_scripted(cell, || self.stream(&frames), golden)).await;
    }

    pub(crate) async fn truncated_after_text(&self, golden: impl FnOnce(&EffectLog)) {
        let frames = self.shape.text_prefix(&self.recorded(self.text_stream));
        self.stream_row(&TRUNCATED_AFTER_TEXT, frames, golden).await;
    }

    pub(crate) async fn truncated_after_tool_call(&self, golden: impl FnOnce(&EffectLog)) {
        let frames = self.shape.tool_prefix(&self.recorded(self.tool_stream));
        self.stream_row(&TRUNCATED_AFTER_TOOL_CALL, frames, golden)
            .await;
    }

    /// The in-band error frame's facts, as the funnel reports them on this
    /// shape (`SseShape::error_{code,message,status}`).
    pub(crate) async fn error_after_text(&self, golden: impl FnOnce(&EffectLog)) {
        let cell = Cell {
            fault: Some(Fault::ErrorAfterText {
                code: self.shape.error_code(),
                message: self.shape.error_message(),
                status: self.shape.error_status(),
            }),
            ..ERROR_AFTER_TEXT
        };
        let frames = self.shape.error_frames(&self.recorded(self.text_stream));
        self.stream_row(&cell, frames, golden).await;
    }

    pub(crate) async fn filtered_with_text(&self, golden: impl FnOnce(&EffectLog)) {
        let frames = self.shape.filtered(&self.recorded(self.text_stream), true);
        self.stream_row(&FILTERED_WITH_TEXT, frames, golden).await;
    }

    pub(crate) async fn filtered_empty(&self, golden: impl FnOnce(&EffectLog)) {
        let frames = self.shape.filtered(&self.recorded(self.text_stream), false);
        self.stream_row(&FILTERED_EMPTY, frames, golden).await;
    }

    /// Row 10: no request reaches the wire; the transport answers nothing.
    pub(crate) async fn failing_load(&self, golden: impl FnOnce(&EffectLog)) {
        self.stream_row(&FAILING_LOAD, Vec::new(), golden).await;
    }

    pub(crate) async fn failing_load_streamed(&self, golden: impl FnOnce(&EffectLog)) {
        self.stream_row(&FAILING_LOAD_STREAMED, Vec::new(), golden)
            .await;
    }

    /// Row 11: a bare `Cancelled` at the first tool-call delta; the stream is
    /// left to its handler, the tool never dispatched. The recorded tool turn
    /// is served whole by the sequenced transport: a cancelled run makes one
    /// request, and the recording holds two.
    pub(crate) async fn cancel_at_first_tool_call_delta(&self, golden: impl FnOnce(&EffectLog)) {
        let wire = self.stream(&self.recorded(self.tool_stream));
        capture_world_programs(cancel_at(
            &wire,
            &super::cells::HOOKS_PATCH_TOOL_ARGS_STREAMED,
            Cut::FirstToolCallDelta,
            golden,
        ))
        .await;
    }

    pub(crate) async fn status_429(&self, golden: impl FnOnce(&EffectLog)) {
        let cell = self.status_cell(429, true, STATUS_429);
        let wire = || self.unary(vec![self.reply(429, true)]);
        capture_world_programs(run_scripted(&cell, wire, golden)).await;
    }

    pub(crate) async fn status_503(&self, golden: impl FnOnce(&EffectLog)) {
        let cell = self.status_cell(503, false, STATUS_503);
        let wire = || self.unary(vec![self.reply(503, false)]);
        capture_world_programs(run_scripted(&cell, wire, golden)).await;
    }

    /// World-only: the default budget re-issues the completion three times.
    pub(crate) async fn status_503_retried(&self, golden: impl FnOnce(&EffectLog)) {
        let cell = self.status_cell(503, false, STATUS_503_RETRIED);
        let wire = self.unary((0..4).map(|_| self.reply(503, false)).collect());
        capture_world_programs(run_world(&wire, &cell, golden)).await;
    }
}

/// Row 11 once the terminal record has landed and before `Fold`: a whole
/// completion, the run cancelled, despawned at once.
pub(crate) async fn cancel_after_terminal<W, T>(
    wire: &Wire<rig::driver::Model<W, T>>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    cancel_at(wire, cell, Cut::AfterTerminal, golden).await;
}

/// `lookup_orchard_label` that fails every call (row 8's second tool).
#[derive(Clone)]
pub(crate) struct FailingOrchard;

impl rig_core::tool::Tool for FailingOrchard {
    const NAME: &'static str = "lookup_orchard_label";
    type Error = rig_core::tool::ToolExecutionError;
    type Args = crate::support::EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the beta signal marker.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(rig_core::tool::ToolExecutionError::other(BROKEN_ORCHARD))
    }
}
