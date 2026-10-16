//! The producer: a cell's program on rig-agent's builder over a cassette,
//! ending as the program says, its log the golden
//! (`crate::goldens::golden_effects`).

use futures::StreamExt;
use rig_cassette::agent::AgentReplayExt;

use rig_agent::agent::AgentBuilder;
use rig_agent::handlers::Handlers;

use rig_agent::agent::MultiTurnStreamItem;

use rig_agent::completion::PromptError;

use rig_core::effect::HandlerKey;

use rig_core::error::ErrorKind;

use rig_core::error::ErrorReport;

use rig_core::serve::ErasedHandler;

use rig_core::serve::adapters::ModelAdapter;

use super::cells::{Bus, Cell, ToolKind};
use super::corpus::{self, Ending, LayerAt, Lookup, Nesting, Program};
use super::faults::FailingOrchard;
use super::{OWNER, Wire};
use crate::goldens::{Adder, FailingAdd, NoteTaker, WriteNote, add_tool_under, families};
use crate::support::{AlphaSignal, BetaSignal};

/// Each record's kind in a line, for a failed shape assertion.
pub(crate) fn record_summary(log: &rig_cassette::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .map(|record| match &record.kind {
            rig_core::effect::EffectKind::ToolCall { name, args } => format!("tool {name}({args})"),
            rig_core::effect::EffectKind::Completion { request, .. } => {
                format!("completion({} messages)", request.chat_history.len())
            }
            other => format!("{:?}", other.family()),
        })
        .collect()
}

/// How a run failed, as the runner reported it.
#[derive(Debug)]
enum RunFailure {
    Cancelled(String),
    MaxTurns,
    Report(ErrorReport),
}

fn classify_prompt(error: PromptError) -> RunFailure {
    match error {
        PromptError::Cancelled { reason, .. } => RunFailure::Cancelled(reason),
        PromptError::MaxTurns { .. } => RunFailure::MaxTurns,
        PromptError::Report(report) => RunFailure::Report(report),
        PromptError::Provider(error) => RunFailure::Report(ErrorReport::from(&error)),
        other => panic!("the run fails as one of the program's endings, not {other:?}"),
    }
}

/// What a streamed run ended in: the final answer, or how it failed (the
/// first error item; a fault ends the stream).
async fn final_output(
    stream: &mut rig_agent::agent::StreamingResult,
) -> Result<rig_agent::agent::PromptResponse, RunFailure> {
    let mut output = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response),
            Ok(_) => {}
            Err(error) => return Err(classify_prompt(error)),
        }
    }
    Ok(output.expect("a final response"))
}

/// The run ended as the program says; an answer is returned, a failure
/// yields an empty output.
fn expect_ending(result: Result<String, RunFailure>, ending: Ending, fixture: &str) -> String {
    match (result, ending) {
        (Ok(output), Ending::Answer) => output,
        (Err(RunFailure::Cancelled(reason)), Ending::Cancelled(expected)) => {
            assert_eq!(reason, expected, "{fixture}: the hook's reason");
            String::new()
        }
        (Err(RunFailure::MaxTurns), Ending::MaxTurns) => String::new(),
        (Err(RunFailure::Report(report)), Ending::ProviderError)
            if report.kind == ErrorKind::ProviderResponse =>
        {
            String::new()
        }
        (Err(RunFailure::Report(report)), Ending::Failed(kind)) if report.kind == kind => {
            String::new()
        }
        (other, ending) => panic!("{fixture}: ends in {ending:?}, got {other:?}"),
    }
}

/// The program's builder settings, as the corpus's `build_agent` applies
/// them; the wire's model is already the builder's.
fn configure(
    mut builder: AgentBuilder,
    program: &Program,
    settlement: Option<&super::reasoning::SettlementCapture>,
) -> AgentBuilder {
    builder = builder.name(OWNER);
    builder = match program.preamble {
        Some(preamble) => builder.preamble(preamble),
        None => builder.without_preamble(),
    };
    if let Some(temperature) = program.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = program.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(params) = program.additional_params {
        builder = builder.additional_params(params());
    }
    if let Some(options) = program.options {
        let options = options();
        builder = builder
            .options(options.generation)
            .provider_options(options.provider);
    }
    if let Some(choice) = program.tool_choice {
        builder = builder.tool_choice(choice.tool_choice());
    }
    if let Some(schema) = program.output_schema {
        builder = builder
            .output_schema_raw(serde_json::from_value(schema()).expect("the schema is a schema"));
    }
    if let Some(mode) = program.output_mode {
        builder = builder.output_mode(mode.mode());
    }
    if let Some(default_max_turns) = program.default_max_turns {
        builder = builder.default_max_turns(default_max_turns);
    }
    if let Some(capture) = settlement {
        assert_eq!(program.hooks, &[corpus::Hook::RecordSettled]);
        builder.add_hook(super::reasoning::RecordSettled(capture.clone()))
    } else {
        corpus::with_program_hooks(builder, program)
    }
}

/// Apply the cell's route, then build.
fn finish<W, T>(
    mut builder: AgentBuilder,
    wire: &Wire<rig::driver::Model<W, T>>,
    program: &Program,
) -> rig_agent::agent::Agent
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    if let Some(label) = program.route {
        builder = builder.model_route(label, wire.route());
    }
    builder.build()
}

fn typed_tool(builder: AgentBuilder, cell: &Cell, tool: ToolKind) -> AgentBuilder {
    use super::long_loop::{ListFiles, ReadFile, RunTests, WriteFile, repo};
    match tool {
        ToolKind::CheckpointStep => builder.tool(super::checkpoint::CheckpointStep),
        ToolKind::CheckpointBatch => builder.tool(super::checkpoint::CheckpointBatch),
        ToolKind::CheckpointLarge => builder.tool(super::checkpoint::CheckpointLarge),
        ToolKind::RepoListFiles => builder.tool(ListFiles(repo(cell))),
        ToolKind::RepoReadFile => builder.tool(ReadFile(repo(cell))),
        ToolKind::RepoWriteFile => builder.tool(WriteFile(repo(cell))),
        ToolKind::RepoRunTests => builder.tool(RunTests(repo(cell))),
        ToolKind::Adder => builder.tool(Adder),
        ToolKind::Alpha => builder.tool(AlphaSignal),
        ToolKind::Beta => builder.tool(BetaSignal),
        ToolKind::WriteNote => builder.tool(WriteNote),
        ToolKind::BrokenAdder => builder.tool(FailingAdd),
        ToolKind::BrokenBeta => builder.tool(FailingOrchard),
        ToolKind::Lookup => unreachable!("the nesting tool is a tool server's"),
    }
}

/// The cell's tools on the builder, then [`finish`].
fn grant<W, T>(
    builder: AgentBuilder,
    wire: &Wire<rig::driver::Model<W, T>>,
    cell: &Cell,
    program: &Program,
) -> rig_agent::agent::Agent
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let layered_tool = program.layers.iter().any(|spec| spec.at == LayerAt::Tool);
    if layered_tool {
        assert_eq!(cell.tools, [ToolKind::Adder], "the layer cells grant add");
        let server = add_tool_under(|adder| corpus::layered(adder, program, LayerAt::Tool, &None));
        return finish(builder.tool_server_handle(server), wire, program);
    }
    match cell.tools {
        [] => finish(builder, wire, program),
        [ToolKind::Lookup] => {
            let nesting: Nesting = program.nesting.expect("the nesting program");
            let server = rig_agent::tool::server::ToolServer::new()
                .owner(OWNER)
                .registered_tool(
                    rig_agent::tool::RegisteredTool::from_handler(Lookup {
                        nesting,
                        model_key: HandlerKey::from(format!("{OWNER}/model:default")),
                    })
                    .expect("a tool-family handler"),
                )
                .run();
            finish(builder.tool_server_handle(server), wire, program)
        }
        [first, rest @ ..] => {
            let mut builder = match first {
                ToolKind::CheckpointStep => builder.tool(super::checkpoint::CheckpointStep),
                ToolKind::CheckpointBatch => builder.tool(super::checkpoint::CheckpointBatch),
                ToolKind::CheckpointLarge => builder.tool(super::checkpoint::CheckpointLarge),
                ToolKind::RepoListFiles => {
                    builder.tool(super::long_loop::ListFiles(super::long_loop::repo(cell)))
                }
                ToolKind::RepoReadFile | ToolKind::RepoWriteFile | ToolKind::RepoRunTests => {
                    unreachable!("the repository toolset is granted in its registration order")
                }
                ToolKind::Adder => builder.tool(Adder),
                ToolKind::Alpha => builder.tool(AlphaSignal),
                ToolKind::Beta => builder.tool(BetaSignal),
                ToolKind::WriteNote => builder.tool(WriteNote),
                ToolKind::BrokenAdder => builder.tool(FailingAdd),
                ToolKind::BrokenBeta => builder.tool(FailingOrchard),
                ToolKind::Lookup => unreachable!("the nesting tool is granted alone"),
            };
            for tool in rest {
                builder = typed_tool(builder, cell, *tool);
            }
            finish(builder, wire, program)
        }
    }
}

/// Run the program's prompts on `agent`, ending as the program says.
async fn run_prompts(
    agent: &rig_agent::agent::Agent,
    cell: &Cell,
    program: &Program,
) -> Vec<rig_agent::agent::PromptResponse> {
    let first = match cell.image {
        Some(image) => super::image::prompt_message(image, program.prompt),
        None => rig_core::message::Message::user(program.prompt),
    };
    let prompts: Vec<rig_core::message::Message> = std::iter::once(first)
        .chain(cell.followup.map(rig_core::message::Message::user))
        .collect();
    let last = prompts.len() - 1;
    let mut outputs: Vec<rig_agent::agent::PromptResponse> = Vec::new();
    for (n, prompt) in prompts.into_iter().enumerate() {
        let mut runner = agent.prompt(prompt);
        if let Some(history) = program.history {
            runner = runner.history(history());
        } else if !outputs.is_empty() {
            // A follow-up continues the conversation the earlier runs made.
            runner = runner.history(
                outputs
                    .iter()
                    .flat_map(|response| response.messages.iter().cloned()),
            );
        }
        if let Some(max_turns) = program.max_turns {
            runner = runner.max_turns(max_turns);
        }
        if let Some(concurrency) = program.tool_concurrency {
            runner = runner.tool_concurrency(concurrency);
        }
        runner = runner.unhandled_invalid_tool_call(match program.unhandled {
            corpus::Unhandled::Fail => rig_agent::run::UnhandledInvalidToolCall::Fail,
            corpus::Unhandled::Ignore => rig_agent::run::UnhandledInvalidToolCall::Ignore,
        });
        let ending = if n == last {
            program.ending
        } else {
            Ending::Answer
        };
        let output = if program.streamed {
            let mut stream = runner.stream();
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(180),
                final_output(&mut stream),
            )
            .await
            .expect("the stream ends");
            drop(stream);
            let output = expect_ending(
                result.map(|response| {
                    let output = response.output();
                    outputs.push(response);
                    output
                }),
                ending,
                program.fixture,
            );
            if ending != Ending::Answer {
                // The engine settles an in-flight cancel or fault after the
                // consumer saw it: give it the window the corpus producers
                // do before reading the log.
                for _ in 0..64 {
                    tokio::task::yield_now().await;
                }
            }
            output
        } else {
            let result = runner
                .await
                .map(|response| {
                    let output = response.output();
                    outputs.push(response);
                    output
                })
                .map_err(classify_prompt);
            expect_ending(result, ending, program.fixture)
        };
        if let (Ending::Answer, Some(expected)) = (ending, program.expected_output) {
            assert_eq!(output, expected, "{}: the replaced answer", program.fixture);
        }
    }
    outputs
}

/// The producer's cell: the program over `wire` on rig-agent's builder,
/// its log written as the golden `golden`.
pub(crate) async fn run_agent<W, T>(
    wire: &Wire<rig::driver::Model<W, T>>,
    cell: &Cell,
    golden: impl FnOnce(&rig_cassette::effect_log::EffectLog),
) -> rig_cassette::effect_log::EffectLog
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let program = wire.program(cell);
    let settlement: Option<super::reasoning::SettlementCapture> =
        (cell.reasoning == Some(super::cells::ReasoningCase::Capped)).then(Default::default);
    let recorder = if cell.events {
        rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig_cassette::effect_log::EffectLogRecorder::new()
    };
    let (log, responses) = if cell.bus == Bus::Own {
        let builder = AgentBuilder::new(wire.model.clone());
        let builder = configure(builder, &program, settlement.as_ref()).record_to(recorder.clone());
        let agent = grant(builder, wire, cell, &program);
        if let Some(label) = program.late_route {
            agent.register_model(label, wire.route());
        }
        let responses = run_prompts(&agent, cell, &program).await;
        (agent.stamp(recorder.take()), responses)
    } else {
        // A host's handlers: the host registers the model under the
        // agent's key and its note taker, and records; the agent stamps the
        // log.
        let handlers = Handlers::default();
        let model_key = HandlerKey::from(format!("{OWNER}/model:default"));
        handlers
            .register_erased(
                model_key.clone(),
                ErasedHandler::new(ModelAdapter::new("default", wire.model.clone())),
            )
            .expect("a fresh key");
        if cell.notes {
            handlers
                .register_erased(
                    HandlerKey::from(corpus::NOTE_KEY),
                    ErasedHandler::new(NoteTaker),
                )
                .expect("a fresh key");
        }
        handlers.record_to(recorder.clone());
        let builder = AgentBuilder::over_handlers(handlers.clone(), OWNER, model_key);
        let builder = configure(builder, &program, settlement.as_ref());
        let agent = grant(builder, wire, cell, &program);
        let responses = run_prompts(&agent, cell, &program).await;
        let log = agent.stamp(recorder.take());
        drop(agent);
        (log, responses)
    };
    if cell.reasoning.is_some() {
        if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode == "record") {
            eprintln!(
                "REASONING_ATTEMPT {}",
                serde_json::to_string(&log).expect("the log serializes")
            );
        }
        super::reasoning::assert_log(cell, wire.thinking, &log);
        if let Some(capture) = settlement {
            let settled = capture
                .lock()
                .expect("settlement capture")
                .take()
                .expect("the error settled");
            super::reasoning::assert_history(cell, &log, &settled.messages);
            let error = settled.error.expect("the capped run failed");
            assert!(
                error.contains(&rig_agent::completion::FinishReason::Length.no_answer_message()),
                "{error}"
            );
        }
        for response in &responses {
            super::reasoning::assert_history(cell, &log, &response.messages);
            assert_eq!(
                response.output(),
                corpus::golden_answer(&log),
                "the reasoning is not the answer"
            );
        }
    }
    if cell.name.starts_with("checkpoint_") {
        super::checkpoint::write_attempt(cell, &log);
        super::checkpoint::assert_log(cell, &log);
    }
    if super::long_loop::is_long_loop(cell) {
        super::long_loop::write_attempt(cell, &log);
        super::long_loop::assert_log(cell, wire.thinking, &log);
        for response in &responses {
            super::long_loop::assert_transcript(cell, &log, &response.messages);
        }
    }
    if cell.image.is_some() {
        super::image::assert_log(cell, &log);
        for (n, response) in responses.iter().enumerate() {
            super::image::assert_history(
                cell,
                &response.messages,
                &format!("{}: run {n} history", cell.name),
            );
        }
        let answer = &responses.last().expect("a run").output();
        super::image::assert_answer(cell, answer);
        assert_eq!(*answer, corpus::golden_answer(&log));
    }
    if !cell.families.is_empty() {
        assert_eq!(
            families(&log),
            cell.families,
            "{}: the record's families; the records: {:?}",
            cell.name,
            record_summary(&log)
        );
    }
    assert_eq!(
        log.header.hooks,
        corpus::program_hooks(&program, OWNER),
        "{}: the header names the program",
        cell.name
    );
    // The caller names the golden at its own call site
    // (`crate::goldens::golden_effects("…", log)`), where the pairing
    // guard reads it.
    golden(&log);
    log
}
