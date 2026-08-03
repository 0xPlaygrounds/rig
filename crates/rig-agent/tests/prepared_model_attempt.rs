use std::collections::BTreeSet;

use rig_agent::agent::run::{
    ModelTurn, StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent,
};
use rig_agent::agent::{
    AgentConfig, AgentRun, AgentRunStep, InvalidToolCallAction, ModelTurnOutcome, OutputMode,
    RequestPatch, ToolCatalog, prepare_request,
};
use rig_core::{
    OneOrMany,
    completion::Usage,
    message::{AssistantContent, ToolCall, ToolFunction},
    schemars::Schema,
    streaming::StreamedAssistantContent,
};
use serde_json::json;

fn object_schema_requiring(field: &str) -> Result<Schema, serde_json::Error> {
    serde_json::from_value(json!({
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
    }))
}

fn hand_driven_streamed_recovery_preserves_patched_contract(
    action: InvalidToolCallAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = AgentConfig::new();
    config.output_mode = OutputMode::Tool;
    config.output_schema = Some(object_schema_requiring("baseline")?);
    config.output_tool_name = Some("final_result".to_string());

    let mut run = AgentRun::new("stream the patched record")
        .max_turns(2)
        .max_invalid_tool_call_retries(1);
    let (prompt, history, attempt_id) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt,
            history,
            attempt_id,
            ..
        } => (prompt, history, attempt_id),
        other => return Err(format!("expected first model step, got {other:?}").into()),
    };
    let patched_schema = object_schema_requiring("patched")?;
    let patch = RequestPatch::new().output_schema(patched_schema.clone());
    let prepared = prepare_request(
        &config,
        &ToolCatalog::default(),
        false,
        prompt,
        &history,
        run.output_tool_name(),
        run.inherited_output_contract(),
        Some(&attempt_id),
        Some(&patch),
    )?;
    let mut assembler = prepared.model_attempt.into_streamed_turn_assembler();
    let events = assembler.ingest(&StreamedAssistantContent::ToolCall {
        tool_call: ToolCall::new(
            "invalid-call".to_string(),
            ToolFunction::new("default_api".to_string(), json!({})),
        ),
        internal_call_id: "internal-invalid-call".to_string(),
    })?;
    let invalid = events
        .into_iter()
        .find_map(|event| match event {
            StreamedTurnEvent::InvalidToolCall(invalid) => Some(*invalid),
            _ => None,
        })
        .ok_or("stream did not surface the invalid tool call")?;
    let partial = assembler.partial_turn(None);
    let resolution = run.resolve_streamed_invalid_tool_call(&partial, &invalid, action)?;
    if !matches!(resolution, StreamedResolution::TurnAbandoned { .. }) {
        return Err("recovery did not abandon the invalid streamed turn".into());
    }
    assembler.resolve_pending_invalid(&resolution);
    run.record_streamed_completion_call(&attempt_id, Usage::new())?;

    let (prompt, history, attempt_id) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt,
            history,
            attempt_id,
            ..
        } => (prompt, history, attempt_id),
        other => return Err(format!("expected corrective model step, got {other:?}").into()),
    };
    let inherited = run
        .inherited_output_contract()
        .ok_or("corrective request lost the attempt output contract")?;
    if inherited.output_tool_name != "final_result" || inherited.effective_schema != patched_schema
    {
        return Err("corrective run state did not retain the patched contract".into());
    }
    let corrective = prepare_request(
        &config,
        &ToolCatalog::default(),
        false,
        prompt,
        &history,
        run.output_tool_name(),
        run.inherited_output_contract(),
        Some(&attempt_id),
        None,
    )?;
    let corrective_contract = corrective
        .model_attempt
        .output_contract()
        .ok_or("corrective prepared request lost Tool mode")?;
    if corrective_contract.output_tool_name != "final_result"
        || corrective_contract.effective_schema != patched_schema
    {
        return Err("corrective request fell back to the configured schema".into());
    }
    Ok(())
}

#[test]
fn hand_driven_streamed_retry_preserves_patched_attempt_contract()
-> Result<(), Box<dyn std::error::Error>> {
    hand_driven_streamed_recovery_preserves_patched_contract(InvalidToolCallAction::retry(
        "use the output tool",
    ))
}

#[test]
fn hand_driven_streamed_skip_preserves_patched_attempt_contract()
-> Result<(), Box<dyn std::error::Error>> {
    hand_driven_streamed_recovery_preserves_patched_contract(InvalidToolCallAction::skip(
        "skip the invalid call",
    ))
}

#[test]
fn hand_driven_streamed_schema_commits_with_its_prepared_attempt()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = AgentConfig::new();
    config.output_mode = OutputMode::Tool;
    config.output_schema = Some(object_schema_requiring("baseline")?);
    config.output_tool_name = Some("final_result".to_string());

    let mut run = AgentRun::new("stream the patched record").max_turns(2);
    let (prompt, history, attempt_id) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt,
            history,
            attempt_id,
            ..
        } => (prompt, history, attempt_id),
        other => return Err(format!("expected first model step, got {other:?}").into()),
    };
    let patched_schema = object_schema_requiring("patched")?;
    let patch = RequestPatch::new().output_schema(patched_schema.clone());
    let prepared = prepare_request(
        &config,
        &ToolCatalog::default(),
        false,
        prompt.clone(),
        &history,
        run.output_tool_name(),
        run.inherited_output_contract(),
        Some(&attempt_id),
        Some(&patch),
    )?;

    let attempt_id = prepared.model_attempt.attempt_id().to_string();
    let output_tool_name = prepared
        .model_attempt
        .output_contract()
        .ok_or("missing Tool-mode attempt contract")?
        .output_tool_name
        .clone();
    let final_choice = OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
        "streamed-output-call".to_string(),
        ToolFunction::new(output_tool_name, json!({"patched": "streamed"})),
    )));
    let turn = prepared
        .model_attempt
        .into_streamed_turn_assembler()
        .finish(None, final_choice, Usage::new());

    let accepted = run.streamed_turn(turn)?;
    if accepted.prompt() != &prompt {
        return Err("streamed attempt committed the wrong prompt".into());
    }
    if accepted.attempt_id() != attempt_id {
        return Err("streamed attempt committed the wrong identity".into());
    }
    if accepted
        .output_contract()
        .ok_or("missing committed Tool-mode contract")?
        .effective_schema
        != patched_schema
    {
        return Err("streamed attempt committed the baseline schema".into());
    }
    if run.output_tool_name() != Some("final_result") {
        return Err("streamed attempt did not promote its output-tool name".into());
    }

    run.continue_model_turn()?;
    let response = match run.next_step()? {
        AgentRunStep::Done(response) => response,
        other => {
            return Err(format!("patched streamed output did not finalize; got {other:?}").into());
        }
    };
    if response.output != r#"{"patched":"streamed"}"# {
        return Err("streamed patched output was not finalized".into());
    }
    Ok(())
}

#[test]
fn hand_driven_patched_schema_commits_with_its_prepared_attempt()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = AgentConfig::new();
    config.output_mode = OutputMode::Tool;
    config.output_schema = Some(object_schema_requiring("baseline")?);
    config.output_tool_name = Some("final_result".to_string());

    let mut run = AgentRun::new("return the patched record").max_turns(2);
    let (prompt, history, attempt_id) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt,
            history,
            attempt_id,
            ..
        } => (prompt, history, attempt_id),
        other => return Err(format!("expected first model step, got {other:?}").into()),
    };
    let patched_schema = object_schema_requiring("patched")?;
    let patch = RequestPatch::new().output_schema(patched_schema.clone());
    let prepared = prepare_request(
        &config,
        &ToolCatalog::default(),
        false,
        prompt.clone(),
        &history,
        run.output_tool_name(),
        run.inherited_output_contract(),
        Some(&attempt_id),
        Some(&patch),
    )?;

    let attempt_id = prepared.model_attempt.attempt_id().to_string();
    let output_tool_name = prepared
        .model_attempt
        .output_contract()
        .ok_or("missing Tool-mode attempt contract")?
        .output_tool_name
        .clone();
    let turn = prepared.model_attempt.into_model_turn(
        None,
        OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
            "output-call".to_string(),
            ToolFunction::new(output_tool_name, json!({"patched": "ok"})),
        ))),
        Usage::new(),
    );

    let accepted = match run.model_response(turn)? {
        ModelTurnOutcome::Continue(accepted) => accepted,
        other => return Err(format!("expected accepted patched response, got {other:?}").into()),
    };
    if accepted.prompt() != &prompt {
        return Err("unary attempt committed the wrong prompt".into());
    }
    if accepted.attempt_id() != attempt_id {
        return Err("unary attempt committed the wrong identity".into());
    }
    if accepted
        .output_contract()
        .ok_or("missing committed Tool-mode contract")?
        .effective_schema
        != patched_schema
    {
        return Err("unary attempt committed the baseline schema".into());
    }
    if run.output_tool_name() != Some("final_result") {
        return Err("unary attempt did not promote its output-tool name".into());
    }

    run.continue_model_turn()?;
    let response = match run.next_step()? {
        AgentRunStep::Done(response) => response,
        other => return Err(format!("patched output did not finalize; got {other:?}").into()),
    };
    if response.output != r#"{"patched":"ok"}"# {
        return Err("unary patched output was not finalized".into());
    }
    Ok(())
}

#[test]
fn stale_unary_attempt_is_rejected_without_poisoning_the_reissue()
-> Result<(), Box<dyn std::error::Error>> {
    let mut run = AgentRun::new("hello").max_turns(2);
    let first_attempt_id = match run.next_step()? {
        AgentRunStep::CallModel { attempt_id, .. } => attempt_id,
        other => return Err(format!("expected first model step, got {other:?}").into()),
    };
    let stale_turn = ModelTurn::new(
        first_attempt_id,
        None,
        OneOrMany::one(AssistantContent::text("stale")),
        Usage::new(),
        BTreeSet::new(),
        BTreeSet::new(),
    );

    if !run.abandon_pending_model_call() {
        return Err("first attempt was not pending".into());
    }
    let current_attempt_id = match run.next_step()? {
        AgentRunStep::CallModel { attempt_id, .. } => attempt_id,
        other => return Err(format!("expected reissued model step, got {other:?}").into()),
    };

    let stale_error = match run.model_response(stale_turn) {
        Err(error) => error,
        Ok(_) => return Err("the abandoned response committed to its reissue".into()),
    };
    if !stale_error.to_string().contains("stale or different") {
        return Err(format!("unexpected stale-attempt error: {stale_error}").into());
    }
    if !run.completion_calls().is_empty() {
        return Err("a rejected stale response recorded usage".into());
    }

    let current_turn = ModelTurn::new(
        current_attempt_id,
        None,
        OneOrMany::one(AssistantContent::text("current")),
        Usage::new(),
        BTreeSet::new(),
        BTreeSet::new(),
    );
    if !matches!(
        run.model_response(current_turn)?,
        ModelTurnOutcome::Continue(_)
    ) {
        return Err("the current response was not accepted after stale rejection".into());
    }
    Ok(())
}

#[test]
fn prepared_turn_receipt_cannot_commit_twice() -> Result<(), Box<dyn std::error::Error>> {
    let mut run = AgentRun::new("hello");
    let (prompt, history, attempt_id) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt,
            history,
            attempt_id,
            ..
        } => (prompt, history, attempt_id),
        other => return Err(format!("expected model step, got {other:?}").into()),
    };
    let prepared = prepare_request(
        &AgentConfig::new(),
        &ToolCatalog::default(),
        false,
        prompt,
        &history,
        run.output_tool_name(),
        run.inherited_output_contract(),
        Some(&attempt_id),
        None,
    )?;
    let turn = prepared.model_attempt.into_model_turn(
        None,
        OneOrMany::one(AssistantContent::text("done")),
        Usage::new(),
    );
    let replay = turn.clone();

    if !matches!(run.model_response(turn)?, ModelTurnOutcome::Continue(_)) {
        return Err("first receipt commit was not accepted".into());
    }
    if run.model_response(replay).is_ok() {
        return Err("a cloned committed turn committed twice".into());
    }
    if run.completion_calls().len() != 1 {
        return Err("replaying a receipt recorded a second completion".into());
    }
    Ok(())
}

#[test]
fn stale_stream_state_and_final_usage_cannot_commit_to_a_reissue()
-> Result<(), Box<dyn std::error::Error>> {
    let mut run = AgentRun::new("stream it")
        .max_turns(2)
        .max_invalid_tool_call_retries(1);
    let first_attempt_id = match run.next_step()? {
        AgentRunStep::CallModel { attempt_id, .. } => attempt_id,
        other => return Err(format!("expected first model step, got {other:?}").into()),
    };

    let mut invalid_assembler =
        StreamedTurnAssembler::new(first_attempt_id.clone(), BTreeSet::new(), BTreeSet::new());
    let invalid = invalid_assembler
        .ingest(&StreamedAssistantContent::ToolCall {
            tool_call: ToolCall::new(
                "stale-call".to_string(),
                ToolFunction::new("unknown".to_string(), json!({})),
            ),
            internal_call_id: "stale-internal".to_string(),
        })?
        .into_iter()
        .find_map(|event| match event {
            StreamedTurnEvent::InvalidToolCall(invalid) => Some(*invalid),
            _ => None,
        })
        .ok_or("expected invalid streamed call")?;
    let stale_partial = invalid_assembler.partial_turn(None);
    let stale_turn =
        StreamedTurnAssembler::new(first_attempt_id.clone(), BTreeSet::new(), BTreeSet::new())
            .finish(
                None,
                OneOrMany::one(AssistantContent::text("stale")),
                Usage::new(),
            );

    if !run.abandon_pending_model_call() {
        return Err("first streamed attempt was not pending".into());
    }
    let current_attempt_id = match run.next_step()? {
        AgentRunStep::CallModel { attempt_id, .. } => attempt_id,
        other => return Err(format!("expected reissued model step, got {other:?}").into()),
    };

    if run
        .record_streamed_completion_call(&first_attempt_id, Usage::new())
        .is_ok()
    {
        return Err("a late final from the abandoned attempt was accepted".into());
    }
    if run
        .resolve_streamed_invalid_tool_call(
            &stale_partial,
            &invalid,
            InvalidToolCallAction::retry("try again"),
        )
        .is_ok()
    {
        return Err("a stale partial turn mutated the reissue".into());
    }
    if run.streamed_turn(stale_turn).is_ok() {
        return Err("a stale completed stream committed to the reissue".into());
    }
    if !run.completion_calls().is_empty() || run.usage() != Usage::new() {
        return Err("stale streamed state poisoned completion accounting".into());
    }

    let current_usage = Usage {
        total_tokens: 5,
        ..Usage::new()
    };
    run.record_streamed_completion_call(&current_attempt_id, current_usage)?;
    let current_turn =
        StreamedTurnAssembler::new(current_attempt_id, BTreeSet::new(), BTreeSet::new()).finish(
            None,
            OneOrMany::one(AssistantContent::text("current")),
            current_usage,
        );
    run.streamed_turn(current_turn)?;
    if run.completion_calls().len() != 1 || run.usage() != current_usage {
        return Err("current streamed attempt did not retain its own usage".into());
    }
    Ok(())
}
