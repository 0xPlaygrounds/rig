use rig_agent::agent::{
    AgentConfig, AgentRun, AgentRunStep, ModelTurnOutcome, OutputMode, RequestPatch, ToolCatalog,
    prepare_request,
};
use rig_core::{
    OneOrMany,
    completion::Usage,
    message::{AssistantContent, ToolCall, ToolFunction},
    schemars::Schema,
};
use serde_json::json;

fn object_schema_requiring(field: &str) -> Result<Schema, serde_json::Error> {
    serde_json::from_value(json!({
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
    }))
}

#[test]
fn hand_driven_streamed_schema_commits_with_its_prepared_attempt()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = AgentConfig::new();
    config.output_mode = OutputMode::Tool;
    config.output_schema = Some(object_schema_requiring("baseline")?);
    config.output_tool_name = Some("final_result".to_string());

    let mut run = AgentRun::new("stream the patched record").max_turns(2);
    let (prompt, history) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt, history, ..
        } => (prompt, history),
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
    let assembled =
        prepared
            .model_attempt
            .streamed_turn_assembler()
            .finish(None, final_choice, Usage::new());
    let turn = prepared.model_attempt.into_streamed_turn(assembled);

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
    let (prompt, history) = match run.next_step()? {
        AgentRunStep::CallModel {
            prompt, history, ..
        } => (prompt, history),
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
