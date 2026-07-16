#![cfg(not(target_family = "wasm"))]

use std::{path::PathBuf, sync::OnceLock, time::Duration};

use rig_candle::{CandleModel, ModelArtifacts, ModelData};
use rig_core::{
    completion::CompletionModel,
    test_utils::{
        buffered_streaming_text_parity, cancellation_and_max_turns, complex_tool_arguments,
        hook_rewrites_and_request_patch, invalid_tool_recovery, optional_argument, parallel_tools,
        sequential_tools, streaming_structured_after_tool, streaming_tool, structured_after_tool,
        structured_extraction, tool_output_serialization, zero_argument_tool,
    },
};

static MODEL: OnceLock<Result<CandleModel, String>> = OnceLock::new();

fn model() -> Result<CandleModel, Box<dyn std::error::Error + Send + Sync>> {
    let result = MODEL.get_or_init(|| -> Result<CandleModel, String> {
        let directory = PathBuf::from(
            std::env::var_os("RIG_CANDLE_TEST_MODEL_DIR")
                .ok_or_else(|| "RIG_CANDLE_TEST_MODEL_DIR is not set".to_string())?,
        );
        let data = ModelData {
            config: std::fs::read(directory.join("config.json"))
                .map_err(|error| error.to_string())?,
            tokenizer: std::fs::read(directory.join("tokenizer.json"))
                .map_err(|error| error.to_string())?,
            weights: std::fs::read(directory.join("model.gguf"))
                .map_err(|error| error.to_string())?,
        };
        CandleModel::builder_from_artifacts(ModelArtifacts::Gguf(data))
            .temperature(0.0)
            .seed(42)
            .max_tokens(384)
            .max_concurrent_requests(1)
            .build()
            .map_err(|error| error.to_string())
    });
    result.clone().map_err(Into::into)
}

fn print_report(report: &rig_core::test_utils::ScenarioReport) {
    let seconds = report.duration.as_secs_f64();
    let throughput = if seconds > 0.0 {
        report.generated_tokens as f64 / seconds
    } else {
        0.0
    };
    println!(
        "PASS {} prompt_tokens={} generated_tokens={} tool_calls={} history_messages={} duration={:.2}s end_to_end_generated_throughput={:.2}tok/s output={:?}",
        report.name,
        report.prompt_tokens,
        report.generated_tokens,
        report.tool_calls,
        report.history_messages,
        seconds,
        throughput,
        report.response,
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "downloads are opt-in; run tests/download_qwen3.sh and set RIG_CANDLE_TEST_MODEL_DIR"]
async fn pinned_qwen3_model_contract() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let loaded_model = model()?;

    let simple = tokio::time::timeout(Duration::from_secs(300), async {
        loaded_model
            .completion(
                loaded_model
                    .completion_request("Answer with only the capital of France.")
                    .temperature(0.0)
                    .max_tokens(32)
                    .build(),
            )
            .await
    })
    .await??;
    if !simple.raw_response.text.contains("Paris") {
        return Err(format!(
            "model-quality failure in simple completion: {:?}",
            simple.raw_response.text
        )
        .into());
    }
    println!(
        "PASS simple_buffered prompt_tokens={} generated_tokens={} tool_calls=0 duration={}ms throughput={:?} output={:?}",
        simple.raw_response.prompt_tokens,
        simple.raw_response.generated_tokens,
        simple.raw_response.generation_duration_ms,
        simple.raw_response.tokens_per_second,
        simple.raw_response.text,
    );

    let text_parity = tokio::time::timeout(
        Duration::from_secs(600),
        buffered_streaming_text_parity(loaded_model.clone()),
    )
    .await??;
    print_report(&text_parity);

    let optional = tokio::time::timeout(
        Duration::from_secs(900),
        optional_argument(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&optional);

    let parallel = tokio::time::timeout(
        Duration::from_secs(900),
        parallel_tools(
            loaded_model.clone(),
            |builder| builder.temperature(0.0).max_tokens(384),
            None,
        ),
    )
    .await??;
    print_report(&parallel);

    let serial_parallel = tokio::time::timeout(
        Duration::from_secs(900),
        parallel_tools(
            loaded_model.clone(),
            |builder| builder.temperature(0.0).max_tokens(384),
            Some(1),
        ),
    )
    .await??;
    print_report(&serial_parallel);

    let zero_argument = tokio::time::timeout(
        Duration::from_secs(900),
        zero_argument_tool(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&zero_argument);

    let serialized_outputs = tokio::time::timeout(
        Duration::from_secs(900),
        tool_output_serialization(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&serialized_outputs);

    let complex_arguments = tokio::time::timeout(
        Duration::from_secs(900),
        complex_tool_arguments(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&complex_arguments);

    let recovery = tokio::time::timeout(
        Duration::from_secs(900),
        invalid_tool_recovery(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&recovery);

    let hooks = tokio::time::timeout(
        Duration::from_secs(900),
        hook_rewrites_and_request_patch(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&hooks);

    let run_controls = tokio::time::timeout(
        Duration::from_secs(900),
        cancellation_and_max_turns(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&run_controls);

    let extraction = tokio::time::timeout(
        Duration::from_secs(900),
        structured_extraction(loaded_model.clone()),
    )
    .await??;
    print_report(&extraction);

    let sequential = tokio::time::timeout(
        Duration::from_secs(1200),
        sequential_tools(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&sequential);

    let streaming = tokio::time::timeout(
        Duration::from_secs(900),
        streaming_tool(loaded_model.clone(), |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&streaming);

    let structured = tokio::time::timeout(
        Duration::from_secs(900),
        structured_after_tool(loaded_model, |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&structured);

    let streaming_structured = tokio::time::timeout(
        Duration::from_secs(900),
        streaming_structured_after_tool(model()?, |builder| {
            builder.temperature(0.0).max_tokens(384)
        }),
    )
    .await??;
    print_report(&streaming_structured);

    let choices = tokio::time::timeout(
        Duration::from_secs(900),
        rig_core::test_utils::tool_choice_modes(model()?),
    )
    .await??;
    print_report(&choices);

    Ok(())
}
