//! The ECS contract matrix on the OpenAI Chat Completions wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default; the route `gpt-5-nano`): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix_chat.rs`, and asserted against that producer's golden,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(client: &rig::providers::openai::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client
            .clone()
            .completions_api()
            .completion_model(GPT_5_MINI),
        route: Some(
            client
                .clone()
                .completions_api()
                .completion_model(GPT_5_NANO),
        ),
        temperature: None,
        additional_params: None,
    }
}

#[tokio::test]
async fn endings_tool_dispatch_cancelled() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_tool_dispatch_cancelled",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TOOL_DISPATCH_CANCELLED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_tool_dispatch_cancelled",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_tool_outcome_cancelled() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_tool_outcome_cancelled",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TOOL_OUTCOME_CANCELLED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_tool_outcome_cancelled",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_answer_outcome_cancelled() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_answer_outcome_cancelled",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_ANSWER_OUTCOME_CANCELLED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_answer_outcome_cancelled",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_turn_finished_stop() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_turn_finished_stop",
        |client| async move {
            run_world(&wire(&client), &cells::ENDINGS_TURN_FINISHED_STOP, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_endings_turn_finished_stop", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_answer_turn_stop() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_answer_turn_stop",
        |client| async move {
            run_world(&wire(&client), &cells::ENDINGS_ANSWER_TURN_STOP, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_endings_answer_turn_stop", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_text_delta_stop() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_text_delta_stop",
        |client| async move {
            run_world(&wire(&client), &cells::ENDINGS_TEXT_DELTA_STOP, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_endings_text_delta_stop", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_tool_call_delta_stop() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_tool_call_delta_stop",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TOOL_CALL_DELTA_STOP,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_tool_call_delta_stop",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_tool_dispatch_cancelled_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_tool_dispatch_cancelled_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TOOL_DISPATCH_CANCELLED_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_tool_dispatch_cancelled_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_turn_finished_stop_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_turn_finished_stop_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TURN_FINISHED_STOP_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_turn_finished_stop_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn endings_tool_outcome_cancelled_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/endings_tool_outcome_cancelled_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::ENDINGS_TOOL_OUTCOME_CANCELLED_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_endings_tool_outcome_cancelled_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_observe_everything() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_observe_everything",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_OBSERVE_EVERYTHING, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_observe_everything", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_patch_tool_args() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_patch_tool_args",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_PATCH_TOOL_ARGS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_patch_tool_args", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_patch_tool_args_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_patch_tool_args_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOOKS_PATCH_TOOL_ARGS_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_hooks_patch_tool_args_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_deny_tool() {
    with_openai_cassette("corpus_matrix_chat/hooks_deny_tool", |client| async move {
        run_world(&wire(&client), &cells::HOOKS_DENY_TOOL, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_hooks_deny_tool", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn hooks_deny_tool_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_deny_tool_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_DENY_TOOL_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_deny_tool_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_replace_tool_result() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_replace_tool_result",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_REPLACE_TOOL_RESULT, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_replace_tool_result", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_replace_answer() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_replace_answer",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_REPLACE_ANSWER, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_replace_answer", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_preamble_override() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_preamble_override",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_PREAMBLE_OVERRIDE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_preamble_override", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_demand_done() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_demand_done",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_DEMAND_DONE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_demand_done", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_lookup_before_run() {
    with_openai_cassette(
        "corpus_matrix_chat/hooks_lookup_before_run",
        |client| async move {
            run_world(&wire(&client), &cells::HOOKS_LOOKUP_BEFORE_RUN, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_hooks_lookup_before_run", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_two_hooks() {
    with_openai_cassette("corpus_matrix_chat/hooks_two_hooks", |client| async move {
        run_world(&wire(&client), &cells::HOOKS_TWO_HOOKS, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_hooks_two_hooks", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn host_custom_at_start() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_start",
        |client| async move {
            run_world(&wire(&client), &cells::HOST_CUSTOM_AT_START, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_host_custom_at_start", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_at_completion_call() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_completion_call",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOST_CUSTOM_AT_COMPLETION_CALL,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_host_custom_at_completion_call",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_at_outcome() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_outcome",
        |client| async move {
            run_world(&wire(&client), &cells::HOST_CUSTOM_AT_OUTCOME, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_host_custom_at_outcome", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_at_settled() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_settled",
        |client| async move {
            run_world(&wire(&client), &cells::HOST_CUSTOM_AT_SETTLED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_host_custom_at_settled", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_start_and_settled() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_start_and_settled",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOST_CUSTOM_START_AND_SETTLED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_host_custom_start_and_settled",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_twice_serial() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_twice_serial",
        |client| async move {
            run_world(&wire(&client), &cells::HOST_CUSTOM_TWICE_SERIAL, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_host_custom_twice_serial", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_twice_concurrent() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_twice_concurrent",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOST_CUSTOM_TWICE_CONCURRENT,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_host_custom_twice_concurrent",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_at_start_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_start_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOST_CUSTOM_AT_START_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_host_custom_at_start_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_at_outcome_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_at_outcome_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::HOST_CUSTOM_AT_OUTCOME_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_host_custom_at_outcome_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn host_custom_unserved() {
    with_openai_cassette(
        "corpus_matrix_chat/host_custom_unserved",
        |client| async move {
            run_world(&wire(&client), &cells::HOST_CUSTOM_UNSERVED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_host_custom_unserved", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_serial_concurrency_one() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_serial_concurrency_one",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SERVING_SERIAL_CONCURRENCY_ONE,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_serving_serial_concurrency_one",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_concurrent_concurrency_one() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_concurrent_concurrency_one",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SERVING_CONCURRENT_CONCURRENCY_ONE,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_serving_concurrent_concurrency_one",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_concurrent_concurrency_two() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_concurrent_concurrency_two",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SERVING_CONCURRENT_CONCURRENCY_TWO,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_serving_concurrent_concurrency_two",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_concurrent_concurrency_two_events() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_concurrent_concurrency_two_events",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_serving_concurrent_concurrency_two_events",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_capacity_one() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_capacity_one",
        |client| async move {
            run_world(&wire(&client), &cells::SERVING_CAPACITY_ONE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_serving_capacity_one", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_serial_memory_tools() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_serial_memory_tools",
        |client| async move {
            run_world(&wire(&client), &cells::SERVING_SERIAL_MEMORY_TOOLS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_serving_serial_memory_tools", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_model_route() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_model_route",
        |client| async move {
            run_world(&wire(&client), &cells::SERVING_MODEL_ROUTE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_serving_model_route", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_model_route_unselected() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_model_route_unselected",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SERVING_MODEL_ROUTE_UNSELECTED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_serving_model_route_unselected",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn serving_host_bus() {
    with_openai_cassette("corpus_matrix_chat/serving_host_bus", |client| async move {
        run_world(&wire(&client), &cells::SERVING_HOST_BUS, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_serving_host_bus", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn serving_host_bus_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/serving_host_bus_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::SERVING_HOST_BUS_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_serving_host_bus_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_deny_tool() {
    with_openai_cassette("corpus_matrix_chat/layers_deny_tool", |client| async move {
        run_world(&wire(&client), &cells::LAYERS_DENY_TOOL, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_layers_deny_tool", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn layers_patch_tool_args() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_patch_tool_args",
        |client| async move {
            run_world(&wire(&client), &cells::LAYERS_PATCH_TOOL_ARGS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_layers_patch_tool_args", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_replace_tool_result() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_replace_tool_result",
        |client| async move {
            run_world(&wire(&client), &cells::LAYERS_REPLACE_TOOL_RESULT, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_layers_replace_tool_result", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_two_layers() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_two_layers",
        |client| async move {
            run_world(&wire(&client), &cells::LAYERS_TWO_LAYERS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_layers_two_layers", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_host_deny_over_host_bus() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_host_deny_over_host_bus",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::LAYERS_HOST_DENY_OVER_HOST_BUS,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_layers_host_deny_over_host_bus",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_patch_beneath_hook_patch() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_patch_beneath_hook_patch",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::LAYERS_PATCH_BENEATH_HOOK_PATCH,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_layers_patch_beneath_hook_patch",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn layers_memory_load_replaced() {
    with_openai_cassette(
        "corpus_matrix_chat/layers_memory_load_replaced",
        |client| async move {
            run_world(&wire(&client), &cells::LAYERS_MEMORY_LOAD_REPLACED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_layers_memory_load_replaced", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_clear_at_start() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_clear_at_start",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_CLEAR_AT_START, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_clear_at_start", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_clear_at_settled() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_clear_at_settled",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_CLEAR_AT_SETTLED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_clear_at_settled", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_two_runs() {
    with_openai_cassette("corpus_matrix_chat/memory_two_runs", |client| async move {
        run_world(&wire(&client), &cells::MEMORY_TWO_RUNS, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_memory_two_runs", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn memory_two_runs_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_two_runs_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_TWO_RUNS_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_two_runs_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_clear_at_settled_two_runs() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_clear_at_settled_two_runs",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::MEMORY_CLEAR_AT_SETTLED_TWO_RUNS,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_memory_clear_at_settled_two_runs",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_clear_at_start_two_runs() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_clear_at_start_two_runs",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::MEMORY_CLEAR_AT_START_TWO_RUNS,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_memory_clear_at_start_two_runs",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_history_bypass() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_history_bypass",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_HISTORY_BYPASS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_history_bypass", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_host_bus_memory() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_host_bus_memory",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_HOST_BUS_MEMORY, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_host_bus_memory", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_serial_two_tools() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_serial_two_tools",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_SERIAL_TWO_TOOLS, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_serial_two_tools", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_failing_append() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_failing_append",
        |client| async move {
            run_world(&wire(&client), &cells::MEMORY_FAILING_APPEND, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_memory_failing_append", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn memory_failing_append_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/memory_failing_append_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::MEMORY_FAILING_APPEND_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_memory_failing_append_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_unary() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_unary",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_TOOL_UNARY, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_tool_unary", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_TOOL_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_tool_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_prompted_unary() {
    with_openai_cassette(
        "corpus_matrix_chat/output_prompted_unary",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_PROMPTED_UNARY, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_prompted_unary", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_prompted_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/output_prompted_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_PROMPTED_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_prompted_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_with_real_tool() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_with_real_tool",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_TOOL_WITH_REAL_TOOL, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_tool_with_real_tool", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_prompted_with_real_tool() {
    with_openai_cassette(
        "corpus_matrix_chat/output_prompted_with_real_tool",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::OUTPUT_PROMPTED_WITH_REAL_TOOL,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_output_prompted_with_real_tool",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_choice_specific_output() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_choice_specific_output",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_output_tool_choice_specific_output",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_choice_required() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_choice_required",
        |client| async move {
            run_world(&wire(&client), &cells::OUTPUT_TOOL_CHOICE_REQUIRED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_output_tool_choice_required", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_under_none_degrades() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_under_none_degrades",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_output_tool_under_none_degrades",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_tool_thinking() {
    with_openai_cassette(
        "corpus_matrix_chat/output_tool_thinking",
        |client| async move {
            run_world(
                &reasoning_wire(&client),
                &cells::OUTPUT_TOOL_THINKING,
                |log| crate::ecs_goldens::golden_effects("openai_chat_output_tool_thinking", log),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_tool_choice_required_first() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_tool_choice_required_first",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_TOOL_CHOICE_REQUIRED_FIRST,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_tool_choice_required_first",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_tool_choice_none_on_committed_output() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_tool_choice_none_on_committed_output",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_tool_choice_none_on_committed_output",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_extra_context() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_extra_context",
        |client| async move {
            run_world(&wire(&client), &cells::SHAPING_EXTRA_CONTEXT, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_shaping_extra_context", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_extra_context_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_extra_context_streamed",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_EXTRA_CONTEXT_STREAMED,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_extra_context_streamed",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_merged_three() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_merged_three",
        |client| async move {
            run_world(&wire(&client), &cells::SHAPING_MERGED_THREE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_shaping_merged_three", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_route_on_first_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_route_on_first_turn",
        |client| async move {
            run_world(&wire(&client), &cells::SHAPING_ROUTE_ON_FIRST_TURN, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_shaping_route_on_first_turn", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_late_route() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_late_route",
        |client| async move {
            run_world(&wire(&client), &cells::SHAPING_LATE_ROUTE, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_shaping_late_route", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_max_tokens_second_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_max_tokens_second_turn",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_MAX_TOKENS_SECOND_TURN,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_max_tokens_second_turn",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[ignore = "gpt-5-mini low reported zero reasoning tokens on the second turn in all three attempts; record-openai-chat-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
#[tokio::test]
async fn shaping_thinking_second_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_thinking_second_turn",
        |client| async move {
            run_world(
                &reasoning_wire(&client),
                &cells::SHAPING_THINKING_SECOND_TURN,
                |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_preamble_second_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_preamble_second_turn",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_PREAMBLE_SECOND_TURN,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_preamble_second_turn",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_active_tools_none_second_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_active_tools_none_second_turn",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_shaping_active_tools_none_second_turn",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn shaping_history_first_turn() {
    with_openai_cassette(
        "corpus_matrix_chat/shaping_history_first_turn",
        |client| async move {
            run_world(&wire(&client), &cells::SHAPING_HISTORY_FIRST_TURN, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_shaping_history_first_turn", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn causal_completion_serial() {
    with_openai_cassette(
        "corpus_matrix_chat/causal_completion_serial",
        |client| async move {
            run_world(&wire(&client), &cells::CAUSAL_COMPLETION_SERIAL, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_causal_completion_serial", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn causal_completion_concurrent() {
    with_openai_cassette(
        "corpus_matrix_chat/causal_completion_concurrent",
        |client| async move {
            run_world(
                &wire(&client),
                &cells::CAUSAL_COMPLETION_CONCURRENT,
                |log| {
                    crate::ecs_goldens::golden_effects(
                        "openai_chat_causal_completion_concurrent",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn causal_completion_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/causal_completion_streamed",
        |client| async move {
            run_world(&wire(&client), &cells::CAUSAL_COMPLETION_STREAMED, |log| {
                crate::ecs_goldens::golden_effects("openai_chat_causal_completion_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn resume_tool_turn() {
    with_openai_cassette("corpus_matrix_chat/resume_tool_turn", |client| async move {
        run_world(&wire(&client), &cells::RESUME_TOOL_TURN, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_resume_tool_turn", log)
        })
        .await;
    })
    .await;
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(
    client: &rig::providers::openai::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client
            .clone()
            .completions_api()
            .completion_model(rig::providers::openai::GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

#[ignore = "gpt-5-mini at reasoning_effort low reported zero reasoning tokens in attempts 1, 2 and 3 (2026-09-13, record-openai-chat-text-unary-attempt-{1,2,3}.log); exhausted the prompt's three-attempt limit"]
#[tokio::test]
async fn reasoning_text_unary() {
    with_openai_cassette("reasoning_matrix_chat/text_unary", |client| async move {
        run_world(
            &reasoning_wire(&client),
            &cells::REASONING_TEXT_UNARY,
            |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn reasoning_text_streamed() {
    with_openai_cassette("reasoning_matrix_chat/text_streamed", |client| async move {
        run_world(
            &reasoning_wire(&client),
            &cells::REASONING_TEXT_STREAMED,
            |log| crate::ecs_goldens::golden_effects("openai_chat_reasoning_text_streamed", log),
        )
        .await;
    })
    .await;
}

#[tokio::test]
#[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in attempts 1, 2 and 3; record-openai-chat-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
async fn reasoning_tool_unary() {
    with_openai_cassette("reasoning_matrix_chat/tool_unary", |client| async move {
        run_world(
            &reasoning_wire(&client),
            &cells::REASONING_TOOL_UNARY,
            |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
        )
        .await;
    })
    .await;
}

#[tokio::test]
#[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in all three attempts; record-openai-chat-tool-streamed-attempt-{1,2,3}.log; three attempts exhausted"]
async fn reasoning_tool_streamed() {
    with_openai_cassette("reasoning_matrix_chat/tool_streamed", |client| async move {
        run_world(
            &reasoning_wire(&client),
            &cells::REASONING_TOOL_STREAMED,
            |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn reasoning_off() {
    with_openai_cassette("reasoning_matrix_chat/off", |client| async move {
        run_world(&reasoning_wire(&client), &cells::REASONING_OFF, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_reasoning_off", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn reasoning_capped() {
    with_openai_cassette("reasoning_matrix_chat/capped", |client| async move {
        run_world(&reasoning_wire(&client), &cells::REASONING_CAPPED, |log| {
            crate::ecs_goldens::golden_effects("openai_chat_reasoning_capped", log)
        })
        .await;
    })
    .await;
}
