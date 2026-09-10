//! Pins the exact request the futures driver hands the model for a scripted
//! tool turn — preamble, static context, a `CompletionCall` patch
//! (preamble/temperature/max_tokens/tool_choice/active_tools/
//! additional_params/extra_context), an output schema in Tool mode — so
//! `crate::run::prepare::prepare_request` cannot drift from what the agent sent before
//! request preparation moved into the protocol crate. The golden values
//! were captured from the pre-move driver; the cassette suites replay the
//! same bodies against recorded provider traffic.

use super::*;
use crate::agent::{
    AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, HookContext,
};
use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockTurn};
use rig_core::completion::Document;
use serde_json::json;

struct GoldenPatchHook;

impl AgentHook for GoldenPatchHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(
            RequestPatch::new()
                .preamble("patched preamble")
                .temperature(0.25)
                .max_tokens(512)
                .tool_choice(ToolChoice::Required)
                .active_tools(["add"])
                .additional_params(json!({"injected": true, "shared": "hook"}))
                .context(Document {
                    id: "extra".into(),
                    text: "extra context".into(),
                    additional_props: Default::default(),
                }),
        )
    }
}

fn golden_model() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
        MockTurn::text("done"),
    ])
}

fn golden_agent(model: MockCompletionModel) -> Agent {
    AgentBuilder::new(model)
        .preamble("base preamble")
        .context("static context")
        .temperature(0.9)
        .max_tokens(64)
        .additional_params(json!({"base": 1, "shared": "agent"}))
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .output_schema_raw(
            serde_json::from_value(json!({
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"]
            }))
            .expect("valid schema"),
        )
        .add_hook(GoldenPatchHook)
        .build()
}

#[tokio::test]
async fn scripted_tool_turn_requests_match_golden() {
    let model = golden_model();
    let agent = golden_agent(model.clone());
    let _ = agent.runner("add 2 and 3").max_turns(3).run().await;

    let requests = model
        .requests()
        .into_iter()
        .map(|request| serde_json::to_value(&request).expect("serializable request"))
        .collect::<Vec<_>>();
    assert_eq!(
        requests.len(),
        3,
        "a tool turn, a plain-text turn that Tool mode re-prompts, and the retry"
    );

    let golden: Vec<serde_json::Value> = serde_json::from_str(GOLDEN).expect("golden JSON");
    for (turn, (actual, expected)) in requests.iter().zip(&golden).enumerate() {
        assert_eq!(
            actual,
            expected,
            "request for turn {} differs from golden:\n{}",
            turn + 1,
            serde_json::to_string_pretty(actual).unwrap_or_default()
        );
    }
}

const GOLDEN: &str = r#"
[
  {
    "additional_params": {
      "base": 1,
      "injected": true,
      "shared": "hook"
    },
    "chat_history": [
      {
        "content": "patched preamble\n\nWhen you have gathered enough information to answer, call the `final_result` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text.",
        "role": "system"
      },
      {
        "content": [
          {
            "text": "add 2 and 3",
            "type": "text"
          }
        ],
        "role": "user"
      }
    ],
    "documents": [
      {
        "id": "static_doc_0",
        "text": "static context"
      },
      {
        "id": "extra",
        "text": "extra context"
      }
    ],
    "max_tokens": 512,
    "model": null,
    "output_schema": null,
    "temperature": 0.25,
    "tool_choice": "required",
    "tools": [
      {
        "description": "Add x and y together",
        "name": "add",
        "parameters": {
          "properties": {
            "x": {
              "description": "The first number to add",
              "type": "number"
            },
            "y": {
              "description": "The second number to add",
              "type": "number"
            }
          },
          "required": [
            "x",
            "y"
          ],
          "type": "object"
        }
      },
      {
        "description": "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.",
        "name": "final_result",
        "parameters": {
          "properties": {
            "answer": {
              "type": "integer"
            }
          },
          "required": [
            "answer"
          ],
          "type": "object"
        }
      }
    ]
  },
  {
    "additional_params": {
      "base": 1,
      "injected": true,
      "shared": "hook"
    },
    "chat_history": [
      {
        "content": "patched preamble\n\nWhen you have gathered enough information to answer, call the `final_result` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text.",
        "role": "system"
      },
      {
        "content": [
          {
            "text": "add 2 and 3",
            "type": "text"
          }
        ],
        "role": "user"
      },
      {
        "content": [
          {
            "additional_params": null,
            "function": {
              "arguments": {
                "x": 2,
                "y": 3
              },
              "name": "add"
            },
            "id": "tc1",
            "provider": {
              "call_id": "tc1"
            },
            "signature": null,
            "type": "toolcall"
          }
        ],
        "id": null,
        "role": "assistant"
      },
      {
        "content": [
          {
            "call": "tc1",
            "content": [
              {
                "type": "json",
                "value": 5
              }
            ],
            "name": "add",
            "provider": {
              "call_id": "tc1"
            },
            "type": "toolresult"
          }
        ],
        "role": "user"
      }
    ],
    "documents": [
      {
        "id": "static_doc_0",
        "text": "static context"
      },
      {
        "id": "extra",
        "text": "extra context"
      }
    ],
    "max_tokens": 512,
    "model": null,
    "output_schema": null,
    "temperature": 0.25,
    "tool_choice": "required",
    "tools": [
      {
        "description": "Add x and y together",
        "name": "add",
        "parameters": {
          "properties": {
            "x": {
              "description": "The first number to add",
              "type": "number"
            },
            "y": {
              "description": "The second number to add",
              "type": "number"
            }
          },
          "required": [
            "x",
            "y"
          ],
          "type": "object"
        }
      },
      {
        "description": "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.",
        "name": "final_result",
        "parameters": {
          "properties": {
            "answer": {
              "type": "integer"
            }
          },
          "required": [
            "answer"
          ],
          "type": "object"
        }
      }
    ]
  },
  {
    "additional_params": {
      "base": 1,
      "injected": true,
      "shared": "hook"
    },
    "chat_history": [
      {
        "content": "patched preamble\n\nWhen you have gathered enough information to answer, call the `final_result` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text.",
        "role": "system"
      },
      {
        "content": [
          {
            "text": "add 2 and 3",
            "type": "text"
          }
        ],
        "role": "user"
      },
      {
        "content": [
          {
            "additional_params": null,
            "function": {
              "arguments": {
                "x": 2,
                "y": 3
              },
              "name": "add"
            },
            "id": "tc1",
            "provider": {
              "call_id": "tc1"
            },
            "signature": null,
            "type": "toolcall"
          }
        ],
        "id": null,
        "role": "assistant"
      },
      {
        "content": [
          {
            "call": "tc1",
            "content": [
              {
                "type": "json",
                "value": 5
              }
            ],
            "name": "add",
            "provider": {
              "call_id": "tc1"
            },
            "type": "toolresult"
          }
        ],
        "role": "user"
      },
      {
        "content": [
          {
            "text": "done",
            "type": "text"
          }
        ],
        "id": null,
        "role": "assistant"
      },
      {
        "content": [
          {
            "text": "Provide your final answer by calling the `final_result` tool with the structured result as its arguments, not as plain text.",
            "type": "text"
          }
        ],
        "role": "user"
      }
    ],
    "documents": [
      {
        "id": "static_doc_0",
        "text": "static context"
      },
      {
        "id": "extra",
        "text": "extra context"
      }
    ],
    "max_tokens": 512,
    "model": null,
    "output_schema": null,
    "temperature": 0.25,
    "tool_choice": "required",
    "tools": [
      {
        "description": "Add x and y together",
        "name": "add",
        "parameters": {
          "properties": {
            "x": {
              "description": "The first number to add",
              "type": "number"
            },
            "y": {
              "description": "The second number to add",
              "type": "number"
            }
          },
          "required": [
            "x",
            "y"
          ],
          "type": "object"
        }
      },
      {
        "description": "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.",
        "name": "final_result",
        "parameters": {
          "properties": {
            "answer": {
              "type": "integer"
            }
          },
          "required": [
            "answer"
          ],
          "type": "object"
        }
      }
    ]
  }
]
"#;
