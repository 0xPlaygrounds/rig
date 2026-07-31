//! Generator/evaluator loop. The generator is an [`Agent`]; the evaluator is
//! structured extraction, which is a free function now — the extractor builder
//! is gone, so the evaluator's instructions are appended to the classic
//! extraction preamble instead.
//!
//! Both stages share one piece of plain data: an
//! `openai::functions::Config` (which names the model), wrapped in
//! [`ProviderConfig`] for the agent and passed straight to the extraction call.
//! Requires `OPENAI_API_KEY`.
use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::provider::Runtime;

use rig::providers::openai;

use schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema, serde::Serialize, Debug)]
struct Evaluation {
    evaluation_status: EvalStatus,
    feedback: String,
}
#[derive(serde::Deserialize, JsonSchema, serde::Serialize, Debug, PartialEq)]
enum EvalStatus {
    Pass,
    NeedsImprovement,
    Fail,
}
const TASK: &str = "Implement a Stack with:
1. push(x)
2. pop()
3. getMin()
All operations should be O(1).
";
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Providers are plain data: one config names the model, and it is shared
    // (cloned) by both the generator agent and the evaluator extraction.
    let cfg = openai::functions::Config::from_env(openai::GPT_4)?;

    let generator_agent = AgentBuilder::new(cfg.clone())
        .preamble(
            "
            Your goal is to complete the task based on <user input>. If there are feedback
            from your previous generations, you should reflect on them to improve your solution

            Output your answer concisely in the following format:

            Thoughts:
            [Your understanding of the task and feedback and how you plan to improve]

            Response:
            [Your code implementation here]
        ",
        )
        .build();

    const EVALUATOR_ROLE: &str = "
            Evaluate this following code implementation for:
            1. code correctness
            2. time complexity
            3. style and best practices

            You should be evaluating only and not attempting to solve the task.

            Only output \"PASS\" if all criteria are met and you have no further suggestions for improvements.

            Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.

            Only output JSON.
        ";
    let classic = ExtractOptions::classic_extractor();
    let extraction_preamble = classic.preamble.clone().unwrap_or_default();
    let evaluator_options = classic.with_preamble(format!(
        "{extraction_preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{EVALUATOR_ROLE}"
    ));
    let evaluator_provider = ProviderConfig::OpenAi(cfg);
    let rt = Arc::new(Runtime::new());

    let mut memories: Vec<String> = Vec::new();
    let mut response = generator_agent.prompt(TASK).await?;
    memories.push(response.clone());

    loop {
        let eval_result = extract_with_options::<Evaluation>(
            AgentConfig::new(),
            evaluator_provider.clone(),
            rt.clone(),
            format!("{TASK}\n\n{response}"),
            evaluator_options.clone(),
        )
        .await?
        .value;
        if eval_result.evaluation_status == EvalStatus::Pass {
            break;
        } else {
            let context = format!("{TASK}\n\n{}", eval_result.feedback);
            response = generator_agent.prompt(context).await?;
            memories.push(response.clone());
        }
    }

    println!("Response: {response}");
    Ok(())
}
