//! Generator/evaluator loop. The generator is an [`Agent`], and the evaluator
//! uses the same agent's non-generic fluent extraction runner.
//! Requires `OPENAI_API_KEY`.
use rig::extract::ExtractOptions;
use rig::prelude::*;

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
    let client = openai::Client::from_env()?;
    let generator_agent = client
        .agent(openai::GPT_4)
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
    let evaluator_preamble = format!(
        "{extraction_preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{EVALUATOR_ROLE}"
    );
    let evaluator_agent = client.agent(openai::GPT_4).build();

    let mut memories: Vec<String> = Vec::new();
    let mut response = generator_agent.prompt(TASK).await?;
    memories.push(response.clone());

    loop {
        let eval_result: Evaluation = evaluator_agent
            .extractor(format!("{TASK}\n\n{response}"))
            .classic()
            .preamble(evaluator_preamble.clone())
            .run()
            .await?;
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
