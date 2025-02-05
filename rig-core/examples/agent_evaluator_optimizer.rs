use std::env;

use rig::{completion::Prompt, providers::openai::Client};
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
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let generator_agent = openai_client
        .agent("gpt-4")
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

    let evaluator_agent = openai_client.extractor::<Evaluation>("gpt-4")
        .preamble("
            Evaluate this following code implementation for:
            1. code correctness
            2. time complexity
            3. style and best practices

            You should be evaluating only and not attempting to solve the task.

            Only output \"PASS\" if all criteria are met and you have no further suggestions for improvements.

            Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.

            Only output JSON.
        ")
        .build();

    let mut memories: Vec<String> = Vec::new();

    let mut response = generator_agent.prompt(TASK).await.unwrap();
    memories.push(response.clone());

    loop {
        let eval_result = evaluator_agent
            .extract(&format!("{TASK}\n\n{response}"))
            .await
            .unwrap();

        if eval_result.evaluation_status == EvalStatus::Pass {
            break;
        } else {
            let context = format!("{TASK}\n\n{}", eval_result.feedback);

            response = generator_agent.prompt(context).await.unwrap();
            memories.push(response.clone());
        }
    }

    println!("Response: {response}");

    Ok(())
}
