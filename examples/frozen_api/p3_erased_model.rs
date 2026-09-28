use rig::DynModel;
use rig::error::ProviderError;
use rig::operation::Completion;
use rig::providers::openai::{self, OpenAI};

#[derive(Clone)]
struct AppState {
    model: DynModel<Completion>,
}

/// A request handler: it names the operation, never the provider.
async fn ask(state: AppState, question: String) -> Result<String, ProviderError> {
    let response = state.model.call(question).await?;
    Ok(response.text())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState {
        model: OpenAI::from_env()?.completion(openai::GPT_5_2).erase(),
    };

    let handler = tokio::spawn(ask(state.clone(), "What is a monad?".to_owned()));
    println!("{}", handler.await??);
    Ok(())
}
