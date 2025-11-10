use anyhow::Context;
use google_cloud_aiplatform_v1 as vertexai;

// Example of using vertexai without Rig in order to put the Rig integration into context

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    const MODEL: &str = "gemini-2.5-flash-lite";
    // google-cloud-auth does not read ~/.config/gcloud/configurations so requiring that
    // project be set by env var for this example
    let project_id: String = std::env::var("GOOGLE_CLOUD_PROJECT")
        .context("GOOGLE_CLOUD_PROJECT env var must be set to run this example")?;

    // implicit ADC auth here, but builder can include a .with_credentials method
    let client = vertexai::client::PredictionService::builder()
        .build()
        .await?;

    let model = format!("projects/{project_id}/locations/global/publishers/google/models/{MODEL}");

    // generating content means sending an Iterable of Content objects that contain role / data
    let user_part = vertexai::model::Part::new()
        .set_text("Name a significant contributor to the Rust programming language?");

    let user_content = vertexai::model::Content::new()
        .set_role("user")
        .set_parts([user_part]);

    // The GenerationConfig can set things like max tokens, temperature, response schema, etc
    let generation_config = vertexai::model::GenerationConfig::new().set_candidate_count(1);

    let response = client
        .generate_content()
        .set_model(&model)
        .set_contents([user_content])
        .set_generation_config(generation_config)
        .send()
        .await;

    // see response:#? for full response (list of candidates, token usage, etc)
    let response = response?;
    let candidate = response
        .candidates
        .first()
        .context("No candidates in response")?;
    let content = candidate
        .content
        .as_ref()
        .context("No content in candidate")?;
    let part = content.parts.first().context("No parts in content")?;

    let output = part.text().context("Part does not contain text data")?;

    println!("OUTPUT = {output}");
    Ok(())
}
