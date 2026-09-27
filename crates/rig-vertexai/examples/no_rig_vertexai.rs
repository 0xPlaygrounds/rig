use google_cloud_aiplatform_v1 as vertexai;
use rig_core::RigError;
use rig_core::error::ErrorKind;

// Example of using vertexai without Rig in order to put the Rig integration into context

#[tokio::main]
async fn main() -> Result<(), RigError> {
    const MODEL: &str = "gemini-2.5-flash-lite";
    // google-cloud-auth does not read ~/.config/gcloud/configurations so requiring that
    // project be set by env var for this example
    let project_id: String = rig_core::client::env::required("GOOGLE_CLOUD_PROJECT")?;

    // implicit ADC auth here, but builder can include a .with_credentials method
    let client = vertexai::client::PredictionService::builder()
        .build()
        .await
        .map_err(RigError::other)?;

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
    let response = response.map_err(RigError::other)?;
    let candidate = response
        .candidates
        .first()
        .ok_or_else(|| RigError::new(ErrorKind::Other, "No candidates in response"))?;
    let content = candidate
        .content
        .as_ref()
        .ok_or_else(|| RigError::new(ErrorKind::Other, "No content in candidate"))?;
    let part = content
        .parts
        .first()
        .ok_or_else(|| RigError::new(ErrorKind::Other, "No parts in content"))?;

    let output = part
        .text()
        .ok_or_else(|| RigError::new(ErrorKind::Other, "Part does not contain text data"))?;

    println!("OUTPUT = {output}");
    Ok(())
}
