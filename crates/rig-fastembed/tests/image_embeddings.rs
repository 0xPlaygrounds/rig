use anyhow::{Result, ensure};
use fastembed::UserDefinedImageEmbeddingModel;
use rig_core::driver::{Local, Model};
use rig_core::embeddings::image_document;
use rig_core::error::ProviderError;
use rig_core::operation::ImageEmbedding;
use rig_core::wire::{Capabilities, Wire};
use rig_fastembed::{FastembedError, FastembedImage, FastembedImageModel, image_embeddings};

const MODEL: &[u8] = include_bytes!("fixtures/image_mean.onnx");
const PREPROCESSOR: &[u8] = include_bytes!("fixtures/image_preprocessor.json");
const RED: &[u8] = include_bytes!("fixtures/red.png");
const GREEN: &[u8] = include_bytes!("fixtures/green.png");

fn fixture_runtime() -> Result<FastembedImage, FastembedError> {
    FastembedImage::from_user_defined(UserDefinedImageEmbeddingModel::new(
        MODEL.to_vec(),
        PREPROCESSOR.to_vec(),
    ))
}

fn fixture_model(
    runtime: FastembedImage,
    ndims: usize,
) -> Model<Local<ImageEmbedding>, FastembedImage> {
    let wire = Local::new("fastembed")
        .with_id("fixture-channel-means")
        .with_capabilities(Capabilities::embedding(1024, ndims).declaring(Some(ndims)));
    Model::new(wire, runtime)
}

#[test]
fn named_model_uses_metadata_or_explicit_dimensions() {
    let wire = image_embeddings(&FastembedImageModel::ClipVitB32, None);
    let descriptor = wire.describe();
    assert_eq!(descriptor.name, "fastembed");
    assert_eq!(descriptor.model, Some("ClipVitB32"));
    assert_eq!(descriptor.capabilities.ndims, 512);
    assert_eq!(descriptor.capabilities.declared, Some(512));

    let wire = image_embeddings(&FastembedImageModel::ClipVitB32, Some(256));
    assert_eq!(wire.describe().capabilities.ndims, 256);
    assert_eq!(wire.describe().capabilities.declared, Some(256));
}

#[tokio::test]
async fn embeds_images_in_order_through_erased_model() -> Result<()> {
    let model = fixture_model(fixture_runtime()?, 3).erase();
    let response = model
        .call(vec![RED.to_vec(), GREEN.to_vec(), RED.to_vec()])
        .await?;
    ensure!(response.embeddings.len() == 3);
    let expected = [
        (RED, [1.0, 0.0, 0.0]),
        (GREEN, [0.0, 1.0, 0.0]),
        (RED, [1.0, 0.0, 0.0]),
    ];
    for (embedding, (bytes, vector)) in response.embeddings.iter().zip(expected) {
        ensure!(embedding.document == image_document(bytes));
        ensure!(embedding.vec.len() == 3);
        for (actual, expected) in embedding.vec.iter().zip(vector) {
            ensure!(actual.is_finite() && (actual - expected).abs() < 1e-6);
        }
    }
    ensure!(response.provider == "fastembed");
    ensure!(response.raw.is_null());
    ensure!(response.provider_request_id.is_none());
    ensure!(response.usage == Default::default());
    Ok(())
}

#[tokio::test]
async fn empty_batch_returns_no_embeddings() -> Result<()> {
    let model = fixture_model(fixture_runtime()?, 3);
    let response = model.call(Vec::new()).await?;
    ensure!(response.embeddings.is_empty());
    Ok(())
}

#[tokio::test]
async fn invalid_image_fails_batch_and_runtime_remains_usable() -> Result<()> {
    let runtime = fixture_runtime()?;
    let model = fixture_model(runtime.clone(), 3);
    let result = model
        .call(vec![RED.to_vec(), b"invalid image".to_vec()])
        .await;
    ensure!(matches!(result, Err(ProviderError::Provider(_))));

    let response = fixture_model(runtime, 3).call(vec![GREEN.to_vec()]).await?;
    ensure!(response.embeddings.len() == 1);
    ensure!(
        response
            .embeddings
            .first()
            .is_some_and(|embedding| embedding.document == image_document(GREEN))
    );
    Ok(())
}

#[tokio::test]
async fn rejects_vectors_that_disagree_with_declared_dimensions() -> Result<()> {
    let model = fixture_model(fixture_runtime()?, 4);
    let result = model.call(vec![RED.to_vec()]).await;
    ensure!(matches!(
        result,
        Err(ProviderError::MismatchedDimensions { .. })
    ));
    Ok(())
}

#[test]
fn invalid_model_returns_initialization_error() {
    let result = FastembedImage::from_user_defined(UserDefinedImageEmbeddingModel::new(
        b"invalid onnx".to_vec(),
        PREPROCESSOR.to_vec(),
    ));
    assert!(matches!(result, Err(FastembedError::Initialization(_))));
}
