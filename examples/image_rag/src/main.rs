//! Multimodal RAG with local CLIP embeddings: index images, find the one a
//! text question is about, and show it to a vision model.
//!
//! The CLIP text and vision models embed into one space, so a text query
//! finds images. Each image is stored as a `Document`, and the agent's
//! dynamic context attaches the retrieved one, image included, to the prompt.
//!
//! Set `OPENAI_API_KEY`, then run (the first run downloads both CLIP models):
//!
//! ```text
//! cargo run -p image_rag -- "Which picture shows a cat?" cat.jpg dog.jpg car.png
//! ```

use anyhow::{Context, Result, ensure};
use rig::completion::Document;
use rig::fastembed::{Fastembed, FastembedImage, FastembedImageModel, FastembedModel};
use rig::message::Image;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let question = args.next().context("pass a question, then image files")?;
    let paths: Vec<String> = args.collect();
    ensure!(!paths.is_empty(), "pass at least one image file");

    let text_model = Fastembed::load(&FastembedModel::ClipVitB32)?
        .embedding(&FastembedModel::ClipVitB32, None)?
        .erase();
    let image_model = FastembedImage::load(&FastembedImageModel::ClipVitB32)?
        .embedding(&FastembedImageModel::ClipVitB32, None)?;

    let mut documents = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = std::fs::read(path).with_context(|| format!("failed to read {path}"))?;
        documents.push(Document::new(path.as_str(), "").with_image(Image::from_bytes(&bytes)));
    }

    // Images go to the CLIP vision model; a document's text, if it had
    // any, would go to the CLIP text model.
    let embeddings = EmbeddingsBuilder::new(text_model.clone())
        .image_model(image_model)
        .documents(documents)?
        .build()
        .await?;
    let index = InMemoryVectorStore::from_documents(embeddings).index(text_model);

    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("Answer from the image you are given, and name its file.")
        .dynamic_context(1, index)
        .build();
    println!("{}", agent.prompt(question).await?.output());

    Ok(())
}
