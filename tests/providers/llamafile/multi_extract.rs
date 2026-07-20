//! Llamafile live coverage for batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::client::CompletionClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

use super::support;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Names {
    names: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Topics {
    topics: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Sentiment {
    sentiment: f64,
    confidence: f64,
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn batch_multi_extract_chain() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client();
    let model = support::model_name();
    let names_extractor = client
        .extractor::<Names>(model.clone())
        .preamble("Extract names from the given text.")
        .retries(2)
        .build();
    let topics_extractor = client
        .extractor::<Topics>(model.clone())
        .preamble("Extract topics from the given text.")
        .retries(2)
        .build();
    let sentiment_extractor = client
        .extractor::<Sentiment>(model)
        .preamble(
            "Extract sentiment and confidence from the given text. \
             Return sentiment normalized to the range [-1.0, 1.0] and confidence normalized to [0.0, 1.0].",
        )
        .retries(2)
        .build();

    let inputs = vec![
        "Ada Lovelace discussed analytical engines and early programming with Charles Babbage.",
        "Grace said she hates rainy weather but still walked her dog to the park.",
        "Linus is going to the store to buy milk and bread for dinner.",
    ];
    let responses: Vec<String> = futures::stream::iter(inputs)
        .map(|text| {
            let names_extractor = &names_extractor;
            let topics_extractor = &topics_extractor;
            let sentiment_extractor = &sentiment_extractor;
            async move {
                let (names, topics, sentiment) = futures::try_join!(
                    names_extractor.extract(text),
                    topics_extractor.extract(text),
                    sentiment_extractor.extract(text),
                )?;
                anyhow::Ok(format!(
                    "Extracted names: {}\nExtracted topics: {}\nExtracted sentiment: {} ({})",
                    names.names.join(", "),
                    topics.topics.join(", "),
                    sentiment.sentiment,
                    sentiment.confidence,
                ))
            }
        })
        .buffered(4)
        .try_collect()
        .await?;

    anyhow::ensure!(responses.len() == 3);
    for response in responses {
        assert_nonempty_response(&response);
    }

    Ok(())
}
