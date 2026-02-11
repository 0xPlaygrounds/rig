use rig::prelude::*;
use rig::{completion::Prompt, providers::anthropic};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A structured book review
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct BookReview {
    /// The title of the book
    title: String,
    /// Information about the author
    author: Author,
    /// A rating from 1 to 5
    rating: u8,
    /// A brief summary of the book's plot or content
    summary: String,
    /// Key themes explored in the book, with descriptions
    themes: Vec<Theme>,
    /// The reviewer's recommendation
    recommendation: Recommendation,
}

/// Information about an author
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Author {
    /// The author's full name
    name: String,
    /// The author's nationality
    nationality: String,
    /// Other notable works by this author
    other_works: Vec<String>,
}

/// A theme present in the book
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Theme {
    /// The name of the theme (e.g. "Totalitarianism", "Surveillance")
    name: String,
    /// A short explanation of how this theme is explored in the book
    description: String,
}

/// The reviewer's recommendation
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Recommendation {
    /// Who this book is best suited for
    target_audience: String,
    /// Similar books the reader might enjoy
    similar_books: Vec<SimilarBook>,
}

/// A book similar to the one being reviewed
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct SimilarBook {
    /// The title of the similar book
    title: String,
    /// The author of the similar book
    author: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = anthropic::Client::from_env();

    // Build an agent with a structured output schema.
    // The provider will constrain the model's response to valid JSON matching the schema.
    let agent = client
        .agent("claude-sonnet-4-5")
        .preamble("You are a literary critic. Provide thoughtful and concise book reviews.")
        .output_schema::<BookReview>()
        .build();

    let response = agent
        .prompt("Write a review of '1984' by George Orwell.")
        .await?;

    // The response is a JSON string conforming to the BookReview schema.
    let review: BookReview = serde_json::from_str(&response)?;

    println!("{}", serde_json::to_string_pretty(&review)?);

    Ok(())
}
