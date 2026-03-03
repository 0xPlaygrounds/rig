use rig::prelude::*;
use rig::{completion::Prompt, providers::gemini};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The difficulty level of a recipe
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// A structured recipe
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct RecipeInfo {
    /// The name of the dish
    name: String,
    /// The type of cuisine (e.g. "Italian", "Japanese", "Mexican")
    cuisine: String,
    /// Time breakdown for the recipe
    timing: Timing,
    /// List of ingredients with quantities
    ingredients: Vec<Ingredient>,
    /// Ordered preparation steps
    steps: Vec<Step>,
    /// Nutritional information per serving
    nutrition: Nutrition,
    /// The difficulty level of the recipe
    difficulty: Difficulty,
}

/// Time breakdown for preparing the recipe
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Timing {
    /// Preparation time in minutes
    prep_minutes: u32,
    /// Cooking time in minutes
    cook_minutes: u32,
    /// Total time in minutes
    total_minutes: u32,
}

/// A single ingredient with its quantity
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Ingredient {
    /// The name of the ingredient
    name: String,
    /// The quantity needed (e.g. "200g", "2 cups", "1 tbsp")
    quantity: String,
    /// Whether a substitution is possible for dietary restrictions
    optional: bool,
}

/// A single preparation step
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Step {
    /// The step number (starting from 1)
    number: u32,
    /// Description of what to do
    instruction: String,
    /// Estimated duration of this step in minutes
    duration_minutes: u32,
}

/// Approximate nutritional information per serving
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Nutrition {
    /// Number of servings this recipe yields
    servings: u32,
    /// Calories per serving
    calories: u32,
    /// Protein in grams per serving
    protein_g: f64,
    /// Fat in grams per serving
    fat_g: f64,
    /// Carbohydrates in grams per serving
    carbs_g: f64,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = gemini::Client::from_env();

    // Build an agent with a structured output schema.
    // The provider will constrain the model's response to valid JSON matching the schema.
    let agent = client
        .agent("gemini-3-flash-preview")
        .preamble("You are a professional chef. Provide detailed and accurate recipes.")
        .output_schema::<RecipeInfo>()
        .build();

    let response = agent
        .prompt("Give me a recipe for spaghetti carbonara.")
        .await?;

    // The response is a JSON string conforming to the RecipeInfo schema.
    let recipe: RecipeInfo = serde_json::from_str(&response)?;

    println!("{}", serde_json::to_string_pretty(&recipe)?);

    Ok(())
}
