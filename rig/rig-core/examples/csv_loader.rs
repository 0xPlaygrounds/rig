//! Example demonstrating the CSV loader functionality.
//!
//! This example shows how to load CSV files and use them as context for an agent.
//!
//! Run with: `cargo run --example csv_loader --features csv`

use rig::{
    completion::Prompt,
    loaders::{CsvConfig, CsvFileLoader},
    providers::openai,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Example 1: Load a single CSV file
    println!("=== Example 1: Load single CSV file ===\n");
    
    // Create a sample CSV file for demonstration
    let csv_content = r#"name,role,department
Alice Smith,Engineer,Engineering
Bob Johnson,Designer,Design
Carol Williams,Manager,Engineering
David Brown,Analyst,Finance"#;
    
    // Load from bytes (simulating file load)
    let document = CsvFileLoader::from_bytes(csv_content.as_bytes(), CsvConfig::default())?;
    println!("Loaded CSV as document:\n{}\n", document);

    // Example 2: Load CSV files with glob pattern
    println!("=== Example 2: Load with glob pattern ===\n");
    
    // In a real scenario, you would use:
    // let loader = CsvFileLoader::with_glob("data/*.csv")?;
    // For this example, we'll show the API:
    println!("CsvFileLoader::with_glob(\"data/*.csv\")?.load().ignore_errors()");
    println!("  -> Returns an iterator of String documents\n");

    // Example 3: Load rows individually
    println!("=== Example 3: Load rows individually ===\n");
    
    let rows: Vec<String> = vec![
        "name: Alice Smith\nrole: Engineer\ndepartment: Engineering".to_string(),
        "name: Bob Johnson\nrole: Designer\ndepartment: Design".to_string(),
    ];
    
    for (i, row) in rows.iter().enumerate() {
        println!("Row {}:\n{}\n", i + 1, row);
    }

    // Example 4: Use with different delimiters (TSV)
    println!("=== Example 4: TSV support ===\n");
    
    let tsv_content = "name\trole\tdepartment\nAlice\tEngineer\tEngineering";
    let tsv_document = CsvFileLoader::from_bytes(tsv_content.as_bytes(), CsvConfig::tsv())?;
    println!("TSV document:\n{}\n", tsv_document);

    // Example 5: Use CSV as agent context
    println!("=== Example 5: Using CSV as agent context ===\n");
    
    // This requires OPENAI_API_KEY to be set
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let openai_client = openai::Client::from_env();
        
        let agent = openai_client
            .agent("gpt-4")
            .preamble("You are a helpful assistant that answers questions about company employees based on the provided data.")
            .context(&document)
            .build();

        let response = agent
            .prompt("Who works in the Engineering department?")
            .await?;

        println!("Agent response: {}", response);
    } else {
        println!("Skipping agent example (OPENAI_API_KEY not set)");
        println!("To run with an agent, set OPENAI_API_KEY environment variable");
    }

    // Example 6: Load with path tracking
    println!("\n=== Example 6: Load with path tracking ===\n");
    println!("CsvFileLoader::with_glob(\"data/*.csv\")?");
    println!("    .load_with_path()");
    println!("    .ignore_errors()");
    println!("  -> Returns (PathBuf, String) pairs\n");

    // Example 7: Custom configuration
    println!("=== Example 7: Custom configuration ===\n");
    
    let custom_config = CsvConfig::new()
        .delimiter(b';')
        .has_headers(true)
        .trim(true)
        .flexible(false);
    
    let semicolon_csv = "name;age;city\nAlice;30;New York\nBob;25;Los Angeles";
    let custom_document = CsvFileLoader::from_bytes(semicolon_csv.as_bytes(), custom_config)?;
    println!("Semicolon-delimited CSV:\n{}\n", custom_document);

    println!("=== CSV Loader Examples Complete ===");
    
    Ok(())
}
