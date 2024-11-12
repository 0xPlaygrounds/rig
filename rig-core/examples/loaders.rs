use rig::loaders::FileLoader;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    FileLoader::with_glob("cargo.toml")?
        .read()
        .into_iter()
        .for_each(|result| match result {
            Ok(content) => println!("{}", content),
            Err(e) => eprintln!("Error reading file: {}", e),
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example() -> Result<(), anyhow::Error> {
        main()
    }
}
