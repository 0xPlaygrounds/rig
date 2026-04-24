fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    compile_gemini_protos()
}

fn compile_gemini_protos() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile_protos(&["proto/gemini.proto"], &["proto"])?;

    Ok(())
}
