fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    compile_xai_protos()
}

/// Generate the messages and the `Chat` service client from xAI's protos,
/// and the descriptor set the proto3 JSON transcode reads.
fn compile_xai_protos() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let well_known = protoc_bin_vendored::include_path()?;
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("xai_descriptor.bin"))
        .compile_protos(
            &[std::path::PathBuf::from("proto/xai/api/v1/chat.proto")],
            &[std::path::PathBuf::from("proto"), well_known],
        )?;
    Ok(())
}
