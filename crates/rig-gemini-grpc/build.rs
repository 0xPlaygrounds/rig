fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    compile_gemini_protos()
}

/// Generate the messages and client, and the descriptor set the canonical
/// proto3 JSON transcode reads, with the `protoc` that `PROTOC` names, else
/// the bundled one.
fn compile_gemini_protos() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("cargo:rerun-if-env-changed=PROTOC");
    let mut includes = vec![std::path::PathBuf::from("proto")];
    if std::env::var_os("PROTOC").is_none() {
        // SAFETY: the build script is single-threaded here.
        unsafe {
            std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
        }
        includes.push(protoc_bin_vendored::include_path()?);
    }
    // A system `protoc` named by `PROTOC` finds the well-known types in its
    // own include directory, and the bundled crate may have no binary for
    // this host at all.
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("gemini_descriptor.bin"))
        .compile_protos(&[std::path::PathBuf::from("proto/gemini.proto")], &includes)?;

    Ok(())
}
