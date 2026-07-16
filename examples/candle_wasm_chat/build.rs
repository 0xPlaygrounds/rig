use std::{
    env, fs,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

struct Artifact {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

const ARTIFACTS: [Artifact; 3] = [
    Artifact {
        name: "config.json",
        size: 846,
        sha256: "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee",
    },
    Artifact {
        name: "tokenizer.json",
        size: 2_104_556,
        sha256: "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    },
    Artifact {
        name: "model.gguf",
        size: 270_590_880,
        sha256: "2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2",
    },
];

fn validate_artifact(path: &Path, artifact: &Artifact) -> io::Result<()> {
    let actual_size = path.metadata()?.len();
    if actual_size != artifact.size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} has {actual_size} bytes; expected {}. Run ./download_model.sh again",
                path.display(),
                artifact.size
            ),
        ));
    }

    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} returned an invalid read length", path.display()),
            )
        })?;
        hasher.update(chunk);
    }
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != artifact.sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} failed SHA-256 verification. Run ./download_model.sh again",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "CARGO_MANIFEST_DIR is not set")
        })?);
    let output_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is not set"))?,
    );
    let model_dir = env::var_os("MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("model"));
    let present_artifacts = ARTIFACTS
        .iter()
        .filter(|artifact| model_dir.join(artifact.name).is_file())
        .count();
    let complete_model = present_artifacts == ARTIFACTS.len();
    if present_artifacts != 0 && !complete_model {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} contains only {present_artifacts} of {} required artifacts; run ./download_model.sh again",
                model_dir.display(),
                ARTIFACTS.len()
            ),
        ));
    }

    println!("cargo:rerun-if-env-changed=MODEL_DIR");
    println!("cargo:rerun-if-changed={}", model_dir.display());
    for artifact in &ARTIFACTS {
        let source = model_dir.join(artifact.name);
        let destination = output_dir.join(artifact.name);
        println!("cargo:rerun-if-changed={}", source.display());
        if complete_model {
            validate_artifact(&source, artifact)?;
            fs::copy(source, destination)?;
        } else {
            fs::write(destination, [])?;
        }
    }

    println!(
        "cargo:rustc-env=RIG_CANDLE_WASM_MODEL_EMBEDDED={}",
        u8::from(complete_model)
    );
    Ok(())
}
