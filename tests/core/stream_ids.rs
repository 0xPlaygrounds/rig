//! Block ids are minted by the stream machinery, never by a handler: the
//! accumulator, the shared minter (`SyntheticIds`, `AdapterOutput`), the
//! provider adapters (which know their wire's boundaries) and the bus's own
//! fold re-emitter may construct a `BlockId::minted`; a bus handler, a mock,
//! a fixture or an agent-side test writes through `StreamWriter` and names
//! none.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Where a minted id may be constructed by hand.
fn may_mint(relative: &str) -> bool {
    relative.starts_with("crates/rig-core/src/streaming/")
        || relative.starts_with("crates/rig-core/src/providers/")
        || relative == "crates/rig-core/src/serve/handler.rs"
        || relative == "crates/rig-core/src/serve/writer.rs"
        || relative.starts_with("crates/rig-candle/src/")
}

fn scan(dir: &Path, offenders: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if name == "target" || name == ".git" {
                continue;
            }
            scan(&path, offenders);
        } else if name.ends_with(".rs") {
            let relative = path
                .strip_prefix(repo())
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if may_mint(&relative) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            // Built at run time so this file does not match itself.
            let needle = ["BlockId::", "minted("].concat();
            for (number, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                if line.contains(&needle) {
                    offenders.push(format!("{relative}:{}: {}", number + 1, line.trim()));
                }
            }
        }
    }
}

#[test]
fn no_handler_mock_or_fixture_mints_a_block_id() {
    let mut offenders = Vec::new();
    for dir in ["crates", "tests", "examples"] {
        scan(&repo().join(dir), &mut offenders);
    }
    assert!(
        offenders.is_empty(),
        "a hand-minted block id outside the stream machinery; write through `StreamWriter`:\n{}",
        offenders.join("\n")
    );
}
