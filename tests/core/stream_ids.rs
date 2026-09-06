//! Block ids are minted by the stream machinery, never by a handler: the
//! accumulator, the shared minter (`SyntheticIds`, `AdapterOutput`), the
//! provider adapters (which know their wire's boundaries) and the bus's own
//! fold re-emitter may construct a `BlockId::minted`; a bus handler, a mock,
//! a fixture or an agent-side test writes through `StreamWriter` and names
//! none. Structural assembler/serialization tests may allocate keys through
//! `SyntheticIds` when they need to preserve explicit durable metadata. The
//! typed durable correlation constructor is a separate, narrowly checked owner:
//! it creates a completion-local identity, not a handler-emitted stream block.

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

/// Only the exact durable-ID constructor may construct a key in message.rs.
/// Keeping the exception at the constructor body prevents this permission from
/// spreading to unrelated message conversion code or its tests.
fn is_durable_identity_constructor(relative: &str, source: &str, line: usize) -> bool {
    if relative != "crates/rig-core/src/completion/message.rs" {
        return false;
    }
    let constructor = [
        "    pub fn minted(index: u64) -> Self {\n",
        "        Self::from_block(&crate::streaming::BlockId::",
        "minted(\n",
        "            crate::streaming::MintKind::Tool,\n",
        "            index,\n",
        "        ))\n",
        "    }",
    ]
    .concat();
    source.match_indices(&constructor).any(|(offset, _)| {
        let start = source[..offset]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count();
        line == start + 1
    })
}

#[test]
fn durable_identity_exception_is_limited_to_its_constructor() {
    let constructor = [
        "    pub fn minted(index: u64) -> Self {\n",
        "        Self::from_block(&crate::streaming::BlockId::",
        "minted(\n",
        "            crate::streaming::MintKind::Tool,\n",
        "            index,\n",
        "        ))\n",
        "    }",
    ]
    .concat();
    let path = "crates/rig-core/src/completion/message.rs";
    assert!(is_durable_identity_constructor(path, &constructor, 1));
    assert!(!is_durable_identity_constructor(path, &constructor, 0));
    assert!(!is_durable_identity_constructor(
        "crates/rig-core/src/completion/message/tests.rs",
        &constructor,
        1,
    ));
    let unrelated = format!("{constructor}\n{}", ["BlockId::", "minted(0)"].concat());
    assert!(!is_durable_identity_constructor(path, &unrelated, 7));
    assert!(!is_durable_identity_constructor(
        path,
        &constructor.replace("pub fn minted", "pub fn other"),
        1,
    ));
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
                if line.contains(&needle)
                    && !is_durable_identity_constructor(&relative, &text, number)
                {
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
