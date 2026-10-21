//! The single source-validated OMP compatibility target.

use rig_harness::rig_core::completion::CacheRetention;

pub(crate) const VERSION: &str = "2.1.280";
pub(crate) const SDK: &str = "0.112.1";
pub(crate) const RUNTIME: &str = "v26.3.0";
pub(crate) const OAUTH_BETA: &str = "oauth-2025-04-20";
pub(crate) const UTILITY_BETAS: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,structured-outputs-2025-12-15";
pub(crate) const USER_AGENT: &str = "claude-cli/2.1.280 (external, cli)";
pub(crate) const IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
pub(crate) const ENTRYPOINT: &str = "cli";
pub(crate) const BETAS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07";
pub(crate) const CACHE: CacheRetention = CacheRetention::Long;
pub(crate) const GZIP_THRESHOLD: usize = 1024;
pub(crate) const SUFFIX_SALT: &str = "59cf53e54c78";
pub(crate) const CCH_SEED: u64 = 0x4d659218e32a3268;

pub(crate) fn suffix(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let units: Vec<_> = text.encode_utf16().collect();
    let selected: Vec<u16> = [4, 7, 20]
        .into_iter()
        .map(|i| units.get(i).copied().unwrap_or(u16::from(b'0')))
        .collect();
    let hash = format!(
        "{:x}",
        Sha256::digest(format!(
            "{SUFFIX_SALT}{}{VERSION}",
            String::from_utf16_lossy(&selected)
        ))
    );
    hash.chars().take(3).collect()
}
