use super::{Family, family};

#[test]
fn flash_prefix_wins_before_m3_in_any_case() {
    assert!(family("MINIMAX-M3.1-Flash-preview") == Family::M31Flash);
    assert!(family("MiniMax-M3.10") == Family::M31Flash);
    assert!(family("MiniMax-M3-preview") == Family::M3);
    assert!(family("MiniMax-M2.7") == Family::Other);
    assert!(family("unknown") == Family::Other);
}
