use super::*;

/// A queued `/update` says so in the status line, with its own cancel.
#[test]
fn a_queued_update_shows_as_an_update() {
    let queued = |rebuild| ReloadStatus::Queued {
        agent: Entity::PLACEHOLDER,
        rebuild,
    };
    let update = text(&queued(Rebuild::Update));
    let reload = text(&queued(Rebuild::Build));
    assert!(
        update.contains("/update cancel") && !update.contains("reload"),
        "{update}"
    );
    assert!(
        reload.contains("/reload cancel") && !reload.contains("update"),
        "{reload}"
    );
}
