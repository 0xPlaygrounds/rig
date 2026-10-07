use super::*;

/// The rule an id the catalog does not list falls back to: Kimi K2 before
/// K2.5 and the `moonshot-v1` text models read no images; later Kimi models
/// and the `moonshot-v1` vision models do.
#[test]
fn an_unlisted_kimi_id_reads_images_by_its_family() {
    for (model, images) in [
        ("kimi-k2", false),
        ("kimi-k2-unlisted-preview", false),
        ("moonshot-v1-999k", false),
        ("moonshot-v1-999k-vision-preview", true),
        ("kimi-k9-unlisted", true),
    ] {
        assert_eq!(reads_images(model), images, "{model}");
    }
}
