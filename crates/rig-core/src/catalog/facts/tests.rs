use super::*;
use crate::catalog::Pricing;
use crate::providers::registry::ProviderId;

fn anthropic() -> ProviderId {
    ProviderId::catalog("anthropic").expect("a catalog vendor")
}

/// The built-in catalog with Claude Sonnet 4.6 at a price of its own.
fn repriced() -> Catalog {
    let mut catalog = Catalog::builtin().clone();
    let sonnet = catalog
        .get_exact(anthropic(), "claude-sonnet-4-6")
        .expect("listed")
        .clone();
    catalog.insert(sonnet.with_pricing(Pricing::new(1.0, 2.0)));
    catalog
}

fn input_price(spec: Option<&ModelSpec>) -> Option<f64> {
    spec.and_then(|spec| spec.pricing.as_ref())
        .map(|pricing| pricing.input)
}

/// Facts from a catalog answer for the model they were made for as written,
/// a dated snapshot included, and for its listed id; every other model of
/// any vendor comes from that catalog, never from the built-in one.
#[test]
fn facts_answer_from_the_catalog_they_came_from() {
    let catalog = repriced();
    let facts = catalog.facts(anthropic(), "claude-sonnet-4-6-20990101");
    assert_eq!(
        facts.spec().map(|spec| spec.id.as_str()),
        Some("claude-sonnet-4-6")
    );
    for model in ["claude-sonnet-4-6-20990101", "claude-sonnet-4-6"] {
        assert_eq!(
            input_price(facts.for_model("anthropic", model)),
            Some(1.0),
            "{model}"
        );
    }
    assert_eq!(
        input_price(facts.for_model("anthropic", "claude-sonnet-4-6-20880101")),
        Some(1.0),
        "another snapshot reads the same catalog"
    );
    assert!(facts.for_model("anthropic", "claude-haiku-4-5").is_some());
    assert!(facts.for_model("anthropic", "not-in-the-catalog").is_none());
    assert!(
        facts.for_model("openrouter", "claude-sonnet-4-6").is_none(),
        "the bound spec answers only under its own vendor"
    );

    let unlisted = catalog.facts(anthropic(), "not-in-the-catalog");
    assert!(unlisted.spec().is_none());
    assert_eq!(
        input_price(unlisted.for_model("anthropic", "claude-sonnet-4-6")),
        Some(1.0)
    );
}

/// Facts made from a spec answer for it, and for anything else from the
/// built-in catalog unless given another; the default binds nothing.
#[test]
fn facts_of_a_spec_fall_back_to_their_catalog() {
    let custom = ModelSpec::new(anthropic(), "claude-custom").with_pricing(Pricing::new(9.0, 9.0));
    let facts = ModelFacts::new(custom);
    assert_eq!(
        input_price(facts.for_model("anthropic", "claude-custom")),
        Some(9.0)
    );
    let builtin = input_price(
        Catalog::builtin()
            .get_exact(anthropic(), "claude-sonnet-4-6")
            .map(|spec| spec as &ModelSpec),
    );
    assert_eq!(
        input_price(facts.for_model("anthropic", "claude-sonnet-4-6")),
        builtin
    );
    let facts = facts.with_catalog(&repriced());
    assert_eq!(
        input_price(facts.for_model("anthropic", "claude-sonnet-4-6")),
        Some(1.0)
    );
    assert_eq!(
        input_price(facts.for_model("anthropic", "claude-custom")),
        Some(9.0)
    );

    let default = ModelFacts::default();
    assert!(default.spec().is_none());
    assert_eq!(
        input_price(default.for_model("anthropic", "claude-sonnet-4-6")),
        builtin
    );
}

/// Facts never make two wires differ, so every pair of facts is equal.
#[test]
fn facts_take_no_part_in_equality() {
    assert_eq!(
        ModelFacts::default(),
        repriced().facts(anthropic(), "claude-sonnet-4-6")
    );
}

/// An id the catalog lists reads images as its entry says; any other id is
/// read by the rule the caller passes, never by a default.
#[test]
fn an_unlisted_model_reads_images_by_its_vendor_rule() {
    let facts = ModelFacts::default();
    assert!(!facts.reads_images_or("openai", "gpt-3.5-turbo", |_| true));
    assert!(facts.reads_images_or("openai", "gpt-4o", |_| false));
    assert!(!facts.reads_images_or("minimax", "minimax-m2.5", |_| false));
    assert!(facts.reads_images_or("minimax", "minimax-m2.5", |_| true));
}

/// A descriptor given no facts answers from the built-in catalog, by the
/// one lookup rule, and has no spec for a wire that addresses no model.
#[test]
fn a_descriptor_without_facts_reads_the_builtin_catalog() {
    use crate::wire::Descriptor;

    let spec = Descriptor::new("anthropic")
        .model("claude-haiku-4-5-20990101")
        .spec()
        .expect("a snapshot of a listed model");
    assert_eq!(spec.id, "claude-haiku-4-5");
    assert!(Descriptor::new("anthropic").spec().is_none());
    assert!(
        Descriptor::new("anthropic")
            .model("claude-unlisted")
            .spec()
            .is_none()
    );
}
