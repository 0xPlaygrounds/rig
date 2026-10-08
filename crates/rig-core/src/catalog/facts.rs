//! The facts a completion wire encodes with, and that its replies are priced
//! by.

use std::sync::Arc;

use super::{Catalog, ModelSpec};

/// The facts a completion wire encodes with: the spec of the model it was
/// connected to, and the catalog that spec came from, which answers for any
/// other model a request names. Every completion wire holds one, and its
/// [`Descriptor`](crate::wire::Descriptor) hands it to the completion fold,
/// which prices usage from it.
///
/// The default binds no spec and answers from [`Catalog::builtin`]. Facts
/// are not part of a wire's serialized form or of its equality: a wire read
/// back from storage answers from the built-in catalog until it is given
/// facts again. Clones are cheap.
///
/// ```
/// use rig_core::catalog::{Catalog, ModelFacts, Pricing};
/// use rig_core::providers::registry::ProviderId;
///
/// let anthropic = ProviderId::catalog("anthropic").ok_or("a known vendor")?;
/// let mut catalog = Catalog::builtin().clone();
/// let spec = catalog.get_exact(anthropic, "claude-haiku-4-5").ok_or("listed")?;
/// catalog.insert(spec.clone().with_pricing(Pricing::new(0.5, 2.5)));
///
/// let facts = catalog.facts(anthropic, "claude-haiku-4-5-20990101");
/// let spec = facts.for_model("anthropic", "claude-haiku-4-5-20990101").ok_or("bound")?;
/// assert_eq!(spec.pricing.as_ref().map(|pricing| pricing.input), Some(0.5));
/// # Ok::<(), &str>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct ModelFacts {
    bound: Option<Arc<Bound>>,
    catalog: Option<Catalog>,
}

/// The model a wire was connected to: the id it addresses and its spec.
#[derive(Debug)]
struct Bound {
    id: String,
    spec: Arc<ModelSpec>,
}

impl ModelFacts {
    /// Facts that bind no spec and answer from the built-in catalog, as the
    /// default does.
    pub(crate) fn builtin() -> &'static ModelFacts {
        static BUILTIN: ModelFacts = ModelFacts {
            bound: None,
            catalog: None,
        };
        &BUILTIN
    }

    /// The facts of `spec`, for the model it lists. Any other model is
    /// answered from [`Catalog::builtin`].
    pub fn new(spec: impl Into<Arc<ModelSpec>>) -> Self {
        let spec = spec.into();
        Self {
            bound: Some(Arc::new(Bound {
                id: spec.id.clone(),
                spec,
            })),
            catalog: None,
        }
    }

    /// The same facts, answering every model but the bound one from
    /// `catalog`.
    pub fn with_catalog(mut self, catalog: &Catalog) -> Self {
        self.catalog = Some(catalog.clone());
        self
    }

    /// The spec of the model the wire was connected to, when it was
    /// connected through the catalog or given one.
    pub fn spec(&self) -> Option<&ModelSpec> {
        self.bound.as_deref().map(|bound| &*bound.spec)
    }

    /// The catalog that answers for every model but the bound one.
    pub fn catalog(&self) -> &Catalog {
        self.catalog.as_ref().unwrap_or_else(|| Catalog::builtin())
    }

    /// The facts of `vendor`'s `model`: the bound spec when `model` is the id
    /// the wire was connected to (or the id the spec lists), otherwise the
    /// catalog's entry by [`Catalog::get`]'s rule. `None` for a model the
    /// catalog does not list. `vendor` is the provider's descriptor name
    /// (`"anthropic"`, `"aws_bedrock"`).
    pub fn for_model(&self, vendor: &str, model: &str) -> Option<&ModelSpec> {
        if let Some(bound) = self.bound.as_deref()
            && bound.spec.provider.vendor() == vendor
            && (bound.id == model || bound.spec.id == model)
        {
            return Some(&bound.spec);
        }
        self.catalog()
            .get_vendor(vendor, model)
            .map(|resolved| resolved.spec)
    }

    /// Whether `vendor`'s `model` reads images: what its facts list, or, for
    /// a model the catalog does not list, what `rule` (the vendor's naming
    /// rule) says of its id. Every wire that filters images reads this, so
    /// an id the catalog does not list (a gateway's spelling, another case,
    /// a deployment name) keeps its vendor's naming rule.
    pub(crate) fn reads_images_or(
        &self,
        vendor: &str,
        model: &str,
        rule: impl FnOnce(&str) -> bool,
    ) -> bool {
        self.for_model(vendor, model)
            .map_or_else(|| rule(model), |spec| spec.input.image)
    }
}

/// Facts take no part in a wire's equality, as they take none in its
/// serialized form: two wires that send the same requests are equal.
impl PartialEq for ModelFacts {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ModelFacts {}

impl Catalog {
    /// The facts of `provider`'s `model` from this catalog: its entry by
    /// [`Self::get`]'s rule, bound to `model` as written, and this catalog
    /// for every other model a request names. A model the catalog does not
    /// list binds nothing.
    pub fn facts(
        &self,
        provider: crate::providers::registry::ProviderId,
        model: &str,
    ) -> ModelFacts {
        let bound = self.entry_for(provider.vendor(), model).map(|spec| {
            Arc::new(Bound {
                id: model.to_owned(),
                spec,
            })
        });
        ModelFacts {
            bound,
            catalog: Some(self.clone()),
        }
    }
}

#[cfg(test)]
mod tests;
