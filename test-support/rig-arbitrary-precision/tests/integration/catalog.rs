use rig_core::catalog::Catalog;

#[test]
fn the_builtin_catalog_loads_every_row_with_its_tiered_prices() -> anyhow::Result<()> {
    // `builtin` panics when the data skips a row.
    let tiers: Vec<_> = Catalog::builtin()
        .iter()
        .flat_map(|spec| spec.details.cost.tiers.iter())
        .collect();
    anyhow::ensure!(!tiers.is_empty(), "the catalog lists tiered prices");
    anyhow::ensure!(
        tiers.iter().all(|tier| tier.rates.input.is_some()),
        "every tier keeps its input price"
    );
    Ok(())
}
