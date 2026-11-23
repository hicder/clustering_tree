//! SST bundle ingestion into a target database.

use super::*;

/// Ingest a pre-built [`SstBundle`] into an empty or non-existent `target` collection directory.
pub fn ingest_ssts(target: impl AsRef<Path>, bundle: &SstBundle) -> Result<()> {
    ingest_ssts_checked(target.as_ref(), bundle, None, None)
}

#[doc(hidden)]
pub fn ingest_ssts_with_hook(
    target: impl AsRef<Path>,
    bundle: &SstBundle,
    expected_config: Option<&CollectionConfig>,
    mut after_cf: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    ingest_ssts_checked(target.as_ref(), bundle, expected_config, Some(&mut after_cf))
}

pub(super) type AfterCfHook<'a> = &'a mut dyn FnMut(&str) -> Result<()>;

pub(super) fn ingest_ssts_checked(
    target: &Path,
    bundle: &SstBundle,
    expected_config: Option<&CollectionConfig>,
    mut after_cf: Option<AfterCfHook<'_>>,
) -> Result<()> {
    let col_cfg = match expected_config {
        Some(cfg) => {
            bundle.validate_against(cfg)?;
            cfg.clone()
        }
        None => {
            let schema = CollectionSchema::decode(&bundle.manifest.schema_bytes)?;
            let mut cfg = match &bundle.manifest.ann_meta {
                Some(am) => {
                    let am = am.to_ann_meta()?;
                    CollectionConfig::new(izzydb::config::TreeConfig::new(am.dim as usize, am.metric))
                }
                None => CollectionConfig::without_vectors(),
            };
            cfg.segment.seal_rows = bundle.manifest.segment_info.seal_rows;
            for entry in schema.entries() {
                cfg.attributes.push(izzydb::schema::AttributeConfig {
                    name: entry.name.clone(),
                    attr_type: entry.attr_type,
                    filterable: entry.filterable,
                    full_text_search: entry.full_text_search.clone(),
                    storage: entry.storage,
                    column: entry.column,
                });
            }
            cfg
        }
    };

    let storage = Storage::open_for_bulk_ingest(target, &col_cfg)?;
    let move_files = same_filesystem(&bundle.dir, target);
    let build_id = rand::random::<u64>();
    storage.put_bulk_import_marker(build_id)?;

    for cf in ALL_DATA_CF_NAMES {
        let files = bundle.files(cf);
        if files.is_empty() {
            continue;
        }
        storage.ingest_cf_with_move(cf, &files, move_files)?;
        if let Some(ref mut hook) = after_cf {
            hook(cf)?;
        }
    }

    let default_files = bundle.files(DEFAULT_CF_NAME);
    storage.ingest_cf_with_move(DEFAULT_CF_NAME, &default_files, move_files)?;
    if let Some(ref mut hook) = after_cf {
        hook(DEFAULT_CF_NAME)?;
    }

    storage.clear_bulk_import_marker()?;
    Ok(())
}
