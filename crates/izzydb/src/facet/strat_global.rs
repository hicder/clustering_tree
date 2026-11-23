//! Strategy 1: global live counts (`default[0x06]`).

use izzydb_core::error::Result;
use izzydb_core::schema::ColumnStorage;
use izzydb_storage::codec::encode_present_attr_key;

use super::*;

impl AttrAccs {
    // ---- Strategy 1: global counts (`default[0x06]`) -------------------------------------

    pub(super) fn load_global_counts(&mut self, env: &FacetEnv<'_>) -> Result<()> {
        for acc in &mut self.accs {
            let is_detached = env
                .schema
                .get_by_id(acc.attr_id)
                .is_some_and(|e| e.storage == ColumnStorage::Detached);
            let col_gen = if is_detached {
                crate::detached::current_detached_gen(env.storage, acc.attr_id)?
            } else {
                0
            };
            if acc.need_values {
                let sel = acc.sel.clone();
                let values = &mut acc.values;
                let cb = |val: &[u8], c: u64| -> Result<bool> {
                    if sel.past_end(val) {
                        return Ok(false);
                    }
                    if c > 0 {
                        values.insert(val.to_vec(), c);
                    }
                    Ok(true)
                };
                if is_detached {
                    env.storage.visit_detached_counts(
                        acc.attr_id,
                        col_gen,
                        acc.type_tag,
                        sel.start(),
                        cb,
                    )?;
                } else {
                    env.storage
                        .visit_attr_counts(acc.attr_id, acc.type_tag, sel.start(), cb)?;
                }
            }
            if acc.need_present {
                let key = encode_present_attr_key(acc.attr_id);
                acc.present = if is_detached {
                    env.storage.get_detached_count(col_gen, &key)?
                } else {
                    env.storage.get_attr_count(&key)?
                };
            }
        }
        Ok(())
    }
}
