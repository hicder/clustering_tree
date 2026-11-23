//! Zone verification.

use std::collections::HashMap;
use roaring::RoaringBitmap;
use izzydb_core::config::{
    AttrId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_storage::codec::{
    ZONE_TAG, ZoneEntry,
};
use izzydb_storage::Storage;


pub(super) fn verify_zones(
    storage: &Storage,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    expected_zones: &HashMap<(SegmentId, AttrId, u8), ZoneEntry>,
) -> Result<()> {
    let mut disk_zones: HashMap<(SegmentId, AttrId, u8), ZoneEntry> = HashMap::new();
    storage.visit_default(&[ZONE_TAG], |k, v| {
        if k.len() != 7 {
            return Err(TreeError::InvariantViolation(
                "0x12 zone key suffix expected 7 bytes".into(),
            ));
        }
        let seg = SegmentId::from_be_bytes(k[0..4].try_into().unwrap());
        if dropped.contains(seg) {
            return Ok(());
        }
        if seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            let aid = u16::from_be_bytes([k[4], k[5]]);
            let tag = k[6];
            disk_zones.insert((seg, aid, tag), ZoneEntry::decode(v)?);
        }
        Ok(())
    })?;

    for (k, exp) in expected_zones {
        let Some(disk) = disk_zones.get(k) else {
            return Err(TreeError::InvariantViolation(format!(
                "missing 0x12 zone entry for {k:?}"
            )));
        };
        if disk.min > exp.min || disk.max < exp.max {
            return Err(TreeError::InvariantViolation(format!(
                "zone {k:?} does not cover live bounds"
            )));
        }
    }
    Ok(())
}
