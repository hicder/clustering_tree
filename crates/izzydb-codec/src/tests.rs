use super::*;
use izzydb_core::config::{DistanceMetric, ROW_BLOCK_SIZE, RowId, TreeConfig};
use izzydb_core::filter::Filter;
use izzydb_core::schema::{
    AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
};
use roaring::RoaringBitmap;

#[test]
fn key_ordering_and_splitting() {
    // row_key order == RowId order
    let rows = [
        RowId::new(0, 0),
        RowId::new(0, 255),
        RowId::new(0, 256),
        RowId::new(1, 0),
        RowId::new(u32::MAX, 10),
    ];
    let rkeys: Vec<[u8; 8]> = rows.iter().map(|&r| row_key(r)).collect();
    for w in rkeys.windows(2) {
        assert!(w[0] < w[1]);
    }
    for (&r, &k) in rows.iter().zip(rkeys.iter()) {
        assert_eq!(decode_row_key(&k).unwrap(), r);
    }

    // posting_key for (attr, tag, val) scans segments ascending, even with embedded 0x00 in strings
    let s = "a\0b\0c";
    let mut val_bytes = Vec::new();
    encode_str_ordered(s.as_bytes(), &mut val_bytes);
    let pk1 = posting_key(7, TAG_STRING, &val_bytes, 2);
    let pk2 = posting_key(7, TAG_STRING, &val_bytes, 10);
    assert!(pk1 < pk2);
    let (pre1, seg1) = split_posting_key(&pk1).unwrap();
    let (pre2, seg2) = split_posting_key(&pk2).unwrap();
    assert_eq!(seg1, 2);
    assert_eq!(seg2, 10);
    assert_eq!(pre1, pre2);
    let (a, tag, vb, seg) = split_posting_key_parts(&pk1).unwrap();
    assert_eq!((a, tag, seg), (7, TAG_STRING, 2));
    assert_eq!(decode_str_ordered(vb).unwrap(), s);

    // detached posting key
    let dpk = detached_posting_key(3, 99, TAG_STRING, &val_bytes, 5);
    let (dpre, dseg) = split_detached_posting_key(&dpk).unwrap();
    assert_eq!(dseg, 5);
    assert!(dpre.starts_with(&detached_posting_prefix(3, 99, TAG_STRING)));
}

#[test]
fn fixed_codecs_roundtrip_and_corruption() {
    let m = Manifest {
        open_segment: Some(3),
        next_segment_id: 4,
        seal_rows: 65_536,
        live_rows: 1000,
        total_rows: 1200,
    };
    assert_eq!(Manifest::decode(&m.encode()).unwrap(), m);
    let m2 = Manifest::new(1024);
    assert_eq!(Manifest::decode(&m2.encode()).unwrap(), m2);

    let sm = SegmentMeta {
        state: SegmentState::Forwarding,
        flags: SEG_FLAG_HAS_COLUMNS | SEG_FLAG_HAS_VECTORS,
        row_count: 500,
        deleted_count: 100,
        forward_target: 9,
        forward_base: 42,
        forwarded_at_epoch: 7,
        created_seq: 111,
        sealed_seq: 222,
    };
    assert_eq!(SegmentMeta::decode(&sm.encode()).unwrap(), sm);

    let mut ze = ZoneEntry::singleton(b"hello");
    assert!(ze.widen(b"alpha"));
    assert!(ze.widen(b"zeta"));
    assert!(!ze.widen(b"mid"));
    assert_eq!(ZoneEntry::decode(&ze.encode()).unwrap(), ze);

    let dm = DetachedMeta {
        state: 1,
        current_gen: 4,
        pending_gen: 5,
        retired_below: 3,
        built_against_watermark: RowId::new(2, 99),
    };
    assert_eq!(DetachedMeta::decode(&dm.encode()).unwrap(), dm);

    let mut block = [0u64; ROW_BLOCK_SIZE];
    block[0] = 11;
    block[255] = u64::MAX;
    let enc_b = encode_doc_id_block(&block);
    assert_eq!(enc_b.len(), DOC_ID_BLOCK_BYTES);
    assert_eq!(*decode_doc_id_block(&enc_b).unwrap(), block);

    // Random / invalid lengths rejected
    for bad_len in [0usize, 1, 5, 31, 33, 43, 45, 2047, 2049] {
        let buf = vec![0u8; bad_len];
        assert!(Manifest::decode(&buf).is_err());
        assert!(SegmentMeta::decode(&buf).is_err());
        assert!(DetachedMeta::decode(&buf).is_err());
        assert!(decode_doc_id_block(&buf).is_err());
    }
    assert!(ZoneEntry::decode(&[0u8; 3]).is_err());
    assert!(ZoneEntry::decode(&ZoneEntry { min: vec![b'z'], max: vec![b'a'] }.encode()).is_err());
}

#[test]
fn seg_posting_roundtrip_sparse_and_roaring() {
    let mut sp = SegPosting::default();
    for i in 0..32u32 {
        sp.insert(i * 1000);
    }
    let b1 = sp.encode().unwrap();
    assert_eq!(b1[0], SEG_POSTING_SPARSE);
    assert_eq!(SegPosting::decode(&b1).unwrap(), sp);

    let mut bm = RoaringBitmap::new();
    SegPostingView::parse(&b1).unwrap().or_into(&mut bm).unwrap();
    assert_eq!(&bm, sp.as_bitmap());

    sp.insert(99_999);
    let b2 = sp.encode().unwrap();
    assert_eq!(b2[0], SEG_POSTING_ROARING);
    assert_eq!(SegPosting::decode(&b2).unwrap(), sp);

    // Unsorted sparse payload rejected
    let mut bad = vec![SEG_POSTING_SPARSE];
    bad.extend_from_slice(&5u32.to_le_bytes());
    bad.extend_from_slice(&5u32.to_le_bytes());
    assert!(SegPostingView::parse(&bad).is_err());
}

#[test]
fn row_view_v3_roundtrip_and_detached_skip() {
    let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared))
        .with_attribute("tag", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_detached_attribute("pop", AttributeType::Float, true);
    let schema = CollectionSchema::from_config(&cfg).unwrap();

    let mut attrs = Attributes::new();
    attrs.insert("tag".into(), "alpha".into());
    attrs.insert("score".into(), 3.5f64.into());
    attrs.insert("pop".into(), 99.0f64.into());

    let bytes = encode_row(&schema, &attrs, true).unwrap();
    let view = RowView::parse(&bytes).unwrap();
    assert!(view.has_vector());
    assert_eq!(view.len(), 2); // detached 'pop' omitted from row

    let decoded = decode_row_attributes(&schema, &bytes).unwrap();
    assert!(!decoded.contains_key("pop"));
    assert_eq!(decoded.get("tag"), Some(&AttributeValue::String("alpha".into())));
    assert_eq!(
        view.get_numeric_f64(schema.get_by_name("score").unwrap().id).unwrap(),
        Some(3.5)
    );
    assert!(
        view.matches_filter(
            &schema,
            &Filter::Eq("tag".into(), "alpha".into()),
            &mut |_, _, _, _, _| false
        )
        .unwrap()
    );
}
