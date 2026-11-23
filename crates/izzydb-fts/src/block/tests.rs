
use izzydb_core::config::RowBlock;
use super::*;

#[test]
fn test_fts_meta_and_term_record_roundtrip() {
    let mut meta = FtsMeta::default();
    assert_eq!(meta.alloc_term_id().unwrap(), 0);
    assert_eq!(meta.alloc_term_id().unwrap(), 1);
    meta.per_attr_stats.insert(
        2,
        AttrFtsStats {
            doc_count: 100,
            total_tokens: 4500,
        },
    );
    let bytes = meta.encode();
    assert_eq!(FtsMeta::decode(&bytes).unwrap(), meta);

    let mut ts = TermStats::default();
    assert_eq!(ts.min_dl_over_tf, f32::INFINITY);
    ts.doc_freq = 42;
    ts.observe(9, 27);
    ts.observe(3, 12);
    assert_eq!(ts.max_tf, 9);
    assert!((ts.min_dl_over_tf - 3.0).abs() < 1e-6);
    assert_eq!(TermStats::decode(&ts.encode()).unwrap(), ts);

    let rec = TermRecord {
        term_id: 17,
        stats: ts,
        inline_block: None,
    };
    assert_eq!(TermRecord::decode(&rec.encode()).unwrap(), rec);

    let mut dls = DocLenBlock::default();
    dls.set(3, 10);
    let mut blk = PackedPostingBlock::default();
    blk.upsert(3, 2, &dls);
    let rec_inline = TermRecord {
        term_id: 18,
        stats: ts,
        inline_block: Some((RowBlock::new(2, 4), blk.encode())),
    };
    assert_eq!(TermRecord::decode(&rec_inline.encode()).unwrap(), rec_inline);
}

#[test]
fn test_skip_block_roundtrip() {
    let mut sb = SkipBlock::default();
    sb.upsert(12, 4, 3.5);
    sb.upsert(2, 120, 1.25);
    sb.upsert(250, 2, 10.0);
    let enc = sb.encode();
    let dec = SkipBlock::decode(&enc).unwrap();
    assert_eq!(dec, sb);
    let view = SkipBlockView::parse(&enc).unwrap();
    assert_eq!(view.find_ge(0), Some(0));
    assert_eq!(view.find_ge(3), Some(1));
    assert_eq!(view.find_ge(251), None);
    assert!(sb.remove(12));
    assert!(!sb.remove(99));
    assert_eq!(sb.rel_block_ids, vec![2, 250]);
}

#[test]
fn test_doc_len_block_widths_roundtrip() {
    for max_dl in [10u32, 300, 100_000] {
        let mut dls = DocLenBlock::default();
        dls.set(0, 1);
        dls.set(17, max_dl);
        dls.set(255, 5);
        assert_eq!(dls.count, 3);

        let enc = dls.encode();
        let dec = DocLenBlock::decode(&enc).unwrap();
        assert_eq!(dec, dls);

        dls.remove(17);
        assert_eq!(dls.count, 2);
        assert_eq!(dls.get(17), 0);
    }
}

#[test]
fn test_posting_block_sparse_dense_and_widths_roundtrip() {
    let mut dls = DocLenBlock::default();
    dls.set(2, 12);
    dls.set(10, 25);
    dls.set(255, 100);

    let mut blk = PackedPostingBlock::default();
    blk.upsert(10, 3, &dls);
    blk.upsert(2, 5, &dls);
    blk.upsert(255, 1, &dls);
    assert_eq!(blk.slots, vec![2, 10, 255]);
    assert_eq!(blk.max_tf, 5);
    assert!((blk.min_dl_over_tf - (12.0 / 5.0)).abs() < 1e-5);

    let enc_u8 = blk.encode();
    let hdr_u8 = PostingBlockHeader::parse(&enc_u8).unwrap();
    assert!(!hdr_u8.is_dense());
    assert_eq!(hdr_u8.tf_width(), IntWidth::U8);
    assert_eq!(PackedPostingBlock::decode(&enc_u8).unwrap(), blk);

    // Overflow into u16 and u32
    dls.set(10, 120_000);
    blk.upsert(10, 500, &dls);
    let enc_u16 = blk.encode();
    assert_eq!(PostingBlockHeader::parse(&enc_u16).unwrap().tf_width(), IntWidth::U16);
    assert_eq!(PackedPostingBlock::decode(&enc_u16).unwrap(), blk);

    blk.upsert(10, 70_000, &dls);
    let enc_u32 = blk.encode();
    assert_eq!(PostingBlockHeader::parse(&enc_u32).unwrap().tf_width(), IntWidth::U32);
    assert_eq!(PackedPostingBlock::decode(&enc_u32).unwrap(), blk);

    assert!(blk.remove(10, &dls));
    assert!(!blk.remove(99, &dls));
    assert_eq!(blk.max_tf, 5);
    assert!((blk.min_dl_over_tf - (12.0 / 5.0)).abs() < 1e-5);

    // Dense layout (>= POSTING_DENSE_THRESHOLD)
    let mut dense_dls = DocLenBlock::default();
    let mut dense_blk = PackedPostingBlock::default();
    for s in (0..=254u8).step_by(2) {
        dense_dls.set(s, 20 + s as u32);
        dense_blk.upsert(s, 1 + (s % 5) as u32, &dense_dls);
    }
    assert!(dense_blk.len() >= POSTING_DENSE_THRESHOLD);
    let enc_dense = dense_blk.encode();
    let hdr_dense = PostingBlockHeader::parse(&enc_dense).unwrap();
    assert!(hdr_dense.is_dense());
    assert_eq!(hdr_dense.tf_width(), IntWidth::U8);
    assert_eq!(PackedPostingBlock::decode(&enc_dense).unwrap(), dense_blk);
}

#[test]
fn test_position_block_roundtrip() {
    let mut pblk = PackedPositionBlock::default();
    pblk.upsert(5, vec![0, 3, 128, 300]);
    pblk.upsert(1, vec![2, 4]);
    let enc = pblk.encode();
    let view = PositionBlockView::parse(&enc).unwrap();
    assert!(!view.u32_offsets);
    assert_eq!(view.slot_positions(1).unwrap(), Some(vec![2, 4]));
    assert_eq!(view.slot_positions(5).unwrap(), Some(vec![0, 3, 128, 300]));
    assert!(view.slot_contains_pos(5, 128).unwrap());
    assert!(!view.slot_contains_pos(5, 129).unwrap());
    assert_eq!(view.slot_positions(9).unwrap(), None);
    assert_eq!(PackedPositionBlock::decode(&enc).unwrap(), pblk);
}
