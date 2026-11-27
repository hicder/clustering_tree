
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

#[test]
fn test_compact_posting_and_fts_chunk_roundtrip() {
    let mut dls = DocLenBlock::default();
    for s in 0..=255u8 {
        dls.set(s, 10 + (s as u32 % 50));
    }

    let mut chunks_in: Vec<(u8, u32, f32, Vec<u8>)> = Vec::new();
    let mut expected_blocks: Vec<(u8, PackedPostingBlock)> = Vec::new();

    // Rel 0: sparse TfOne
    let b0 = PackedPostingBlock::from_sorted_with_dl(&[(1, 1), (5, 1), (200, 1)], |s| dls.get(s));
    let c0 = b0.encode_compact();
    assert_eq!(c0.len(), 4);
    assert_eq!(
        PackedPostingBlock::decode_compact(&c0, b0.max_tf, b0.min_dl_over_tf).unwrap(),
        b0
    );
    chunks_in.push((0, b0.max_tf, b0.min_dl_over_tf, c0));
    expected_blocks.push((0, b0));

    // Rel 2: sparse U2
    let b2 = PackedPostingBlock::from_sorted_with_dl(&[(2, 2), (7, 4), (19, 1)], |s| dls.get(s));
    let c2 = b2.encode_compact();
    assert_eq!(
        PackedPostingBlock::decode_compact(&c2, b2.max_tf, b2.min_dl_over_tf).unwrap(),
        b2
    );
    chunks_in.push((2, b2.max_tf, b2.min_dl_over_tf, c2));
    expected_blocks.push((2, b2));

    // Rel 5: dense TfOne (40 slots >= 33)
    let entries5: Vec<(u8, u32)> = (0..40u8).map(|i| (i * 5, 1)).collect();
    let b5 = PackedPostingBlock::from_sorted_with_dl(&entries5, |s| dls.get(s));
    let c5 = b5.encode_compact();
    assert_eq!(c5.len(), 33);
    assert_eq!(
        PackedPostingBlock::decode_compact(&c5, b5.max_tf, b5.min_dl_over_tf).unwrap(),
        b5
    );
    chunks_in.push((5, b5.max_tf, b5.min_dl_over_tf, c5));
    expected_blocks.push((5, b5));

    // Rel 10: dense U4
    let entries10: Vec<(u8, u32)> = (0..120u8).map(|i| (i * 2, (i as u32 % 16) + 1)).collect();
    let b10 = PackedPostingBlock::from_sorted_with_dl(&entries10, |s| dls.get(s));
    let c10 = b10.encode_compact();
    assert_eq!(c10.len(), 161);
    assert_eq!(
        PackedPostingBlock::decode_compact(&c10, b10.max_tf, b10.min_dl_over_tf).unwrap(),
        b10
    );
    chunks_in.push((10, b10.max_tf, b10.min_dl_over_tf, c10));
    expected_blocks.push((10, b10));

    let refs: Vec<(u8, u32, f32, &[u8])> = chunks_in
        .iter()
        .map(|(r, m, mr, b)| (*r, *m, *mr, b.as_slice()))
        .collect();
    let enc = FtsChunk::encode_from_blocks(&refs);
    let view = FtsChunkView::parse(&enc).unwrap();
    assert_eq!(view.count, 4);
    assert_eq!(view.chunk_max_tf, 16);
    assert_eq!(view.find_ge(3), Some(2));
    assert_eq!(view.find_exact(5), Some(2));
    assert_eq!(view.find_exact(6), None);

    for (i, (rel, exp_blk)) in expected_blocks.iter().enumerate() {
        assert_eq!(view.rel_block_id(i), *rel);
        let dec = PackedPostingBlock::decode_compact(
            view.block_bytes(i),
            view.max_tf(i),
            view.min_dl_over_tf(i),
        )
        .unwrap();
        assert_eq!(&dec, exp_blk);
    }

    // Upsert a block in the middle (rel = 3)
    let b3 = PackedPostingBlock::from_sorted_with_dl(&[(11, 500)], |s| dls.get(s));
    let enc2 = FtsChunk::upsert_block(
        Some(&enc),
        3,
        b3.max_tf,
        b3.min_dl_over_tf,
        &b3.encode_compact(),
    )
    .unwrap();
    let view2 = FtsChunkView::parse(&enc2).unwrap();
    assert_eq!(view2.count, 5);
    assert_eq!(view2.chunk_max_tf, 500);
    assert_eq!(view2.rel_block_ids, &[0, 2, 3, 5, 10]);
}
