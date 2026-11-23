use izzydb_core::schema::{AttributeType, AttributeValue};

use super::*;

#[test]
fn column_block_roundtrip_all_types_and_for_widths() {
    // Test Bool (ALL_PRESENT)
    let mut b = ColumnBlockBuilder::new(AttributeType::Bool);
    for i in 0..256u16 {
        b.set_attr_value(i as u8, &AttributeValue::Bool(i % 3 == 0))
            .unwrap();
    }
    let enc = b.encode();
    let view = ColumnBlockView::parse(&enc).unwrap();
    assert_eq!(view.present_count, 256);
    assert_ne!(view.flags & COL_FLAG_ALL_PRESENT, 0);
    for i in 0..256u16 {
        assert_eq!(view.get_bool(i as u8), Some(i % 3 == 0));
    }

    // Test Int FOR widths: 0 (const), 1, 2, 4, 8
    for (min_v, step) in [
        (-500i64, 0i64),
        (-100, 1),
        (10_000, 200),
        (-1_000_000, 100_000),
        (i64::MIN / 2, (i64::MAX / 256)),
    ] {
        let mut b = ColumnBlockBuilder::new(AttributeType::Int);
        for i in 0..256u16 {
            if i % 7 != 0 {
                b.set_attr_value(i as u8, &AttributeValue::Int(min_v + (i as i64) * step))
                    .unwrap();
            }
        }
        let enc = b.encode();
        let view = ColumnBlockView::parse(&enc).unwrap();
        for i in 0..256u16 {
            let expect = (i % 7 != 0).then_some(min_v + (i as i64) * step);
            assert_eq!(view.get_i64(i as u8), expect);
        }
    }

    // Test Uint FOR widths
    for (min_v, step) in [(42u64, 0u64), (10, 1), (0, 250), (1, 100_000), (0, u64::MAX / 256)] {
        let mut b = ColumnBlockBuilder::new(AttributeType::Uint);
        for i in 0..200u16 {
            if i % 5 != 0 {
                b.set_attr_value(i as u8, &AttributeValue::Uint(min_v + (i as u64) * step))
                    .unwrap();
            }
        }
        let enc = b.encode();
        let view = ColumnBlockView::parse(&enc).unwrap();
        for i in 0..256u16 {
            let expect = (i < 200 && i % 5 != 0).then_some(min_v + (i as u64) * step);
            assert_eq!(view.get_u64(i as u8), expect);
        }
    }

    // Test Float preserving -0.0 and NaN/arbitrary bits
    let mut b = ColumnBlockBuilder::new(AttributeType::Float);
    b.set_attr_value(0, &AttributeValue::Float(-0.0)).unwrap();
    b.set_attr_value(1, &AttributeValue::Float(0.0)).unwrap();
    b.set_attr_value(5, &AttributeValue::Float(3.5)).unwrap();
    let enc = b.encode();
    let view = ColumnBlockView::parse(&enc).unwrap();
    assert_eq!(view.get_f64(0).unwrap().to_bits(), (-0.0f64).to_bits());
    assert_eq!(view.get_f64(1).unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(view.get_f64(2), None);
    assert_eq!(view.get_f64(5), Some(3.5));

    // Test String + arrays
    let mut bs = ColumnBlockBuilder::new(AttributeType::String);
    bs.set_attr_value(2, &AttributeValue::String("alpha".into()))
        .unwrap();
    bs.set_attr_value(10, &AttributeValue::String(String::new()))
        .unwrap();
    let enc = bs.encode();
    let vs = ColumnBlockView::parse(&enc).unwrap();
    assert_eq!(vs.get_str(0), None);
    assert_eq!(vs.get_str(2), Some(b"alpha".as_slice()));
    assert_eq!(vs.get_str(10), Some(b"".as_slice()));

    let mut bia = ColumnBlockBuilder::new(AttributeType::IntArray);
    bia.set_attr_value(3, &AttributeValue::IntArray(vec![-1, 0, 42]))
        .unwrap();
    bia.set_attr_value(7, &AttributeValue::IntArray(Vec::new()))
        .unwrap();
    let enc = bia.encode();
    let via = ColumnBlockView::parse(&enc).unwrap();
    assert_eq!(
        via.decode_attr_value(3).unwrap(),
        AttributeValue::IntArray(vec![-1, 0, 42])
    );
    assert_eq!(
        via.decode_attr_value(7).unwrap(),
        AttributeValue::IntArray(Vec::new())
    );

    let mut bsa = ColumnBlockBuilder::new(AttributeType::StringArray);
    bsa.set_attr_value(1, &AttributeValue::StringArray(vec!["x".into(), "".into(), "yz".into()]))
        .unwrap();
    let enc = bsa.encode();
    let vsa = ColumnBlockView::parse(&enc).unwrap();
    assert_eq!(
        vsa.decode_attr_value(1).unwrap(),
        AttributeValue::StringArray(vec!["x".into(), "".into(), "yz".into()])
    );
    assert_eq!(vsa.decode_attr_value(0).unwrap(), AttributeValue::Null);
}
