use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, StringBuilder, UInt64Builder};
use arrow_schema::{DataType, Field, Schema};
use izzydb_bench::{
    DEFAULT_FTS_PARQUET_PATH, DEFAULT_FTS_ROWS, FTS_MAX_DOC_LEN, FTS_MIN_DOC_LEN, FTS_VECTOR_DIM,
    FTS_VOCAB_SIZE,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn generate_fts_parquet(path: &Path, n: usize) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).unwrap();
    }

    let mut rng = SmallRng::seed_from_u64(42);
    let vocab: Vec<String> = (0..FTS_VOCAB_SIZE).map(|i| format!("term{i:04}")).collect();

    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                FTS_VECTOR_DIM as i32,
            ),
            false,
        ),
        Field::new("body", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder()
        .set_max_row_group_size(65_536)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        arrow_schema.clone(),
        Some(props),
    )
    .unwrap();

    let batch_size = 65_536usize;
    let mut written = 0usize;
    let mut text_buf = String::with_capacity(FTS_MAX_DOC_LEN * 10);

    while written < n {
        let cur = (n - written).min(batch_size);
        let mut id_b = UInt64Builder::with_capacity(cur);
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), FTS_VECTOR_DIM as i32)
            .with_field(Field::new("item", DataType::Float32, false));
        let mut body_b = StringBuilder::with_capacity(cur, cur * 256);

        for j in 0..cur {
            let id = (written + j) as u64;
            id_b.append_value(id);

            for _ in 0..FTS_VECTOR_DIM {
                vec_b.values().append_value(rng.random_range(-1.0f32..1.0f32));
            }
            vec_b.append(true);

            let doc_len = rng.random_range(FTS_MIN_DOC_LEN..=FTS_MAX_DOC_LEN);
            text_buf.clear();
            for t in 0..doc_len {
                if t > 0 {
                    text_buf.push(' ');
                }
                // Zipf-like skew: favor lower-index terms to generate dense blocks, mix rare terms.
                let idx = if rng.random_bool(0.5) {
                    rng.random_range(0..50.min(vocab.len()))
                } else {
                    rng.random_range(0..vocab.len())
                };
                text_buf.push_str(&vocab[idx]);
            }
            body_b.append_value(&text_buf);
        }

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(id_b.finish()),
                Arc::new(vec_b.finish()),
                Arc::new(body_b.finish()),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        written += cur;
    }

    writer.close().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut parquet_path = PathBuf::from(DEFAULT_FTS_PARQUET_PATH);
    let mut rows = DEFAULT_FTS_ROWS;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--parquet" => {
                i += 1;
                parquet_path = PathBuf::from(args.get(i).expect("missing value for --parquet"));
            }
            "--rows" => {
                i += 1;
                rows = args
                    .get(i)
                    .expect("missing value for --rows")
                    .parse()
                    .expect("invalid --rows integer");
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    let t0 = Instant::now();
    generate_fts_parquet(&parquet_path, rows);
    let elapsed_s = t0.elapsed().as_secs_f64();
    let size_mib = fs::metadata(&parquet_path).unwrap().len() as f64 / (1024.0 * 1024.0);
    println!(
        "Generated Parquet: {} (rows={rows}, size={size_mib:.2} MiB, elapsed={elapsed_s:.2}s)",
        parquet_path.display()
    );
}
