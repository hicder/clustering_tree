use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use rayon::prelude::*;

use izzydb::error::{Result, TreeError};

const MAX_MERGE_FANIN: usize = 64;

pub trait SortItem: Sized + Send {
    fn cmp_key(a: &Self, b: &Self) -> Ordering;
    fn estimated_bytes(&self) -> usize;
    fn write_to(&self, w: &mut impl Write) -> io::Result<()>;
    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>>;
}

/// Parallel external merge-sorter with bounded memory budget and LZ4-framed run files.
pub struct ExternalSorter<T: SortItem> {
    name: String,
    run_dir: PathBuf,
    shard_budget_bytes: usize,
    total_budget_bytes: usize,
    next_run_id: Arc<AtomicUsize>,
    runs: Arc<Mutex<Vec<PathBuf>>>,
    in_mem_shards: Arc<Mutex<Vec<Vec<T>>>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: SortItem> ExternalSorter<T> {
    pub fn new(
        name: impl Into<String>,
        run_dir: impl AsRef<Path>,
        memory_budget_bytes: usize,
        num_shards: usize,
    ) -> Result<Self> {
        let run_dir = run_dir.as_ref().to_path_buf();
        fs::create_dir_all(&run_dir).map_err(|e| {
            TreeError::Storage(format!("create sorter dir {}: {e}", run_dir.display()))
        })?;
        let total_budget_bytes = memory_budget_bytes.max(1024 * 1024);
        let shard_budget_bytes = (total_budget_bytes / num_shards.max(1)).max(256 * 1024);
        Ok(Self {
            name: name.into(),
            run_dir,
            shard_budget_bytes,
            total_budget_bytes,
            next_run_id: Arc::new(AtomicUsize::new(0)),
            runs: Arc::new(Mutex::new(Vec::new())),
            in_mem_shards: Arc::new(Mutex::new(Vec::new())),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn shard(&self) -> SorterShard<T> {
        SorterShard {
            name: self.name.clone(),
            run_dir: self.run_dir.clone(),
            budget_bytes: self.shard_budget_bytes,
            used_bytes: 0,
            buf: Vec::new(),
            next_run_id: Arc::clone(&self.next_run_id),
            runs: Arc::clone(&self.runs),
            in_mem_shards: Arc::clone(&self.in_mem_shards),
        }
    }

    /// Sort remaining buffered items and return a streaming iterator in ascending `T::cmp_key` order.
    pub fn into_sorted_iter(self) -> Result<SortedIter<T>> {
        let mut runs = std::mem::take(&mut *self.runs.lock().unwrap());
        let mut tails = std::mem::take(&mut *self.in_mem_shards.lock().unwrap());

        if runs.is_empty() {
            let total: usize = tails.iter().map(Vec::len).sum();
            let mut merged = Vec::with_capacity(total);
            for mut t in tails {
                merged.append(&mut t);
            }
            merged.par_sort_unstable_by(T::cmp_key);
            return Ok(SortedIter::InMemory(merged.into_iter()));
        }

        // Spill any non-empty tail buffers so all data goes through k-way merge.
        for mut tail in tails.drain(..) {
            if tail.is_empty() {
                continue;
            }
            tail.par_sort_unstable_by(T::cmp_key);
            let run_id = self.next_run_id.fetch_add(1, AtomicOrdering::Relaxed);
            let path = self.run_dir.join(format!("{}-{:06}.lz4", self.name, run_id));
            write_run_file(&path, &tail)?;
            runs.push(path);
        }

        // Multi-pass merge if there are more than MAX_MERGE_FANIN runs.
        while runs.len() > MAX_MERGE_FANIN {
            let mut next_runs = Vec::with_capacity(runs.len().div_ceil(MAX_MERGE_FANIN));
            for chunk in runs.chunks(MAX_MERGE_FANIN) {
                let run_id = self.next_run_id.fetch_add(1, AtomicOrdering::Relaxed);
                let out_path = self.run_dir.join(format!("{}-{:06}.lz4", self.name, run_id));
                merge_runs_to_file::<T>(chunk, &out_path)?;
                for p in chunk {
                    let _ = fs::remove_file(p);
                }
                next_runs.push(out_path);
            }
            runs = next_runs;
        }

        let _ = self.total_budget_bytes;
        let iter = KWayMergeIter::open(runs)?;
        Ok(SortedIter::KWay(iter))
    }
}

pub struct SorterShard<T: SortItem> {
    name: String,
    run_dir: PathBuf,
    budget_bytes: usize,
    used_bytes: usize,
    buf: Vec<T>,
    next_run_id: Arc<AtomicUsize>,
    runs: Arc<Mutex<Vec<PathBuf>>>,
    in_mem_shards: Arc<Mutex<Vec<Vec<T>>>>,
}

impl<T: SortItem> SorterShard<T> {
    pub fn push(&mut self, item: T) -> Result<()> {
        self.used_bytes += item.estimated_bytes() + std::mem::size_of::<T>();
        self.buf.push(item);
        if self.used_bytes >= self.budget_bytes {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.buf.sort_unstable_by(T::cmp_key);
        let run_id = self.next_run_id.fetch_add(1, AtomicOrdering::Relaxed);
        let path = self.run_dir.join(format!("{}-{:06}.lz4", self.name, run_id));
        write_run_file(&path, &self.buf)?;
        self.runs.lock().unwrap().push(path);
        self.buf.clear();
        self.used_bytes = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if !self.buf.is_empty() {
            if self.runs.lock().unwrap().is_empty() {
                self.in_mem_shards.lock().unwrap().push(self.buf);
            } else {
                self.spill()?;
            }
        }
        Ok(())
    }
}

fn write_run_file<T: SortItem>(path: &Path, items: &[T]) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| TreeError::Storage(format!("create run file {}: {e}", path.display())))?;
    let mut enc = FrameEncoder::new(BufWriter::with_capacity(256 * 1024, file));
    for item in items {
        item.write_to(&mut enc)
            .map_err(|e| TreeError::Storage(format!("write run {}: {e}", path.display())))?;
    }
    enc.finish()
        .map_err(|e| TreeError::Storage(format!("finish lz4 run {}: {e}", path.display())))?;
    Ok(())
}

fn merge_runs_to_file<T: SortItem>(inputs: &[PathBuf], out_path: &Path) -> Result<()> {
    let mut merge = KWayMergeIter::<T>::open(inputs.to_vec())?;
    let file = File::create(out_path)
        .map_err(|e| TreeError::Storage(format!("create merge run {}: {e}", out_path.display())))?;
    let mut enc = FrameEncoder::new(BufWriter::with_capacity(256 * 1024, file));
    while let Some(item) = merge.next_item()? {
        item.write_to(&mut enc)
            .map_err(|e| TreeError::Storage(format!("write merge run {}: {e}", out_path.display())))?;
    }
    enc.finish()
        .map_err(|e| TreeError::Storage(format!("finish lz4 merge {}: {e}", out_path.display())))?;
    Ok(())
}

pub enum SortedIter<T: SortItem> {
    InMemory(std::vec::IntoIter<T>),
    KWay(KWayMergeIter<T>),
}

impl<T: SortItem> SortedIter<T> {
    pub fn next_item(&mut self) -> Result<Option<T>> {
        match self {
            Self::InMemory(it) => Ok(it.next()),
            Self::KWay(it) => it.next_item(),
        }
    }
}

struct HeapNode<T: SortItem> {
    item: T,
    run_idx: usize,
}

impl<T: SortItem> PartialEq for HeapNode<T> {
    fn eq(&self, other: &Self) -> bool {
        T::cmp_key(&self.item, &other.item) == Ordering::Equal && self.run_idx == other.run_idx
    }
}

impl<T: SortItem> Eq for HeapNode<T> {}

impl<T: SortItem> PartialOrd for HeapNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: SortItem> Ord for HeapNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap.
        T::cmp_key(&other.item, &self.item).then_with(|| other.run_idx.cmp(&self.run_idx))
    }
}

pub struct KWayMergeIter<T: SortItem> {
    readers: Vec<(FrameDecoder<BufReader<File>>, PathBuf)>,
    heap: BinaryHeap<HeapNode<T>>,
}

impl<T: SortItem> KWayMergeIter<T> {
    fn open(paths: Vec<PathBuf>) -> Result<Self> {
        let mut readers = Vec::with_capacity(paths.len());
        let mut heap = BinaryHeap::with_capacity(paths.len());
        for (idx, path) in paths.into_iter().enumerate() {
            let file = File::open(&path)
                .map_err(|e| TreeError::Storage(format!("open run {}: {e}", path.display())))?;
            let mut dec = FrameDecoder::new(BufReader::with_capacity(128 * 1024, file));
            if let Some(first) = T::read_from(&mut dec)
                .map_err(|e| TreeError::Storage(format!("read run {}: {e}", path.display())))?
            {
                heap.push(HeapNode {
                    item: first,
                    run_idx: idx,
                });
            }
            readers.push((dec, path));
        }
        Ok(Self { readers, heap })
    }

    pub fn next_item(&mut self) -> Result<Option<T>> {
        let Some(node) = self.heap.pop() else {
            return Ok(None);
        };
        let run_idx = node.run_idx;
        if let Some(next_val) = T::read_from(&mut self.readers[run_idx].0).map_err(|e| {
            TreeError::Storage(format!(
                "read run {}: {e}",
                self.readers[run_idx].1.display()
            ))
        })? {
            self.heap.push(HeapNode {
                item: next_val,
                run_idx,
            });
        }
        Ok(Some(node.item))
    }
}

impl<T: SortItem> Drop for KWayMergeIter<T> {
    fn drop(&mut self) {
        for (_, path) in &self.readers {
            let _ = fs::remove_file(path);
        }
    }
}

pub fn write_bytes_lp(w: &mut impl Write, b: &[u8]) -> io::Result<()> {
    w.write_all(&(b.len() as u32).to_le_bytes())?;
    w.write_all(b)
}

pub fn read_bytes_lp(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut out = vec![0u8; len];
    r.read_exact(&mut out)?;
    Ok(Some(out))
}

pub fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_u16(w: &mut impl Write, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn read_u64_opt(r: &mut impl Read) -> io::Result<Option<u64>> {
    let mut b = [0u8; 8];
    match r.read_exact(&mut b) {
        Ok(()) => Ok(Some(u64::from_le_bytes(b))),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn read_u16_opt(r: &mut impl Read) -> io::Result<Option<u16>> {
    let mut b = [0u8; 2];
    match r.read_exact(&mut b) {
        Ok(()) => Ok(Some(u16::from_le_bytes(b))),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn read_u32_req(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_bytes_req(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u32_req(r)? as usize;
    let mut out = vec![0u8; len];
    r.read_exact(&mut out)?;
    Ok(out)
}

pub struct KvItem {
    pub key: Vec<u8>,
    pub val: Vec<u8>,
}

impl SortItem for KvItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> Ordering {
        a.key.cmp(&b.key)
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.key.len() + self.val.len() + 32
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_bytes_lp(w, &self.key)?;
        write_bytes_lp(w, &self.val)
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(key) = read_bytes_lp(r)? else {
            return Ok(None);
        };
        let val = read_bytes_req(r)?;
        Ok(Some(Self { key, val }))
    }
}
