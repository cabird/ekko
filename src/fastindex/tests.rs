//! Unit tests for the parallel driver and the pure selection helpers.
//!
//! Nothing here touches the network: [`drive`] is exercised with plain
//! futures, and the queue helpers are pure.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chrono::{TimeZone, Utc};
use futures_util::future::BoxFuture;

use super::*;

/// Tracks in-flight count and the high-water mark.
#[derive(Default)]
struct Gauge {
    live: AtomicUsize,
    peak: AtomicUsize,
    started: AtomicUsize,
}

impl Gauge {
    fn enter(&self) {
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.started.fetch_add(1, Ordering::SeqCst);
        self.peak.fetch_max(live, Ordering::SeqCst);
    }
    fn exit(&self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

fn item(gauge: Arc<Gauge>, outcome: ItemOutcome) -> BoxFuture<'static, ItemOutcome> {
    Box::pin(async move {
        gauge.enter();
        tokio::time::sleep(Duration::from_millis(20)).await;
        gauge.exit();
        outcome
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drive_respects_the_worker_bound() {
    let gauge = Arc::new(Gauge::default());
    let futures: Vec<BoxFuture<'static, ItemOutcome>> = (0..20)
        .map(|_| item(gauge.clone(), ItemOutcome::Done))
        .collect();

    let mut seen: Vec<String> = Vec::new();
    let counters = drive(futures, 3, &mut |c| seen.push(c.render())).await;

    assert_eq!(counters.total, 20);
    assert_eq!(counters.done, 20);
    assert_eq!(counters.completed, 20);
    assert_eq!(gauge.started.load(Ordering::SeqCst), 20);
    assert!(
        gauge.peak.load(Ordering::SeqCst) <= 3,
        "peak concurrency {} exceeded the 3-worker bound",
        gauge.peak.load(Ordering::SeqCst)
    );
    // One progress line per completion, counting up.
    assert_eq!(seen.len(), 20);
    assert_eq!(seen[0], "1/20");
    assert_eq!(seen[19], "20/20");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drive_isolates_per_item_failures() {
    let gauge = Arc::new(Gauge::default());
    let outcomes = [
        ItemOutcome::Done,
        ItemOutcome::Failed,
        ItemOutcome::Skipped,
        ItemOutcome::Done,
        ItemOutcome::Failed,
    ];
    let futures: Vec<BoxFuture<'static, ItemOutcome>> =
        outcomes.iter().map(|o| item(gauge.clone(), *o)).collect();

    let counters = drive(futures, 2, &mut |_| {}).await;

    // A failing item neither aborts the run nor stops later items.
    assert_eq!(gauge.started.load(Ordering::SeqCst), 5);
    assert_eq!(counters.total, 5);
    assert_eq!(counters.completed, 5);
    assert_eq!(counters.done, 2);
    assert_eq!(counters.failed, 2);
    assert_eq!(counters.skipped, 1);
    assert_eq!(counters.render(), "5/5 (2 failed, 1 claimed elsewhere)");
}

#[tokio::test]
async fn drive_of_an_empty_queue_is_a_no_op() {
    let mut calls = 0usize;
    let counters = drive(Vec::new(), 4, &mut |_| calls += 1).await;
    assert_eq!(counters, PassCounters::default());
    assert_eq!(calls, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_single_worker_serializes() {
    let gauge = Arc::new(Gauge::default());
    let futures: Vec<BoxFuture<'static, ItemOutcome>> = (0..5)
        .map(|_| item(gauge.clone(), ItemOutcome::Done))
        .collect();
    drive(futures, 0, &mut |_| {}).await; // 0 is clamped to 1
    assert_eq!(gauge.peak.load(Ordering::SeqCst), 1);
}

#[test]
fn stability_gate() {
    let now = Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap();
    // Two hours old → stable.
    assert!(is_stable(Some("2025-06-15T10:00:00Z"), now, 1.0));
    // Ten minutes old → still in flight.
    assert!(!is_stable(Some("2025-06-15T11:50:00Z"), now, 1.0));
    // Exactly one hour → stable under is_stable's `>=`. (The SQL gate's
    // strict `<` disagrees for exactly this one second — irrelevant
    // behaviorally, noted for honesty.).
    assert!(is_stable(Some("2025-06-15T11:00:00Z"), now, 1.0));
    // Naive timestamps are read as UTC, as everywhere else in ekko.
    assert!(is_stable(Some("2025-06-15 08:00:00"), now, 1.0));
    // Offsets are honored: 13:30+05:30 == 08:00 UTC.
    assert!(is_stable(Some("2025-06-15T13:30:00+05:30"), now, 1.0));
    // Unknown/missing/garbage timestamps are treated as unstable.
    assert!(!is_stable(None, now, 1.0));
    assert!(!is_stable(Some(""), now, 1.0));
    assert!(!is_stable(Some("whenever"), now, 1.0));
    // A zero window admits everything parseable.
    assert!(is_stable(Some("2025-06-15T12:00:00Z"), now, 0.0));
}

#[test]
fn counter_rendering() {
    let mut c = PassCounters {
        total: 3,
        ..PassCounters::default()
    };
    assert_eq!(c.render(), "0/3");
    c.record(ItemOutcome::Done);
    assert_eq!(c.render(), "1/3");
    c.record(ItemOutcome::Failed);
    assert_eq!(c.render(), "2/3 (1 failed)");
    c.record(ItemOutcome::Skipped);
    assert_eq!(c.render(), "3/3 (1 failed, 1 claimed elsewhere)");
}

#[test]
fn options_clamp_workers() {
    let opts = IndexOptions {
        workers: 0,
        ..IndexOptions::default()
    };
    assert_eq!(opts.effective_workers(), 1);
    assert_eq!(IndexOptions::default().effective_workers(), 7);
    assert_eq!(IndexOptions::default().min_age_hours, MIN_AGE_HOURS);
}
