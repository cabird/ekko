//! Singleton role leases with heartbeat renewal.
//!
//! `summary_claims` deliberately cannot renew before expiry (its guard is
//! `claimed_until < now`), which is fine for one-shot summarization work
//! and useless for hosting: a web server needs to keep a lease alive for
//! hours. This table's acquire is also the renewal — the same holder always
//! wins — so one primitive covers take, keep, and take-over-after-death.
//!
//! Election order matters and lives in the caller (`web::server`): bind the
//! port FIRST, then acquire the lease, so a failed bind can never strand a
//! lease pointing at a server that isn't listening.

use rusqlite::{OptionalExtension, params};

use super::{Database, DbResult};

/// The web UI hosting lease.
pub const WEB_LEASE: &str = "web_server";

impl Database {
    /// Acquire or renew `name` for `holder` until `now + ttl_secs`.
    /// Succeeds when the lease is free, expired, or already held by this
    /// holder; fails (false) when another holder's lease is still live.
    pub fn lease_acquire(&self, name: &str, holder: &str, ttl_secs: i64) -> DbResult<bool> {
        let now = unix_now();
        let conn = self.connection()?;
        let changed = conn.execute(
            "INSERT INTO leases (name, holder, expires_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at
             WHERE leases.expires_at < ?4 OR leases.holder = excluded.holder",
            params![name, holder, now + ttl_secs.max(1), now],
        )?;
        Ok(changed > 0)
    }

    /// Release `name` if `holder` still owns it. Idempotent.
    pub fn lease_release(&self, name: &str, holder: &str) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "DELETE FROM leases WHERE name = ? AND holder = ?",
            params![name, holder],
        )?;
        Ok(())
    }

    /// Current `(holder, seconds_until_expiry)` for `name`; negative
    /// seconds mean the lease is stale (holder died without releasing).
    pub fn lease_info(&self, name: &str) -> DbResult<Option<(String, i64)>> {
        let conn = self.connection()?;
        conn.query_row(
            "SELECT holder, expires_at FROM leases WHERE name = ?",
            params![name],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)? - unix_now(),
                ))
            },
        )
        .optional()
        .map_err(Into::into)
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Database::new(dir.path().join("t.db")).unwrap(), dir)
    }

    #[test]
    fn acquire_renew_block_steal_release() {
        let (db, _d) = db();
        assert!(db.lease_acquire(WEB_LEASE, "a:3556", 60).unwrap(), "free lease");
        // The capability summary_claims lacks: renewal BEFORE expiry.
        assert!(db.lease_acquire(WEB_LEASE, "a:3556", 60).unwrap(), "same holder renews");
        assert!(!db.lease_acquire(WEB_LEASE, "b:3556", 60).unwrap(), "live lease blocks others");
        let (holder, remaining) = db.lease_info(WEB_LEASE).unwrap().unwrap();
        assert_eq!(holder, "a:3556");
        assert!((0..=60).contains(&remaining), "{remaining}");

        // Wrong holder's release is a no-op; right holder frees it.
        db.lease_release(WEB_LEASE, "b:3556").unwrap();
        assert!(db.lease_info(WEB_LEASE).unwrap().is_some());
        db.lease_release(WEB_LEASE, "a:3556").unwrap();
        assert!(db.lease_info(WEB_LEASE).unwrap().is_none());
        assert!(db.lease_acquire(WEB_LEASE, "b:3556", 60).unwrap(), "released lease is takeable");
    }

    #[test]
    fn expired_lease_is_taken_over() {
        let (db, _d) = db();
        // ttl clamps to >= 1s, so plant an already-expired row directly.
        let conn = rusqlite::Connection::open(db.path()).unwrap();
        conn.execute(
            "INSERT INTO leases (name, holder, expires_at) VALUES ('web_server', 'dead:3556', 1)",
            [],
        )
        .unwrap();
        let (_, remaining) = db.lease_info(WEB_LEASE).unwrap().unwrap();
        assert!(remaining < 0, "stale lease reports negative remaining");
        assert!(db.lease_acquire(WEB_LEASE, "new:3556", 60).unwrap(), "expired lease is stolen");
        assert_eq!(db.lease_info(WEB_LEASE).unwrap().unwrap().0, "new:3556");
    }
}
