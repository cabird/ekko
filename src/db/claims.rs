//! Summary claims: atomic coordination between background indexer
//! instances, via an `INSERT … ON CONFLICT DO UPDATE … WHERE` that only
//! succeeds when the existing claim has expired.

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use super::{Database, DbResult};

/// Current Unix time in whole seconds.
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl Database {
    /// Atomically try to claim a session for summarization. The UPSERT's
    /// WHERE clause only takes over claims whose TTL has expired, so
    /// exactly one instance wins. Returns `true` when the claim was
    /// acquired.
    pub fn try_claim_summary(
        &self,
        session_id: &str,
        content_hash: &str,
        instance_id: &str,
        ttl_seconds: i64,
    ) -> DbResult<bool> {
        let now = unix_now();
        let expires = now + ttl_seconds;
        let conn = self.connection()?;
        let changed = conn.execute(
            "INSERT INTO summary_claims (session_id, content_hash, claimed_until, claimed_by)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(session_id, content_hash)
             DO UPDATE SET
                 claimed_until = excluded.claimed_until,
                 claimed_by = excluded.claimed_by
             WHERE summary_claims.claimed_until < ?",
            params![session_id, content_hash, expires, instance_id, now],
        )?;
        Ok(changed > 0)
    }

    /// Release a claim after successful summarization.
    pub fn release_claim(&self, session_id: &str, content_hash: &str) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "DELETE FROM summary_claims WHERE session_id = ? AND content_hash = ?",
            params![session_id, content_hash],
        )?;
        Ok(())
    }

    /// Remove expired claims (housekeeping).
    pub fn cleanup_expired_claims(&self) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "DELETE FROM summary_claims WHERE claimed_until < ?",
            params![unix_now()],
        )?;
        Ok(())
    }
}
