//! Telemetry storage: record, export, aggregate summary, retention
//! cleanup, and LLM cost statistics.

use rusqlite::params;
use rusqlite::types::Value as SqlValue;
use serde_json::Value;

use super::{Database, DbError, DbResult, json_dumps, utc_now_iso};

/// Optional LLM usage fields attached to a telemetry event.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlmUsage {
    pub model: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub thinking_tokens: Option<i64>,
    pub cost: Option<f64>,
}

/// One exported telemetry row; `safe_args` is parsed back from JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryEvent {
    pub id: i64,
    pub instance_id: String,
    pub timestamp: String,
    pub operation: String,
    pub duration_ms: Option<i64>,
    pub safe_args: Option<Value>,
    pub model: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub thinking_tokens: Option<i64>,
    pub cost: Option<f64>,
}

/// Per-operation timing stats, as reported by `get_telemetry_summary`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationStats {
    pub count: i64,
    /// Rounded average; `None` when the SQL AVG is NULL **or exactly 0**.
    pub avg_duration_ms: Option<i64>,
    pub min_duration_ms: Option<i64>,
    pub max_duration_ms: Option<i64>,
}

/// Aggregated telemetry summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetrySummary {
    pub total_events: i64,
    pub first_event: Option<String>,
    pub last_event: Option<String>,
    /// `(operation, stats)` ordered by count DESC.
    pub operations: Vec<(String, OperationStats)>,
}

/// One cost bucket within [`CostStats`].
#[derive(Debug, Clone, PartialEq)]
pub struct CostBucket {
    /// Total cost rounded to 6 decimal places.
    pub cost: f64,
    /// `"$X.XXXX"` display string built from the unrounded total.
    pub cost_display: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub thinking_tokens: i64,
    pub calls: i64,
}

/// Per-operation cost row, as carried in `by_operation_7d`.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationCost {
    pub cost: f64,
    pub cost_display: String,
    pub calls: i64,
}

/// Cost statistics derived from the telemetry table.
#[derive(Debug, Clone, PartialEq)]
pub struct CostStats {
    /// Present only when an `instance_id` was supplied.
    pub this_instance: Option<CostBucket>,
    pub today: CostBucket,
    pub last_7_days: CostBucket,
    /// `(operation, cost)` ordered by cost DESC, top 10, only nonzero
    /// totals.
    pub by_operation_7d: Vec<(String, OperationCost)>,
}

fn round6(x: f64) -> f64 {
    crate::utils::round_decimal(x, 6)
}

impl Database {
    /// Record a telemetry event. `None` **or an empty map** for `safe_args`
    /// stores NULL.
    pub fn record_telemetry(
        &self,
        instance_id: &str,
        operation: &str,
        duration_ms: Option<i64>,
        safe_args: Option<&serde_json::Map<String, Value>>,
        usage: &LlmUsage,
    ) -> DbResult<()> {
        let timestamp = utc_now_iso();
        let safe_args_json = match safe_args {
            Some(map) if !map.is_empty() => Some(json_dumps(&Value::Object(map.clone()))),
            _ => None,
        };
        let conn = self.connection()?;
        conn.execute(
            "INSERT INTO telemetry (
                 instance_id, timestamp, operation, duration_ms, safe_args,
                 model, input_tokens, output_tokens, thinking_tokens, cost
             )
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                instance_id,
                timestamp,
                operation,
                duration_ms,
                safe_args_json,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.thinking_tokens,
                usage.cost,
            ],
        )?;
        Ok(())
    }

    /// Export telemetry data. `Some(0)` for `days` means all data.
    pub fn export_telemetry(&self, days: Option<i64>) -> DbResult<Vec<TelemetryEvent>> {
        let conn = self.connection()?;
        let (query, sql_params): (&str, Vec<SqlValue>) = match days.filter(|d| *d != 0) {
            Some(d) => (
                "SELECT * FROM telemetry
                 WHERE timestamp >= datetime('now', ?)
                 ORDER BY timestamp",
                vec![SqlValue::Text(format!("-{d} days"))],
            ),
            None => ("SELECT * FROM telemetry ORDER BY timestamp", Vec::new()),
        };
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(sql_params.iter()), |row| {
            Ok((
                TelemetryEvent {
                    id: row.get("id")?,
                    instance_id: row.get("instance_id")?,
                    timestamp: row.get("timestamp")?,
                    operation: row.get("operation")?,
                    duration_ms: row.get("duration_ms")?,
                    safe_args: None,
                    model: row.get("model")?,
                    input_tokens: row.get("input_tokens")?,
                    output_tokens: row.get("output_tokens")?,
                    thinking_tokens: row.get("thinking_tokens")?,
                    cost: row.get("cost")?,
                },
                row.get::<_, Option<String>>("safe_args")?,
            ))
        })?;
        let mut events = Vec::new();
        for row in rows {
            let (mut event, raw_args) = row?;
            // An empty string is left unparsed rather than treated as JSON.
            if let Some(raw) = raw_args.filter(|s| !s.is_empty()) {
                event.safe_args =
                    Some(serde_json::from_str(&raw).map_err(|source| DbError::Json {
                        context: "telemetry safe_args",
                        source,
                    })?);
            }
            events.push(event);
        }
        Ok(events)
    }

    /// Aggregated telemetry summary. `Some(0)` for `days` means all data.
    pub fn get_telemetry_summary(&self, days: Option<i64>) -> DbResult<TelemetrySummary> {
        let conn = self.connection()?;
        let (time_filter, sql_params): (&str, Vec<SqlValue>) = match days.filter(|d| *d != 0) {
            Some(d) => (
                "WHERE timestamp >= datetime('now', ?)",
                vec![SqlValue::Text(format!("-{d} days"))],
            ),
            None => ("", Vec::new()),
        };

        let query = format!(
            "SELECT
                 operation,
                 COUNT(*) as count,
                 AVG(duration_ms) as avg_duration_ms,
                 MIN(duration_ms) as min_duration_ms,
                 MAX(duration_ms) as max_duration_ms
             FROM telemetry
             {time_filter}
             GROUP BY operation
             ORDER BY count DESC"
        );
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(sql_params.iter()), |row| {
            let avg: Option<f64> = row.get("avg_duration_ms")?;
            Ok((
                row.get::<_, String>("operation")?,
                OperationStats {
                    count: row.get("count")?,
                    // An average of exactly 0.0 reports as `None`, not 0.
                    avg_duration_ms: avg
                        .filter(|a| *a != 0.0)
                        .map(|a| a.round_ties_even() as i64),
                    min_duration_ms: row.get("min_duration_ms")?,
                    max_duration_ms: row.get("max_duration_ms")?,
                },
            ))
        })?;
        let operations = rows.collect::<rusqlite::Result<Vec<_>>>()?;

        let total_query = format!(
            "SELECT COUNT(*) as total, MIN(timestamp) as first, MAX(timestamp) as last
             FROM telemetry
             {time_filter}"
        );
        let (total_events, first_event, last_event) = conn.query_row(
            &total_query,
            rusqlite::params_from_iter(sql_params.iter()),
            |row| {
                Ok((
                    row.get::<_, i64>("total")?,
                    row.get::<_, Option<String>>("first")?,
                    row.get::<_, Option<String>>("last")?,
                ))
            },
        )?;

        Ok(TelemetrySummary {
            total_events,
            first_event,
            last_event,
            operations,
        })
    }

    /// Remove telemetry older than `days` days (365 at the call site).
    pub fn cleanup_old_telemetry(&self, days: i64) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "DELETE FROM telemetry WHERE timestamp < datetime('now', ?)",
            params![format!("-{days} days")],
        )?;
        Ok(())
    }

    /// LLM cost statistics from telemetry.
    pub fn get_cost_stats(&self, instance_id: Option<&str>) -> DbResult<CostStats> {
        let conn = self.connection()?;

        let sum_costs = |where_clause: &str, sql_params: &[SqlValue]| -> DbResult<CostBucket> {
            let query = format!(
                "SELECT
                     COALESCE(SUM(cost), 0) as total_cost,
                     COALESCE(SUM(input_tokens), 0) as input_tokens,
                     COALESCE(SUM(output_tokens), 0) as output_tokens,
                     COALESCE(SUM(thinking_tokens), 0) as thinking_tokens,
                     COUNT(*) as call_count
                 FROM telemetry
                 WHERE cost IS NOT NULL {where_clause}"
            );
            conn.query_row(
                &query,
                rusqlite::params_from_iter(sql_params.iter()),
                |row| {
                    let total_cost: f64 = row.get("total_cost")?;
                    Ok(CostBucket {
                        cost: round6(total_cost),
                        cost_display: format!("${total_cost:.4}"),
                        input_tokens: row.get("input_tokens")?,
                        output_tokens: row.get("output_tokens")?,
                        thinking_tokens: row.get("thinking_tokens")?,
                        calls: row.get("call_count")?,
                    })
                },
            )
            .map_err(Into::into)
        };

        let this_instance = match instance_id {
            Some(id) => Some(sum_costs(
                "AND instance_id = ?",
                &[SqlValue::Text(id.to_string())],
            )?),
            None => None,
        };
        let today = sum_costs("AND date(timestamp) = date('now')", &[])?;
        let last_7_days = sum_costs("AND timestamp >= datetime('now', '-7 days')", &[])?;

        let mut stmt = conn.prepare(
            "SELECT
                 operation,
                 COALESCE(SUM(cost), 0) as total_cost,
                 COUNT(*) as call_count
             FROM telemetry
             WHERE cost IS NOT NULL
               AND timestamp >= datetime('now', '-7 days')
             GROUP BY operation
             HAVING total_cost > 0
             ORDER BY total_cost DESC
             LIMIT 10",
        )?;
        let rows = stmt.query_map([], |row| {
            let total_cost: f64 = row.get("total_cost")?;
            Ok((
                row.get::<_, String>("operation")?,
                OperationCost {
                    cost: round6(total_cost),
                    cost_display: format!("${total_cost:.4}"),
                    calls: row.get("call_count")?,
                },
            ))
        })?;
        let by_operation_7d = rows.collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(CostStats {
            this_instance,
            today,
            last_7_days,
            by_operation_7d,
        })
    }
}
