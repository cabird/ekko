//! JSON-Schema sanitization for tool listings.
//!
//! Claude clients have been observed to silently drop **every** tool a
//! server advertises when one schema contains either
//!
//! - a nullable type union — `"type": ["string", "null"]` — which is what
//!   schemars emits for `Option<T>`, or
//! - an integer `"format"` (`"int64"`, `"uint32"`, …).
//!
//! Rather than contorting the parameter structs (which would cost us
//! `null`-tolerant deserialization), the schemas are rewritten after
//! schemars produces them:
//!
//! - `"type": [T, "null"]` → `"type": T`, and the schema is marked
//!   `"nullable": true`-free — the field is simply optional (it is already
//!   absent from `required`, because `Option` fields are never required).
//! - `"type": [A, B, …]` with more than one non-null member → the union is
//!   collapsed to the first member (never happens with ekko's parameters;
//!   kept so the walker has no silent hole).
//! - `"format"` is removed wherever the type is `integer` or `number`.
//! - `"default": null` is removed: schemars emits it for `Option` fields,
//!   and it contradicts the (now non-nullable) declared type. Clients that
//!   validate defaults against the schema would reject it, and it tells a
//!   model nothing.
//!
//! The walk is generic over the whole schema tree (`properties`, `items`,
//! `$defs`, `anyOf`, …) so nested shapes are covered too.

use rmcp::model::Tool;
use serde_json::{Map, Value};

/// Rewrite a JSON schema in place, removing the shapes that break Claude
/// clients. Returns the number of edits made (handy for tests/telemetry).
pub fn sanitize(value: &mut Value) -> usize {
    let mut edits = 0;
    walk(value, &mut edits);
    edits
}

fn walk(value: &mut Value, edits: &mut usize) {
    match value {
        Value::Object(obj) => {
            fix_type_union(obj, edits);
            drop_numeric_format(obj, edits);
            drop_null_default(obj, edits);
            for (_key, child) in obj.iter_mut() {
                walk(child, edits);
            }
        }
        Value::Array(items) => {
            for item in items {
                walk(item, edits);
            }
        }
        _ => {}
    }
}

/// `"type": [T, "null"]` → `"type": T`.
fn fix_type_union(obj: &mut Map<String, Value>, edits: &mut usize) {
    let Some(Value::Array(members)) = obj.get("type") else {
        return;
    };
    let first = members
        .iter()
        .find(|m| m.as_str() != Some("null"))
        .cloned()
        // A `["null"]`-only schema has no useful non-null member; leaving the
        // array in place would trip the client, so degrade to a bare string.
        .unwrap_or_else(|| Value::String("null".to_string()));
    obj.insert("type".to_string(), first);
    *edits += 1;
}

/// Integer/number `"format"` keywords are dropped.
fn drop_numeric_format(obj: &mut Map<String, Value>, edits: &mut usize) {
    let is_numeric = matches!(
        obj.get("type").and_then(Value::as_str),
        Some("integer" | "number")
    );
    if is_numeric && obj.remove("format").is_some() {
        *edits += 1;
    }
}

/// `"default": null` (schemars' rendering of an absent `Option`) is dropped.
fn drop_null_default(obj: &mut Map<String, Value>, edits: &mut usize) {
    if obj.get("default").is_some_and(Value::is_null) {
        obj.remove("default");
        *edits += 1;
    }
}

/// Sanitize a tool's input (and output) schema, returning the fixed tool.
pub fn sanitize_tool(mut tool: Tool) -> Tool {
    if let Some(fixed) = sanitize_object(&tool.input_schema) {
        tool.input_schema = std::sync::Arc::new(fixed);
    }
    if let Some(schema) = &tool.output_schema
        && let Some(fixed) = sanitize_object(schema)
    {
        tool.output_schema = Some(std::sync::Arc::new(fixed));
    }
    tool
}

fn sanitize_object(schema: &Map<String, Value>) -> Option<Map<String, Value>> {
    let mut value = Value::Object(schema.clone());
    if sanitize(&mut value) == 0 {
        return None;
    }
    match value {
        Value::Object(obj) => Some(obj),
        _ => None,
    }
}

/// Every forbidden shape found in `schema`, as human-readable paths. Empty
/// means the schema is safe to advertise.
pub fn forbidden_shapes(schema: &Value) -> Vec<String> {
    let mut found = Vec::new();
    scan(schema, "$", &mut found);
    found
}

fn scan(value: &Value, path: &str, found: &mut Vec<String>) {
    match value {
        Value::Object(obj) => {
            if let Some(Value::Array(members)) = obj.get("type") {
                let names: Vec<String> = members.iter().map(|m| m.to_string()).collect();
                found.push(format!("{path}.type is an array: [{}]", names.join(", ")));
            }
            if matches!(
                obj.get("type").and_then(Value::as_str),
                Some("integer" | "number")
            ) && let Some(format) = obj.get("format")
            {
                found.push(format!("{path}.format = {format} on a numeric type"));
            }
            if obj.get("default").is_some_and(Value::is_null) {
                found.push(format!("{path}.default is null"));
            }
            for (key, child) in obj {
                scan(child, &format!("{path}.{key}"), found);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                scan(item, &format!("{path}[{i}]"), found);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nullable_unions_collapse_to_the_concrete_type() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "directory": {"type": ["string", "null"]},
                "days": {"type": ["integer", "null"], "format": "int64"},
            }
        });
        assert!(sanitize(&mut schema) > 0);
        assert_eq!(schema["properties"]["directory"]["type"], json!("string"));
        assert_eq!(schema["properties"]["days"]["type"], json!("integer"));
        assert!(schema["properties"]["days"].get("format").is_none());
        assert!(forbidden_shapes(&schema).is_empty());
    }

    #[test]
    fn integer_formats_are_stripped_even_without_a_union() {
        let mut schema = json!({"type": "integer", "format": "uint32", "minimum": 0});
        assert_eq!(sanitize(&mut schema), 1);
        assert_eq!(schema, json!({"type": "integer", "minimum": 0}));
    }

    #[test]
    fn null_defaults_are_dropped() {
        let mut schema = json!({"type": ["string", "null"], "default": null});
        assert_eq!(sanitize(&mut schema), 2);
        assert_eq!(schema, json!({"type": "string"}));
        // A meaningful default survives.
        let mut schema = json!({"type": "integer", "default": 7});
        assert_eq!(sanitize(&mut schema), 0);
        assert_eq!(schema["default"], json!(7));
    }

    #[test]
    fn string_formats_are_left_alone() {
        let mut schema = json!({"type": "string", "format": "date-time"});
        assert_eq!(sanitize(&mut schema), 0);
        assert_eq!(schema["format"], json!("date-time"));
    }

    #[test]
    fn nested_definitions_are_walked() {
        let mut schema = json!({
            "type": "object",
            "$defs": {"Inner": {"properties": {"x": {"type": ["number", "null"], "format": "double"}}}},
            "items": [{"type": ["boolean", "null"]}]
        });
        assert!(sanitize(&mut schema) >= 2);
        assert!(forbidden_shapes(&schema).is_empty());
    }

    #[test]
    fn forbidden_shapes_reports_paths() {
        let schema = json!({"properties": {"days": {"type": ["integer", "null"]}}});
        let found = forbidden_shapes(&schema);
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with("$.properties.days.type is an array"));
    }
}
