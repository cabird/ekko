//! LLM pricing for token cost calculation.
//!
//! A static table of per-million-token prices, a lookup cascade (exact →
//! version suffix → family fuzzy match) and the cost arithmetic. Prices last
//! updated 2026-02-17.
//!
//! The table intentionally keeps the Gemini entries even though ekko only
//! *calls* OpenAI and Anthropic. Cost lookup is also used for telemetry over
//! historical rows, so dropping rows would change reported numbers rather
//! than only removing dead code.

/// Pricing for a single model (per million tokens).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPricing {
    /// $/M input tokens.
    pub input: f64,
    /// $/M output tokens.
    pub output: f64,
    /// $/M thinking/reasoning tokens.
    pub thinking: f64,
    /// Context window size.
    pub context: u32,
}

impl ModelPricing {
    /// Construct a pricing entry.
    pub const fn new(input: f64, output: f64, thinking: f64, context: u32) -> Self {
        ModelPricing {
            input,
            output,
            thinking,
            context,
        }
    }

    /// Calculate cost in dollars for the given token counts.
    ///
    /// The division-then-multiplication order of the three terms is
    /// load-bearing: the goldens pin the resulting floating-point value, so
    /// re-associating the arithmetic would shift costs in the last digits.
    pub fn calculate_cost(&self, input_tokens: u64, output_tokens: u64, thinking_tokens: u64) -> f64 {
        (input_tokens as f64 / 1_000_000.0) * self.input
            + (output_tokens as f64 / 1_000_000.0) * self.output
            + (thinking_tokens as f64 / 1_000_000.0) * self.thinking
    }
}

/// Pricing table — model id → pricing. Order matters: the version-suffix
/// scan below iterates in table order and takes the first match, so moving
/// an entry can change which price a fuzzy model name resolves to.
pub static PRICING_TABLE: &[(&str, ModelPricing)] = &[
    // OpenAI GPT-5 family
    ("gpt-5-nano", ModelPricing::new(0.05, 0.40, 0.40, 400_000)),
    ("gpt-5-mini", ModelPricing::new(0.25, 2.00, 2.00, 400_000)),
    ("gpt-5", ModelPricing::new(1.25, 10.00, 10.00, 400_000)),
    ("gpt-5.1", ModelPricing::new(1.25, 10.00, 10.00, 400_000)),
    ("gpt-5.2", ModelPricing::new(1.75, 14.00, 14.00, 400_000)),
    // Google Gemini
    ("gemini-3-pro", ModelPricing::new(2.00, 12.00, 12.00, 1_000_000)),
    (
        "gemini/gemini-3-pro",
        ModelPricing::new(2.00, 12.00, 12.00, 1_000_000),
    ),
    (
        "gemini-3-pro-preview",
        ModelPricing::new(2.00, 12.00, 12.00, 1_000_000),
    ),
    // Anthropic Claude
    ("claude-sonnet-4-5", ModelPricing::new(3.00, 15.00, 15.00, 200_000)),
    ("claude-sonnet-4.5", ModelPricing::new(3.00, 15.00, 15.00, 200_000)),
    ("sonnet-4.5", ModelPricing::new(3.00, 15.00, 15.00, 200_000)),
    ("claude-opus-4-5", ModelPricing::new(5.00, 25.00, 25.00, 200_000)),
    ("claude-opus-4.5", ModelPricing::new(5.00, 25.00, 25.00, 200_000)),
    ("opus-4.5", ModelPricing::new(5.00, 25.00, 25.00, 200_000)),
    (
        "claude-opus-4-6",
        ModelPricing::new(5.00, 25.00, 25.00, 1_000_000),
    ),
    (
        "claude-opus-4.6",
        ModelPricing::new(5.00, 25.00, 25.00, 1_000_000),
    ),
    ("opus-4.6", ModelPricing::new(5.00, 25.00, 25.00, 1_000_000)),
];

/// Default pricing for unknown models (conservative estimate).
pub const DEFAULT_PRICING: ModelPricing = ModelPricing::new(1.00, 5.00, 5.00, 128_000);

fn table_get(key: &str) -> ModelPricing {
    PRICING_TABLE
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, pricing)| *pricing)
        // Only reached if a literal below is misspelled; the table above is
        // static so this is a programming error, not a runtime condition.
        .unwrap_or(DEFAULT_PRICING)
}

/// Normalize a model name for lookup: lowercase, trim, strip vendor prefix.
pub fn normalize_model_name(model: &str) -> String {
    // Lowercase first, then trim: lowercasing can change a string's length
    // for some characters, so the order is fixed.
    let name = model.to_lowercase();
    let mut name = name.trim().to_string();
    for prefix in ["openai/", "anthropic/", "google/"] {
        if name.starts_with(prefix) {
            name = name[prefix.len()..].to_string();
        }
    }
    name
}

/// Get pricing for a model (case-insensitive, prefixes stripped).
pub fn get_pricing(model: &str) -> ModelPricing {
    let normalized = normalize_model_name(model);

    // Direct match
    if let Some((_, pricing)) = PRICING_TABLE.iter().find(|(name, _)| *name == normalized) {
        return *pricing;
    }

    // Partial match: version-suffixed models match canonical keys
    // (e.g. "gpt-5-mini-2024-01-01" → "gpt-5-mini"), but not the reverse.
    for (key, pricing) in PRICING_TABLE {
        if normalized.starts_with(&format!("{key}-")) || normalized.starts_with(&format!("{key}_"))
        {
            return *pricing;
        }
    }

    // Fuzzy match by family
    if normalized.contains("gpt-5-nano") || normalized.contains("nano") {
        return table_get("gpt-5-nano");
    }
    if normalized.contains("gpt-5-mini")
        || (normalized.contains("gpt-5") && normalized.contains("mini"))
    {
        return table_get("gpt-5-mini");
    }
    if normalized.contains("gpt-5.2") {
        return table_get("gpt-5.2");
    }
    if normalized.contains("gpt-5.1") {
        return table_get("gpt-5.1");
    }
    if normalized.contains("gpt-5") {
        return table_get("gpt-5");
    }
    if normalized.contains("gemini") && normalized.contains("pro") {
        return table_get("gemini-3-pro");
    }
    if normalized.contains("opus") && normalized.contains("4.6") {
        return table_get("opus-4.6");
    }
    if normalized.contains("opus") {
        return table_get("opus-4.5");
    }
    if normalized.contains("sonnet") {
        return table_get("sonnet-4.5");
    }

    DEFAULT_PRICING
}

/// Calculate cost in dollars for a model call.
pub fn get_cost(model: &str, input_tokens: u64, output_tokens: u64, thinking_tokens: u64) -> f64 {
    get_pricing(model).calculate_cost(input_tokens, output_tokens, thinking_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The golden values below are pinned expected output; update them
    // deliberately when behavior changes intentionally.

    #[test]
    fn table_has_the_pinned_entries() {
        assert_eq!(PRICING_TABLE.len(), 17);
        let names: Vec<&str> = PRICING_TABLE.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "gpt-5-nano",
                "gpt-5-mini",
                "gpt-5",
                "gpt-5.1",
                "gpt-5.2",
                "gemini-3-pro",
                "gemini/gemini-3-pro",
                "gemini-3-pro-preview",
                "claude-sonnet-4-5",
                "claude-sonnet-4.5",
                "sonnet-4.5",
                "claude-opus-4-5",
                "claude-opus-4.5",
                "opus-4.5",
                "claude-opus-4-6",
                "claude-opus-4.6",
                "opus-4.6",
            ]
        );
    }

    #[test]
    fn normalize_strips_prefix_and_case() {
        assert_eq!(normalize_model_name("  GPT-5-Mini  "), "gpt-5-mini");
        assert_eq!(normalize_model_name("openai/gpt-5"), "gpt-5");
        assert_eq!(normalize_model_name("anthropic/claude-opus-4.6"), "claude-opus-4.6");
        assert_eq!(normalize_model_name("google/gemini-3-pro"), "gemini-3-pro");
        // Only the vendor prefixes above are stripped.
        assert_eq!(normalize_model_name("gemini/gemini-3-pro"), "gemini/gemini-3-pro");
    }

    #[test]
    fn pricing_lookup_cascade() {
        // Direct
        assert_eq!(get_pricing("gpt-5-mini").input, 0.25);
        // Version suffix
        assert_eq!(get_pricing("gpt-5-mini-2024-01-01").output, 2.00);
        assert_eq!(get_pricing("gpt-5_preview").output, 10.00);
        // Fuzzy family
        assert_eq!(get_pricing("some-nano-thing").input, 0.05);
        // NOTE: version-suffix match ("gpt-5-" prefix) wins over the fuzzy
        // gpt-5-mini rule, so this resolves to gpt-5 pricing.
        assert_eq!(get_pricing("gpt-5-turbo-mini").input, 1.25);
        assert_eq!(get_pricing("my-gpt-5.2-x").input, 1.75);
        assert_eq!(get_pricing("my-gpt-5.1-x").input, 1.25);
        assert_eq!(get_pricing("weird-gpt-5-thing").input, 1.25);
        assert_eq!(get_pricing("gemini-pro-experimental").input, 2.00);
        assert_eq!(get_pricing("claude-opus-4.6-thinking").input, 5.00);
        assert_eq!(get_pricing("opus-latest").context, 200_000);
        assert_eq!(get_pricing("sonnet-latest").input, 3.00);
        // Unknown → default
        assert_eq!(get_pricing("llama-3-70b"), DEFAULT_PRICING);
        // "gpt" alone must NOT match "gpt-5" (no reverse prefix match)
        assert_eq!(get_pricing("gpt"), DEFAULT_PRICING);
    }

    #[test]
    fn cost_goldens_are_pinned() {
        // get_cost(model, input, output, thinking).
        assert_eq!(get_cost("gpt-5-mini", 1000, 500, 200), 0.00165);
        assert_eq!(get_cost("gpt-5", 100000, 20000, 5000), 0.375);
        assert_eq!(get_cost("gpt-5-nano", 12345, 678, 9), 0.00089205);
        assert_eq!(get_cost("claude-opus-4.6", 250000, 3000, 0), 1.325);
        // The summed terms land on 3.2999999999999996e-05, which is NOT the
        // nearest double to 3.3e-5 — this pins the arithmetic order.
        assert_eq!(get_cost("sonnet-4.5", 1, 1, 1), 3.2999999999999996e-05);
        assert_eq!(get_cost("unknown-model", 1_000_000, 1_000_000, 0), 6.0);
        assert_eq!(get_cost("gpt-5-mini", 0, 0, 0), 0.0);
    }
}
