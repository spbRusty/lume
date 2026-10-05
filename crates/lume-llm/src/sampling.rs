//! Sampling utilities for Ollama.

use lume_core::types::SamplingParams;
use serde_json::json;

/// Convert sampling parameters to Ollama options.
pub fn to_ollama_options(p: &SamplingParams) -> serde_json::Value {
    let mut opts = serde_json::Map::new();
    if let Some(t) = p.temperature {
        opts.insert("temperature".to_string(), json!(t));
    }
    if let Some(tp) = p.top_p {
        opts.insert("top_p".to_string(), json!(tp));
    }
    if let Some(tk) = p.top_k {
        opts.insert("top_k".to_string(), json!(tk));
    }
    if let Some(mt) = p.max_tokens {
        opts.insert("num_predict".to_string(), json!(mt));
    }
    if let Some(s) = p.seed {
        opts.insert("seed".to_string(), json!(s));
    }
    if !p.stop.is_empty() {
        opts.insert("stop".to_string(), json!(p.stop));
    }
    json!(opts)
}
