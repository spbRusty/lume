//! Task routing and complexity classification.

use lume_core::model::ModelTier;

/// Word count below which a task without any structural signal is trivial.
const TRIVIAL_MAX_WORDS: usize = 20;

/// Word count above which a task is hard regardless of other signals.
const HARD_MIN_WORDS: usize = 80;

/// Number of distinct source extensions above which a task is hard.
const HARD_MAX_EXTENSIONS: usize = 2;

/// Whole words that introduce another step of a multi-step request.
const STEP_MARKERS: [&str; 6] = ["then", "afterwards", "finally", "затем", "потом", "после"];

/// Whole words that mark planning work rather than a single one-shot ask.
const PLANNING_MARKERS: [&str; 7] = [
    "plan",
    "design",
    "architect",
    "implement",
    "refactor",
    "migrate",
    "integrate",
];

/// Source file extensions whose presence means the task edits files.
const SOURCE_EXTENSIONS: [&str; 7] = ["rs", "py", "ts", "go", "json", "toml", "md"];

/// Task complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    /// Trivial task.
    Trivial,
    /// Moderate task.
    Moderate,
    /// Hard task.
    Hard,
}

/// Classify task complexity using deterministic heuristic.
///
/// A task is [`TaskComplexity::Hard`] when it carries a structural signal: a
/// fenced code block, a long request, many source extensions, two explicit step
/// markers, or one step marker next to planning vocabulary or a named source
/// file. A task shorter than [`TRIVIAL_MAX_WORDS`] with no signal at all is
/// [`TaskComplexity::Trivial`]; everything else is [`TaskComplexity::Moderate`].
///
/// Step and planning markers are matched on whole words. Substring matching
/// sends ordinary prose ("strengthen the parser", "hasten the release") to the
/// large model, and on a CPU-only box the large model costs roughly three times
/// the latency per token, so a false [`TaskComplexity::Hard`] makes the whole
/// harness unusably slow.
pub fn classify(task: &str) -> TaskComplexity {
    let words = words(task);
    let word_count = words.len();
    let steps = count_markers(&words, &STEP_MARKERS);
    let planning = count_markers(&words, &PLANNING_MARKERS);
    let extensions = count_extensions(task);

    if task.contains("```")
        || word_count > HARD_MIN_WORDS
        || extensions > HARD_MAX_EXTENSIONS
        || steps >= 2
        || (steps >= 1 && (planning >= 1 || extensions >= 1))
    {
        return TaskComplexity::Hard;
    }
    if word_count < TRIVIAL_MAX_WORDS && steps == 0 && extensions == 0 {
        return TaskComplexity::Trivial;
    }
    TaskComplexity::Moderate
}

/// Get model tier for complexity.
pub fn tier_for(c: TaskComplexity) -> ModelTier {
    match c {
        TaskComplexity::Trivial => ModelTier::Small,
        TaskComplexity::Moderate => ModelTier::Small,
        TaskComplexity::Hard => ModelTier::Large,
    }
}

/// Split a task into lowercase alphanumeric words.
fn words(task: &str) -> Vec<String> {
    task.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Count how many words of the task appear in `markers`.
fn count_markers(words: &[String], markers: &[&str]) -> usize {
    words
        .iter()
        .filter(|word| markers.contains(&word.as_str()))
        .count()
}

/// Count distinct source file extensions referenced by the task.
fn count_extensions(task: &str) -> usize {
    let lower = task.to_lowercase();
    SOURCE_EXTENSIONS
        .iter()
        .filter(|ext| lower.contains(&format!(".{ext}")))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a task of exactly `n` filler words.
    fn filler(n: usize) -> String {
        (0..n).map(|_| "word").collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn trivial_short() {
        let c = classify("fix typo");
        assert_eq!(c, TaskComplexity::Trivial);
    }

    #[test]
    fn trivial_ignores_words_that_merely_contain_a_step_marker() {
        let task = "strengthen the parser and hasten the release";
        assert_eq!(classify(task), TaskComplexity::Trivial);
    }

    #[test]
    fn trivial_word_count_boundary() {
        assert_eq!(classify(&filler(19)), TaskComplexity::Trivial);
        assert_eq!(classify(&filler(20)), TaskComplexity::Moderate);
    }

    #[test]
    fn moderate_short_edit_naming_one_source_file() {
        let task = "fix the off by one in src/main.rs";
        assert_eq!(classify(task), TaskComplexity::Moderate);
    }

    #[test]
    fn hard_long_planning() {
        let task = "plan and implement feature then test and document with multiple files";
        let c = classify(task);
        assert_eq!(c, TaskComplexity::Hard);
    }

    #[test]
    fn hard_two_step_markers_without_planning_words() {
        let task = "read the log file, then note the failing test, afterwards run the suite";
        assert_eq!(classify(task), TaskComplexity::Hard);
    }

    #[test]
    fn hard_step_marker_next_to_a_named_source_file() {
        let task = "add a verbose flag in main.rs then print it in doctor";
        assert_eq!(classify(task), TaskComplexity::Hard);
    }

    #[test]
    fn hard_fenced_code_block() {
        let task = "explain this snippet:\n```rust\nfn main() {}\n```";
        assert_eq!(classify(task), TaskComplexity::Hard);
    }

    #[test]
    fn hard_too_many_source_extensions() {
        let task = "align config.toml with schema.json and the notes.md table";
        assert_eq!(classify(task), TaskComplexity::Hard);
    }

    #[test]
    fn hard_word_count_boundary() {
        assert_eq!(classify(&filler(80)), TaskComplexity::Moderate);
        assert_eq!(classify(&filler(81)), TaskComplexity::Hard);
    }

    #[test]
    fn tier_small_for_trivial_and_moderate() {
        assert_eq!(tier_for(TaskComplexity::Trivial), ModelTier::Small);
        assert_eq!(tier_for(TaskComplexity::Moderate), ModelTier::Small);
    }

    #[test]
    fn tier_large_for_hard() {
        assert_eq!(tier_for(TaskComplexity::Hard), ModelTier::Large);
    }
}
