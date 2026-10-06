//! Task routing and complexity classification.

use lume_core::model::ModelTier;

/// Word count below which a task without any structural signal is trivial.
const TRIVIAL_MAX_WORDS: usize = 20;

/// Word count above which a task is hard regardless of other signals.
const HARD_MIN_WORDS: usize = 80;

/// Number of distinct source extensions above which a task is hard.
const HARD_MAX_EXTENSIONS: usize = 2;

/// Lexical score at or above which a task is hard without any structural
/// signal. Each broad-scope or planning word is worth 2, so the threshold is
/// two of those, or one plus two code tokens, and so on.
const HARD_MIN_SCORE: i32 = 4;

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

/// Whole words that widen the ask to everything in scope rather than one
/// named place. These are what separate "fix this line" from "review every
/// caller", and a pair of them is worth [`HARD_MIN_SCORE`] on its own.
const BROAD_MARKERS: [&str; 8] = [
    "review",
    "audit",
    "all",
    "every",
    "whole",
    "entire",
    "everything",
    "each",
];

/// Whole words that name a code-level concern, so the task is about
/// machinery rather than prose. Matched whole-word: `impl` is a token,
/// `implement` is planning vocabulary, and neither catches the other.
const CODE_TOKENS: [&str; 20] = [
    "fn", "struct", "impl", "trait", "enum", "macro", "error", "errors", "bug", "bugs", "fix",
    "test", "tests", "testing", "compile", "compiler", "panic", "unwrap", "lifetime", "borrow",
];

/// Whole words that mark a narrowly scoped, mechanical edit. Each one
/// subtracts 1 from the score, so "fix the typo" nets to zero and stays
/// trivial instead of reading as a code task just because it says `fix`.
const NARROW_MARKERS: [&str; 12] = [
    "typo",
    "typos",
    "spelling",
    "rename",
    "reword",
    "rephrase",
    "wording",
    "whitespace",
    "indent",
    "indentation",
    "punctuation",
    "retitle",
];

/// Whole words that scope the task to the whole project instead of a named
/// file. Naming a specific file is a weak signal (see [`SOURCE_EXTENSIONS`]);
/// naming the whole codebase is a strong one.
const SCOPE_MARKERS: [&str; 5] = ["codebase", "project", "repository", "repo", "workspace"];

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

/// Classify task complexity with a deterministic, pure in-process heuristic.
///
/// The decision has two layers. A task is [`TaskComplexity::Hard`] outright
/// when it carries a structural signal: a fenced code block, a long request
/// (more than [`HARD_MIN_WORDS`] words), more than [`HARD_MAX_EXTENSIONS`]
/// source extensions, two step separators (newlines, semicolons, or
/// [`STEP_MARKERS`]), one step marker next to planning vocabulary or a named
/// source file, or a lexical score of at least [`HARD_MIN_SCORE`].
///
/// The score sums whole-word weights over the task: broad-scope asks
/// ([`BROAD_MARKERS`]) and planning verbs ([`PLANNING_MARKERS`]) are worth 2,
/// code tokens ([`CODE_TOKENS`]) and whole-project scope ([`SCOPE_MARKERS`])
/// are worth 1, and narrow asks ([`NARROW_MARKERS`]) subtract 1. Scoring
/// catches what the structural rules cannot: "refactor the auth module and
/// add tests for every edge case" has no step marker and is short, so the
/// old rule called it trivial and sent it to the weak model, while its three
/// independent concerns score well past [`HARD_MIN_SCORE`].
///
/// A task shorter than [`TRIVIAL_MAX_WORDS`] with no structural signal and a
/// non-positive score is [`TaskComplexity::Trivial`]; everything else is
/// [`TaskComplexity::Moderate`].
///
/// Markers are matched on whole words. Substring matching sends ordinary
/// prose ("strengthen the parser", "hasten the release") to the large model,
/// and on a CPU-only box the large model costs roughly three times the
/// latency per token, so a false [`TaskComplexity::Hard`] makes the whole
/// harness unusably slow. Weights are asymmetric for that reason: a single
/// broad or code word never reaches [`HARD_MIN_SCORE`] by itself.
pub fn classify(task: &str) -> TaskComplexity {
    let words = words(task);
    let word_count = words.len();
    let markers = count_markers(&words, &STEP_MARKERS);
    let planning = count_markers(&words, &PLANNING_MARKERS);
    let extensions = count_extensions(task);
    let steps = markers + separators(task);
    let score: i32 = words.iter().map(|word| word_weight(word)).sum();

    if task.contains("```")
        || word_count > HARD_MIN_WORDS
        || extensions > HARD_MAX_EXTENSIONS
        || steps >= 2
        || (markers >= 1 && (planning >= 1 || extensions >= 1))
        || score >= HARD_MIN_SCORE
    {
        return TaskComplexity::Hard;
    }
    if word_count < TRIVIAL_MAX_WORDS && steps == 0 && extensions == 0 && score <= 0 {
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

/// Count the step separators a task contains: every newline and semicolon
/// divides the request into distinct sub-tasks, just like a [`STEP_MARKERS`]
/// word does, so a three-line ask is three steps of work.
fn separators(task: &str) -> usize {
    task.chars().filter(|c| matches!(c, '\n' | ';')).count()
}

/// Lexical weight of one whole word in the routing score.
///
/// The first matching list wins, so a word that ever appears in two lists
/// is counted once. Broad scope and planning outweigh a single code token,
/// and a narrow ask can cancel one back out to zero.
fn word_weight(word: &str) -> i32 {
    if BROAD_MARKERS.contains(&word) || PLANNING_MARKERS.contains(&word) {
        2
    } else if CODE_TOKENS.contains(&word) || SCOPE_MARKERS.contains(&word) {
        1
    } else if NARROW_MARKERS.contains(&word) {
        -1
    } else {
        0
    }
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

    #[test]
    fn routes_a_trivial_readme_typo_to_the_small_tier() {
        let c = classify("fix the typo in the README");
        assert_eq!(c, TaskComplexity::Trivial);
        assert_eq!(tier_for(c), ModelTier::Small);
    }

    #[test]
    fn routes_a_hard_multi_step_refactor_to_the_large_tier() {
        let task = "refactor the auth module and add tests for every edge case";
        let c = classify(task);
        assert_eq!(c, TaskComplexity::Hard);
        assert_eq!(tier_for(c), ModelTier::Large);
    }

    #[test]
    fn a_single_broad_scope_word_stays_on_the_small_tier() {
        let c = classify("review the changelog");
        assert_eq!(c, TaskComplexity::Moderate);
        assert_eq!(tier_for(c), ModelTier::Small);
    }

    #[test]
    fn two_broad_scope_words_reach_the_large_tier() {
        let c = classify("audit every module");
        assert_eq!(c, TaskComplexity::Hard);
        assert_eq!(tier_for(c), ModelTier::Large);
    }

    #[test]
    fn narrow_scope_words_cancel_a_code_token() {
        assert_eq!(classify("fix the typo"), TaskComplexity::Trivial);
        assert_eq!(
            classify("fix the typo and the spelling"),
            TaskComplexity::Trivial
        );
    }

    #[test]
    fn score_below_the_hard_threshold_stays_moderate() {
        let c = classify("fix the error test");
        assert_eq!(c, TaskComplexity::Moderate);
        assert_eq!(tier_for(c), ModelTier::Small);
    }

    #[test]
    fn score_at_the_hard_threshold_routes_to_the_large_tier() {
        let c = classify("review the error test");
        assert_eq!(c, TaskComplexity::Hard);
        assert_eq!(tier_for(c), ModelTier::Large);
    }

    #[test]
    fn one_step_separator_is_moderate_but_two_are_hard() {
        assert_eq!(
            classify("fix the build; restart the service"),
            TaskComplexity::Moderate
        );
        assert_eq!(
            classify("fix the build\nrestart the service"),
            TaskComplexity::Moderate
        );
        assert_eq!(
            classify("fix the build; restart the service; verify the log"),
            TaskComplexity::Hard
        );
    }

    #[test]
    fn classification_is_deterministic_and_case_insensitive() {
        let task = "Refactor the Auth module and add Tests for EVERY edge case";
        let first = classify(task);
        for _ in 0..8 {
            assert_eq!(
                classify(task),
                first,
                "the same input must classify the same way every time"
            );
        }
        assert_eq!(first, classify(&task.to_lowercase()));
    }
}
