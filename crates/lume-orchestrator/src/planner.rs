//! Task decomposition.

use lume_core::model::ModelTier;

use crate::router::{classify, tier_for};

/// Subtask.
#[derive(Debug, Clone)]
pub struct Subtask {
    /// Description.
    pub description: String,
    /// Tier.
    pub tier: ModelTier,
}

/// Decompose a task into at most `max_subtasks` subtasks.
///
/// Each blank-line separated paragraph becomes a subtask, and a paragraph that
/// is a numbered list (`1.`, `2)`, ...) is split into one subtask per item, so a
/// multi-step plan costs one agent run per step instead of one run for the whole
/// list. Each subtask is classified on its own text, so a plan whose last step is
/// a one-line fix still runs that step on the small model.
///
/// A blank task decomposes to nothing: there is no work to classify or run.
pub fn decompose(task: &str, max_subtasks: usize) -> Vec<Subtask> {
    let mut descriptions: Vec<String> = Vec::new();
    for paragraph in task.split("\n\n") {
        descriptions.extend(items(paragraph));
    }
    descriptions
        .into_iter()
        .take(max_subtasks)
        .map(|description| Subtask {
            tier: tier_for(classify(&description)),
            description,
        })
        .collect()
}

/// Split one paragraph into subtask descriptions: one per list item when the
/// paragraph is a numbered list, otherwise the paragraph itself.
fn items(paragraph: &str) -> Vec<String> {
    let trimmed = paragraph.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() > 1 && lines.iter().all(|line| is_numbered_item(line)) {
        return lines.into_iter().map(str::to_string).collect();
    }
    vec![trimmed.to_string()]
}

/// Whether a line opens a numbered list item, such as `1.` or `12)`.
fn is_numbered_item(line: &str) -> bool {
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let Some(marker) = line[digits..].chars().next() else {
        return false;
    };
    let rest = &line[digits + marker.len_utf8()..];
    matches!(marker, '.' | ')') && (rest.is_empty() || rest.starts_with(char::is_whitespace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_task() {
        let subs = decompose("do something", 5);
        assert!(!subs.is_empty());
    }

    #[test]
    fn blank_task_decomposes_to_nothing() {
        assert!(decompose("", 5).is_empty());
        assert!(decompose("  \n\n\t \n", 5).is_empty());
    }

    #[test]
    fn zero_max_subtasks_decomposes_to_nothing() {
        assert!(decompose("do something", 0).is_empty());
    }

    #[test]
    fn blank_lines_separate_subtasks() {
        let subs = decompose("fix typo\n\nfix other typo", 5);
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn numbered_list_splits_into_one_subtask_per_item() {
        let subs = decompose("1. read the log\n2. summarise the failure", 5);
        let descriptions: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
        assert_eq!(
            descriptions,
            ["1. read the log", "2. summarise the failure"]
        );
    }

    #[test]
    fn multi_line_prose_stays_one_subtask() {
        let subs = decompose("first line of the ask\nsecond line of the ask", 5);
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn max_subtasks_caps_the_output() {
        let task = "a\n\nb\n\nc";
        assert_eq!(decompose(task, 2).len(), 2);
    }

    #[test]
    fn each_subtask_carries_its_own_tier() {
        let task = "fix typo\n\nexplain this:\n```rust\nfn main() {}\n```";
        let subs = decompose(task, 5);
        let tiers: Vec<ModelTier> = subs.iter().map(|s| s.tier).collect();
        assert_eq!(tiers, [ModelTier::Small, ModelTier::Large]);
    }
}
