# Agent Instructions

## Repository workflow

- Never work directly on main or master.
- Always inspect `git status` before modifying files.
- Use one branch per task.
- Prefer small, reviewable changes.
- Do not reformat unrelated files.
- Do not force-push.
- Do not merge pull requests without human approval.

## Handoff protocol

Before finishing, summarize:
- Files changed.
- Tests run.
- Remaining risks.
- Next recommended step.

## GitHub workflow

For non-trivial changes:
- Read the related issue or pull request first.
- Create or use a task-specific branch.
- Open a pull request instead of committing directly to the main branch.
- Leave a handoff comment in the issue or pull request.