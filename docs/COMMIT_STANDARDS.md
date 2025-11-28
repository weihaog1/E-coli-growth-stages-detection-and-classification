# Commit Standards

## Format

```
<type>: <description>
```

Keep the description under 50 characters. Use imperative mood ("Add" not "Added").

## Types

| Type | Use for |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructure (no behavior change) |
| `docs` | Documentation only |
| `style` | Formatting, whitespace |
| `test` | Adding or updating tests |
| `chore` | Dependencies, configs, build |

## Examples

```
feat: Add sliding window detector
fix: Handle empty annotation files
refactor: Simplify IoU calculation
docs: Update training instructions
chore: Bump ultralytics to 8.1.0
```

## Rules

1. One logical change per commit
2. Code must run after each commit
3. Do not commit generated files (weights, logs, cache)
4. Reference issue numbers when applicable: `fix: Handle edge case (#12)`

