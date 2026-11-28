# AGENTS.md

## Code comments

Every `pub` struct, including `pub(crate)` and `pub(super)`, must have `///` doc comments.

On the struct, explain:

- What it does.
- What it's for.
- An example. Put code or data examples in a ` ```text ` block.

Inside the struct, explain what each field is for.

Check comments against the code that uses the struct, not only its definition. Update them when its behavior changes.

## Clippy

Always run `cargo clippy --workspace --all-targets` and make sure all Clippy checks pass before finishing work.

## Documentation

Check `docs/` and `README.md` when making changes and make sure they stay up to date.

## Plans

Put all plans under `docs/plans/`.
