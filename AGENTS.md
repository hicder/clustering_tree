# AGENTS.md

## Code comments

Every `pub` struct, including `pub(crate)` and `pub(super)`, must have `///` doc comments.

On the struct, explain:

- What it does.
- What it's for.
- An example. Put code or data examples in a ` ```text ` block.

Inside the struct, explain what each field is for.

Check comments against the code that uses the struct, not only its definition. Update them when its behavior changes.
