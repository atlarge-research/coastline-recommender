# Code quality standard

The bar Coastline holds itself to. "Too much code written for how little it does" is the
failure mode; the goal is code that reads like a careful human wrote it — one home per
concept, small functions, lean prose, and no more machinery than the problem needs.

Use this as the review lens for every change (and as the acceptance lens for the ongoing
de-slop work).

## Functional (checkable) requirements

1. **One home per concept.** Each vocabulary, table, model, and serializer is defined exactly once; no parallel copies.
2. **Conversions only at the boundary.** Foreign formats convert in one place; every inner layer speaks only the canonical type.
3. **Enums for closed sets.** Policies, presets, goals, normalization are enums defined once — never scattered string `Literal`s. (Pydantic request models may keep `Literal`s where they *are* the API contract.)
4. **Small, single-purpose functions.** A function that resolves-then-builds-then-constructs is split into three; no 70-line kitchen sinks.
5. **No dead code.** Every module, function, and config key has a real (non-test) caller; delete it on discovery.
6. **Trust upstream.** No re-validating what a type, pydantic, or the caller already guarantees ("checking 1+1==2").
7. **Lean docstrings.** One line when the signature is obvious; document *why*, never restate *what*.
8. **Refactors are test-guarded.** Every consolidation has a characterization test pinning the exact prior output, written first.
9. **Errors isolate, never crash.** A bad row or input degrades locally (`feasible=False`); the batch or process keeps going.
10. **One spelling per field.** A field/output column has a single canonical name across CLI, SDK, and UI.

## Non-functional (qualitative) requirements

1. **Readability.** A newcomer understands a unit from its interface alone — without reading its internals or its callers.
2. **Testability.** Every unit is testable in isolation against a clear oracle; tests are fast and hermetic.
3. **Maintainability.** Changing one behaviour means editing one place; no ripple across duplicated copies.
4. **Proportionality.** Code volume matches problem size; prefer deleting over generalizing; never abstract for a single caller.
5. **Consistency.** New code reads like the code around it — same naming, idioms, and structure throughout.

## The two rules that resolve most disputes

- **Delete beats generalize.** Don't build an abstraction with one caller; don't keep dead code "for symmetry." When a boundary layer isn't wired in, wire it or delete it — never leave it dead.
- **Add a conversion at the boundary, don't expand the layer.** When one layer needs another's format, put the conversion at the seam and keep both layers speaking their own canonical type — rather than teaching every layer every format.
