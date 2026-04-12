---
name: jupytext
description: Create, update, sync, and execute jupytext-managed notebooks. Use when working with paired .md/.ipynb notebooks, fixing notebook execution errors, or adding new notebooks to the project.
---

# Jupytext notebook workflow

Notebooks are paired `.md` + `.ipynb` via jupytext. The `.md` file is the
source of truth — always edit the `.md`, never the `.ipynb` directly.

## Creating a new notebook

1. Write the `.md` file with markdown text and fenced `python` code cells.
2. Pair: `jupytext --set-kernel KERNEL --set-formats md,ipynb foo.md`
   — this adds frontmatter and creates the paired `.ipynb`.
3. Sync and execute: `jupytext --sync --execute foo.md`

Check available kernels with `jupyter kernelspec list`. Common values:
`python3`, `conda-env-XXX-py`, or a custom name.

## Executing an existing notebook

```sh
jupytext --sync --execute foo.md
```

This syncs `.md` → `.ipynb`, executes all cells, saves outputs to `.ipynb`,
and syncs timestamps back. One command, no intermediate steps.

Run from the notebook's directory so relative paths resolve correctly.
Prefix with the project's environment manager if needed (e.g. `pixi run`).

## Fixing a broken notebook

1. Read the `.md` to understand the code (not the `.ipynb` — it may have
   stale outputs or error markers).
2. Fix the code in the `.md`.
3. Delete the `.ipynb` and regenerate cleanly:

```sh
rm foo.ipynb
jupytext --sync foo.md            # recreates .ipynb from .md
jupytext --sync --execute foo.md  # execute
```

Deleting the `.ipynb` first avoids stale artifacts (papermill error spans,
`<!-- #region -->` markers) leaking back into the `.md` via sync.

## Cell tags and papermill

Cell metadata (like `tags=["parameters"]`) is set on the code fence line:

    ```python tags=["parameters"]
    x = 1.0
    ```

Tags survive the `.md` → `.ipynb` → `.md` roundtrip only if the `.md` is
newer when you sync. If the `.ipynb` is newer (e.g. after a manual
execution or a failed papermill run), jupytext syncs `.ipynb` → `.md` and
may overwrite your tags. Fix: delete the `.ipynb` and regenerate from `.md`.

When using papermill for parameter injection, sync from `.md` first:

```sh
jupytext --sync foo.md
papermill foo.ipynb foo.ipynb -p x 2.0
```

Don't reach for papermill by default. `jupytext --sync --execute` is the
standard workflow.
