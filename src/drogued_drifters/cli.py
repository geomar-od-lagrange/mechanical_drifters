"""CLI for drogued_drifters cache management.

Usage::

    pixi run save-eom-cache        # save symbolic EOM to .srepr cache file
"""

from pathlib import Path

import click
from drogued_drifters.lagrange_model import _save_eom_cache


@click.command()
def save_eom_cache():
    """Save the symbolic EOM to .srepr cache file.

    This pre-computes the symbolic derivation once and writes it to
    src/drogued_drifters/data/symbolic_eom.srepr for faster imports.
    """
    cache_path = Path(__file__).resolve().parent / "data" / "symbolic_eom.srepr"

    click.echo("Running sympy derivation (this may take a minute)...")
    _save_eom_cache(cache_path)

    click.echo(f"Wrote {cache_path}")
