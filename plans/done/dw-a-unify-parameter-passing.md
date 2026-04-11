# DW-A: Unify parameter passing

## Decision

Two structs that match the two lifetimes:

- `DrifterPhysics` (9 fields) — physical constants, frozen, set once per drifter instance.
  Public: users construct drifters with these params.
- `EOMState` (10 fields) — per-timestep state + forcing, built fresh each evaluation.
  Internal: fields use stereographic coordinates, not part of the public API.

These replace the current `LagrangeParams` (19-field flat NamedTuple) and the `_params()` dict.

The lambdified functions' own signatures are the source of truth for argument ordering.
No manually maintained ordering constant. A packer function is built once by inspecting
the lambda signatures, then reused at zero overhead per call.

## Structs

```python
class DrifterPhysics(NamedTuple):
    m_b: float        # buoy dry mass [kg]
    m_d: float        # drogue dry mass [kg]
    m_hat_d: float    # drogue buoyancy correction [kg]
    m_tilde_d: float  # drogue added mass [kg]
    m_tilde_b: float  # buoy added mass [kg]
    l: float          # pole length [m]
    g: float          # gravitational acceleration [m/s^2]
    k_b: float        # buoy drag coefficient [kg/m]
    k_d: float        # drogue drag coefficient [kg/m]

class EOMState(NamedTuple):
    """Per-timestep state variables and forcing.

    Fields hold scalars (in rhs) or (N,) arrays (in _rhs_batch).
    Not part of the public API — fields use stereographic coordinates.
    """
    u: float     # stereographic u
    v: float     # stereographic v
    xd: float    # buoy x velocity [m/s]
    yd: float    # buoy y velocity [m/s]
    ud: float    # stereographic u velocity [1/s]
    vd: float    # stereographic v velocity [1/s]
    U_b: float   # current at buoy, east [m/s]
    V_b: float   # current at buoy, north [m/s]
    U_d: float   # current at drogue, east [m/s]
    V_d: float   # current at drogue, north [m/s]
```

## Argument ordering — inspect, don't maintain

The lambdified `_raw_M` / `_raw_F` are regular Python functions with inspectable
signatures (parameter names come from `str(sympy_symbol)`). We use this to
**derive** the correct packing order rather than maintaining it manually.

### Why the static symbol substitution exists

Sympy's `dynamicsymbols` (e.g. `x(t)`, `u_st(t)`) carry implicit time dependence.
Their `str()` representations (`"x(t)"`, `"Derivative(x(t), t)"`) are not valid
Python identifiers, so `lambdify` replaces them with anonymous `_Dummy_N` parameters.
These are not inspectable — you cannot map `_Dummy_35` back to a struct field.

The static substitution replaces each dynamic symbol with a plain `sp.Symbol` before
lambdifying. This produces named parameters that match our struct fields, enabling
signature-based packing. The substitution is necessary for inspectability, not just
for lambdify to work numerically.

### Rename static symbols to match struct fields

Currently the static symbols use `_s` suffixes (`u_s`, `v_s`, `xd_s`, ...).
As part of this refactor, rename them to match the struct field names directly:

```python
# In _derive_symbolic, change:
u_s, v_s = sp.symbols("u_s v_s", real=True)      # old
# to:
u_static, v_static = sp.symbols("u v", real=True)  # new — symbol NAME is "u", "v"
```

(Python variable names like `u_static` are just local names within `_derive_symbolic`;
the sympy symbol names — what `str()` returns and what appears in the lambda signature —
are `"u"`, `"v"`, `"xd"`, etc.)

This makes the lambda parameter names match `EOMState` and `DrifterPhysics` field names
directly. No mapping table needed.

### Packer construction (once, at first call via lru_cache)

```python
def _build_packer(raw_func):
    """Inspect raw_func's signature, return a pack_eom_args(physics, state) callable.

    Called once (cached). Maps each lambda parameter name to a field in
    DrifterPhysics or EOMState by name. Returns a closure that assembles
    the positional arg tuple from (physics, state).

    Raises KeyError immediately if a parameter name doesn't map to
    any struct field — no silent ordering bugs.
    """
    param_names = list(inspect.signature(raw_func).parameters)
    physics_fields = DrifterPhysics._fields
    state_fields = EOMState._fields

    indices = []  # list of ('p'|'s', field_index)
    for name in param_names:
        if name in physics_fields:
            indices.append(('p', physics_fields.index(name)))
        elif name in state_fields:
            indices.append(('s', state_fields.index(name)))
        else:
            raise KeyError(
                f"Lambda param {name!r} not found in DrifterPhysics or EOMState fields"
            )

    def pack_eom_args(physics, state):
        return tuple(physics[i] if src == 'p' else state[i] for src, i in indices)

    return pack_eom_args
```

### Runtime cost

- `inspect.signature`: ~13 us — called once, cached inside `_get_eom_callables`.
- `pack_eom_args(physics, state)`: tuple indexing, same cost as manual tuple construction.
- `_raw_M(*args)`: ~2.4 us — unchanged, this is the real work.

### What this buys us

- **Self-checking.** If a struct field is renamed or a sympy symbol changes, packer
  construction fails with a KeyError at first call — not a silent wrong-answer bug.
- **DW-C proof.** If the CSE/exec pipeline is later replaced with `sp.lambdify(cse=True)`,
  the same inspection approach works (verified: standard lambdify produces inspectable
  signatures with named params).
- **No mapping table.** Since static symbols are renamed to match struct fields, there
  is no `_SYMPY_TO_FIELD` dict. The lambda param names *are* the struct field names.

## Information flow

```
CURRENT (3 repacking steps)              PROPOSED (1 repacking step)
===============================          ================================

__init__: 9 self.* attrs                 __init__: self.physics = DrifterPhysics(...)

rhs / _rhs_batch                         rhs / _rhs_batch
  |  unpack state, sample currents         |  unpack state, sample currents
  v                                        v
_params() -> dict         REPACK 1       state = EOMState(u, v, ..., U_d, V_d)
  |                                        |
  v                                        v
M_func(**p, u=u, ...)     REPACK 2       M_func(self.physics, state)
  |                                        v
  v                                      pack_eom_args(physics, state)  REPACK 1
LagrangeParams(...)       REPACK 3         |  (prebuilt via signature inspection)
  |                                        v
  v                                      _raw_M(*args)
_raw_M(*params)
```

## Changes

### 1. `lagrange_model.py`

- Add `DrifterPhysics`, `EOMState` (NamedTuples).
- Add `_build_packer(raw_func)` — inspects signature, returns `pack_eom_args`.
- Rename static symbols in `_derive_symbolic`: `u_s` → `sp.symbols("u")`, etc.
  Add comment explaining why static substitution exists (dynamic symbols produce
  anonymous `_Dummy_N` params in lambdify, breaking inspectability).
- Update `_get_eom_callables()` to also build and return packers (still lru_cached).
- `M_func(physics, state)` / `F_func(physics, state)`: use prebuilt packer, call
  `_raw_M(*pack_eom_args(physics, state))`. Drop 19-kwarg signatures.
- Delete `LagrangeParams`.
- `_derive_symbolic`: `symbol_map` updated to use new symbol names. The ordering it
  produces flows into the lambda signatures, which the packer reads.
- `.srepr` cache: will need regeneration (symbol names change from `u_s` → `u` etc.).
  After regeneration, the packer's signature inspection provides automatic consistency
  checking — stale cache with wrong param names causes KeyError in `_build_packer`.

### 2. `drifter.py`

- `__init__`: build `self.physics = DrifterPhysics(...)`. Drop individual `self.m_b` etc.
  Access via `self.physics.m_b` where needed (only `self.l` is used outside `__init__`,
  in `_z_eff`-related code).
- Delete `_params()`.
- `_eval_M_F`: drop unused `t, x, y` params. Build `EOMState(...)` from remaining args,
  call `M_func(self.physics, state)`. Method becomes ~3 lines (DW-D inlining is next).
- `_rhs_batch`: build `EOMState(...)` from batch arrays, call `M_func(self.physics, state)`.
- Update `rhs` call to `_eval_M_F` to drop `t, x, y` args.

### 3. `__init__.py`

Export `DrifterPhysics` (public). Do not export `EOMState` (internal, uses stereographic
coords). Keep exporting `M_func`, `F_func`.

### 4. Tests

- All `M_func(u=..., m_b=..., ...)` calls become `M_func(DrifterPhysics(...), EOMState(...))`.
- Tests that reference `LagrangeParams` switch to `DrifterPhysics` / `EOMState`.
- The old `LagrangeParams._fields` ordering test is replaced by: verify that
  `_build_packer` succeeds (i.e. all lambda params map to struct fields). This is
  implicitly tested by every test that calls `M_func`/`F_func`, but an explicit
  test is cheap and documents the contract.
- Existing numeric suite covers correctness. No other new tests needed.

### 5. Regenerate `.srepr` cache

Symbol names change (`u_s` → `u`, etc.), so the cache must be regenerated.

### 6. Plan update

Tick off DW-A in `plans/code-review-remarks.md`.

## Order of operations

1. Add `DrifterPhysics`, `EOMState`, `_build_packer` to `lagrange_model.py`.
   Rename static symbols. Update `_derive_symbolic` and `symbol_map`.
   Update `_get_eom_callables` to build packers.
   Change `M_func`/`F_func` signatures. Delete `LagrangeParams`.
2. Regenerate `.srepr` cache.
3. Update `drifter.py` (`__init__`, delete `_params`, update `_eval_M_F` / `_rhs_batch` / `rhs`).
4. Update `__init__.py` exports.
5. Update all tests.
6. Run full test suite.
7. Tick off DW-A.
