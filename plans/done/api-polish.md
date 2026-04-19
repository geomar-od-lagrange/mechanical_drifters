Implemented. See [docs/architecture.md](../../docs/architecture.md).

# Plan: API polish

Follow-up to [model-independence-refactor](model-independence-refactor.md)
(implemented). Small targeted cleanups to the model API and eom internals.

## 1. Physics defaults on the NamedTuple

Put default values directly on NamedTuple fields:

```python
class DroguedDrifterPhysics(NamedTuple):
    m_b: float = 1.0
    m_d: float = 2.7
    m_hat_d: float = 1.0
    m_tilde_d: float = 101.0
    m_tilde_b: float = 1.9
    l: float = 3.0
    g: float = 9.81
    k_b: float = 12.0
    k_d: float = 154.0
```

`DroguedDrifterPhysics()` gives Callies et al. defaults. Delete
`DEFAULT_PHYSICS`. Each model's `__init__` uses its Physics type as the
default arg:

```python
def __init__(self, physics=DroguedDrifterPhysics(), *, backend="numpy"):
    super().__init__(physics, backend=backend)
```

Same for PointSurfaceDrifter.

## 2. `_max_depth` as a property

Replace `_max_depth(self, physics)` with:

```python
@property
def _max_depth(self):
    return self.physics.l
```

Update `parcels.py` to `max_depth = getattr(model, '_max_depth', 0.0)`.

## 3. Rename State types → `_State`

`DroguedDrifterState` → `_State`, `PointSurfaceState` → `_State`. Internal
to each model module. The underscore signals "you don't construct these."
Physics types stay public (callers construct them).

## 4. Extract `caching.py` from `eom.py`

Move disk-cache logic (`_cache_key`, `_load_or_derive`, pickle round-trip)
into `caching.py`. `eom.py` becomes purely sympy → numpy: derive, lambdify,
wrap. A physicist reading `eom.py` sees physics, not `hashlib`.

## 5. Delete `_build_packer`, use `(*physics, *state)`

The lambda arg order in `_derive_symbolic` is Physics fields then State
fields by construction. `_build_packer`'s introspection rediscovers this.
Replace with:

```python
def pack_eom_args(physics, state):
    return (*physics, *state)
```

Add a safety test that catches misordering (parametrized over all models):

```python
def test_pack_order_matches_lambdify_args(model):
    physics = model.Physics(*range(1, len(model.Physics._fields) + 1))
    state = model.State(*(np.array([float(i)])
                          for i in range(100, 100 + len(model.State._fields))))
    _, M_raw, _, pack = _get_eom_callables(model)
    np.testing.assert_array_equal(
        M_raw(*pack(physics, state)),
        M_raw(*physics, *state),
    )
```

Once test passes, delete `_build_packer`.

## 6. Merge `_make_qdd_func` into `_get_eom_callables`

One function, one cache, one entry point:

```python
def _get_eom_callables(model, backend="numpy"):
    # Returns (qdd_func, M_raw, F_raw)
    # qdd_func: backend-wrapped, handles batch/scalar
    # M_raw, F_raw: raw lambdified, for exploration
    ...
```

`base.__init__` becomes:
```python
self._qdd_func = _get_eom_callables(self, backend)[0]
```

## 7. Numba as first-class backend

Add `backend` as a papermill parameter to the long-running baltic
notebooks:

```python tags=["parameters"]
backend = "numba"
```

Verify numba works with the current `integrate()` return shape.
