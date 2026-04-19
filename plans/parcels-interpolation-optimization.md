# Parcels interpolation optimization

## Problem statement

Profiling the Baltic drifter simulation (6 particles, 3457 timesteps,
dt=300 s) shows `fieldset.UV.eval()` costs 30 s total for ~17K calls.
Of that, 24 s (80%) is spent inside `_get_corner_data_Agrid` which
constructs `xr.DataArray` index objects and calls `data.isel()` on every
invocation.  The call chain does redundant work:

1. **Grid search repeats.**  `_search_1d_array` calls `np.searchsorted`
   unconditionally on every `eval()`, even though consecutive calls for
   the same particle at different depth levels share identical `(lat, lon,
   time)` and therefore identical `(yi, xi, ti)`.  The `ei` hint mechanism
   exists but is ignored for rectilinear grids.
2. **xarray indexing overhead.**  `_get_corner_data_Agrid` constructs
   `xr.DataArray` objects for each index dimension (lines 73-80 of
   `_xinterpolators.py`), then calls `data.isel()`.  This creates and
   destroys xarray alignment/index machinery ~17K times.
3. **Per-depth-level `eval()`.**  Our kernel calls `fieldset.UV.eval()`
   D times per timestep (once per depth level, D=5 for typical drogue
   depths).  Each call independently searches the grid, allocates index
   arrays, and fetches corner data -- even though only `zi` changes.

Particles barely move between timesteps (Baltic Sea, dt=300 s), so grid
cell indices `(xi, yi, ti)` are often identical across consecutive calls
within a timestep AND across consecutive timesteps.

## Call chain: `fieldset.UV.eval()` to numpy array access

### Level 0: `VectorField.eval()` (field.py:288)

```python
def eval(self, time: datetime, z, y, x, particles=None):
```

- Calls `_get_positions(self.U, time, z, y, x, particles, _ei)`
- Calls `self._vector_interp_method(particle_positions, grid_positions, self)`
  (which is `XLinear_Velocity` for A-grids)
- **Inputs:** time (datetime), z/y/x (float or array), particles (optional)
- **Outputs:** (u, v) or (u, v, w) arrays
- **Hashable?** time is datetime, z/y/x are numpy arrays (not hashable)
- **Caching:** Could cache at this level keyed on `(time, z_hash, y_hash, x_hash)` but arrays aren't hashable natively

### Level 1: `_get_positions()` (field.py:455)

```python
def _get_positions(field, time, z, y, x, particles, _ei) -> tuple[dict, dict]:
```

- Builds `particle_positions = {"time": time, "z": z, "lat": y, "lon": x}`
- Calls `_search_time_index(field, time)` -> `{"T": {"index": ti, "bcoord": tau}}`
- Calls `field.grid.search(z, y, x, ei=_ei)` -> `{"Z": ..., "Y": ..., "X": ...}`
- Updates particle ei and states
- **Output:** `(particle_positions, grid_positions)` dicts containing int indices and float barycentric coords
- **What stays the same across depth levels:** `(yi, eta, xi, xsi, ti, tau)` -- only `(zi, zeta)` changes
- **What stays the same across timesteps:** often `(yi, xi)` when particles barely move

### Level 2a: `_search_time_index()` (index_search.py:65)

```python
def _search_time_index(field, time) -> dict:
```

- Converts `field.data.time.data` to float via `timedelta_to_float` every call (line 88)
- Calls `_search_1d_array(time_flt, time)`
- **Redundant work:** `timedelta_to_float(field.data.time.data - field.time_interval.left)` recomputes the same float array every call
- **Cache opportunity:** The time coordinate array is immutable; cache the float conversion on the field

### Level 2b: `XGrid.search()` (xgrid.py:294)

```python
def search(self, z, y, x, ei=None):
```

- For rectilinear grids, calls `_search_1d_array` independently for depth, lat, lon
- `ei` hint is **only used for curvilinear grids** (line 302-315); for rectilinear, `searchsorted` runs unconditionally
- **Redundant work:** When called D times per timestep with same `(y, x)` but different `z`, the lat/lon searches repeat identically

### Level 3: `_search_1d_array()` (index_search.py:20)

```python
def _search_1d_array(arr, x) -> tuple[np.ndarray, np.ndarray]:
```

- `np.searchsorted(arr, x, side="right") - 1`
- Clips, computes barycentric coordinate, checks bounds
- Pure function, no side effects, deterministic
- **Inputs:** `arr` (1-D numpy, grid coords), `x` (particle positions)
- **Cost:** cheap per call (~1 us for 6 particles), but called ~85K times total (5 axes x 17K evals)

### Level 4: `XLinear_Velocity()` (_xinterpolators.py:147)

```python
def XLinear_Velocity(particle_positions, grid_positions, vectorfield):
```

- Calls `XLinear(particle_positions, grid_positions, vectorfield.U)` (line 153)
- Calls `XLinear(particle_positions, grid_positions, vectorfield.V)` (line 154)
- Applies spherical conversion if needed
- **Each `XLinear` call independently fetches corner data**

### Level 5: `XLinear()` (_xinterpolators.py:98)

```python
def XLinear(particle_positions, grid_positions, field):
```

- Extracts `(xi, xsi, yi, eta, zi, zeta, ti, tau)` from `grid_positions`
- Calls `field.grid.get_axis_dim_mapping(field.data.dims)` -- dict lookup, cheap but repeated
- Calls `_get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)`
- Does trilinear interpolation on the returned corner array
- **This is where 80% of the time goes** (in `_get_corner_data_Agrid`)

### Level 6: `_get_corner_data_Agrid()` (_xinterpolators.py:37) -- THE BOTTLENECK

```python
def _get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim) -> np.ndarray:
```

- Builds index arrays for the 2x2x2x2 stencil corners (16 points per particle for 4D)
- **Creates `xr.DataArray` objects for each dimension** (lines 73-80):
  ```python
  selection_dict[axis_dim["X"]] = xr.DataArray(xi, dims=("points"))  # line 74
  selection_dict[axis_dim["Y"]] = xr.DataArray(yi, dims=("points"))  # line 76
  ```
- Calls `data.isel(selection_dict)` (line 82) -- this triggers xarray's full alignment machinery
- Reshapes result to `(lenT, lenZ, 2, 2, npart)`
- **Inputs:** `data` (xr.DataArray, not hashable), indices (int arrays, could be made hashable via `.tobytes()`)
- **Output:** numpy array of shape `(lenT, lenZ, 2, 2, npart)` -- pure numeric, cacheable
- **Called 2x per eval** (once for U, once for V via `XLinear`)
- **What's redundant:** When `(ti, yi, xi)` don't change between depth levels, only `zi` differs. The xarray DataArray construction and `isel` call repeat for all 4 dims even when 3 of 4 are identical.

## Caching opportunities

### Opportunity A: Cache `_get_corner_data_Agrid` output

**Cache key:** `(id(data), ti.tobytes(), zi.tobytes(), yi.tobytes(), xi.tobytes(), lenT, lenZ)`

- `id(data)` distinguishes U vs V DataArrays (cheap, hashable)
- `.tobytes()` converts small int arrays to hashable bytes (6 particles = 24-48 bytes)
- **Expected hit rate within a timestep:** 0% (zi changes each depth level).  But within `XLinear_Velocity`, U and V often share `(ti, yi, xi)` on A-grids -- could cache the shared indices.
- **Expected hit rate across timesteps:** High when particles don't change cells.  For 6 particles barely moving, `(ti, yi, xi)` repeat for many consecutive timesteps.
- **Memory cost:** One cached array per unique `(ti, zi, yi, xi)` combo.  At `(2, 2, 2, 2, 6)` float64 = ~768 bytes.  With D=5 depth levels x 2 fields = 10 entries = ~8 KB.  Negligible.
- **Invasiveness:** Requires monkey-patching `_get_corner_data_Agrid` or wrapping it. Moderate -- the function is a module-level function in `_xinterpolators.py`.

**Implementation sketch:**
```python
_corner_cache = {}

def _cached_get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim):
    key = (id(data), ti.tobytes(), zi.tobytes(), yi.tobytes(), xi.tobytes(), lenT, lenZ)
    if key in _corner_cache:
        return _corner_cache[key]
    result = _original_get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)
    _corner_cache[key] = result
    return result
```

Cache invalidation: clear at start of each kernel call (or use bounded dict).

### Opportunity B: Cache grid search results across depth levels

Within one kernel call, `field.grid.search(z, y, x)` is called D times
with identical `(y, x)` but different `z`.  Factor out the spatial search:

```python
# Do spatial search once
yi, eta = _search_1d_array(grid._ds.lat.values, y)
xi, xsi = _search_1d_array(grid._ds.lon.values, x)
ti, tau = _search_time_index(field, time)

# Then for each depth level, only search z
for z_level in depth_levels:
    zi, zeta = _search_1d_array(grid._ds.depth.values, z_arr)
    grid_positions = {"Z": {"index": zi, "bcoord": zeta},
                      "Y": {"index": yi, "bcoord": eta},
                      "X": {"index": xi, "bcoord": xsi},
                      "T": {"index": ti, "bcoord": tau}}
    # call interpolator directly
```

- **Expected speedup:** Eliminates 4*(D-1) redundant `searchsorted` calls per timestep per field.  Small absolute saving (~ms) but architecturally clean.
- **Invasiveness:** Requires bypassing `VectorField.eval()` and calling the interpolator directly.  Medium -- we already own the kernel code.
- **Risk:** Ties us to the internal `grid_positions` dict format.

### Opportunity C: Bypass xarray entirely for corner data extraction

Replace the xarray `data.isel()` call with direct numpy indexing on `data.values`:

```python
values = data.values  # shape (T, Z, Y, X) -- cache this reference
corner_data = values[ti_idx, zi_idx, yi_idx, xi_idx]  # numpy fancy indexing
```

- **Expected speedup:** Eliminates xr.DataArray construction and xarray alignment.  This is where 24 s of the 30 s goes.  Expected 10-20x speedup for this function.
- **Hit rate:** N/A -- not a cache, just avoiding overhead.
- **Memory cost:** Zero (references existing array).
- **Invasiveness:** HIGH if done inside Parcels (modify `_get_corner_data_Agrid`).  LOW if done in our kernel by bypassing `XLinear` entirely.
- **Risk:** `data.values` triggers dask compute if data is lazy.  For our use case (small Baltic datasets loaded eagerly), this is fine.  Guard with `if is_dask_collection(data): data = data.compute()`.

**Implementation sketch (in our kernel):**
```python
# At kernel init or first call, extract and cache numpy arrays
U_np = fieldset.U.data.values  # (T, Z, Y, X)
V_np = fieldset.V.data.values

# Then do our own trilinear interpolation:
# 1. searchsorted for all dims (once for lat/lon/time, per-level for z)
# 2. numpy fancy indexing for corners
# 3. manual trilinear interp (same math as XLinear, ~10 lines)
```

### Opportunity D: Cache `_search_time_index` float conversion

`_search_time_index` (index_search.py:88) calls `timedelta_to_float(field.data.time.data - field.time_interval.left)` every invocation.  This converts the same datetime array to float every time.

- **Cache:** Compute once, store on the field object (or in a module-level dict keyed on `id(field.data.time)`).
- **Saving:** Minor (~ms total) but trivially correct.
- **Invasiveness:** Monkey-patch `_search_time_index` or cache in our kernel.

### Opportunity E: Batch all depth levels into one vectorized `eval()` call

Instead of calling `fieldset.UV.eval()` D times with `(N,)` arrays, call once with `(D*N,)` arrays:

```python
z_all = np.repeat(depth_levels, N)          # (D*N,)
lat_all = np.tile(lat, D)                   # (D*N,)
lon_all = np.tile(lon, D)                   # (D*N,)
u_all, v_all = fieldset.UV.eval(time, z_all, lat_all, lon_all)[:2]
U_profiles = u_all.reshape(D, N)
V_profiles = v_all.reshape(D, N)
```

- **Expected speedup:** Reduces 5 eval calls to 1.  `searchsorted` on 30 elements vs 6 is nearly the same cost.  The big win is that `_get_corner_data_Agrid` is called once with a larger batch rather than 5 times with small batches.  xarray overhead is per-call, not per-element.  Expected 3-4x speedup (5 calls -> 1 call, but the single call does more work).
- **Memory cost:** Negligible (30 elements instead of 6).
- **Invasiveness:** ZERO -- pure change in our kernel, no Parcels modification.
- **Risk:** `XLinear_Velocity` applies spherical conversion using `particle_positions["lat"]`, which would be the tiled lat array.  This is correct because `cos(lat)` is the same for all depths at the same particle.

### Opportunity F: Pre-extract numpy arrays and do our own interpolation

The nuclear option: extract `U.data.values` and `V.data.values` once (numpy arrays, shape `(T, Z, Y, X)`), plus grid coordinate arrays, and implement trilinear interpolation ourselves.  Skip Parcels' eval entirely.

- **Expected speedup:** Eliminates ALL xarray overhead.  30 s -> estimated 0.5-1 s (numpy fancy indexing + arithmetic on 6 particles is ~microseconds per call).
- **Memory cost:** Zero extra (numpy arrays already exist inside the DataArrays).
- **Invasiveness:** HIGH in terms of code we own.  ZERO Parcels changes.  But we lose grid-agnosticism (must handle A-grid layout ourselves).
- **Risk:** Duplicates Parcels interpolation logic.  Must handle edge cases (out-of-bounds, time boundaries, spherical correction).  The A-grid trilinear interpolation is ~15 lines of numpy, manageable.  C-grid support would require reimplementing the Jacobian rotation (~60 lines).

**This is essentially what the old `make_dd_velocity_interpolator` did** (and was removed for good reasons in D-I).  Only revisit if the speedup justifies the maintenance cost.

## Summary of options

| ID | What | Speedup | Invasiveness | Parcels changes? | Risk |
|----|------|---------|--------------|-------------------|------|
| A | Cache `_get_corner_data_Agrid` output | ~2x across timesteps | Medium (monkey-patch) | No | Cache invalidation if field data changes |
| B | Factor out spatial search across depth levels | ~1.1x | Medium (bypass eval) | No | Tied to internal dict format |
| C | Direct numpy indexing in `_get_corner_data_Agrid` | ~10-20x for that function | High (Parcels PR) or Low (our kernel) | Yes or No | Dask-lazy data needs guard |
| D | Cache time float conversion | ~1.01x | Low (monkey-patch) | No | Negligible |
| **E** | **Batch depth levels into one eval()** | **~3-4x** | **Zero** | **No** | **Correct by construction** |
| F | Own numpy interpolation, bypass Parcels | ~30-60x | High (our code) | No | Maintenance burden, grid lock-in |

## Recommended approach

### Phase 1: Quick wins, no Parcels changes (target: 30 s -> ~8 s)

1. **Opportunity E: Batch depth levels** into a single `fieldset.UV.eval()` call.  This is a 5-line change in the kernel (parcels.py).  Expected 3-4x speedup on the Parcels portion (30 s -> ~8 s).  Zero risk.

2. **Opportunity D: Cache time float conversion** via a one-line monkey-patch or wrapper.  Trivial, ~0 risk.

### Phase 2: Bypass xarray for A-grids (target: 8 s -> ~1 s)

3. **Opportunity C+F hybrid: extract numpy arrays, do our own interpolation, but only for A-grids.**  Wrap it in a helper that checks `field.grid._gtype == GridType.RectilinearZGrid` and falls back to `fieldset.UV.eval()` for other grid types.  This gives us 30x speedup on the common case while preserving grid-agnosticism as a fallback.

   Structure:
   ```python
   def _fast_profile_sample_agrid(fieldset, time, lat, lon, depth_levels):
       """A-grid fast path: numpy-only trilinear interpolation."""
       ...

   def _generic_profile_sample(fieldset, time, lat, lon, depth_levels):
       """Generic path: batched fieldset.UV.eval()."""
       ...

   # In the kernel (_extract_profiles):
   if is_rectilinear_agrid(fieldset):
       U_profiles, V_profiles = _fast_profile_sample_agrid(...)
   else:
       U_profiles, V_profiles = _generic_profile_sample(...)
   ```

   The A-grid fast path is ~30 lines of numpy (searchsorted + fancy index + trilinear interp).

### Phase 3: Upstream contribution (optional, long-term)

4. **Propose to Parcels:** a `field.eval_numpy()` path that skips xarray DataArray construction when the data is already a numpy array.  This would benefit all Parcels users, not just us.  The change is ~10 lines in `_get_corner_data_Agrid`:

   ```python
   # Instead of:
   selection_dict[axis_dim["X"]] = xr.DataArray(xi, dims=("points"))
   ...
   return data.isel(selection_dict).data.reshape(...)

   # Use:
   if isinstance(data, np.ndarray):
       # Direct fancy indexing (no xarray overhead)
       return data[ti, zi, yi, xi].reshape(...)
   else:
       # Existing xarray path for dask/lazy data
       ...
   ```

   File: `parcels/interpolators/_xinterpolators.py`, lines 72-82.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Phase 1 (batching) changes particle_positions dict shape | `XLinear_Velocity` indexes `particle_positions["lat"]` for spherical correction.  The tiled array has the right values at the right positions -- test by comparing batched vs per-level output. |
| Phase 2 (numpy bypass) duplicates Parcels logic | Keep it to A-grid trilinear only (~30 lines).  Add a regression test comparing our output against `fieldset.UV.eval()` for the same inputs. |
| Phase 2 breaks on dask-lazy data | Guard: `if is_dask_collection(data): return _generic_profile_sample(...)` |
| Parcels internal API changes | We already depend on `grid._mesh`, `grid.depth`, `grid._gtype`.  Pin Parcels version.  Phase 1 has zero internal API dependency. |
| Cache invalidation (Opportunity A, if pursued) | Not needed for Phases 1-2.  If pursued later, clear cache at each kernel call (simple and correct). |

## Out of scope

- Numba acceleration of the interpolation (covered in [numba-acceleration.md](numba-acceleration.md))
- Warm-starting the ODE (covered in [d-i-parcels-isolation.md](d-i-parcels-isolation.md))
- C-grid or unstructured grid fast paths (wait for a real use case)
- Upstream Parcels PR (Phase 3 is a suggestion, not a commitment)
