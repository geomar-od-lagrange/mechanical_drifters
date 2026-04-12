# D2: Eliminate the phi singularity via stereographic projection

## Decision

**Use stereographic projection (Option C).** The current eps/nu
regularization introduces noise and uncertainty at exactly the most
likely drifter configuration (theta ~ pi), which hampers convergence
detection. A coordinate transform eliminates the singularity cleanly
with no tunable parameters.

## The singularity

The drogued drifter state is parameterized by generalized coordinates
q = (x, y, theta, phi), where theta is the zenith angle of the pole
(theta=0: drogue above buoy, theta=pi: drogue below buoy) and phi is
the azimuthal angle.

The drogue position relative to the buoy is:

    r = l * (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

At theta=pi (drogue hanging straight down), sin(theta)=0, so the
drogue lies on the z-axis regardless of phi. The azimuthal angle phi
becomes degenerate: all values of phi describe the same physical
configuration.

In the EOM M*qdd = F, this manifests as M[3,3] -> 0 when theta -> pi:

    M[3,3] = l^2 * (m_d + m_tilde_d) * sin^2(theta)

The equilibrium has theta routinely within ~0.05 rad of pi, making
sin^2(theta) ~ 0.0025 — small enough to cause numerical stiffness.

## Why the current regularization is inadequate

The eps/nu approach (mass matrix padding + dissipative torque) works
but has a fundamental problem: it introduces artificial dynamics at
exactly the operating point. This means:

- Convergence detection is noisy — the regularization adds fictitious
  oscillations/damping near equilibrium
- Two tunable parameters (eps, nu) with no physical basis
- The dissipative torque removes energy from the system
- Cannot cleanly distinguish "converged" from "regularization-damped"

## Stereographic projection

Project the pole direction from the south pole (theta=0, drogue up)
onto the plane tangent to the north pole (theta=pi, drogue down):

    u = 2 tan((pi - theta)/2) cos(phi)
    v = 2 tan((pi - theta)/2) sin(phi)

or equivalently, with delta = pi - theta (tilt from vertical):

    u = 2 tan(delta/2) cos(phi)
    v = 2 tan(delta/2) sin(phi)

The equilibrium (delta=0) maps to (u, v) = (0, 0). Small tilts give
u, v ~ delta * (cos(phi), sin(phi)) — the projection is approximately
identity near the operating point.

**Properties:**
- Singularity-free near theta=pi (the operating point)
- 2 DOF, same as (theta, phi) — no redundancy, no constraints
- Conformal map — angles preserved locally
- Singularity at theta=0 (drogue pointing up) — physically unreachable
- Equilibrium at the origin — clean for convergence detection

## Derivation strategy

**Substitute at the Lagrangian level, not at the EOM level.**

Two approaches were considered:

1. **Jacobian transform of M, F** (EOM level): Derive M, F in (theta,
   phi), then apply J^T M J. This **fails** because the Jacobian
   d(theta,phi)/d(u,v) contains `1/sin(theta)` terms from the phi
   mapping, reintroducing the `1/(u²+v²)` singularity.

2. **Substitute into T, V, Q** (Lagrangian level): Express the drogue
   position as smooth rational functions of (u, v), then compute T, V,
   and the generalized forces from scratch. Re-derive M, F via
   Euler-Lagrange in (u, v). This **works** because T and V never see
   the singular (theta, phi) coordinates — only the smooth rational
   position functions.

The working approach uses these rational identities:

    sin(theta) cos(phi) = 4u / (u² + v² + 4)
    sin(theta) sin(phi) = 4v / (u² + v² + 4)
    cos(theta)          = (u² + v² - 4) / (u² + v² + 4)

These are smooth everywhere, including at (u,v)=(0,0) where theta=pi.
The Lagrangian is expressed directly in terms of these functions, and
sympy derives the EOM in (u, v) coordinates from scratch.

**Implementation note:** The cos(theta) identity must have the correct
sign: `(s - 4)/(s + 4)`, not `-(s - 4)/(s + 4)`. At s=0 this gives
cos(theta)=-1, i.e. theta=pi (drogue down). A sign error here would
place the equilibrium at theta=0 (drogue up) — physically wrong but
hard to catch because the mass matrix is still well-conditioned.

For the manuscript, present the Lagrangian in spherical coordinates
(intuitive for physicists), then add a short section noting the
coordinate substitution for numerical implementation.

## Impact on D1 (sympy codegen)

D1 and D2 are tightly coupled. The generation script should:

1. Define T, V, Q directly in stereographic (u, v) using the rational
   identities (as `lagrange_model.py` now does)
2. Derive M, F via Euler-Lagrange in (u, v)
3. Apply CSE to the resulting expressions
4. Generate numpy code in (u, v) coordinates

The generated `_generated_eom.py` will have functions:
- `compute_M(u, v, ...)` — mass matrix in stereographic coords
- `compute_F(u, v, ud, vd, U_b, V_b, U_d, V_d, ...)` — force vector

No regularization parameters. No eps, no nu.

## State vector change

Current: Y = (x_b, y_b, theta, phi, xd_b, yd_b, thetad, phid)
New:     Y = (x_b, y_b, u, v, xd_b, yd_b, ud, vd)

The physical outputs (drogue drift velocity, effective depth z_eff)
need to be computed from (u, v) via the inverse transform:

    delta = 2 * atan(sqrt(u^2 + v^2) / 2)
    theta = pi - delta
    phi = atan2(v, u)

These are only needed for output/diagnostics, not during integration.

## Initialization

The default initial condition theta=pi, phi=0 becomes u=0, v=0 — the
origin. This is the cleanest possible initialization: the state vector
is all zeros for the angular part.

For warm-starting from a previous solution, the (u, v) values carry
over directly (no conversion needed between timesteps).

## Verification strategy

1. **Numerical equivalence**: For a grid of test currents, verify that
   the stereographic formulation produces the same steady-state drift
   velocity as the current (theta, phi) + regularization approach.
   Agreement should be to solver tolerance, not just O(eps^2).

2. **Convergence quality**: Compare the number of solve_ivp steps
   needed to reach steady state. The stereographic version should
   converge more cleanly (no fictitious oscillations from eps/nu).

3. **Symbolic verification**: In the generation script, verify that
   M_new is symmetric and positive definite at the origin (u=v=0).
   This confirms the singularity is truly eliminated.

4. **Edge cases**:
   - Zero current (u=v=0 should be a stable equilibrium)
   - Very strong shear (large tilt — u, v far from origin)
   - Rapidly changing currents (test that (u, v) track smoothly)

5. **Regression**: All existing tests must pass with the new
   coordinates. The public API (get_final_drift_batch) returns the
   same (xd, yd, theta, Y_final) — theta is computed from (u, v)
   via the inverse transform.

## Alternatives considered

- **Option A (Cartesian direction cosines)**: Singularity-free
  everywhere but adds a constraint (|n|=1), increasing the system
  size and requiring stabilization. Overkill.
- **Option B (Quaternions)**: Massive overkill for a 2-DOF system.
- **Option D (Rodrigues)**: Similar to stereographic but less standard.
- **Option E (Keep regularization)**: Works but introduces artificial
  dynamics at the equilibrium point. Rejected.

## Implementation order

D2 should be implemented together with D1, since both touch the
coordinate system and code generation:

1. Add the stereographic transform to the sympy derivation
2. Generate numpy code in (u, v) coordinates
3. Update `_rhs_batch` to use (u, v) state vector
4. Update `get_final_drift_batch` to convert (u, v) → theta for output
5. Remove eps/nu parameters and regularization code
6. Verify against current implementation
