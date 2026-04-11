# Architecture diagrams (post DW-A through DW-F, updated for D-I)

## Class diagram

```mermaid
classDiagram
    class DrifterPhysics {
        <<NamedTuple>>
        m_b : float
        m_d : float
        m_hat_d : float
        m_tilde_d : float
        m_tilde_b : float
        l : float
        g : float
        k_b : float
        k_d : float
    }

    class EOMState {
        <<NamedTuple>>
        u : float
        v : float
        xd : float
        yd : float
        ud : float
        vd : float
        U_b : float
        V_b : float
        U_d : float
        V_d : float
    }

    class DroguedDrifter {
        +physics : DrifterPhysics
        +get_uv : Callable
        +get_full_solution(t_span, ...) xr.Dataset
        +get_final_drift(t_span, ...) tuple[xd, yd, max_accel]
        +get_final_drift_batch(sample_uv, ...) tuple[xd, yd, Y, max_accel]
        -_rhs(t, y) np.array
        -_rhs_batch(Y, sample_uv) np.array
        -_z_eff(u, v) np.array
        -_solve(t_span, y0, ...) OdeResult
        -_default_uv(t, x, y, z) tuple[U, V]
    }

    DroguedDrifter --> DrifterPhysics : self.physics
    DroguedDrifter ..> EOMState : builds per timestep
    DroguedDrifter ..> lagrange_model : calls _qdd_func

    class lagrange_model {
        <<module>>
        +M_func(physics, state) np.array
        +F_func(physics, state) np.array
        -_qdd_func(physics, state) np.array
        -_get_eom_callables() tuple
        -_load_or_derive() tuple
        -_build_packer(raw_func) Callable
        -_derive_symbolic() tuple
        -_cache_key() str
    }

    lagrange_model --> DrifterPhysics : arg type
    lagrange_model --> EOMState : arg type

    class parcels_v4 {
        <<module>>
        +make_dd_kernel(dd) Callable
        +DDAdvectEE(particles, fieldset, dd) void
        +make_profile_sampler(depth_levels, U, V) Callable
    }

    parcels_v4 --> DroguedDrifter : calls get_final_drift_batch
```

## Data flow: parameter passing and argument packing

```mermaid
flowchart TD
    subgraph "Construction (once)"
        INIT["DroguedDrifter(**kwargs)"] --> PHYS["self.physics = DrifterPhysics(9 fields)"]
    end

    subgraph "Per timestep"
        UNPACK["Unpack state: u,v,xd,yd,ud,vd from y"] --> SAMPLE
        SAMPLE["Sample currents: U_b,V_b,U_d,V_d"] --> STATE
        STATE["state = EOMState(10 fields)"]
    end

    subgraph "EOM evaluation (lambdify)"
        STATE --> PACK["pack_eom_args(physics, state)"]
        PHYS --> PACK
        PACK --> |"19-element tuple"| RAW["qdd_raw(*args)"]
        RAW --> |"tuple of 4 values"| RESHAPE["np.array or np.column_stack"]
        RESHAPE --> QDD["qdd: (4,) or (N,4)"]
    end

    subgraph "Argument ordering (built once via inspection)"
        LAMBDIFY["sp.lambdify(cse=True)"] --> SIG["inspect.signature(qdd_raw)"]
        SIG --> PACKER["_build_packer maps param names to struct fields"]
        PACKER --> PACK
    end

    style PHYS fill:#e1f5fe
    style STATE fill:#fff3e0
    style QDD fill:#e8f5e9
```

## Coordinate boundary

```mermaid
flowchart LR
    subgraph "Public API (spherical)"
        IN_PUB["theta, phi, thetad, phid"]
        OUT_PUB["theta, phi, thetad, phid"]
    end

    subgraph "Internal (stereographic)"
        UV["u, v, ud, vd"]
        ZEFF["_z_eff: cos_theta = (s-4)/(s+4)"]
        QDD_INT["_qdd_func(physics, state)"]
        SOLVE["solve_ivp with _rhs / _rhs_batch"]
    end

    IN_PUB --> |"_spherical_to_uv"| UV
    UV --> ZEFF
    UV --> QDD_INT
    QDD_INT --> SOLVE
    SOLVE --> |"_uv_to_spherical"| OUT_PUB
```

## Sequence: get_final_drift_batch

```mermaid
sequenceDiagram
    participant User
    participant GFDB as get_final_drift_batch
    participant Conv as _spherical_to_uv
    participant IVP as solve_ivp
    participant RHS as _rhs_batch
    participant ZEFF as _z_eff
    participant SUV as sample_uv(z)
    participant QDD as _qdd_func
    participant Pack as pack_eom_args
    participant Raw as qdd_raw (lambdified)
    participant ConvOut as _uv_to_spherical

    User->>GFDB: sample_uv, t_span, y0 (spherical)
    GFDB->>Conv: y0[:, theta/phi] to u/v
    GFDB->>IVP: rhs_flat, t_span, y0_flat

    loop Each solver step
        IVP->>RHS: t, Y (N,8)
        RHS->>ZEFF: u, v
        ZEFF-->>RHS: z_eff (N,)
        RHS->>SUV: z=0 and z=z_eff
        SUV-->>RHS: U_b,V_b,U_d,V_d
        RHS->>QDD: physics, EOMState(...)
        QDD->>Pack: physics, state
        Pack-->>QDD: 19-tuple
        QDD->>Raw: *args
        Raw-->>QDD: (val0,val1,val2,val3)
        QDD-->>RHS: qdd (N,4)
        RHS-->>IVP: dY (N,8) raveled
    end

    IVP-->>GFDB: sol
    GFDB->>RHS: Y_final (max_accel diagnostic)
    RHS-->>GFDB: dY_final
    GFDB->>ConvOut: u,v,ud,vd final
    ConvOut-->>GFDB: theta,phi,thetad,phid
    GFDB-->>User: (xd, yd, Y_final, max_accel)
```

## Sequence: get_full_solution (scalar)

```mermaid
sequenceDiagram
    participant User
    participant GFS as get_full_solution
    participant Conv as _spherical_to_uv
    participant Solve as _solve
    participant IVP as solve_ivp
    participant RHS as _rhs
    participant ZEFF as _z_eff
    participant UV as self.get_uv(t,x,y,z)
    participant QDD as _qdd_func
    participant ConvOut as _uv_to_spherical
    participant XR as xarray.Dataset

    User->>GFS: t_span, theta, phi, ...
    GFS->>Conv: theta, phi to u0, v0
    GFS->>Solve: t_span, y0_internal
    Solve->>IVP: _rhs, t_span, y0

    loop Each solver step
        IVP->>RHS: t, y (8,)
        RHS->>ZEFF: u, v
        ZEFF-->>RHS: z_d (scalar)
        RHS->>UV: z=0 and z=z_d
        UV-->>RHS: U_b,V_b,U_d,V_d
        RHS->>QDD: physics, EOMState(...)
        QDD-->>RHS: qdd (4,)
        RHS-->>IVP: dy (8,)
    end

    IVP-->>Solve: sol
    Solve-->>GFS: sol
    GFS->>ConvOut: sol.y[IU], sol.y[IV], ...
    ConvOut-->>GFS: theta(t), phi(t), ...
    GFS->>XR: build Dataset
    XR-->>User: ds with x,y,theta,phi,xd,yd,...
```

## Caching and lambdification pipeline

```mermaid
flowchart TD
    subgraph "Symbolic derivation (135s, cached)"
        DERIVE["_derive_symbolic()"] --> |"M(4x4), F(4x1), args"| LUSOLVE
        LUSOLVE["M.LUsolve(F)"] --> |"qdd_exprs (4 scalar)"| PICKLE
    end

    subgraph "Disk cache (hash-keyed pickle)"
        PICKLE["eom_cache.pkl"] --> |"key = hash(source + sympy version)"| CHECK
        CHECK{key matches?}
        CHECK --> |yes, 70ms| LAMBDIFY
        CHECK --> |no| DERIVE
    end

    subgraph "Lambdification (once per process)"
        LAMBDIFY["sp.lambdify(args, exprs, cse=True)"] --> QDD_RAW["qdd_raw"]
        LAMBDIFY --> M_RAW["M_raw"]
        LAMBDIFY --> F_RAW["F_raw"]
        QDD_RAW --> INSPECT["inspect.signature"]
        INSPECT --> BUILD["_build_packer"]
        BUILD --> PACKER["pack_eom_args closure"]
    end

    subgraph "Runtime (per call)"
        PACKER --> |"physics + state to 19-tuple"| QDD_RAW
        QDD_RAW --> |"5 us scalar, 214 us batch"| RESULT["qdd (4,) or (N,4)"]
    end

    style DERIVE fill:#ffcdd2
    style PICKLE fill:#fff9c4
    style LAMBDIFY fill:#c8e6c9
    style RESULT fill:#e8f5e9
```
