# Dataset contract

One row represents one engine cycle. Required inputs are `EngineID`, `Cycle`, `Altitude`, `Mach`,
`Tamb`, `Pamb`, `RPM`, `FuelFlow`, and station measurements `P2`, `T2`, `P3`, `T3`, `P4`, `T4`.
Training additionally requires `CompressorHealth`, `CombustorHealth`, `TurbineHealth`,
`OverallHealth`, `Thrust`, and `TSFC`.

All quantities use SI: metres, kelvin, pascals, revolutions/minute, kg/s, newtons, and kg/(N s).
Health is dimensionless on `[0, 1]`. Splits are grouped by engine to prevent leakage.
