# TurboEngine-HAL: PDF-First Physics Cross-Check

## Scope and authority

This document preserves every problem listed in `turbojet_problems_analysis.md`, but solves or assesses only items that are directly about gas-turbine propulsion physics or engineering. Items about machine learning, estimation algorithms, sensor-data processing, uncertainty methods, application deployment, and project operations are retained for traceability and marked **Not a physics problem — not solved here**.

**Authoritative source:** *Elements of Gas Turbine Propulsion*, Jack D. Mattingly, uploaded as `ELEMENTS OF GAS TURBINE PROPULTION2.pdf` (called “the PDF” below). Where the source analysis conflicts with, extends beyond, or mixes assumptions from the PDF, the PDF treatment is used.

The PDF’s analysis is primarily a **steady, one-dimensional cycle analysis**. It distinguishes an **ideal turbojet** (isentropic components, constant-pressure combustion, nozzle expanded to ambient) from a **real turbojet** (component pressure ratios, polytropic efficiencies, burner efficiency, mechanical efficiency, fuel addition, and nozzle pressure mismatch). Do not combine equations from one model with assumptions from the other without stating the change in model.

## Notation and units

| Symbol | Meaning | Typical SI unit |
|---|---|---|
| \(F\) | uninstalled thrust | N |
| \(\dot m_0\) | inlet air mass flow | kg/s |
| \(f\) | fuel/air ratio, \(\dot m_f/\dot m_0\) | dimensionless |
| \(M_0, M_9\) | flight and nozzle-exit Mach number | dimensionless |
| \(T_t, P_t\) | total (stagnation) temperature and pressure | K, Pa |
| \(c_p\) | specific heat at constant pressure | kJ/(kg K) or J/(kg K) |
| \(h_{PR}\) | fuel heating value | kJ/kg |
| \(\pi\) | total-pressure ratio | dimensionless |
| \(\tau\) | total-temperature ratio | dimensionless |

## Part 1 — Critical problems

### Problem inventory

| Source problem | Physics disposition |
|---|---|
| 1.1 Inadequate combustor pressure-loss model | Assessed below |
| 1.2 Oversimplified thrust model | Assessed below |
| 1.3 Constant mass flow and no bleed air | Assessed below |
| 2.1 Synthetic data distribution mismatch | Not a physics problem — not solved here |
| 2.2 Hybrid physics+ML model degrades on small data | Not a physics problem — not solved here |
| 3.1 EKF assumes constant degradation rate | Not a physics problem — not solved here |
| 3.2 Observation model assumes identity Jacobian | Not a physics problem — not solved here |
| 4.1 No sensor fault detection | Not a physics problem — not solved here |
| 4.2 No handling of transient/startup conditions | Physics-engineering portion assessed below |
| 5.1 Conformal prediction overestimates coverage | Not a physics problem — not solved here |
| 5.2 Validation uses synthetic rather than prognostic metric | Not a physics problem — not solved here |
| 6.1 No graceful-degradation failure modes | Not a physics problem — not solved here |

### 1.1 Inadequate combustor pressure-loss model

**Verdict: Partially correct.** The criticism that a real-engine model should account for burner pressure loss is supported by the PDF. The proposed FAR-dependent numerical formula, its stated percentages, and its predicted thrust error are **not verifiable from the PDF** and must not be presented as PDF-derived results.

#### PDF treatment

For an **ideal turbojet**, the PDF assumes constant-pressure combustion; equivalently the burner pressure ratio is unity. In the ideal-cycle derivation the burner energy balance is

\[
\dot m_0 c_p T_{t3}+\dot m_f h_{PR}=\dot m_0 c_p T_{t4},
\]

which gives

\[
f=\frac{c_p T_0}{h_{PR}}\left(\tau_\lambda-\tau_r\tau_c\right).
\]

Here \(\tau_\lambda=T_{t4}/T_0\), \(\tau_r\) is the ram temperature ratio, and \(\tau_c\) is the compressor temperature ratio. This ideal result is appropriate only if the model explicitly adopts the PDF’s ideal assumptions.

For a **real turbojet**, the PDF includes a burner total-pressure ratio \(\pi_b\) as part of the total-pressure-ratio chain and uses burner efficiency \(\eta_b\) in the burner energy balance. Its real-cycle fuel/air-ratio expression is

\[
f=\frac{\tau_\lambda-\tau_r\tau_d\tau_c}
{\eta_b h_{PR}/(c_{pc}T_0)-\tau_\lambda},
\]

with the associated real component pressure ratios and efficiencies carried through the nozzle/thrust calculation. This is the PDF-consistent starting point for a loss-inclusive model.

#### Cross-check of the source analysis

The source model expression \(p_3=p_2[0.96-0.03(1-h_{comb})]\) does represent a burner pressure-ratio assumption. It is not inherently invalid as a **chosen calibration model**, but it is insufficiently identified: it silently selects a fixed relationship rather than exposing \(\pi_b\) as a stated model input or validated component characteristic.

The source’s proposed replacement,

\[
\pi_b=0.96-0.02f\quad\text{(plus a health correction)},
\]

is **not in the PDF**. The PDF discusses combustion behavior as dependent on pressure, temperature, vaporization, and mixing, and presents pressure loss as a real-engine component parameter; it does not prescribe this FAR-linear loss law. Likewise, the claimed 3–8% loss range, FAR variation, and 200 Pa / 2–4% thrust consequences are unsupported by the PDF.

#### PDF-first solution / modeling procedure

1. Choose one analysis level and state it: ideal cycle or real cycle.
2. For ideal-cycle charts, set \(\pi_b=1\) and \(\eta_b=1\), and do not add a separate pressure-loss penalty.
3. For real-cycle calculations, specify \(\pi_b\) and \(\eta_b\) as documented inputs, with \(0<\pi_b\le1\) and \(0<\eta_b\le1\).
4. Use the PDF’s burner energy balance to calculate \(f\), then use the turbine work balance and nozzle/thrust equations consistently.
5. Only introduce \(\pi_b=f(\mathrm{FAR},\ldots)\) after calibration to component data; label it as an external empirical submodel, not a PDF formula.

**PDF references:** Chapter 5, §5-7, Eqs. (5-24)–(5-26), pp. 258–259; Chapter 7, Eqs. (7-8)–(7-10), pp. 373–375; Chapter 10, §10-6, pp. 814–816.

### 1.2 Oversimplified thrust model

**Verdict: Correct criticism.** The source’s fitted equation is not the PDF’s turbojet thrust calculation. Replace it with the PDF’s momentum-plus-pressure thrust evaluation, using a consistent nozzle state and mass flow.

#### PDF treatment

For the ideal turbojet with ambient nozzle expansion and neglected fuel mass, the PDF gives specific thrust as

\[
\frac{F}{\dot m_0}=\frac{a_0}{g_c}\left(\frac{V_9}{a_0}-M_0\right),
\]

where \(V_0=a_0M_0\). The exit velocity follows the cycle state:

\[
\frac{V_9}{a_0}=
\left[\frac{2}{\gamma-1}\tau_\lambda\tau_t\left(1-\frac{1}{\tau_r\tau_c\tau_t}\right)\right]^{1/2}.
\]

The exact symbols should be taken from the selected PDF equation set; the essential point is that exit velocity comes from the thermodynamic pressure/temperature state, not a post-hoc shape calculation.

For the real turbojet, the PDF starts from the uninstalled-thrust equation containing both momentum and pressure terms:

\[
F=\frac{1}{g_c}\left(\dot m_9V_9-\dot m_0V_0\right)+A_9(P_9-P_0).
\]

It then uses \(\dot m_9/\dot m_0=1+f\). Thus fuel mass and a non-ambient exit pressure are retained when relevant. This is the correct PDF framework for a nozzle that is not perfectly expanded.

#### Step-by-step PDF-first calculation sequence

1. **Flight condition:** supply \(M_0\), \(T_0\) (K), and the gas-property assumptions. Compute \(a_0=\sqrt{\gamma Rg_cT_0}\) and \(V_0=a_0M_0\).
2. **Inlet/diffuser:** obtain ram quantities and apply the selected diffuser recovery / pressure ratio for the real cycle.
3. **Compressor:** apply compressor pressure ratio and the applicable ideal or real efficiency relation to obtain \(T_{t3}\) and \(P_{t3}\).
4. **Burner:** apply \(\pi_b\), \(\eta_b\), \(T_{t4}\), and the energy balance to obtain \(f\).
5. **Turbine:** apply the PDF turbine–compressor power balance to determine the turbine temperature ratio/state.
6. **Nozzle:** determine exit pressure/Mach/velocity from the nozzle pressure ratio and nozzle model. Include \(A_9(P_9-P_0)\) if the nozzle exit is not at ambient pressure.
7. **Thrust:** evaluate the real uninstalled-thrust equation above, or the ideal specific-thrust equation only when all ideal conditions apply.
8. **TSFC:** calculate \(S=f/(F/\dot m_0)\), with its unit stated, e.g. \(\mathrm{kg/(N\,s)}\).

#### Discrepancies in the source analysis

The empirical form

\[
F=k_1(\mathrm{RPM})\pi_{noz}+k_2\dot m_f-k_3V_0+c
\]

is **not a PDF propulsion formula**. It omits the explicit exit-velocity calculation, the air-plus-fuel mass ratio, and the pressure-thrust term. The source’s proposed “real nozzle” pseudocode is directionally closer, but its displayed return value \(\dot m_{exit}V_{exit}\times\text{recovery}\) still omits the PDF pressure-thrust term and inlet momentum term. It is therefore incomplete for general uninstalled thrust.

Claims such as a 30% unstart loss, a 12–18% overprediction at a named condition, and fixed high-Mach spillage losses are **not verifiable from this PDF**. The PDF does support the general need to model inlet recovery, nozzle operating condition, and the flight Mach number.

#### Chart / trend interpretation

The PDF’s ideal-turbojet plots (Figs. 5-8 and 5-9) show performance varying with compressor pressure ratio and flight Mach number; specific thrust has an optimum compressor pressure ratio for a given condition rather than being proportional to RPM alone. The real-turbojet discussion and Fig. 7-6 show that nozzle pressure mismatch affects TSFC; for small stated mismatch the change is gradual, but it is not generally zero. These figures support a state-based cycle calculation rather than one global set of fitted thrust coefficients.

**PDF references:** Chapter 5, §5-7, Eqs. (5-23)–(5-32), pp. 256–260; Chapter 7, Eqs. (7-1)–(7-5), pp. 372–374; Fig. 7-6 and discussion, pp. 384–385.

### 1.3 Hidden assumption: constant mass flow and no bleed air

**Verdict: Partially correct.** A full engine-performance calculation cannot treat mass flow as an unconstrained constant independent of flow area and operating state. However, the PDF does not substantiate the source’s specific bleed percentages, fouling rates, sensor-drift figures, or claimed health-estimation consequences.

#### PDF treatment

The PDF defines the mass-flow parameter (MFP) as a compressible-flow function of Mach number and uses it with total pressure, total temperature, and flow area to determine mass flow. In this framework, mass flow is constrained by the station state and effective area; it is not determined by RPM alone. The real-engine/off-design analysis also uses continuity and MFP to connect component operating states.

For an idealized cycle, \(\dot m_0\) can be used as the reference air mass flow and the exit flow is approximated by \(\dot m_9=\dot m_0\). For the real turbojet, the PDF explicitly uses

\[
\dot m_9=(1+f)\dot m_0.
\]

That distinction matters for thrust and energy balance even before any cooling/bleed streams are introduced.

#### PDF-first assessment

The source is right to flag an RPM–ambient-only mass-flow correlation as a simplification. A more physically traceable model should establish a reference station, its effective area, total conditions, and Mach/MFP relation, then satisfy continuity through the engine.

The source’s statements that a single-spool turbojet has 3–5% bleed, fouling reduces flow 2–4% per 100 hours, and IGV changes cause ±8% flow are **not verifiable from the PDF**. Nor does the PDF support the proposed inverse method as written; isentropic efficiency and measured \(P,T\) alone do not uniquely determine mass flow without geometry, a compressor characteristic/operating relation, and the remaining boundary conditions.

#### PDF-consistent procedure

1. Define the station at which \(\dot m\) is evaluated and state the effective flow area \(A\).
2. Determine the total state \((P_t,T_t)\) and station Mach number \(M\).
3. Use the PDF MFP relation to calculate \(\dot m\) from \(A\), \(P_t\), \(T_t\), and MFP.
4. Propagate mass conservation between stations; for the real turbojet include fuel through \(\dot m_9=(1+f)\dot m_0\).
5. If cooling, customer bleed, or secondary flows are added, define each extraction/injection station and mass fraction explicitly. These are extensions beyond the simple turbojet summary equations and require component data.

**PDF references:** Chapter 3, Mass Flow Parameter, pp. 123–126; Chapter 7, Eqs. (7-1)–(7-4), pp. 372–374; Chapter 8, off-design continuity/MFP treatment.

### 2.1 Synthetic data distribution mismatch

**Not a physics problem — not solved here.** Specific environmental, degradation, noise, fleet-study, and model-accuracy claims are not verifiable from the PDF.

### 2.2 Hybrid physics+ML model degrades on small data

**Not a physics problem — not solved here.** The PDF does not validate ML residual learning, training-set size, or reported metrics.

### 3.1 EKF assumes constant degradation rate

**Not a physics problem — not solved here.** The PDF is not a validation source for EKF state-transition assumptions or RUL distributions.

### 3.2 Observation model assumes identity Jacobian

**Not a physics problem — not solved here.** The PDF provides forward cycle relations, but it does not validate an EKF Jacobian implementation.

### 4.1 No sensor fault detection

**Not a physics problem — not solved here.** Sensor fault detection and the numerical claims require instrumentation specifications and data, not this propulsion text.

### 4.2 No handling of transient/startup conditions

**Verdict: Partially correct as an engineering concern.** The PDF’s parametric cycle equations are steady-state relations, so applying them directly during starts, rapid throttle movements, surge, or recovery events requires an additional transient model. The detailed durations, temperature overshoots, sensor lags, and numerical residual example in the source are **not verifiable from the PDF**.

The PDF does support the underlying operating-limit point: off-design operation depends on compressor/nozzle matching; an operating line nearer the stall/surge line is undesirable because transient operation may cause compressor stall or surge. It also notes that variable nozzle area changes backpressure and corrected compressor mass flow, and describes an advantage of increased nozzle throat area during starting: lower backpressure permits required turbine power at lower turbine inlet temperature and allows starting at lower engine speed.

**PDF-first treatment:** classify each input sample as steady operation only after its state can reasonably be represented by steady inlet, compressor, burner, turbine, and nozzle conditions. For a transient-capable engine model, add spool dynamics, component maps, volumes/thermal storage, fuel-control logic, and sensor dynamics; these are not supplied by the Chapter 5/7 cycle equations alone. Do not label a transient residual as a degradation residual until the transient model or a steady-state gate has been applied.

**PDF references:** Chapter 10, compressor/nozzle matching and transient stall/surge discussion, pp. 801–802; Chapter 8, off-design engine performance.

### 5.1 Conformal prediction overestimates coverage

**Not a physics problem — not solved here.**

### 5.2 Validation uses synthetic rather than prognostic metric

**Not a physics problem — not solved here.**

### 6.1 No graceful-degradation failure modes

**Not a physics problem — not solved here.**

## Part 2 — Secondary problems

### Secondary-problem inventory and assessment

| Source no. | Problem | Physics disposition |
|---|---|---|
| 2.1 | No inlet total-pressure recovery model | Assessed below |
| 2.2 | Constant specific heats | Assessed below |
| 2.3 | No compressor map | Assessed below |
| 2.4 | Combustor exit temperature clamped to 1900 K | Assessed below |
| 2.5 | Thrust assumes no inlet spillage at high Mach | Assessed below |
| 3.1 | Healthy-baseline feature engineering is slow | Not a physics problem — not solved here |
| 3.2 | No robust feature scaling | Not a physics problem — not solved here |
| 3.3 | Stacking model not used in practice | Not a physics problem — not solved here |
| 3.4 | No model retraining pipeline | Not a physics problem — not solved here |
| 3.5 | Hyperparameters tuned on full training set | Not a physics problem — not solved here |
| 4.1 | EKF fixed process noise | Not a physics problem — not solved here |
| 4.2 | UKF not compared with EKF | Not a physics problem — not solved here |
| 4.3 | No multi-hypothesis filtering | Not a physics problem — not solved here |
| 4.4 | Health states unbounded | Not a physics problem — not solved here |
| 5.1 | No sensor-consistency cross-checks | Not a physics problem — not solved here |
| 5.2 | No environmental contamination model | Not a physics problem — not solved here |
| 5.3 | Same sensor-error distribution for all engines | Not a physics problem — not solved here |
| 5.4 | No handling of missing cycles | Not a physics problem — not solved here |
| 6.1 | Quantile regression not compared with conformal | Not a physics problem — not solved here |
| 6.2 | No out-of-distribution detection | Not a physics problem — not solved here |
| 6.3 | Confidence intervals not validated on hold-out set | Not a physics problem — not solved here |
| 6.4 | Ad-hoc bootstrap uncertainty | Not a physics problem — not solved here |
| 7.1 | CAD rendering lacks uncertainty visualization | Not a physics problem — not solved here |
| 7.2 | Dashboard update frequency unknown | Not a physics problem — not solved here |
| 7.3 | No flight-data-system integration | Not a physics problem — not solved here |
| 7.4 | RUL omits maintenance history | Not a physics problem — not solved here |

### 2.1 No inlet total-pressure recovery model

**Verdict: Correct.** In the real turbojet, the PDF uses diffuser pressure recovery / diffuser total-pressure ratio in the cycle pressure-ratio chain. The ideal cycle’s isentropic inlet assumption is not an acceptable silent substitute for it in a real-cycle model.

For the real cycle, set a documented diffuser recovery \(\pi_d\) (the PDF also denotes its maximum/recovery form in the real turbojet equations), calculate the inlet total state, and carry it through compressor, burner, turbine, and nozzle. The source’s fixed 0.5–5% and ±1–3% numbers are **not verifiable from the PDF**. The PDF does include a Mach-dependent real-inlet recovery treatment, including a different expression above \(M_0=1\); use that PDF relation when its assumptions apply.

**PDF references:** Chapter 7, summary equations (7-20), pp. 375–376.

### 2.2 Constant specific heats

**Verdict: Partially correct.** The source correctly identifies an approximation, but its stated 20–50 K error is **not verifiable from the PDF**. The PDF’s real-turbojet analysis improves on a single \(c_p\) by assuming one constant \(c_{pc}\) upstream of the burner and a different constant \(c_{pt}\) downstream. It still calls this an approximation to variable specific heats.

**PDF-first solution:** for the Chapter 7 real-cycle equations, use \((\gamma_c,c_{pc})\) for stations 0–3 and \((\gamma_t,c_{pt})\) for stations 4–9, with units used consistently. Do not claim fully temperature-dependent properties unless an additional property model is provided and validated.

**PDF references:** Chapter 7 introduction, p. 372; Eqs. (7-1)–(7-20), pp. 372–376.

### 2.3 No compressor map

**Verdict: Correct as an off-design engineering issue.** The PDF treats off-design component and engine behavior using operating maps/operating lines, corrected speed, corrected mass flow, matching, and surge/stall margins. A hard-coded one-dimensional efficiency curve cannot represent all combinations of corrected speed and corrected flow.

**PDF-first solution:** use a compressor map or a clearly limited steady-state surrogate parameterized by the operating variables. Couple it to continuity/MFP and turbine/nozzle matching; reject or flag states beyond the represented map instead of extrapolating silently. The source’s ±2–5% error is **not verifiable from the PDF**.

**PDF references:** Chapter 8, off-design performance; Chapter 10, compressor map and operating-line discussion around Fig. 10-53, pp. 800–802.

### 2.4 Combustor exit temperature clamped to 1900 K

**Verdict: Not verifiable as a fault from the PDF alone.** A temperature limit may be a valid engine-design/control constraint, but the PDF does not establish whether 1900 K is the correct limit for this particular engine or code. In the PDF, \(T_{t4}\) is an explicit input/design limit for the cycle, not an arbitrary universal constant.

**PDF-first solution:** expose \(T_{t4}\) as a documented input/limit, specify the source for its value, and make clear whether an out-of-limit request is clipped, rejected, or handled by a control/transient model. The claim that the clamp necessarily hides high-energy transients and misjudges RUL is not verifiable from the PDF.

**PDF references:** Chapter 5, ideal turbojet inputs, pp. 259–260; Chapter 7, real turbojet inputs, pp. 375–376.

### 2.5 Thrust formula assumes no inlet spillage at high Mach

**Verdict: Partially correct.** The PDF’s real-cycle procedure includes inlet/diffuser pressure recovery and flight Mach number; these are required to capture inlet losses in the cycle model. However, the source’s 2–3% loss at Mach 2 and ±5% RUL conclusion are **not verifiable from the PDF**.

**PDF-first solution:** use the appropriate PDF diffuser recovery relation and calculate thrust from the resulting engine state. Do not apply a separate fixed “spillage-loss percentage” on top of a diffuser model unless external inlet data define that additional installation loss and double-counting has been ruled out.

**PDF references:** Chapter 7, real turbojet diffuser-recovery equations, pp. 375–376; Chapter 1, distinction between uninstalled thrust and installation losses.

## Part 3 — Timeline items

No additional, distinct physics problem is added from the source timeline. Its physics-relevant entries—out-of-envelope Mach, thermal/startup transients, and multi-fault interaction—restate Problems 1.2, 4.2, and non-physics prognostics concerns respectively. Their stated timing and numerical outcomes are not verifiable from the PDF.

## Part 4 — Physics-relevant fix items

| Source fix | PDF-first disposition |
|---|---|
| Add FAR-dependent combustor pressure loss | Replace with an explicit real-cycle \(\pi_b\) and \(\eta_b\) treatment first. Add FAR dependence only with external component data. |
| Implement nozzle mass-flow conservation | Required, but calculate full uninstalled thrust: inlet momentum, exit momentum, and pressure thrust where applicable. |
| Multi-mode degradation curves | Not a physics-cycle fix in the PDF; not solved here. |
| “Real compass nozzle” choking logic | Use the PDF’s compressible-flow/MFP and nozzle-state analysis; spelling preserved from source but the intended item is evidently a real nozzle model. |

## Final conclusions

1. **Use the real turbojet equations when representing a real engine.** They require explicit diffuser/burner/nozzle pressure ratios, component efficiencies, turbine–compressor power balance, fuel addition, and the pressure term in thrust.
2. **Do not claim the PDF supports the source’s invented numerical ranges or empirical correction laws.** The PDF supports the physical categories, not those particular coefficients, percentages, or predicted errors.
3. **Mass flow, thrust, and component states must be coupled.** A fitted thrust equation or RPM-only mass-flow correlation may be a calibrated surrogate, but it is not the PDF’s cycle model.
4. **Keep the model regime explicit.** The Chapter 5/7 equations are steady-state cycle analyses; starts, throttle transients, and surge require separate dynamic/component-map treatment.

## Source-reference index

- Mattingly, *Elements of Gas Turbine Propulsion*, Chapter 3: compressible flow and mass-flow parameter.
- Chapter 5, §5-7: ideal turbojet cycle analysis; Eqs. (5-23)–(5-32); Figs. 5-8 and 5-9.
- Chapter 7: real turbojet cycle analysis; Eqs. (7-1)–(7-20); Figs. 7-5 through 7-7.
- Chapter 8: off-design engine-performance analysis.
- Chapter 10, §10-6: combustion systems; compressor/nozzle matching and transient stall/surge discussion.
