////////////////////////////////////////////////////////////////////////////////////////////////
// Gravothermal evolution SIDM (dark matter only, no baryons)                                 //
// Crank-Nicolson conduction + hydrostatic relaxation (Newton), same physics as original      //
// code, except:                                                                              //
//  - no Eigen: all linear systems are tridiagonal -> O(N) instead of O(N^3)                  //
//    * conduction: Thomas algorithm (diagonally dominant matrix)                             //
//    * relaxation: tridiagonal LU with partial pivoting on the EXACT system (the hydrostatic //
//      matrix is NOT symmetric, so ldlt was incorrect)                                       //
//  - rho and p updated exactly from R (not linearized)                                       //
//  - Newton stops as soon as it converges (max|dR/R| < NEWTON_TOL) instead of always 10 iter.//
//  - step rejected and retried with halved dt (rollback) if relaxation fails to converge     //
//    or if conduction creates a NEW entropy inversion                                        //
//  - SPEED: pow() replaced by cbrt() (rho^(2/3), rho^(5/3), a); entropy reused from s.a      //
//    (pow in hasInversion only if needed); Newton stops after correction < 1e-10 (quadratic  //
//    convergence: no verification-only iteration needed); zero allocations in relaxation;    //
//    kinv computed once per setup; output via std::to_chars (if available).                  //
//  - ADAPTIVE MESHING: every ADAPT_EVERY steps ALL shells are scanned; any shell with        //
//    ln(R_i/R_{i-1}) > ADAPT_LOG_GAP is split into two shells of mass m_i/2 each.            //
//    The scan is REPEATED (up to ADAPT_MAX_PASS passes) until no shell violates the          //
//    criterion; then quantities are recomputed and ONE global hydrostatic relaxation is      //
//    performed. The sole splitting criterion is based on radii.                              //
//    (splitting criterion and method UNCHANGED compared to evolutionCN-nobar8.cpp)           //
//  - COARSENING (new): splitting alone can never remove layers; those created while a shell  //
//    was "wide" remain even when, after recompression, the region becomes very dense (in the //
//    N vs r plot: the peak at r/rs ~ 1e-2). Before splitting, every ADAPT_EVERY steps,       //
//    adjacent shell pairs are merged (in repeated passes) whose merged shell has             //
//    ln(R_out/R_in) <= ADAPT_MERGE_FRAC * split threshold and is no coarser in mass than the //
//    coarsest shell of the INITIAL grid (protects the core). Mass, volume, and internal      //
//    energy are conserved exactly; merging cannot worsen entropy.                            //
//  - SPEED/SIZE (nobar8-2): fine epsilon raised to 0.01 (ONLY after Rho(0)>=RHO_FINE;        //
//    EPS_NORMAL unchanged: raising it shifts t_c noticeably); sparse saves before the        //
//    threshold (every 500 steps) and, afterward, driven by central density (one snapshot     //
//    every SAVE_DLOG10_RHO decades of Rho(0)), 10 digits instead of 17, final snapshot       //
//    guaranteed, aggregated adaptive meshing log. Physics is UNCHANGED.                      //
// Compilation: g++ -O3 -std=c++17 -march=native -fno-math-errno -DNDEBUG evolutionCN-nobar8-2.cpp -o evolve //
// Usage: evolve [tag] [inputDir] [outputFile] [totalStep] [adaptGap] [epsFine] [saveDlog10]  (no args: default)//
////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::vector<double>;

// fractional powers via cbrt (much faster than std::pow)
static inline double pow23(double x) { double c = std::cbrt(x); return c * c; }       // x^(2/3)
static inline double pow53(double x) { double c = std::cbrt(x); return x * c * c; }   // x^(5/3)

// ---------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------
constexpr int    DEFAULT_TOTAL_STEPS      = 1000000000;
constexpr int    DEFAULT_SAVE_STEPS       = 500;     // save every N steps BEFORE threshold RHO_SAVE_DENSE
constexpr double EPS_NORMAL               = 0.003;   // max |du/u| per step, normal regime
constexpr double EPS_FINE                 = 0.01;    // same when Rho(0) >= RHO_FINE (was 0.001; test: 0.01 shifts t(Rho0=1e8) by +0.017 out of ~423, with ~8x fewer steps)
constexpr double RHO_FINE                 = 1e5;
constexpr double RHO_SAVE_DENSE           = 1e6;     // from here on, saves track central density
constexpr double SAVE_DLOG10_RHO          = 0.05;    // after threshold: new snapshot when log10(Rho(0)) has grown by this amount since the last one
constexpr int    SAVE_MIN_GAP_AFTER       = 100;     // ... but never more often than once every 100 steps
constexpr int    SAVE_MAX_GAP_AFTER       = 20000;   // ... and at least every 20000 steps anyway (if Rho(0) stagnates)
constexpr int    SAVE_PRECISION           = 10;      // decimal places in scientific notation (was 17)
constexpr int    ADAPT_LOG_EVERY          = 20;      // adaptive meshing log line every N calls (aggregated)
constexpr double DEFAULT_DENSITY_THRESHOLD = 1e30;   // stop if Rho(0) exceeds this threshold
constexpr int    MAX_NEWTON_ITER          = 30;      // maximum relaxation iterations
constexpr double NEWTON_TOL               = 1e-10;   // convergence: correction max|dR/R| < tol (after applying it, error is ~tol^2)
constexpr int    MAX_HALVINGS_NEWTON      = 40;      // backtracking inside Newton
constexpr int    MAX_STEP_RETRIES         = 60;      // dt halvings for a rejected step
constexpr double ENTROPY_TOL              = 1e-6;    // relative entropy inversion tolerance
constexpr bool   STOP_ON_NEW_INVERSION    = false;   // true: stop instead of reducing dt (like the old check)

// --- adaptive meshing ---
constexpr bool   ADAPTIVE_MESHING         = true;
constexpr int    ADAPT_EVERY              = 50;      // how often (in steps) the grid is checked
constexpr double ADAPT_LOG_GAP            = 0.15;    // threshold on gap g_i = log(R_i/R_{i-1}): SOLE criterion
constexpr bool   ADAPT_USE_LN             = true;    // true: NATURAL log (0.15 = 0.065 dex); false: log10
                                                     // NB: the initial grid has ln(R_i/R_{i-1}) = 0.061 (0.026 dex),
                                                     // so with log10 the 0.15 threshold is 5.7x the initial spacing
constexpr int    ADAPT_MAX_PASS           = 60;      // maximum splitting passes per call
                                                     // (each pass reduces the gap by ~0.1 dex at most)
constexpr int    ADAPT_MAX_LAYERS         = 20000;   // safety cap on the number of shells
constexpr double ADAPT_MERGE_FRAC         = 0.65;    // merge if merged shell has gap <= FRAC * split threshold (FRAC<1: hysteresis)
constexpr bool   ADAPT_COARSEN            = true;    // merge oversampled shells (false = nobar8 behavior)

// ---------------------------------------------------------------------------------------------
class Logger {
    bool verbose;
public:
    explicit Logger(bool v = false) : verbose(v) {}
    void info(const std::string& m) const { std::cout << m << std::endl; }
    void debug(const std::string& m) const { if (verbose) std::cout << "[DEBUG] " << m << '\n'; }
    void error(const std::string& m) const { std::cerr << "[ERROR] " << m << std::endl; }
};

// ---------------------------------------------------------------------------------------------
struct Params {
    int    totalStep = DEFAULT_TOTAL_STEPS;
    int    saveStep  = DEFAULT_SAVE_STEPS;
    double epsilon   = EPS_NORMAL;
    double totalTime = 0.0;
    double a = 2.257, b = 1.385, c = 0.753;   // SIDM conductivity parameters
    double sigma = 0.001;                       // normalized cross section
    double adaptGap = ADAPT_LOG_GAP;          // threshold on gap log(R_i/R_{i-1}) (argv[5])
    double epsFine  = EPS_FINE;               // epsilon after Rho(0)>=RHO_FINE (argv[6])
    double saveDlog = SAVE_DLOG10_RHO;        // step in log10(Rho(0)) between snapshots after threshold (argv[7]); 0 = every SAVE_MIN_GAP_AFTER steps
    std::string tag       = "20261006_180Layers_nobar";
    std::string inputDir  = "";
    std::string outputFile = "";

    void setDefaultPaths() {
        inputDir   = "./initial/" + tag + "/";
        outputFile = "./results8/result_" + tag + ".txt";
    }
    void display(const Logger& lg) const {
        lg.info("Initial profile: " + tag);
        lg.info("Initial time: " + std::to_string(totalTime));
        lg.info("Total step: " + std::to_string(totalStep));
        lg.info("Save step (before density threshold): " + std::to_string(saveStep));
        lg.info("Save dlog10(Rho0) (after threshold): " + std::to_string(saveDlog));
        lg.info("Epsilon normal / fine: " + std::to_string(epsilon) + " / " + std::to_string(epsFine));
        lg.info(std::string("Adaptive gap threshold (") + (ADAPT_USE_LN ? "ln" : "log10") + "): " + std::to_string(adaptGap));
        lg.info("Cross section (sigma): " + std::to_string(sigma));
        lg.info("Conduction parameter (a, b, c): " + std::to_string(a) + ", " + std::to_string(b) + ", " + std::to_string(c));
    }
};

// ---------------------------------------------------------------------------------------------
static Vec readColumn(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    Vec v; std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line); double x;
        if (s >> x) v.push_back(x);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// Tridiagonal solvers
// ---------------------------------------------------------------------------------------------
// Thomas (without pivoting): safe only for diagonally dominant matrices (here: CN matrix).
static void solveThomas(const Vec& lo, const Vec& di, const Vec& up, Vec rhs, Vec& x, Vec& cp) {
    const int n = (int)di.size();
    cp.assign(n, 0.0);
    x.assign(n, 0.0);
    double den = di[0];
    cp[0] = up[0] / den;
    rhs[0] /= den;
    for (int i = 1; i < n; ++i) {
        den = di[i] - lo[i] * cp[i-1];
        if (i < n-1) cp[i] = up[i] / den;
        rhs[i] = (rhs[i] - lo[i] * rhs[i-1]) / den;
    }
    x[n-1] = rhs[n-1];
    for (int i = n-2; i >= 0; --i) x[i] = rhs[i] - cp[i] * x[i+1];
}

// Tridiagonal LU with partial pivoting (LAPACK dgtsv style), IN PLACE: dl, d, du, b are
// overwritten; du2 (size max(n-2,0)) and x are work buffers. Zero allocations.
// dl[i] = T(i+1,i), d[i] = T(i,i), du[i] = T(i,i+1).
static bool solveTridiagPivotInPlace(Vec& dl, Vec& d, Vec& du, Vec& b, Vec& x, Vec& du2) {
    const int n = (int)d.size();
    std::fill(du2.begin(), du2.end(), 0.0);
    for (int i = 0; i < n-1; ++i) {
        if (std::fabs(d[i]) >= std::fabs(dl[i])) {
            if (d[i] == 0.0) return false;
            double f = dl[i] / d[i];
            d[i+1] -= f * du[i];
            b[i+1] -= f * b[i];
        } else {
            double f = d[i] / dl[i];
            d[i] = dl[i];
            double t = d[i+1];
            d[i+1] = du[i] - f * t;
            if (i < n-2) { du2[i] = du[i+1]; du[i+1] = -f * du2[i]; }
            du[i] = t;
            std::swap(b[i], b[i+1]);
            b[i+1] -= f * b[i];
        }
    }
    if (d[n-1] == 0.0) return false;
    x[n-1] = b[n-1] / d[n-1];
    if (n > 1) x[n-2] = (b[n-2] - du[n-2] * x[n-1]) / d[n-2];
    for (int i = n-3; i >= 0; --i) x[i] = (b[i] - du[i] * x[i+1] - du2[i] * x[i+2]) / d[i];
    return true;
}

// ---------------------------------------------------------------------------------------------
struct State {
    Vec R, Rho, M, u, L, v, p, a;   // a = adiabatic variable (entropy)
    int n = 0;                      // number of shells (the last one, n-1, is fixed)
};

class Simulator {
    Params params;
    State  s;
    Logger logger;
    int    N = 0;                   // movable shells = n-1

    // reused buffers (avoid allocations at every step)
    Vec dUc, C, m, lo, di, up, rhs, xs, cp;
    Vec hl, hd, hu, hb, dR;          // tridiagonal hydrostatic matrix and right-hand side
    Vec Rtry, Rhotry, ptry;
    Vec kv;                                      // 1/(R_i^3 - R_{i-1}^3)
    Vec rdsc, rdl, rdd, rdu, rbs, rxs, rdu2;     // relaxation buffers (zero allocations per iteration)
    Vec bu, bp, ba, bR, bRho;                    // backup for step rollback
    std::string saveBuf;                         // output buffer
    long adaptCalls = 0, adaptSplitsAcc = 0, adaptMergesAcc = 0; int adaptLayersFrom = 0, lastAdaptStep = 0;   // aggregated log
    double mergeGapR = 0.0, mergeGapM = 0.0;     // merge thresholds (ln R, ln M), set from initial grid

public:
    Simulator(const Params& p, const Logger& lg) : params(p), logger(lg) {}

    // -----------------------------------------------------------------------------------------
    void initialize() {
        const std::string base = params.inputDir;
        s.R   = readColumn(base + "RList-"   + params.tag + ".txt");
        s.Rho = readColumn(base + "RhoList-" + params.tag + ".txt");
        s.M   = readColumn(base + "MList-"   + params.tag + ".txt");
        s.u   = readColumn(base + "uList-"   + params.tag + ".txt");
        s.L   = readColumn(base + "LList-"   + params.tag + ".txt");
        s.n = (int)s.R.size();
        if (s.n < 4 || (int)s.Rho.size() != s.n || (int)s.M.size() != s.n ||
            (int)s.u.size() != s.n || (int)s.L.size() != s.n)
            throw std::runtime_error("Input files with inconsistent sizes");
        N = s.n - 1;

        s.v.resize(s.n); s.p.resize(s.n); s.a.resize(s.n);
        for (int i = 0; i < s.n; ++i) {
            s.v[i] = std::sqrt(2.0/3.0) * std::sqrt(s.u[i]);
            s.p[i] = (2.0/3.0) * s.Rho[i] * s.u[i];
            s.a[i] = (2.0/3.0) * s.u[i] / pow23(s.Rho[i]);
        }
        resizeBuffers();

        // Coarsening thresholds: in ln R the merged shell does not exceed ADAPT_MERGE_FRAC * split threshold
        // (hysteresis: a freshly merged shell is not immediately re-split, and the two children of a split
        // merged back together would give ~ the full threshold, so they are not re-merged); in ln M it does
        // not exceed the coarsest shell of the initial grid (the core is never coarsened).
        {
            double gR0 = 0.0, gM0 = 0.0;
            for (int i = 1; i < s.n; ++i) {
                gR0 = std::max(gR0, std::log(s.R[i] / s.R[i-1]));
                gM0 = std::max(gM0, std::log(s.M[i] / s.M[i-1]));
            }
            const double splitLn = ADAPT_USE_LN ? params.adaptGap : params.adaptGap * std::log(10.0);
            (void)gR0;
            mergeGapR = ADAPT_MERGE_FRAC * splitLn;
            mergeGapM = gM0;
            logger.info("Coarsening thresholds: ln R gap <= " + std::to_string(mergeGapR) +
                        ", ln M gap <= " + std::to_string(mergeGapM) + (ADAPT_COARSEN ? "" : "  (coarsening OFF)"));
        }
        // consistency check for shell 0 (solid sphere): mass = Rho R^3 / 3
        {
            const double r0 = s.Rho[0] * s.R[0] * s.R[0] * s.R[0] / (3.0 * s.M[0]);
            if (std::fabs(r0 - 1.0) > 1e-2)
                logger.info("WARNING: in initial file Rho[0]*R[0]^3/(3*M[0]) = " + std::to_string(r0) +
                            " (expected 1): Rho[0] is inconsistent with M[0]; left unchanged.");
        }

        logger.info("Number of layers: " + std::to_string(s.n));
        logger.info("Initial r inner most: " + std::to_string(s.R[0]));
        logger.info("Initial r next to outer most: " + std::to_string(s.R[s.n-1]));
        logger.info("Initial rho inner most: " + std::to_string(s.Rho[0]));
        logger.info("Initial rho next to outer most: " + std::to_string(s.Rho[s.n-1]));

        std::ofstream f(params.outputFile, std::ofstream::out | std::ofstream::trunc);   // trunc: no more appended runs
        if (!f.is_open()) throw std::runtime_error("Failed to open output file: " + params.outputFile);
        f << "Input profile name: " << params.tag << '\n'
          << "Initial time: " << params.totalTime << '\n'
          << "Number of layers: " << s.n << '\n'
          << "Initial r inner most: " << s.R[0] << '\n'
          << "Initial r outer most: " << s.R[s.n-1] << '\n'
          << "Initial rho inner most: " << s.Rho[0] << '\n'
          << "Initial rho outer most: " << s.Rho[s.n-1] << '\n'
          << "Total step: " << params.totalStep << '\n'
          << "Save step: " << params.saveStep << '\n'
          << "Abs(delta u/u): " << params.epsilon << '\n'
          << "Cross section (sigma): " << params.sigma << '\n'
          << "Conduction parameter (a, b, c): " << params.a << ", " << params.b << ", " << params.c << '\n'
          << "time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity" << '\n'
          << std::scientific << std::setprecision(SAVE_PRECISION) << params.totalTime << " " << 0 << '\n';
        writeVec(f, s.R); writeVec(f, s.Rho); writeVec(f, s.M); writeVec(f, s.u); writeVec(f, s.L);
        logger.info("Evolution without baryon starts!");
    }

    // -----------------------------------------------------------------------------------------
    void run() {
        std::ofstream out(params.outputFile, std::ofstream::out | std::ofstream::app);
        if (!out.is_open()) throw std::runtime_error("Failed to open output file: " + params.outputFile);

        bool   dense = false;                 // true after Rho(0) >= RHO_SAVE_DENSE
        int    lastSaveStep = 0, lastDone = 0;
        double lastSaveLogRho = std::log10(s.Rho[0]);
        double timeComp = 0.0;
        double dtScale  = 1.0;       // memory of the dt reduction factor
        long   rejected = 0;

        for (int tstep = 1; tstep < params.totalStep + 1; ++tstep) {
            try {
                params.epsilon = (s.Rho[0] >= RHO_FINE) ? params.epsFine : EPS_NORMAL;
                const bool inverted0 = hasInversion();   // already inverted before the step?

                // --- step with rollback: conduction + relaxation, dt halved if rejected ---
                bu.assign(s.u.begin(), s.u.end()); bp.assign(s.p.begin(), s.p.end()); ba.assign(s.a.begin(), s.a.end());
                bR.assign(s.R.begin(), s.R.end()); bRho.assign(s.Rho.begin(), s.Rho.end());
                double scale = std::min(1.0, 2.0 * dtScale);
                double deltat = 0.0;
                bool ok = false;
                std::string why;
                for (int attempt = 0; attempt < MAX_STEP_RETRIES; ++attempt) {
                    deltat = conductionStep(scale);
                    if (!inverted0 && hasInversion()) {
                        if (STOP_ON_NEW_INVERSION) throw std::runtime_error("entropy inversion");
                        why = "entropy inversion";
                    } else {
                        try { relaxationStep(); ok = true; }
                        catch (const std::exception& e) { why = e.what(); }
                    }
                    if (ok) break;
                    // rollback
                    s.u = bu; s.p = bp; s.a = ba; s.R = bR; s.Rho = bRho;
                    scale *= 0.5; ++rejected;
                }
                if (!ok) throw std::runtime_error("step rejected " + std::to_string(MAX_STEP_RETRIES) +
                                                  " times, last reason: " + why);
                dtScale = scale;

                // compensated (Kahan) time summation
                double y = deltat - timeComp;
                double tmp = params.totalTime + y;
                timeComp = (tmp - params.totalTime) - y;
                params.totalTime = tmp;

                updateProfiles();

                lastDone = tstep;
                if (!dense && s.Rho[0] >= RHO_SAVE_DENSE) {
                    dense = true;
                    logger.info("Density threshold " + std::to_string(RHO_SAVE_DENSE) + " reached! Snapshots now every " +
                                std::to_string(params.saveDlog) + " dex of Rho(0).");
                }
                if (abnormal()) break;

                bool doSave;
                if (!dense) doSave = (tstep % params.saveStep == 0);
                else {
                    const int    gap = tstep - lastSaveStep;
                    const double dl  = std::log10(s.Rho[0]) - lastSaveLogRho;
                    doSave = (gap >= SAVE_MIN_GAP_AFTER && dl >= params.saveDlog) || gap >= SAVE_MAX_GAP_AFTER;
                }
                if (doSave) {
                    saveResults(out, tstep);
                    lastSaveStep = tstep; lastSaveLogRho = std::log10(s.Rho[0]);
                    logger.debug("Saved at time = " + std::to_string(params.totalTime) + ", step = " +
                                 std::to_string(tstep) + ", Rho(0) = " + std::to_string(s.Rho[0]) +
                                 ", layers = " + std::to_string(s.n) + ", dt scale = " + std::to_string(dtScale));
                }

                // adaptive meshing (after saving: the step snapshot remains on the old grid)
                if (ADAPTIVE_MESHING && tstep % ADAPT_EVERY == 0) adaptMesh(tstep);
            } catch (const std::exception& e) {
                logger.error("Error during simulation step " + std::to_string(tstep) + ": " + e.what());
                break;
            }
        }
        // final snapshot (consistent state: last completed step; not if R is nan/negative)
        if (lastDone > lastSaveStep && std::isfinite(s.R[0]) && s.R[0] > 0.0) {
            saveResults(out, lastDone);
            logger.info("Final snapshot saved at step " + std::to_string(lastDone));
        }
        flushAdaptLog(true);
        out.close();
        logger.info("Rejected steps (dt halved): " + std::to_string(rejected));
        logger.info("Evolution ends");
    }

private:
    static void writeVec(std::ofstream& f, const Vec& v) {
        for (size_t i = 0; i < v.size(); ++i) f << (i ? " " : "") << v[i];
        f << '\n';
    }

    bool abnormal() const {
        if (s.Rho[0] > DEFAULT_DENSITY_THRESHOLD) { logger.info("Rho reaches threshold!"); return true; }
        if (s.R[0] < 0)                           { logger.info("R(0) is negative!");      return true; }
        if (std::isnan(s.R[0]))                   { logger.info("R is nan!");              return true; }
        return false;
    }

    // entropy a = (2/3) u rho^(-2/3) must be non-decreasing outward
    bool hasInversion() const {
        // Fast path: if EVERY interface has a_{i+1} >= a_i (1 + 1e-9) there is no inversion
        // (margin >> rounding error, so the result matches the exact formula). s.a is
        // consistent with (u, rho). No pow.
        bool clear = true;
        for (int i = 0; i < N; ++i)
            if (!(s.a[i+1] >= s.a[i] * (1.0 + 1e-9))) { clear = false; break; }   // true even with NaN
        if (clear) return false;

        // Exact path (identical to the original code). Needed when two shells have nearly equal
        // entropy (e.g. children of a split): here the outcome depends on rounding, so the
        // exact same formula as before is kept to avoid changing behavior.
        double a0 = (2.0/3.0) * s.u[0] * std::pow(s.Rho[0], -2.0/3.0);
        for (int i = 0; i < N; ++i) {
            double a1 = (2.0/3.0) * s.u[i+1] * std::pow(s.Rho[i+1], -2.0/3.0);
            if (!(a1 >= a0 * (1.0 - ENTROPY_TOL))) return true;   // true even with NaN
            a0 = a1;
        }
        return false;
    }

    // -----------------------------------------------------------------------------------------
    // Crank-Nicolson conduction (at fixed rho, R). dt = scale * epsilon / max|(dL/dm)/u|
    // -----------------------------------------------------------------------------------------
    double conductionStep(double scale) {
        double maxc = 0.0;
        dUc[0] = -(s.L[0] / s.M[0]) / s.u[0];
        maxc = std::fabs(dUc[0]);
        for (int i = 1; i < N; ++i) {
            dUc[i] = -((s.L[i] - s.L[i-1]) / (s.M[i] - s.M[i-1])) / s.u[i];
            maxc = std::max(maxc, std::fabs(dUc[i]));
        }
        const double deltat = scale * params.epsilon / maxc;

        const double sigma2 = params.sigma * params.sigma;
        const double abc_s  = params.a * params.b * params.c * params.sigma;
        auto term = [&](int i) {
            double vi = s.v[i], vi2 = vi * vi, vi3 = vi2 * vi;
            return (s.Rho[i] * vi3) / (params.a * params.c * sigma2 * s.Rho[i] * vi2 + params.b);
        };

        // coefficients C_i such that L_i = -C_i (u_{i+1} - u_i)
        C[0] = (s.R[0] * s.R[0] / s.R[1]) * abc_s * (term(0) + term(1));
        for (int i = 1; i < N; ++i)
            C[i] = (s.R[i] * s.R[i] / (s.R[i+1] - s.R[i-1])) * abc_s * (term(i) + term(i+1));

        m[0] = s.M[0];
        for (int i = 1; i < N; ++i) m[i] = s.M[i] - s.M[i-1];

        // A u^{n+1} = B u^n + boundary; A tridiagonal diagonally dominant -> Thomas
        for (int i = 0; i < N; ++i) {
            double al = (i > 0) ? 0.5 * (deltat / m[i]) * C[i-1] : 0.0;
            double ar = 0.5 * (deltat / m[i]) * C[i];
            lo[i] = -al;
            di[i] = 1.0 + al + ar;
            up[i] = -ar;
            double r = (1.0 - al - ar) * s.u[i];
            if (i > 0) r += al * s.u[i-1];
            if (i < N-1) r += ar * s.u[i+1];
            else         r += 2.0 * ar * s.u[N];          // fixed outer boundary
            rhs[i] = r;
        }
        solveThomas(lo, di, up, rhs, xs, cp);
        for (int i = 0; i < N; ++i) s.u[i] = xs[i];

        for (int i = 0; i < s.n; ++i) {
            s.p[i] = (2.0/3.0) * s.Rho[i] * s.u[i];
            s.a[i] = (2.0/3.0) * s.u[i] / pow23(s.Rho[i]);
        }
        return deltat;
    }

    // -----------------------------------------------------------------------------------------
    // Tridiagonal hydrostatic matrix: H(i,i-1)=hl[i], H(i,i)=hd[i], H(i,i+1)=hu[i]; RHS hb
    // (same formulas as the original code)
    // -----------------------------------------------------------------------------------------
    void setupHydrostatic() {
        const Vec& R = s.R; const Vec& rho = s.Rho; const Vec& p = s.p; const Vec& M = s.M;
        for (int i = 1; i < N; ++i) {              // 1/(R_i^3 - R_{i-1}^3), once per setup
            double d = R[i] - R[i-1];
            kv[i] = 1.0 / (d * (R[i]*R[i] + R[i]*R[i-1] + R[i-1]*R[i-1]));
        }
        // row 0
        {
            double R0 = R[0], R02 = R0*R0, R03 = R02*R0, R04 = R02*R02, R1 = R[1];
            double k1 = kv[1];
            hd[0] = 8.0*R0*(p[1]-p[0]) + 20.0*R04*(p[0]/R03 + p[1]*k1)
                  + 3.0*M[0]*R1*R02*(-rho[0]/R03 + rho[1]*k1);
            hu[0] = M[0]*(rho[0]+rho[1]) - 20.0*R02*p[1]*R1*R1*k1 - 3.0*M[0]*rho[1]*R1*R1*R1*k1;
            hb[0] = -4.0*R02*(p[1]-p[0]) - M[0]*(rho[0]+rho[1])*R1;
        }
        // intermediate rows 1 .. N-2
        for (int i = 1; i < N-1; ++i) {
            double Ri = R[i], Ri2 = Ri*Ri, Ri4 = Ri2*Ri2;
            double Rm = R[i-1], Rm2 = Rm*Rm, Rp = R[i+1], Rp2 = Rp*Rp;
            double km = kv[i], kp = kv[i+1];
            double dRr = Rp - Rm;
            hl[i] = -M[i]*(rho[i]+rho[i+1]) - 20.0*Ri2*p[i]*Rm2*km + 3.0*M[i]*dRr*rho[i]*Rm2*km;
            hd[i] = 8.0*Ri*(p[i+1]-p[i]) + 20.0*Ri4*(p[i]*km + p[i+1]*kp)
                  + 3.0*M[i]*dRr*Ri2*(-rho[i]*km + rho[i+1]*kp);
            hu[i] = M[i]*(rho[i]+rho[i+1]) - 20.0*Ri2*p[i+1]*Rp2*kp - 3.0*M[i]*dRr*rho[i+1]*Rp2*kp;
            hb[i] = -4.0*Ri2*(p[i+1]-p[i]) - M[i]*(rho[i]+rho[i+1])*dRr;
        }
        // last movable row N-1 (R[N] fixed)
        {
            int l = N-1;
            double Rl = R[l], Rl2 = Rl*Rl, Rl4 = Rl2*Rl2, Rm = R[l-1], Rm2 = Rm*Rm, R1 = R[l+1];
            double km = kv[l], dRr = R1 - Rm;
            hl[l] = -M[l]*(rho[l]+rho[l+1]) - 20.0*Rl2*p[l]*Rm2*km + 3.0*M[l]*dRr*rho[l]*Rm2*km;
            hd[l] = 8.0*Rl*(p[l+1]-p[l]) + 20.0*Rl4*p[l]*km - 3.0*M[l]*dRr*Rl2*rho[l]*km;
            hb[l] = -4.0*Rl2*(p[l+1]-p[l]) - M[l]*(rho[l]+rho[l+1])*dRr;
        }
    }

    // volume (R_i^3 - R_{i-1}^3) in stable form
    static inline double shellVol(double Ri, double Rm) {
        return (Ri - Rm) * (Ri*Ri + Ri*Rm + Rm*Rm);
    }

    // Candidate R + sc*dR: exact R, rho, p (adiabatic: rho ~ 1/V, p ~ V^(-5/3)). Returns false if inadmissible.
    bool buildTrial(double sc, std::string* msg = nullptr) {
        double prev = 0.0;
        for (int i = 0; i < N; ++i) {
            double Rn = s.R[i] + sc * dR[i];
            if (!std::isfinite(Rn) || Rn <= prev) {
                if (msg) *msg = "R non-increasing/non-finite at shell " + std::to_string(i);
                return false;
            }
            double Vold = shellVol(s.R[i], i ? s.R[i-1] : 0.0);
            double Vnew = shellVol(Rn, prev);
            double ratio = Vold / Vnew;
            double rn = s.Rho[i] * ratio;
            double pn = s.p[i] * pow53(ratio);
            if (!(rn > 0.0) || !(pn > 0.0) || !std::isfinite(rn) || !std::isfinite(pn)) {
                if (msg) *msg = "invalid rho or p at shell " + std::to_string(i);
                return false;
            }
            Rtry[i] = Rn; Rhotry[i] = rn; ptry[i] = pn;
            prev = Rn;
        }
        if (!(prev < s.R[N])) { if (msg) *msg = "shell N-1 reaches the fixed shell"; return false; }
        return true;
    }

    // -----------------------------------------------------------------------------------------
    // Hydrostatic relaxation: Newton with exact tridiagonal solve, diagonal equilibration,
    // and backtracking. Stops as soon as it converges.
    // -----------------------------------------------------------------------------------------
    void relaxationStep() {
        for (int it = 0; it < MAX_NEWTON_ITER; ++it) {
            setupHydrostatic();

            // symmetric equilibration H' = D H D, b' = D b, deltaR = D x'
            for (int i = 0; i < N; ++i) rdsc[i] = 1.0 / std::sqrt(std::max(std::fabs(hd[i]), 1e-300));
            for (int i = 0; i < N; ++i) {
                rdd[i] = rdsc[i] * hd[i] * rdsc[i];
                rbs[i] = rdsc[i] * hb[i];
                if (i < N-1) {
                    rdu[i] = rdsc[i]   * hu[i]   * rdsc[i+1];   // H(i,i+1)
                    rdl[i] = rdsc[i+1] * hl[i+1] * rdsc[i];     // H(i+1,i)
                }
            }
            if (!solveTridiagPivotInPlace(rdl, rdd, rdu, rbs, rxs, rdu2))
                throw std::runtime_error("relaxation: singular hydrostatic matrix (iter " + std::to_string(it) + ")");
            double maxrel = 0.0;
            for (int i = 0; i < N; ++i) {
                dR[i] = rdsc[i] * rxs[i];
                maxrel = std::max(maxrel, std::fabs(dR[i] / s.R[i]));
            }

            // backtracking: halve step until admissible
            double sc = 1.0; int h = 0; std::string msg;
            while (!buildTrial(sc, &msg) && h < MAX_HALVINGS_NEWTON) { sc *= 0.5; ++h; }
            if (h >= MAX_HALVINGS_NEWTON && !buildTrial(sc, &msg))
                throw std::runtime_error("relaxation: no admissible step (iter " + std::to_string(it) +
                                         "): " + msg);
            for (int i = 0; i < N; ++i) { s.R[i] = Rtry[i]; s.Rho[i] = Rhotry[i]; s.p[i] = ptry[i]; }

            // quadratic convergence: after applying a full correction < NEWTON_TOL the residual error
            // is ~NEWTON_TOL^2, so an additional "verification" iteration would just waste time
            if (sc == 1.0 && maxrel < NEWTON_TOL) return;
        }
        // Did not converge within MAX_NEWTON_ITER: accepted only if the last correction is small
        // (otherwise the step is rejected and retried with halved dt)
        setupHydrostatic();
        double res = 0.0;
        for (int i = 0; i < N; ++i) res = std::max(res, std::fabs(dR[i] / s.R[i]));
        if (res > 1e-6) throw std::runtime_error("relaxation: Newton fails to converge (|dR/R|=" + std::to_string(res) + ")");
    }

    // -----------------------------------------------------------------------------------------
    // Resize work buffers after a change in the number of shells
    // -----------------------------------------------------------------------------------------
    void resizeBuffers() {
        N = s.n - 1;
        dUc.assign(s.n, 0.0); C.assign(N, 0.0); m.assign(N, 0.0);
        lo.assign(N, 0.0); di.assign(N, 0.0); up.assign(N, 0.0); rhs.assign(N, 0.0);
        hl.assign(N, 0.0); hd.assign(N, 0.0); hu.assign(N, 0.0); hb.assign(N, 0.0); dR.assign(N, 0.0);
        Rtry.assign(N, 0.0); Rhotry.assign(N, 0.0); ptry.assign(N, 0.0);
        kv.assign(N, 0.0);
        rdsc.assign(N, 0.0); rdd.assign(N, 0.0); rbs.assign(N, 0.0); rxs.assign(N, 0.0);
        rdl.assign(std::max(N-1, 0), 0.0); rdu.assign(std::max(N-1, 0), 0.0); rdu2.assign(std::max(N-2, 0), 0.0);
    }

    // -----------------------------------------------------------------------------------------
    // Adaptive meshing.
    // Convention: shell i lies between R[i-1] and R[i], has mass m_i = M[i]-M[i-1], and uniform
    // density Rho[i]. SOLE criterion: g_i = log(R_i/R_{i-1}) > ADAPT_LOG_GAP (shells i = 1..N, i.e.,
    // all except shell 0, which has no R[-1]).
    //
    // For each shell violating the criterion, a node R_new is inserted such that the two resulting
    // shells each have mass m_i/2:
    //   - uniform density in the shell  =>  equal volume for both halves  =>
    //        R_new^3 = (R_i^3 + R_{i-1}^3)/2
    //   - M_new = M[i-1] + m_i/2
    //   - the adiabatic variable a (entropy) remains that of the original shell in both halves;
    //     rho follows from mass/volume, p = a rho^(5/3), u = 1.5 a rho^(2/3), v from u, L from
    //     updateProfiles.
    // Thus mass, internal energy, and entropy are conserved.
    //
    // The scan is REPEATED across the entire grid (splitPass) until no shell exceeds the threshold
    // (or ADAPT_MAX_PASS / ADAPT_MAX_LAYERS is reached): a single pass is insufficient because
    // splitting a shell with R_out >> R_in reduces the gap by only ~0.1 dex (0.23 in ln).
    // Finally, ONE global hydrostatic relaxation is performed at frozen time. If it fails,
    // the previous grid is restored.
    // -----------------------------------------------------------------------------------------

    // Single pass: splits ALL shells violating the criterion. Returns the number of splits.
    int splitPass() {
        if (s.n >= ADAPT_MAX_LAYERS) return 0;

        Vec new_R, new_M, new_Rho, new_a, new_u, new_v, new_p, new_L;
        const size_t cap = (size_t)s.n * 2;
        new_R.reserve(cap); new_M.reserve(cap); new_Rho.reserve(cap); new_a.reserve(cap);
        new_u.reserve(cap); new_v.reserve(cap); new_p.reserve(cap); new_L.reserve(cap);

        // The first node (R[0]) remains unchanged
        new_R.push_back(s.R[0]);   new_M.push_back(s.M[0]);
        new_Rho.push_back(s.Rho[0]); new_a.push_back(s.a[0]);
        new_u.push_back(s.u[0]);   new_v.push_back(s.v[0]);
        new_p.push_back(s.p[0]);   new_L.push_back(s.L[0]);

        int splits = 0;
        for (int i = 1; i < s.n; ++i) {
            const double Rin  = s.R[i-1];
            const double Rout = s.R[i];
            const double gap  = ADAPT_USE_LN ? std::log(Rout / Rin) : std::log10(Rout / Rin);

            if (gap > params.adaptGap && (int)new_R.size() + 2 <= ADAPT_MAX_LAYERS) {
                const double Rnew   = std::cbrt(0.5 * (Rout*Rout*Rout + Rin*Rin*Rin));
                const double mShell = s.M[i] - s.M[i-1];
                const double Mnew   = s.M[i-1] + 0.5 * mShell;
                const double Vold   = shellVol(Rout, Rin);
                const double Vlo    = shellVol(Rnew, Rin);
                const double Vup    = shellVol(Rout, Rnew);
                const double rho0   = s.Rho[i];
                const double a0     = s.a[i];
                const double rhoLo  = rho0 * (0.5 * Vold) / Vlo;
                const double rhoUp  = rho0 * (0.5 * Vold) / Vup;

                // inner child (intermediate node Rnew)
                new_R.push_back(Rnew);  new_M.push_back(Mnew);
                new_Rho.push_back(rhoLo); new_a.push_back(a0);
                new_u.push_back(0.0);   new_v.push_back(0.0);
                new_p.push_back(a0 * pow53(rhoLo)); new_L.push_back(0.0);

                // outer child (node Rout)
                new_R.push_back(Rout);  new_M.push_back(s.M[i]);
                new_Rho.push_back(rhoUp); new_a.push_back(a0);
                new_u.push_back(0.0);   new_v.push_back(0.0);
                new_p.push_back(a0 * pow53(rhoUp)); new_L.push_back(0.0);

                ++splits;
            } else {
                // normal shell: direct copy
                new_R.push_back(Rout);  new_M.push_back(s.M[i]);
                new_Rho.push_back(s.Rho[i]); new_a.push_back(s.a[i]);
                new_u.push_back(s.u[i]); new_v.push_back(s.v[i]);
                new_p.push_back(s.p[i]); new_L.push_back(s.L[i]);
            }
        }
        if (splits == 0) return 0;

        s.R   = std::move(new_R);   s.M = std::move(new_M);
        s.Rho = std::move(new_Rho); s.a = std::move(new_a);
        s.u   = std::move(new_u);   s.v = std::move(new_v);
        s.p   = std::move(new_p);   s.L = std::move(new_L);
        s.n   = (int)s.R.size();
        resizeBuffers();
        updateProfiles();           // u, v, L consistent with a and rho of the new grid
        return splits;
    }

    // Coarsening: single pass. Merges DISJOINT pairs of adjacent shells (i, i+1), 1 <= i <= N-2
    // (shell 0 and fixed shell N are untouched), if the merged shell [R[i-1], R[i+1]] is no
    // coarser than the initial grid: ln(R[i+1]/R[i-1]) <= mergeGapR and ln(M[i+1]/M[i-1]) <= mergeGapM.
    // Merged shell: mass m = m_i + m_{i+1}, volume V = V_i + V_{i+1} (Rho = volume-weighted average,
    // consistent with m = Rho V / 3), conserved internal energy u = (m_i u_i + m_{i+1} u_{i+1}) / m,
    // entropy a = (2/3) u Rho^(-2/3) and p = a Rho^(5/3) recomputed. The merge is rejected if
    // it would introduce an entropy inversion. Returns the number of merges.
    int mergePass() {
        if (!ADAPT_COARSEN || s.n < 8) return 0;
        const int n = s.n;

        Vec new_R, new_M, new_Rho, new_a, new_u, new_v, new_p, new_L;
        new_R.reserve(n); new_M.reserve(n); new_Rho.reserve(n); new_a.reserve(n);
        new_u.reserve(n); new_v.reserve(n); new_p.reserve(n); new_L.reserve(n);
        auto keep = [&](int k) {
            new_R.push_back(s.R[k]);     new_M.push_back(s.M[k]);
            new_Rho.push_back(s.Rho[k]); new_a.push_back(s.a[k]);
            new_u.push_back(s.u[k]);     new_v.push_back(s.v[k]);
            new_p.push_back(s.p[k]);     new_L.push_back(s.L[k]);
        };

        keep(0);                                   // shell 0: unchanged
        int merges = 0;
        int i = 1;
        while (i < n) {
            bool done = false;
            if (i <= N - 2) {                      // pair (i, i+1) must lie among movable shells
                const double gR = std::log(s.R[i+1] / s.R[i-1]);
                const double gM = std::log(s.M[i+1] / s.M[i-1]);
                if (gR <= mergeGapR && gM <= mergeGapM) {
                    const double m1 = s.M[i]   - s.M[i-1];
                    const double m2 = s.M[i+1] - s.M[i];
                    const double V1 = shellVol(s.R[i],   s.R[i-1]);
                    const double V2 = shellVol(s.R[i+1], s.R[i]);
                    const double rho = (s.Rho[i] * V1 + s.Rho[i+1] * V2) / (V1 + V2);
                    const double u   = (m1 * s.u[i] + m2 * s.u[i+1]) / (m1 + m2);
                    const double a   = (2.0/3.0) * u / pow23(rho);
                    // entropy must not worsen: the number of inverted interfaces (same tolerance as hasInversion)
                    // must not increase (an ALREADY inverted region can be merged, a monotonic one stays monotonic)
                    const double aPrev = new_a.back(), aNext = s.a[i+2];
                    auto bad = [](double lo, double hi) { return !(hi >= lo * (1.0 - ENTROPY_TOL)) ? 1 : 0; };
                    const int badBefore = bad(aPrev, s.a[i]) + bad(s.a[i], s.a[i+1]) + bad(s.a[i+1], aNext);
                    const int badAfter  = bad(aPrev, a) + bad(a, aNext);
                    if (std::isfinite(a) && badAfter <= badBefore) {
                        new_R.push_back(s.R[i+1]);   new_M.push_back(s.M[i+1]);
                        new_Rho.push_back(rho);      new_a.push_back(a);
                        new_u.push_back(u);          new_v.push_back(std::sqrt(2.0/3.0) * std::sqrt(u));
                        new_p.push_back(a * pow53(rho)); new_L.push_back(0.0);
                        ++merges;
                        i += 2;
                        done = true;
                    }
                }
            }
            if (!done) { keep(i); ++i; }
        }
        if (merges == 0) return 0;

        s.R   = std::move(new_R);   s.M = std::move(new_M);
        s.Rho = std::move(new_Rho); s.a = std::move(new_a);
        s.u   = std::move(new_u);   s.v = std::move(new_v);
        s.p   = std::move(new_p);   s.L = std::move(new_L);
        s.n   = (int)s.R.size();
        resizeBuffers();
        updateProfiles();           // u, v, L consistent with a and rho of the new grid
        return merges;
    }

    void adaptMesh(int tstep) {
        const State backup = s;
        const int nBefore = s.n;

        // 1) coarsening (before splitting: a freshly split shell has gap >= threshold/2 and cannot
        //    be re-merged; a freshly merged shell has gap <= threshold/2 and cannot be re-split)
        int merged = 0, mpasses = 0;
        for (int pass = 0; pass < ADAPT_MAX_PASS; ++pass) {
            int k = mergePass();
            if (k == 0) break;
            merged += k;
            ++mpasses;
        }

        // 2) splitting (criterion and method unchanged)
        int total = 0, passes = 0;
        if (s.n < ADAPT_MAX_LAYERS) {
            for (int pass = 0; pass < ADAPT_MAX_PASS; ++pass) {
                int k = splitPass();
                if (k == 0) break;
                total += k;
                ++passes;
            }
        }
        if (total == 0 && merged == 0) return;

        // Single global relaxation on the entire modified grid
        try {
            updateProfiles();
            relaxationStep();
            updateProfiles();
        } catch (const std::exception& e) {
            s = backup;
            resizeBuffers();
            logger.info("Adaptive meshing canceled at step " + std::to_string(tstep) + ": " + e.what());
            return;
        }

        if (adaptCalls == 0) adaptLayersFrom = nBefore;
        ++adaptCalls; adaptSplitsAcc += total; adaptMergesAcc += merged;
        lastAdaptStep = tstep;
        flushAdaptLog(false);
    }

    // aggregated log line every ADAPT_LOG_EVERY calls that modified the grid
    void flushAdaptLog(bool force) {
        if (adaptCalls == 0 || (!force && adaptCalls % ADAPT_LOG_EVERY != 0)) return;
        std::ostringstream os;
        os << "Adaptive meshing: up to step " << lastAdaptStep << ", t = " << std::setprecision(10) << params.totalTime
           << ", " << adaptCalls << " grid modifications (" << adaptSplitsAcc << " splits, " << adaptMergesAcc
           << " merges), layers " << adaptLayersFrom << " -> " << s.n;
        logger.info(os.str());
        adaptCalls = 0; adaptSplitsAcc = 0; adaptMergesAcc = 0;
    }

    // -----------------------------------------------------------------------------------------
    void updateProfiles() {
        for (int i = 0; i < s.n; ++i) {
            s.u[i] = 1.5 * s.a[i] * pow23(s.Rho[i]);
            s.v[i] = std::sqrt(2.0/3.0) * std::sqrt(s.u[i]);
        }
        const double sigma2 = params.sigma * params.sigma;
        const double abc_s  = params.a * params.b * params.c * params.sigma;
        auto term = [&](int i) {
            double vi = s.v[i], vi2 = vi * vi, vi3 = vi2 * vi;
            return s.Rho[i] * vi3 / (params.a * params.c * sigma2 * s.Rho[i] * vi2 + params.b);
        };
        s.L[0] = -(s.u[1] - s.u[0]) / s.R[1] * s.R[0] * s.R[0] * abc_s * (term(0) + term(1));
        for (int i = 1; i < N; ++i)
            s.L[i] = -(s.u[i+1] - s.u[i]) / (s.R[i+1] - s.R[i-1]) * s.R[i] * s.R[i] * abc_s * (term(i) + term(i+1));
    }

    void saveResults(std::ofstream& f, int tstep) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
        // same format as std::scientific + setprecision(SAVE_PRECISION), but ~10x faster than iostream
        saveBuf.clear();
        char tmp[64];
        auto put = [&](double x) {
            auto r = std::to_chars(tmp, tmp + sizeof(tmp), x, std::chars_format::scientific, SAVE_PRECISION);
            saveBuf.append(tmp, r.ptr);
        };
        auto line = [&](const Vec& v) {
            for (size_t i = 0; i < v.size(); ++i) { if (i) saveBuf += ' '; put(v[i]); }
            saveBuf += '\n';
        };
        put(params.totalTime); saveBuf += ' '; saveBuf += std::to_string(tstep); saveBuf += '\n';
        line(s.R); line(s.Rho); line(s.M); line(s.u); line(s.L);
        f.write(saveBuf.data(), (std::streamsize)saveBuf.size());
#else
        f << std::scientific << std::setprecision(SAVE_PRECISION) << params.totalTime << " " << tstep << '\n';
        writeVec(f, s.R); writeVec(f, s.Rho); writeVec(f, s.M); writeVec(f, s.u); writeVec(f, s.L);
#endif
    }
};

// ---------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    clock_t start = clock();
    try {
        Logger logger(true);
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
        logger.info("WARNING: compiled with fast-math. Recompile without -ffast-math/-Ofast.");
#endif
        Params params;
        if (argc > 1) params.tag = argv[1];
        params.setDefaultPaths();
        if (argc > 2) params.inputDir = argv[2];
        if (argc > 3) params.outputFile = argv[3];
        if (argc > 4) params.totalStep = std::atoi(argv[4]);
        if (argc > 5) params.adaptGap = std::atof(argv[5]);
        if (argc > 6) params.epsFine  = std::atof(argv[6]);
        if (argc > 7) params.saveDlog = std::atof(argv[7]);
        params.display(logger);

        Simulator sim(params, logger);
        sim.initialize();
        sim.run();

        float seconds = (float)(clock() - start) / CLOCKS_PER_SEC;
        logger.info("Computation time = " + std::to_string(seconds) + " s");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
