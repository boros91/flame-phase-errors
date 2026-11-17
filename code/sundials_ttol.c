/* sundials_ttol.c -- flame problem y' = y^q (1-y), y(0) = delta, 0 <= t <= 2/((q-1) delta^(q-1)),
   solved with CVODE (SUNDIALS, BDF, dense direct linear solver) in three ways:

     std : the original equation, RelTol = tol, AbsTol = tol/100
     sv  : time as a dependent variable (Sundman transformation dt/dtau = 1/rho(y)),
           standard tolerance interface CVodeSVtolerances: scalar RelTol = tol for both
           components, vector AbsTol = (tol/100, tol)
     wf  : the same transformed system with a user-supplied error weight function
           (CVodeWFtolerances): W_y = 1/(tol*|y| + tol/100), W_t = 1/(1e-13*|t| + tol),
           i.e. an ABSOLUTE tolerance tol on t (the tiny relative part 1e-13 only keeps the
           tolerance on t above the spacing of doubles when t ~ 1e8; the same values
           rtol_t = 1e-13, atol_t = tol are used with LSODA in exp_sundman_table.py)

   rho = y^(q-1) ("power") or rho = |f(y)|/|y| + |f'(y)| ("generic").

   Usage:  sundials_ttol q delta tol std|sv|wf power|generic outfile.csv [bdf|adams] [atol_y]
           (defaults: BDF, atol_y = tol/100)
   Output: every accepted step (t, y); the last line holds the CVODE counters.
   Build:  gcc -O2 -o sundials_ttol sundials_ttol.c -lsundials_cvode -lsundials_nvecserial \
               -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lm
   Tested with SUNDIALS 6.4.1. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

typedef struct { int q; int generic; double tol; double atoly; } Data;

static double fq(double y, int q)  { return pow(y, q) * (1.0 - y); }
static double fqp(double y, int q) { return q * pow(y, q - 1) - (q + 1) * pow(y, q); }
static double rho(double y, const Data *d)
{
    if (d->generic) return fabs(fq(y, d->q)) / fabs(y) + fabs(fqp(y, d->q));
    return pow(y, d->q - 1);
}

static int rhs_std(sunrealtype t, N_Vector u, N_Vector du, void *ud)
{
    const Data *d = (const Data *)ud;
    NV_Ith_S(du, 0) = fq(NV_Ith_S(u, 0), d->q);
    return 0;
}

static int rhs_sund(sunrealtype tau, N_Vector u, N_Vector du, void *ud)
{
    const Data *d = (const Data *)ud;
    double y = NV_Ith_S(u, 0), r = rho(y, d);
    NV_Ith_S(du, 0) = fq(y, d->q) / r;
    NV_Ith_S(du, 1) = 1.0 / r;
    return 0;
}

/* root function for the transformed system: stop when t reaches tend */
static double g_tend;
static int root_t(sunrealtype tau, N_Vector u, sunrealtype *g, void *ud)
{
    g[0] = NV_Ith_S(u, 1) - g_tend;
    return 0;
}

/* error weights: relative + absolute on y, absolute on t */
#define RELTOL_T 1e-13
static int ewt_abs_t(N_Vector u, N_Vector w, void *ud)
{
    const Data *d = (const Data *)ud;
    NV_Ith_S(w, 0) = 1.0 / (d->tol * fabs(NV_Ith_S(u, 0)) + d->atoly);
    NV_Ith_S(w, 1) = 1.0 / (RELTOL_T * fabs(NV_Ith_S(u, 1)) + d->tol);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage: q delta tol std|sv|wf power|generic out.csv\n"); return 1; }
    Data d; d.q = atoi(argv[1]); double delta = atof(argv[2]); d.tol = atof(argv[3]);
    const char *form = argv[4]; d.generic = strcmp(argv[5], "generic") == 0;
    FILE *fo = fopen(argv[6], "w");
    int lmm = (argc > 7 && strcmp(argv[7], "adams") == 0) ? CV_ADAMS : CV_BDF;
    d.atoly = (argc > 8) ? atof(argv[8]) : d.tol / 100.0;
    double tend = 2.0 / ((d.q - 1) * pow(delta, d.q - 1));
    int std = strcmp(form, "std") == 0, n = std ? 1 : 2;

    SUNContext ctx; SUNContext_Create(NULL, &ctx);
    N_Vector u = N_VNew_Serial(n, ctx);
    NV_Ith_S(u, 0) = delta; if (!std) NV_Ith_S(u, 1) = 0.0;
    void *mem = CVodeCreate(lmm, ctx);
    CVodeInit(mem, std ? rhs_std : rhs_sund, 0.0, u);
    CVodeSetUserData(mem, &d);
    if (std) {
        CVodeSStolerances(mem, d.tol, d.atoly);
        CVodeSetStopTime(mem, tend);
    } else if (strcmp(form, "sv") == 0) {
        N_Vector at = N_VNew_Serial(2, ctx);
        NV_Ith_S(at, 0) = d.atoly; NV_Ith_S(at, 1) = d.tol;
        CVodeSVtolerances(mem, d.tol, at);
    } else {
        CVodeWFtolerances(mem, ewt_abs_t);
    }
    if (!std) { g_tend = tend; CVodeRootInit(mem, 1, root_t); }
    SUNMatrix A = SUNDenseMatrix(n, n, ctx);
    SUNLinearSolver LS = SUNLinSol_Dense(u, A, ctx);
    CVodeSetLinearSolver(mem, LS, A);
    CVodeSetMaxNumSteps(mem, 10000000);

    fprintf(fo, "t,y\n%.17g,%.17g\n", 0.0, delta);
    double tout = std ? tend : 1e3 * tend, tret = 0.0;
    int flag = 0;
    for (;;) {
        flag = CVode(mem, tout, u, &tret, CV_ONE_STEP);
        if (flag < 0) break;
        double t = std ? tret : NV_Ith_S(u, 1);
        fprintf(fo, "%.17g,%.17g\n", t, NV_Ith_S(u, 0));
        if (flag == CV_ROOT_RETURN || t >= tend * (1 - 1e-14)) break;
    }
    long nst, nfe, nfeLS, nje, nsetups, netf;
    CVodeGetNumSteps(mem, &nst); CVodeGetNumRhsEvals(mem, &nfe);
    CVodeGetNumLinRhsEvals(mem, &nfeLS); CVodeGetNumJacEvals(mem, &nje);
    CVodeGetNumLinSolvSetups(mem, &nsetups); CVodeGetNumErrTestFails(mem, &netf);
    fprintf(fo, "#stats,flag=%d,nst=%ld,nfe=%ld,nfeLS=%ld,nje=%ld,nsetups=%ld,netf=%ld\n",
            flag, nst, nfe, nfeLS, nje, nsetups, netf);
    fclose(fo);
    N_VDestroy(u); SUNLinSolFree(LS); SUNMatDestroy(A); CVodeFree(&mem); SUNContext_Free(&ctx);
    return flag < 0 ? 2 : 0;
}
