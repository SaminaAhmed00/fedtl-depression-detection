# ----------------------------- statistics & LaTeX export -----------------------------
import json, os
import numpy as np, pandas as pd
from scipy import stats

EP_CODE = {"bin_ea": "BinEA", "bin_t1": "BinOne", "bin_t2": "BinTwo", "sev3": "Sev", "bin_mixed": "BinMix"}
EP_SHORT = {"sev3": "3-class", "bin_t1": r"Binary $\theta{=}1$", "bin_t2": r"Binary $\theta{=}2$"}
EP_NAME = {"bin_ea": r"Binary, EEG + audio (depressed vs.\ not)",
           "bin_t1": r"Binary, all silos, $\theta=1$ (depressed vs.\ not)",
           "bin_t2": r"Binary, all silos, $\theta=2$ (severe vs.\ non-severe)",
           "sev3": "Three-class severity",
           "bin_mixed": r"Binary, original mixed endpoint"}
M_CODE = {"standalone": "Stand", "local_proto": "Loc", "fedproto": "FP", "local_tgp": "LocT", "fedtgp": "TGP"}
M_NAME = {"standalone": "Standalone probe", "local_proto": "Local (matched to FedProto)",
          "fedproto": "FedProto-style", "local_tgp": "Local (matched to FedTGP)", "fedtgp": "FedTGP-style"}
M_SHORT = {"standalone": "Standalone", "local_proto": "Local$_{\\mathrm{P}}$", "fedproto": "FedProto",
           "local_tgp": "Local$_{\\mathrm{T}}$", "fedtgp": "FedTGP"}
S_CODE = {"eeg": "EEG", "audio": "Aud", "text": "Txt", "all": "All"}
S_NAME = {"eeg": "EEG", "audio": "Audio", "text": "Text", "all": "Overall"}
PAIRS = [("fedproto", "local_proto", "FPvLoc"), ("fedtgp", "local_tgp", "TGPvLocT"),
         ("fedproto", "standalone", "FPvStand"), ("fedtgp", "standalone", "TGPvStand"),
         ("fedtgp", "fedproto", "TGPvFP"), ("local_proto", "standalone", "LocvStand")]
PAIR_NAME = {"FPvLoc": "FedProto $-$ matched local", "TGPvLocT": "FedTGP $-$ matched local",
             "FPvStand": "FedProto $-$ standalone", "TGPvStand": "FedTGP $-$ standalone",
             "TGPvFP": "FedTGP $-$ FedProto", "LocvStand": "Matched local $-$ standalone"}
BAL_CODE = {"natural": "Nat", "balanced": "Bal"}
CONTRASTS = [  # (name, (endpoint, balance) minus (endpoint, balance),
    ("SevVsOne", ("sev3", "natural"), ("bin_t1", "natural"), r"granularity: 3-class $-$ binary $\theta{=}1$"),
    ("SevVsTwo", ("sev3", "natural"), ("bin_t2", "natural"), r"granularity: 3-class $-$ binary $\theta{=}2$"),
    ("TwoVsOne", ("bin_t2", "natural"), ("bin_t1", "natural"), r"threshold: $\theta{=}2$ $-$ $\theta{=}1$"),
    ("BalSev", ("sev3", "balanced"), ("sev3", "natural"), r"balance (3-class): balanced $-$ natural"),
    ("BalOne", ("bin_t1", "balanced"), ("bin_t1", "natural"), r"balance ($\theta{=}1$): balanced $-$ natural"),
    ("BalTwo", ("bin_t2", "balanced"), ("bin_t2", "natural"), r"balance ($\theta{=}2$): balanced $-$ natural"),
]
TBD = r"\tbd"


def sgn(x, d=3):
    """Signed number with a proper minus sign for LaTeX text mode."""
    return f"{'$-$' if x < 0 else '$+$'}{abs(x):.{d}f}"


class Fmt:
    def __init__(self, placeholder):
        self.ph = placeholder
    def f(self, x, d=3, sign=False):
        if self.ph:
            return TBD
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--"
        return sgn(x, d) if sign else f"{x:.{d}f}"
    def p(self, x):
        if self.ph:
            return TBD
        if x is None or np.isnan(x):
            return "--"
        return "<0.001" if x < 0.001 else f"{x:.3f}"


def select_final(runs, tune_seed=0):
    """Keep, for every learned method, only runs at the LR selected on validation (seed..."""
    runs = runs.copy()
    std = runs[runs.method == "standalone"]
    fl = runs[runs.method != "standalone"]
    if fl.empty:
        return std
    key = ["study", "endpoint", "balance", "partition", "method"]
    t = fl[fl.seed == tune_seed].groupby(key + ["lr"]).val_pooled.first().reset_index()
    best = t.loc[t.groupby(key).val_pooled.idxmax(), key + ["lr"]].rename(columns={"lr": "lr_sel"})
    fl = fl.merge(best, on=key)
    fl = fl[np.isclose(fl.lr.astype(float), fl.lr_sel.astype(float))].drop(columns="lr_sel")
    return pd.concat([std, fl], ignore_index=True)


def partition_table(final):
    """One score per study/endpoint/balance/method/partition/silo (mean over seeds) + 'all' =..."""
    g = (final.groupby(["study", "endpoint", "balance", "method", "partition", "silo"])
              .agg(f1=("test_f1", "mean"), f1_seed_sd=("test_f1", lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0),
                   n_seeds=("test_f1", "size"), n_train=("n_train", "mean"), n_test=("n_test", "mean"))
              .reset_index())
    allr = (g.groupby(["study", "endpoint", "balance", "method", "partition"])
              .agg(f1=("f1", "mean"), f1_seed_sd=("f1_seed_sd", "mean"), n_seeds=("n_seeds", "min"),
                   n_train=("n_train", "sum"), n_test=("n_test", "sum")).reset_index())
    allr["silo"] = "all"
    return pd.concat([g, allr], ignore_index=True)


def nb_corrected(d, ratio):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-partition differences d."""
    d = np.asarray(d, float); R = len(d)
    if R < 2:
        return np.nan, np.nan, np.nan, np.nan
    v = np.var(d, ddof=1)
    se = np.sqrt((1.0 / R + ratio) * v)
    if se == 0:
        return d.mean(), d.mean(), np.nan, 1.0
    t = d.mean() / se
    q = stats.t.ppf(0.975, R - 1)
    return d.mean() - q * se, d.mean() + q * se, t, 2 * stats.t.sf(abs(t), R - 1)


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = ~np.isnan(p); idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]; m = len(order); running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def compare(pt, study, endpoint, balance, a, b, silo):
    A = pt[(pt.study == study) & (pt.endpoint == endpoint) & (pt.balance == balance) & (pt.method == a) & (pt.silo == silo)]
    B = pt[(pt.study == study) & (pt.endpoint == endpoint) & (pt.balance == balance) & (pt.method == b) & (pt.silo == silo)]
    m = A.merge(B, on="partition", suffixes=("_a", "_b"))
    if m.empty:
        return None
    d = (m.f1_a - m.f1_b).to_numpy()
    ratio = float((m.n_test_a / m.n_train_a).mean())
    lo, hi, t, p_nb = nb_corrected(d, ratio)
    sd = np.std(d, ddof=1) if len(d) > 1 else np.nan
    t_plain, p_plain = (stats.ttest_rel(m.f1_a, m.f1_b) if len(d) > 1 and sd > 0 else (np.nan, np.nan))
    try:
        p_w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = np.nan
    return dict(diff=d.mean(), sd=sd, lo=lo, hi=hi, t_corr=t, p_corr=p_nb, t=t_plain, p=p_plain,
                p_wilcoxon=p_w, dz=(d.mean() / sd if sd and sd > 0 else np.nan), R=len(d),
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))


def all_comparisons(pt, endpoints):
    rows = []
    for (study, endpoint, balance), _ in pt.groupby(["study", "endpoint", "balance"]):
        if study != "main":
            continue
        silos = list(endpoints[endpoint]["silos"]) + ["all"]
        fam = []
        for a, b, code in PAIRS:
            for s in silos:
                r = compare(pt, study, endpoint, balance, a, b, s)
                if r is not None:
                    fam.append(dict(endpoint=endpoint, balance=balance, pair=code, a=a, b=b, silo=s, **r))
        if fam:
            # Holm over the family (one endpoint); the 'all' rows are summaries and are excluded
            pv = np.array([f["p_corr"] if f["silo"] != "all" else np.nan for f in fam])
            for f, ph in zip(fam, holm(pv)):
                f["p_holm"] = ph
            rows += fam
    return pd.DataFrame(rows)


def factor_analysis(pt):
    """Δ = FedProto − matched local per partition, for every (endpoint, balance) cell, plus..."""
    cells, deltas = [], {}
    for (endpoint, balance), _ in pt.groupby(["endpoint", "balance"]):
        for s in pt[pt.endpoint == endpoint].silo.unique():
            sub = pt[(pt.endpoint == endpoint) & (pt.balance == balance) & (pt.silo == s)]
            fp = sub[sub.method == "fedproto"].set_index("partition").f1
            lc = sub[sub.method == "local_proto"].set_index("partition").f1
            idx = fp.index.intersection(lc.index)
            if len(idx) == 0:
                continue
            d = (fp[idx] - lc[idx])
            ratio = float((sub.n_test / sub.n_train).mean())
            lo, hi, t, p = nb_corrected(d.to_numpy(), ratio)
            deltas[(endpoint, balance, s)] = (d, ratio)
            cells.append(dict(endpoint=endpoint, balance=balance, silo=s, delta=d.mean(), lo=lo, hi=hi, p_corr=p, R=len(d)))
    contr = []
    for name, c1, c2, desc in CONTRASTS:
        for s in ("eeg", "audio", "text", "all"):
            k1, k2 = (*c1, s), (*c2, s)
            if k1 in deltas and k2 in deltas:
                d1, r1 = deltas[k1]; d2, r2 = deltas[k2]
                idx = d1.index.intersection(d2.index)
                dd = (d1[idx] - d2[idx]).to_numpy()
                lo, hi, t, p = nb_corrected(dd, (r1 + r2) / 2)
                contr.append(dict(contrast=name, desc=desc, silo=s, diff=dd.mean(), lo=lo, hi=hi, p_corr=p, R=len(dd)))
    cdf = pd.DataFrame(contr)
    if not cdf.empty:
        cdf["p_holm"] = np.nan
        m = cdf.silo != "all"
        cdf.loc[m, "p_holm"] = holm(cdf.loc[m, "p_corr"].to_numpy())
    return pd.DataFrame(cells), cdf


def perclass_table(final, endpoint="sev3", methods=("local_proto", "fedproto")):
    rows = []
    sub = final[(final.study == "main") & (final.endpoint == endpoint) & (final.balance == "natural")]
    for m in methods:
        for s in ("eeg", "text", "audio"):
            q = sub[(sub.method == m) & (sub.silo == s)]
            if q.empty:
                continue
            per = []
            for (part, seed), r in q.groupby(["partition", "seed"]):
                y = np.array(json.loads(r.y.iloc[0])); p = np.array(json.loads(r.p.iloc[0]))
                per.append(f1_per_class(y, p) + [np.bincount(y, minlength=3)])
            f = np.array([x[:3] for x in per]); sup = np.mean([x[3] for x in per], axis=0)
            rows.append(dict(method=m, silo=s, f0=f[:, 0].mean(), f1=f[:, 1].mean(), f2=f[:, 2].mean(),
                             f0sd=f[:, 0].std(ddof=1), f1sd=f[:, 1].std(ddof=1), f2sd=f[:, 2].std(ddof=1),
                             sup="/".join(f"{v:.1f}" for v in sup)))
    return pd.DataFrame(rows)


def f1_per_class(y, p):
    from sklearn.metrics import f1_score
    return list(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0))


# ------------------------------------ writers ------------------------------------
def _ms(F, pt, study, ep, bal, m, s):
    v = pt[(pt.study == study) & (pt.endpoint == ep) & (pt.balance == bal) & (pt.method == m) & (pt.silo == s)].f1
    if F.ph:
        return TBD
    return "--" if v.empty else f"{v.mean():.3f}\\,$\\pm$\\,{(v.std(ddof=1) if len(v) > 1 else 0):.3f}"


def table_endpoint(F, pt, ep, endpoints, methods):
    silos = list(endpoints[ep]["silos"]) + ["all"]
    cols = "l" + "c" * len(methods)
    L = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         r"\textbf{Silo} & " + " & ".join(rf"\textbf{{{M_SHORT[m]}}}" for m in methods) + r" \\", r"\midrule"]
    for s in silos:
        if s == "all":
            L.append(r"\midrule")
        L.append(f"{S_NAME[s]} & " + " & ".join(_ms(F, pt, "main", ep, "natural", m, s) for m in methods) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_stats(F, cmp, ep, pairs=("FPvLoc", "TGPvLocT", "FPvStand", "TGPvStand", "TGPvFP")):
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"\textbf{Comparison} & \textbf{Silo} & \textbf{$\Delta$ [95\% CI]} & \textbf{$d_z$} & "
         r"\textbf{$p_{\mathrm{corr}}$} & \textbf{$p_{\mathrm{Holm}}$} & \textbf{$p_{\mathrm{W}}$} \\", r"\midrule"]
    sub = cmp[cmp.endpoint == ep] if not cmp.empty else cmp
    first = True
    for code in pairs:
        q = sub[(sub.pair == code) & (sub.silo != "all")] if not sub.empty else sub
        if q.empty:
            continue
        if not first:
            L.append(r"\midrule")
        first = False
        for k, (_, r) in enumerate(q.iterrows()):
            ci = TBD if F.ph else f"{sgn(r['diff'])} [{sgn(r.lo)}, {sgn(r.hi)}]"
            L.append(f"{PAIR_NAME[code] if k == 0 else ''} & {S_NAME[r.silo]} & {ci} & {F.f(r.dz, 2, True)} & "
                     f"{F.p(r.p_corr)} & {F.p(r.p_holm)} & {F.p(r.p_wilcoxon)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_factor(F, cells, contr):
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"\textbf{Endpoint} & \textbf{Training balance} & \textbf{EEG} & \textbf{Audio} & \textbf{Text} \\", r"\midrule"]
    for ep in ("sev3", "bin_t1", "bin_t2"):
        for bal in ("natural", "balanced"):
            vals = []
            for s in ("eeg", "audio", "text"):
                q = cells[(cells.endpoint == ep) & (cells.balance == bal) & (cells.silo == s)] if not cells.empty else cells
                vals.append(TBD if F.ph else ("--" if q.empty else f"{sgn(q['delta'].iloc[0])} [{sgn(q.lo.iloc[0])}, {sgn(q.hi.iloc[0])}]"))
            L.append(f"{EP_SHORT[ep] if bal == 'natural' else ''} & {bal} & " + " & ".join(vals) + r" \\")
    L += [r"\midrule", r"\multicolumn{5}{l}{\textit{Contrasts of $\Delta$ (one factor changed)}} \\"]
    for name, c1, c2, desc in CONTRASTS:
        vals = []
        for s in ("eeg", "audio", "text"):
            q = contr[(contr.contrast == name) & (contr.silo == s)] if not contr.empty else contr
            if F.ph:
                vals.append(TBD)
            elif q.empty:
                vals.append("--")
            else:
                star = "$^{*}$" if q.p_holm.iloc[0] < 0.05 else ""
                vals.append(f"{sgn(q['diff'].iloc[0])} [{sgn(q.lo.iloc[0])}, {sgn(q.hi.iloc[0])}]{star}")
        L.append(rf"\multicolumn{{2}}{{l}}{{{desc}}} & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_perclass(F, pc):
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"\textbf{Method} & \textbf{Silo} & \textbf{None-min.} & \textbf{Mild-mod.} & \textbf{Severe} & \textbf{Test support} \\",
         r"\midrule"]
    for m in ("local_proto", "fedproto"):
        for k, silo in enumerate(("eeg", "text", "audio")):
            q = pc[(pc.method == m) & (pc.silo == silo)] if not pc.empty else pc
            lead = M_SHORT[m] if k == 0 else ""
            if F.ph or q.empty:
                L.append(f"{lead} & {S_NAME[silo]} & {TBD} & {TBD} & {TBD} & {TBD} \\\\")
            else:
                r = q.iloc[0]
                L.append(f"{lead} & {S_NAME[silo]} & {r.f0:.3f}\\,$\\pm$\\,{r.f0sd:.3f} & "
                         f"{r.f1:.3f}\\,$\\pm$\\,{r.f1sd:.3f} & {r.f2:.3f}\\,$\\pm$\\,{r.f2sd:.3f} & {r.sup} \\\\")
        if m == "local_proto":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_individual(F, pt, endpoints, methods):
    """Supplementary longtable: per-partition macro-F1 (mean over seeds) for every configuration."""
    P = sorted(pt.partition.unique())
    L = [r"\setlength{\tabcolsep}{2.5pt}", r"\begin{longtable}{lll" + "c" * len(P) + "cc}",
         r"\caption{Individual results for every configuration: test macro-F1 per subject-level partition "
         r"(mean over seeds within the partition), with mean and SD across partitions. Per-seed values are in "
         r"the accompanying \texttt{runs\_long.csv}.}\label{tab:individual}\\", r"\toprule",
         r"\textbf{Endpoint} & \textbf{Method} & \textbf{Silo} & " + " & ".join(rf"\textbf{{P{p + 1}}}" for p in P)
         + r" & \textbf{Mean} & \textbf{SD} \\", r"\midrule", r"\endfirsthead", r"\toprule",
         r"\textbf{Endpoint} & \textbf{Method} & \textbf{Silo} & " + " & ".join(rf"\textbf{{P{p + 1}}}" for p in P)
         + r" & \textbf{Mean} & \textbf{SD} \\", r"\midrule", r"\endhead"]
    for ep in [e for e in EP_CODE if e in pt.endpoint.unique()]:
        for m in methods:
            for s in endpoints[ep]["silos"]:
                q = pt[(pt.study == "main") & (pt.endpoint == ep) & (pt.balance == "natural") & (pt.method == m) & (pt.silo == s)].set_index("partition").f1
                vals = [F.f(q.get(p, np.nan), 2) for p in P]
                L.append(f"{EP_CODE[ep]} & {M_SHORT[m]} & {S_NAME[s]} & " + " & ".join(vals)
                         + f" & {F.f(q.mean() if len(q) else np.nan)} & {F.f(q.std(ddof=1) if len(q) > 1 else np.nan)} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{longtable}")
    return "\n".join(L)


def write_macros(F, pt, cmp, cells, contr, path, n_seeds, endpoints):
    M = {}
    M["Rnpart"] = str(pt.partition.nunique()) if not F.ph else TBD
    M["Rnseeds"] = str(int(pt.n_seeds.max())) if not F.ph else TBD
    for ep in EP_CODE:
        if ep not in endpoints:
            continue
        for m in M_CODE:
            for s in list(endpoints[ep]["silos"]) + ["all"]:
                v = pt[(pt.study == "main") & (pt.endpoint == ep) & (pt.balance == "natural") & (pt.method == m) & (pt.silo == s)].f1
                M[f"R{EP_CODE[ep]}{M_CODE[m]}{S_CODE[s]}Mean"] = F.f(v.mean() if len(v) else np.nan)
                M[f"R{EP_CODE[ep]}{M_CODE[m]}{S_CODE[s]}SD"] = F.f(v.std(ddof=1) if len(v) > 1 else np.nan)
        for a, b, code in PAIRS:
            for s in list(endpoints[ep]["silos"]) + ["all"]:
                q = cmp[(cmp.endpoint == ep) & (cmp.pair == code) & (cmp.silo == s)] if not cmp.empty else cmp
                r = q.iloc[0] if len(q) else None
                base = f"C{EP_CODE[ep]}{code}{S_CODE[s]}"
                M[base + "Diff"] = F.f(r["diff"] if r is not None else np.nan, 3, True)
                M[base + "Lo"] = F.f(r.lo if r is not None else np.nan, 3, True)
                M[base + "Hi"] = F.f(r.hi if r is not None else np.nan, 3, True)
                M[base + "P"] = F.p(r.p_corr if r is not None else np.nan)
                M[base + "Padj"] = F.p(r.p_holm if r is not None and "p_holm" in r else np.nan)
                M[base + "Pw"] = F.p(r.p_wilcoxon if r is not None else np.nan)
                M[base + "Dz"] = F.f(r.dz if r is not None else np.nan, 2, True)
                M[base + "Npos"] = TBD if F.ph else (str(r.n_pos) if r is not None else "--")
    for ep in ("sev3", "bin_t1", "bin_t2"):
        for bal in ("natural", "balanced"):
            for s in ("eeg", "audio", "text", "all"):
                q = cells[(cells.endpoint == ep) & (cells.balance == bal) & (cells.silo == s)] if not cells.empty else cells
                r = q.iloc[0] if len(q) else None
                base = f"F{EP_CODE[ep]}{BAL_CODE[bal]}{S_CODE[s]}"
                M[base + "Delta"] = F.f(r["delta"] if r is not None else np.nan, 3, True)
                M[base + "Lo"] = F.f(r.lo if r is not None else np.nan, 3, True)
                M[base + "Hi"] = F.f(r.hi if r is not None else np.nan, 3, True)
    for name, *_ in CONTRASTS:
        for s in ("eeg", "audio", "text", "all"):
            q = contr[(contr.contrast == name) & (contr.silo == s)] if not contr.empty else contr
            r = q.iloc[0] if len(q) else None
            base = f"Fc{name}{S_CODE[s]}"
            M[base + "Diff"] = F.f(r["diff"] if r is not None else np.nan, 3, True)
            M[base + "Lo"] = F.f(r.lo if r is not None else np.nan, 3, True)
            M[base + "Hi"] = F.f(r.hi if r is not None else np.nan, 3, True)
            M[base + "P"] = F.p(r.p_holm if r is not None else np.nan)
    with open(path, "w") as fh:
        fh.write("% auto-generated by 3_FL_experiments.ipynb -- do not edit by hand\n")
        for k, v in M.items():
            assert k.isalpha(), k
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return M


def write_hparam_table(hp, gate_target, lr_grid, path, lr_counts=None):
    lrs = ", ".join(f"{v:g}" for v in lr_grid)
    rows = [
        ("Projection head $f_s$", rf"$d_s\!\to\!{hp['HIDDEN_DIM']}\!\to\!{hp['HIDDEN_DIM']}\!\to\!{hp['SHARED_DIM']}$; "
                                  rf"Linear--LayerNorm--GELU--Dropout({hp['HEAD_DROPOUT']}) blocks; output L2-normalised"),
        ("Classifier head", rf"${hp['SHARED_DIM']}\!\to\!{hp['HIDDEN_DIM']}\!\to\!K$, LayerNorm, GELU, Dropout({hp['HEAD_DROPOUT']}); local, never shared"),
        ("Input standardisation", "z-score per silo with training-split statistics"),
        ("Optimiser", rf"AdamW, weight decay $10^{{-4}}$, gradient clipping {hp['GRAD_CLIP']}"),
        ("Learning rate", rf"selected per partition and method from \{{{lrs}\}} on pooled validation macro-F1"),
        ("Rounds / local epochs / batch", rf"{hp['N_ROUNDS']} / {hp['LOCAL_EPOCHS']} / {hp['BATCH_SIZE']}; early stop after {hp['EARLY_STOP_PATIENCE']} rounds without validation gain"),
        ("Prototype loss", rf"temperature $\tau={hp['PROTO_TEMPERATURE']}$, inverse-frequency class weights"),
        ("Compactness term", rf"$\lambda_{{\mathrm{{comp}}}}={hp['COMPACT_LAMBDA']}$, supervised-contrastive temperature 0.07"),
        ("Classification loss", rf"focal, $\gamma={hp['FOCAL_GAMMA']}$, weight {hp['CE_WEIGHT']}, inverse-frequency class weights"),
        ("Augmentation", rf"Gaussian embedding noise $\sigma={hp['AUGMENT_NOISE_STD']}$; mixup $\alpha={hp['MIXUP_ALPHA']}$ (classifier path only)"),
        ("FedTGP server", rf"margin $m={hp['TGP_MARGIN']}$ (mean over competing prototypes), Adam lr $10^{{-3}}$, one step per round"),
        ("Reliability weights $r_k^{(s)}$", "per-class validation F1 of nearest-prototype prediction"),
        ("Gate initial value $g_s^{(0)}$", ", ".join(f"{k.upper() if k=='eeg' else k.capitalize()} {v:g}" for k, v in hp["GATE_INIT"].items())),
        ("Gate target", "; ".join(
            (f"{k.upper() if k=='eeg' else k.capitalize()} $\\min({m:g}\\,\\bar r_s,{c:g})$" if c is not None
             else f"{k.upper() if k=='eeg' else k.capitalize()} $\\bar r_s$") for k, (m, c) in gate_target.items())
         + rf"; $\bar r_s$ clipped to [{hp['GATE_MIN']}, {hp['GATE_MAX']}]; EMA momentum {hp['GATE_ADAPT_MOMENTUM']}"),
        ("Binary decision threshold", rf"tuned per silo on validation macro-F1 over {hp['THRESHOLD_GRID']} values in [0.05, 0.95]"),
    ]
    L = [r"\begin{tabular}{p{0.30\textwidth}p{0.64\textwidth}}", r"\toprule",
         r"\textbf{Component} & \textbf{Setting} \\", r"\midrule"]
    L += [f"{a} & {b} \\\\" for a, b in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(L))


def export_all(runs, out, endpoints, placeholder=False, tune_seed=0):
    os.makedirs(out, exist_ok=True)
    F = Fmt(placeholder)
    final = select_final(runs, tune_seed)
    pt = partition_table(final)
    cmp = all_comparisons(pt, endpoints)
    cells, contr = factor_analysis(pt)
    pc = perclass_table(final)
    if not placeholder:
        pt.to_csv(f"{out}/per_partition_scores.csv", index=False)
        cmp.to_csv(f"{out}/comparisons.csv", index=False)
        cells.to_csv(f"{out}/factor_cells.csv", index=False)
        contr.to_csv(f"{out}/factor_contrasts.csv", index=False)
        pc.to_csv(f"{out}/perclass_sev3.csv", index=False)
    meth5 = ["standalone", "local_proto", "fedproto", "local_tgp", "fedtgp"]
    present = set(pt.endpoint.unique())
    for ep in ("bin_ea", "bin_t1", "bin_t2", "sev3", "bin_mixed"):
        if ep in present:
            open(f"{out}/tab_{ep}.tex", "w").write(table_endpoint(F, pt, ep, endpoints, meth5))
            open(f"{out}/tab_stats_{ep}.tex", "w").write(table_stats(F, cmp, ep))
    open(f"{out}/tab_factor.tex", "w").write(table_factor(F, cells, contr))
    open(f"{out}/tab_perclass_sev3.tex", "w").write(table_perclass(F, pc))
    open(f"{out}/tab_individual.tex", "w").write(table_individual(F, pt, endpoints, meth5))
    write_macros(F, pt, cmp, cells, contr, f"{out}/results_macros.tex", None, endpoints)
    return pt, cmp, cells, contr
