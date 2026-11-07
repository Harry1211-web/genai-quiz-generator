"""Main experiment: every model x every prompting strategy on the same documents.

Crash-safe: every (config, strategy, document) "cell" runs in its own folder under
reports/eval/runs/ and saves its work after EACH step (see quizgen/pipeline.py):
  generated.jsonl - after each section is generated
  verified.jsonl  - after each question is verified
  judged.jsonl    - after each question is judged
  cell_stats.json - token/cost/time totals of the cell, accumulated over runs
Run the same command again after a crash or Ctrl+C: finished work is skipped and nothing paid
is paid twice. Use --regenerate to get a NEW set of questions: previous cell folders are moved to
reports/eval/archive/<timestamp>/ (never deleted).

Outputs (rebuilt from the cell folders at the end of every run):
  reports/eval/questions.csv      - every generated question (including rejected ones), with a stable id
  reports/eval/labeling_sheet.csv - HUMAN rating sheet: shuffled, model/strategy HIDDEN (blind rating).
                                    Ratings you already typed are KEPT when the sheet is rebuilt.
  reports/eval/key.csv            - id -> model/strategy key (do not open while rating)
  reports/eval/judge_labels.csv   - LLM-as-judge scores
  reports/eval/comparison.md      - comparison table

Models are numbered in .env (LLM_MODEL_1, LLM_MODEL_2, ...). eval/configs.json says which model
generates and which verifies in each configuration (see eval/configs.example.json); --judge picks the
judge, the same for every configuration. --recheck-quotes verifies again only the questions that were
rejected as quote_not_found (after the quote check was improved), without generating anything again.

Run: python -m eval.run_eval data/private/eval_set --n 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from quizgen.extract import load_sections
from quizgen.generate import gen_prompt_version, sections_version
from quizgen.llm import (
    DEFAULT_ROLES,
    LLM,
    BudgetExceeded,
    LLMRequestError,
    build_llm,
    model_for_role,
    model_from_env,
)
from quizgen.pipeline import (
    GENERATED,
    VERIFIED,
    _append,
    _read_jsonl,
    archive_previous,
    drop_rejections,
    log_stderr,
    run,
)
from quizgen.schema import RUBRIC_KEYS

from .judge import judge_one
from .kappa import read_sheet, write_sheet

JUDGED = "judged.jsonl"
CELL_STATS = "cell_stats.json"
DOC_SUFFIXES = {".pdf", ".pptx", ".txt", ".md"}


def make_llm(spec) -> LLM:
    """A model for an experiment role: the number of a model declared in .env (1, 2, ... -> LLM_MODEL_<n>),
    "fake", or an inline dict with the same keys as .env (model, base_url, api_key_env, price_in, ...)."""
    if isinstance(spec, (int, str)):
        return model_from_env(spec)
    s = {k: v for k, v in spec.items() if k not in ("name", "api_key_env")}
    if spec.get("api_key_env"):
        s["api_key"] = os.getenv(spec["api_key_env"], "")
    return build_llm(s, name=spec.get("name", "fake"))


def config_models(cfg: dict) -> tuple[LLM, LLM]:
    """(generator, verifier) of one configuration: {"name": ..., "generator": 1, "verifier": 2}.
    Older configs that put the generator's settings inline (model, base_url, ...) still work."""
    gen_spec = cfg.get("generator", cfg if "model" in cfg else DEFAULT_ROLES["generator"])
    ver_spec = cfg.get("verifier", DEFAULT_ROLES["verifier"])
    return make_llm(gen_spec), make_llm(ver_spec)


def stable_id(config: str, strategy: str, doc: str, qid: str, question: str) -> str:
    """Short id that does not reveal the configuration (blind rating). It includes the question text,
    so a regenerated question never inherits the rating of the question it replaced, while an
    unchanged question keeps its id (and rating) across runs."""
    key = f"{config}|{strategy}|{doc}|{qid}|{question}"
    return "q" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _usage(m: LLM | None) -> tuple:
    if m is None:
        return (0, 0, 0, 0.0, 0.0)
    u = m.usage
    return (u.calls, u.prompt_tokens, u.completion_tokens, m.cost_usd(), u.seconds)


def _delta(after: tuple, before: tuple) -> tuple:
    return tuple(a - b for a, b in zip(after, before, strict=True))


def judge_cell(judge: LLM, qs, sections, cell: Path, log) -> dict[str, dict]:
    """Judge accepted questions, saving after each one. Network errors are retried on the next run;
    a judge answer that is still invalid JSON after retries is recorded and not paid for again."""
    path = cell / JUDGED
    done = {r["qid"]: r for r in _read_jsonl(path)}
    by_title = {s.title: s for s in sections}
    todo = [q for q in qs if q.ok and q.qid not in done]
    for i, q in enumerate(todo, 1):
        try:
            js = judge_one(judge, q, by_title[q.section_title])
        except BudgetExceeded:
            raise
        except (httpx.HTTPError, LLMRequestError) as e:
            log(f"[judge] {q.qid}: {type(e).__name__} - will retry on the next run")
            continue
        rec = {"qid": q.qid, **({k: getattr(js, k) for k in RUBRIC_KEYS} if js else {"error": "format"})}
        _append(path, rec)
        done[q.qid] = rec
        log(f"[judge {i}/{len(todo)}] {q.qid}: {'error' if js is None else sum(rec[k] for k in RUBRIC_KEYS)}")
    return done


def prune_changed_sections(cell: Path, sections, log) -> int:
    """After the extraction changed, keep the questions of the sections that did NOT change and forget
    the others, so only the changed sections are generated again (and removed ones disappear).
    A section is unchanged when its index, title and pages are the same. The previous files are
    archived first. Use it when sections were removed or trimmed; after a change that alters the text
    of every section (e.g. a new PDF text extractor), regenerate everything instead."""
    sig = {i: (s.title, s.page_start, s.page_end) for i, s in enumerate(sections)}

    def unchanged(r: dict) -> bool:
        i = r.get("section_index")
        if i not in sig:
            return False
        if r.get("empty"):
            return r.get("section") == sig[i][0]
        return (r.get("section"), r.get("page_start"), r.get("page_end")) == sig[i]

    gen = _read_jsonl(cell / GENERATED)
    keep = [r for r in gen if unchanged(r)]
    if len(keep) == len(gen):
        return 0
    qids = {r.get("qid") for r in keep}
    kept = {name: [r for r in _read_jsonl(cell / name) if r.get("qid") in qids]
            for name in (VERIFIED, JUDGED)}
    dest = archive_previous(cell, (GENERATED, VERIFIED, JUDGED))
    for r in keep:
        _append(cell / GENERATED, r)
    for name, rows in kept.items():
        for r in rows:
            _append(cell / name, r)
    dropped = sum(1 for r in gen if not unchanged(r) and not r.get("empty"))
    log(f"[update-sections] {dropped} questions of changed or removed sections forgotten "
        f"(previous files in {dest}); unchanged sections are kept")
    return dropped


def update_cell_stats(cell: Path, gen: tuple, ver: tuple, jud: tuple, n_questions: int,
                      n_accepted: int) -> dict:
    """Add this run's usage to the cell's running totals (resumed runs keep counting)."""
    path = cell / CELL_STATS
    s = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for prefix, d in (("gen", gen), ("verify", ver), ("judge", jud)):
        for name, v in zip(("calls", "tokens_in", "tokens_out", "cost_usd", "seconds"), d, strict=True):
            s[f"{prefix}_{name}"] = round(s.get(f"{prefix}_{name}", 0) + v, 6)
    s.update(questions=n_questions, accepted=n_accepted)
    path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def merge_labeling_sheet(path: Path, new: pd.DataFrame, log=print) -> pd.DataFrame:
    """Keep every rating already typed in the sheet; append rows for new questions only.
    Rows of questions that no longer exist (after --regenerate) are removed from the sheet; if they
    were already rated, they are saved to labeling_sheet.orphaned.csv so no rating work is lost."""
    if not path.exists():
        return new.sample(frac=1, random_state=42)
    old = read_sheet(path, dtype=str).fillna("")
    shutil.copyfile(path, path.with_suffix(".backup.csv"))
    gone = old[~old["id"].isin(new["id"])]
    rated_gone = gone[(gone[RUBRIC_KEYS] != "").any(axis=1)]
    if len(rated_gone):
        orphan_path = path.with_name("labeling_sheet.orphaned.csv")
        prev = read_sheet(orphan_path, dtype=str).fillna("") if orphan_path.exists() else None
        both = rated_gone if prev is None else pd.concat([prev, rated_gone]).drop_duplicates("id")
        write_sheet(both, orphan_path)
        log(f"[sheet] {len(rated_gone)} rated questions no longer exist -> kept in {orphan_path.name}")
    kept = old[old["id"].isin(new["id"])]
    added = new[~new["id"].isin(old["id"])].sample(frac=1, random_state=42)
    if len(added) or len(gone):
        log(f"[sheet] {len(kept)} kept, {len(added)} new questions added, {len(gone)} removed")
    return pd.concat([kept, added.astype(str)], ignore_index=True)


ROLES = ("gen", "verify", "judge")
USAGE = ("calls", "tokens_in", "tokens_out", "cost_usd", "seconds")


def build_tables(st: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Quality table, cost/token table (with a TOTAL row) and run totals from the per-cell stats."""
    usage_cols = [f"{r}_{u}" for r in ROLES for u in USAGE]
    for c in usage_cols:
        if c not in st:
            st[c] = 0
    keys = ["config", "strategy"]
    g = st.groupby(keys)[usage_cols + ["questions", "accepted"]].sum().reset_index()

    quality = g[keys + ["questions", "accepted"]].copy()
    quality["rejected_rate"] = (1 - quality["accepted"] / quality["questions"].clip(lower=1)).round(3)
    if "judge_factual" in df:
        jm = df.groupby(keys)[[f"judge_{k}" for k in RUBRIC_KEYS]].mean().round(3).reset_index()
        quality = quality.merge(jm, on=keys, how="left")

    cost = g[keys + ["questions"]].copy()
    for r in ROLES:
        cost[f"{r}_tokens"] = g[f"{r}_tokens_in"] + g[f"{r}_tokens_out"]
        cost[f"{r}_usd"] = g[f"{r}_cost_usd"]
    cost["total_tokens"] = sum(cost[f"{r}_tokens"] for r in ROLES)
    cost["total_usd"] = sum(cost[f"{r}_usd"] for r in ROLES)
    for r in ROLES:
        cost[f"{r}_min"] = g[f"{r}_seconds"] / 60
    total_row = {c: cost[c].sum() for c in cost.columns if c not in keys}
    total = pd.DataFrame([{"config": "TOTAL", "strategy": "", **total_row}])
    cost = pd.concat([cost, total], ignore_index=True)
    cost["usd_per_100q"] = 100 * cost["total_usd"] / cost["questions"].clip(lower=1)
    for c in cost.columns:
        if c.endswith("_usd") or c == "usd_per_100q":
            cost[c] = cost[c].round(4)
        elif c.endswith("_min"):
            cost[c] = cost[c].round(1)
        elif c not in keys:
            cost[c] = cost[c].astype(int)

    totals = {c: float(round(st[c].sum(), 6)) for c in usage_cols}
    totals["questions"], totals["accepted"] = int(st["questions"].sum()), int(st["accepted"].sum())
    totals["total_tokens"] = int(sum(st[f"{r}_tokens_in"].sum() + st[f"{r}_tokens_out"].sum() for r in ROLES))
    totals["total_cost_usd"] = round(float(sum(st[f"{r}_cost_usd"].sum() for r in ROLES)), 6)
    totals["by_config"] = {cfg: round(float(sum(grp[f"{r}_cost_usd"].sum() for r in ROLES)), 6)
                           for cfg, grp in st.groupby("config")}
    return quality, cost, totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", help="folder of documents (pdf/pptx/txt/md)")
    ap.add_argument("--configs", default="eval/configs.json")
    ap.add_argument("--strategies", default="v1,v2,v3")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--exclude-sections", default=None, metavar="REGEX",
                    help="skip sections whose title matches (e.g. administrative slides, course handout)")
    ap.add_argument("--max-sections", type=int, default=None,
                    help="limit sections per document (trial runs)")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge", default=DEFAULT_ROLES["judge"],
                    help="number of the model in .env (LLM_MODEL_<n>) used as LLM-as-judge")
    ap.add_argument("--regenerate", "--fresh", dest="regenerate", action="store_true",
                    help="generate a NEW set of questions; previous runs are moved to <out>/archive/ "
                         "(ratings already typed are kept)")
    ap.add_argument("--recheck-quotes", action="store_true",
                    help="verify again the questions rejected as quote_not_found (nothing is regenerated)")
    ap.add_argument("--update-sections", action="store_true",
                    help="after the extraction changed: keep questions of unchanged sections, generate only "
                         "the changed ones (ratings of kept questions stay valid)")
    a = ap.parse_args()
    log = log_stderr

    configs = json.loads(Path(a.configs).read_text(encoding="utf-8")) if Path(a.configs).exists() else \
        [{"name": "fake-large", "generator": "fake", "verifier": "fake"},
         {"name": "fake-small", "generator": "fake", "verifier": "fake"}]
    # skip Office lock files ("~$name.pptx", created while the file is open) and hidden files
    files = sorted(p for p in Path(a.docs).iterdir()
                   if p.suffix.lower() in DOC_SUFFIXES and not p.name.startswith(("~$", ".")))
    exclude = re.compile(a.exclude_sections, re.IGNORECASE) if a.exclude_sections else None
    docs = {}
    for f in files:
        secs = load_sections(f)
        if exclude:
            skipped = [s.title for s in secs if exclude.search(s.title)]
            secs = [s for s in secs if not exclude.search(s.title)]
            for t in skipped:
                log(f"[exclude] {f.name}: {t[:80]}")
        docs[f.name] = secs[: a.max_sections] if a.max_sections else secs
    judge = None if a.no_judge else model_for_role("judge", a.judge)
    out = Path(a.out)
    runs = out / "runs"
    if a.regenerate and runs.exists():
        dest = out / "archive" / time.strftime("%Y%m%d-%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(runs), dest / "runs")
        log(f"[regenerate] previous runs moved to {dest}")

    rows, stats = [], []
    for cfg in configs:
        llm, verifier = config_models(cfg)
        if verifier.name == llm.name:
            log(f"[warn] {cfg['name']}: verifier is the same model as the generator ({llm.name})")
        for strat in a.strategies.split(","):
            for fname, secs in docs.items():
                cell = runs / cfg["name"] / strat / Path(fname).stem
                meta = {"model": llm.name, "verifier": verifier.name, "n": a.n,
                        "max_sections": a.max_sections, "exclude_sections": a.exclude_sections,
                        "gen_prompt": gen_prompt_version(strat), "sections": sections_version(secs)}
                cell.mkdir(parents=True, exist_ok=True)
                meta_path = cell / "cell_meta.json"
                old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else meta
                diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
                if diff and a.update_sections and set(diff) == {"sections"}:
                    log(f"\n[update-sections] {cell}")
                    prune_changed_sections(cell, secs, log)
                elif diff:
                    sys.exit(f"{cell} was run with different settings {diff} (gen_prompt = the generator "
                             "prompt changed, sections = the extraction changed). "
                             "Use --regenerate or another --out (or --update-sections if only the "
                             "sections changed).")
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                log(f"\n=== {cfg['name']} | {strat} | {fname} "
                    f"(generator: {llm.name}, verifier: {verifier.name})")
                if a.recheck_quotes:
                    dropped = drop_rejections(cell, "quote_not_found")
                    if dropped:
                        log(f"[recheck] {dropped} quote_not_found questions will be verified again")

                b_gen, b_ver, b_jud = _usage(llm), _usage(verifier), _usage(judge)
                qs, _ = run(secs, llm, verifier, a.n, strat, use_st=False, log=log, out_dir=cell)
                judged = judge_cell(judge, qs, secs, cell, log) if judge else {}
                cs = update_cell_stats(cell, _delta(_usage(llm), b_gen), _delta(_usage(verifier), b_ver),
                                       _delta(_usage(judge), b_jud), len(qs), sum(q.ok for q in qs))
                stats.append({"config": cfg["name"], "strategy": strat, "doc": fname, **cs})
                for q in qs:
                    d = q.to_dict()
                    qid = stable_id(cfg["name"], strat, fname, q.qid, q.mcq.question)
                    d.update(id=qid, config=cfg["name"], doc=fname)
                    j = judged.get(q.qid)
                    if j and "error" not in j:
                        d.update({f"judge_{k}": j[k] for k in RUBRIC_KEYS})
                    rows.append(d)

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[["id"] + [c for c in df.columns if c != "id"]]
    write_sheet(df, out / "questions.csv")

    # Human rating sheet: only verified questions, configuration hidden (blind rating)
    ok = df[df["rejected"].fillna("") == ""]
    sheet = ok[["id", "doc", "section", "question", "options", "answer_index", "explanation", "level",
                "source_quote"]].copy()
    sheet["options"] = sheet["options"].apply(
        lambda o: " | ".join(f"{'ABCD'[i]}. {x}" for i, x in enumerate(o)))
    sheet["answer"] = sheet.pop("answer_index").apply(lambda i: "ABCD"[i])
    for k in RUBRIC_KEYS:
        sheet[k] = ""
    merged = merge_labeling_sheet(out / "labeling_sheet.csv", sheet, log=log)
    write_sheet(merged, out / "labeling_sheet.csv")
    write_sheet(ok[["id", "config", "strategy"]], out / "key.csv")
    if "judge_factual" in df:
        jl = df.dropna(subset=["judge_factual"])[["id"] + [f"judge_{k}" for k in RUBRIC_KEYS]]
        jl.columns = ["id"] + RUBRIC_KEYS
        write_sheet(jl.astype({k: int for k in RUBRIC_KEYS}), out / "judge_labels.csv")

    # Comparison tables per (config, strategy), summed over documents, + run totals
    quality, cost, totals = build_tables(pd.DataFrame(stats), df)
    text = ("## Quality (automatic checks + judge)\n\n" + quality.to_markdown(index=False)
            + "\n\n## Cost and tokens (summed over documents and runs)\n\n" + cost.to_markdown(index=False)
            + "\n")
    (out / "comparison.md").write_text(text, encoding="utf-8")
    (out / "totals.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(text)
    print(f"Total: {totals['total_tokens']:,} tokens, ${totals['total_cost_usd']:.4f} "
          f"(generator ${totals['gen_cost_usd']:.4f}, verifier ${totals['verify_cost_usd']:.4f}, "
          f"judge ${totals['judge_cost_usd']:.4f})")
    print(f"\nWrote {out}/ - rate labeling_sheet.csv by hand (0/1 in each rubric column).")

if __name__ == "__main__":
    main()
