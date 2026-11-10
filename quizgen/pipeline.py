"""Glue for all steps: extract -> generate -> verify -> dedupe. Used by the CLI, the API and the UI.

Crash-safe CLI runs: with an output folder, every paid step is saved the moment it finishes.
  generated.jsonl - appended after EACH section is generated (a crash loses at most one section)
  verified.jsonl  - appended after EACH question is verified (with the verifier's answer and reason)
  run_meta.json   - the settings of the run (file, strategy, n, model)
Running the same command again RESUMES: sections already generated and questions already verified
are skipped, so no tokens are spent twice. Use --regenerate to get a NEW set of questions: the previous
results are moved to <out>/archive/<timestamp>/ (never deleted). Use --reverify to verify the SAME
questions again (e.g. after changing the verifier or its prompt) without paying for generation again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .dedupe import mark_duplicates
from .export import to_csv, to_gift, to_json
from .extract import Section, load_sections
from .generate import (
    STRATEGIES,
    GeneratedQuestion,
    gen_prompt_version,
    generate_for_section,
    load_prompt,
    sections_version,
    split_labels,
)
from .llm import DEFAULT_ROLES, LLM, model_for_role
from .verify import rejection_report, verify_all

GENERATED = "generated.jsonl"
VERIFIED = "verified.jsonl"
META = "run_meta.json"


def _quiet(msg: str) -> None:
    pass


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _append(path: Path, record: dict) -> None:
    """Append one JSON line and flush it to disk immediately."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a crash: ignore it
                pass
    return rows


OUTPUT_FILES = ("all_questions.jsonl", "quiz.json", "quiz.csv", "quiz.gift.txt", "stats.json")
RESULT_FILES = (GENERATED, VERIFIED, META, *OUTPUT_FILES)


def verify_prompt_version() -> str:
    """Short hash of the verifier prompt, saved with every verification for traceability."""
    return hashlib.sha1(load_prompt("verify.md").encode()).hexdigest()[:8]


def archive_previous(out_dir: Path, names: tuple[str, ...] = RESULT_FILES) -> Path | None:
    """Move earlier results into <out_dir>/archive/<timestamp>/ so a new run starts clean
    without destroying anything that was paid for."""
    present = [out_dir / n for n in names if (out_dir / n).exists()]
    if not present:
        return None
    dest = out_dir / "archive" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for p in present:
        shutil.move(str(p), dest / p.name)
    return dest


def _check_meta(out_dir: Path, meta: dict, regenerate: bool) -> Path | None:
    """Refuse to mix results of different settings in one folder.
    With regenerate=True the previous results are archived and a new run starts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / META
    archived = None
    if regenerate:
        archived = archive_previous(out_dir)
    elif meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{out_dir} holds results of a different run {diff} (gen_prompt = the generator "
                             "prompt changed, sections = the extraction changed). Use another --out folder, "
                             "or --regenerate to archive the old results and start a new run.")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return archived


def drop_rejections(out_dir: Path, label: str) -> int:
    """Forget the verification of questions rejected with `label` so the next run verifies them
    again (e.g. after the quote check was fixed). Generation is never repeated. The previous file
    is kept as verified.before-recheck.jsonl."""
    path = out_dir / VERIFIED
    rows = _read_jsonl(path)
    keep = [r for r in rows if label not in (r.get("rejected") or "").split(";")]
    if len(keep) == len(rows):
        return 0
    shutil.copyfile(path, out_dir / "verified.before-recheck.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    return len(rows) - len(keep)


def run(sections: list[Section], llm: LLM, verifier: LLM | None = None, n_per_section: int = 3,
        strategy: str = "v3", max_sections: int | None = None, use_st: bool = True,
        seed: int = 0, log: Callable[[str], None] = _quiet,
        out_dir: Path | None = None) -> tuple[list[GeneratedQuestion], dict]:
    """log: progress callback (e.g. log_stderr).
    out_dir: if given, progress is saved after every section / question and resumed on the next run."""
    t0 = time.perf_counter()
    sections = sections[:max_sections] if max_sections else sections
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / GENERATED if out_dir else None
    ver_path = out_dir / VERIFIED if out_dir else None

    # ---- generation (paid): resume from generated.jsonl, save after each section
    saved = _read_jsonl(gen_path) if gen_path else []
    done_sections = {r["section_index"] for r in saved}
    qs = [GeneratedQuestion.from_dict(r) for r in saved if not r.get("empty")]
    if done_sections:
        log(f"[resume] {len(done_sections)}/{len(sections)} sections already generated "
            f"({len(qs)} questions) - not calling the generator again for them")
    for i, sec in enumerate(sections):
        if i in done_sections:
            continue
        ts = time.perf_counter()
        new = generate_for_section(llm, sec, n=n_per_section, strategy=strategy, seed=seed + i)
        for k, q in enumerate(new):
            q.qid = f"{i}-{k}"
        if gen_path:
            for q in new:
                _append(gen_path, {"section_index": i, **q.to_dict()})
            if not new:  # remember failed sections too, so a resume does not pay for them again
                _append(gen_path, {"section_index": i, "empty": True, "section": sec.title})
        qs += new
        u = llm.usage
        log(f"[generate {i + 1}/{len(sections)}] {sec.title[:50]!r}: {len(new)} questions in "
            f"{time.perf_counter() - ts:.1f}s | {llm.name} this run: {u.prompt_tokens} in / "
            f"{u.completion_tokens} out tokens, ${llm.cost_usd():.4f}, format errors {u.format_errors}")
    qs.sort(key=lambda q: tuple(int(x) for x in q.qid.split("-")) if q.qid else (0, 0))

    # ---- verification: resume from verified.jsonl (failed calls are retried), save after each question
    previous = {r["qid"]: r for r in _read_jsonl(ver_path)} if ver_path else {}
    previous = {k: r for k, r in previous.items() if "verify_error" not in (r.get("rejected") or "")}
    for q in qs:
        if q.qid in previous:
            r = previous[q.qid]
            q.rejected, q.warnings = split_labels(r.get("rejected"), r.get("warnings"))
            q.verifier_answer, q.verifier_reason = r.get("verifier_answer"), r.get("verifier_reason", "")
    if previous:
        log(f"[resume] {len(previous)} questions already verified - skipping them")
    verified_now: set[str] = set()
    prompt_version = verify_prompt_version()

    def on_done(q: GeneratedQuestion) -> None:
        verified_now.add(q.qid)
        if ver_path:
            _append(ver_path, {"qid": q.qid, "rejected": ";".join(q.rejected),
                               "warnings": ";".join(q.warnings),
                               "verifier": verifier.name if verifier else None,
                               "verify_prompt": prompt_version,
                               "verifier_answer": q.verifier_answer, "verifier_reason": q.verifier_reason})

    by_title = {s.title: s for s in sections}
    verify_all(qs, by_title, verifier, log=log, on_done=on_done, skip=set(previous))
    for q in qs:  # never export a question as accepted if verification stopped before it
        if q.qid not in previous and q.qid not in verified_now:
            q.rejected = ["not_verified"]
    mark_duplicates(qs, use_st=use_st)

    stats = rejection_report(qs)
    stats.update(model=llm.name, strategy=strategy, sections=len(sections),
                 seconds=round(time.perf_counter() - t0, 2), llm_calls=llm.usage.calls,
                 prompt_tokens=llm.usage.prompt_tokens, completion_tokens=llm.usage.completion_tokens,
                 format_errors=llm.usage.format_errors, format_failures=llm.usage.format_failures,
                 verifier=verifier.name if verifier else None, verify_prompt=prompt_version,
                 verify_seconds=round(verifier.usage.seconds, 1) if verifier else 0,
                 cost_usd_this_run=round(llm.cost_usd() + (verifier.cost_usd() if verifier else 0), 4))
    return qs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate review questions from a lecture")
    ap.add_argument("file", help="PDF / PPTX / TXT / MD")
    ap.add_argument("--strategy", choices=STRATEGIES, default="v3")
    ap.add_argument("--n", type=int, default=3, help="questions per section")
    ap.add_argument("--max-sections", type=int)
    ap.add_argument("--no-verify", action="store_true", help="skip the answer-back verification (cheaper)")
    ap.add_argument("--generator", default=DEFAULT_ROLES["generator"],
                    help="number of the model in .env (LLM_MODEL_<n>) that writes the questions")
    ap.add_argument("--verifier", default=DEFAULT_ROLES["verifier"],
                    help="number of the model in .env (LLM_MODEL_<n>) that re-answers them")
    ap.add_argument("--out", default="reports/output")
    ap.add_argument("--regenerate", "--fresh", dest="regenerate", action="store_true",
                    help="generate a NEW set of questions; previous results are moved to <out>/archive/")
    ap.add_argument("--reverify", action="store_true",
                    help="keep the generated questions but verify them again "
                         "(previous verification is archived)")
    ap.add_argument("--recheck-quotes", action="store_true",
                    help="verify again only the questions rejected as quote_not_found "
                         "(after the quote check was improved); nothing is generated again")
    ap.add_argument("--quiet", action="store_true", help="do not print progress")
    a = ap.parse_args()

    llm = model_for_role("generator", a.generator)
    verifier = None if a.no_verify else model_for_role("verifier", a.verifier)
    out = Path(a.out)
    sections = load_sections(a.file)
    used = sections[: a.max_sections] if a.max_sections else sections
    archived = _check_meta(out, {"file": Path(a.file).name, "strategy": a.strategy, "n": a.n,
                                 "max_sections": a.max_sections, "model": llm.name,
                                 "gen_prompt": gen_prompt_version(a.strategy),
                                 "sections": sections_version(used)}, a.regenerate)
    if archived:
        print(f"[regenerate] previous results moved to {archived}", file=sys.stderr)
    if a.reverify and not a.regenerate:
        moved = archive_previous(out, (VERIFIED, *OUTPUT_FILES))
        if moved:
            print(f"[reverify] previous verification moved to {moved}; generated questions are reused",
                  file=sys.stderr)
    if a.recheck_quotes:
        print(f"[recheck] {drop_rejections(out, 'quote_not_found')} quote_not_found questions will be "
              "verified again", file=sys.stderr)
    print(f"{len(sections)} sections: " + ", ".join(s.title for s in sections[:8]), file=sys.stderr)
    qs, stats = run(sections, llm, verifier, a.n, a.strategy, a.max_sections,
                    log=_quiet if a.quiet else log_stderr, out_dir=out)
    good = [q.mcq for q in qs if q.ok]
    (out / "all_questions.jsonl").write_text(
        "".join(json.dumps(q.to_dict(), ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    (out / "quiz.json").write_text(to_json(good), encoding="utf-8")
    (out / "quiz.csv").write_text(to_csv(good), encoding="utf-8-sig")  # BOM: Excel shows UTF-8 correctly
    (out / "quiz.gift.txt").write_text(to_gift(good), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
