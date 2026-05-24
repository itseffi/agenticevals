from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import Settings
from .environments import EnvironmentOptions, load_environment
from .data_loop import write_dataset, write_improvement_loop
from .environment_baselines import run_environment_baselines
from .export import export_data, export_trajectories
from .agents.factory import adapter_status
from .authoring import scaffold_task, validate_task_quality
from .baselines import run_baselines
from .calibration import calibrate_judge_file, write_calibration_set
from .recompute import recompute_rewards
from .release_gate import evaluate_release_gate
from .review_cli import filtered_review_rows, format_review_rows
from .runner import run_task
from .schema import TaskSpec
from .suites import run_suite
from .trials import run_trials
from .canonical import write_normalized_jsonl
from .viewer import write_review, write_viewer
from .verify import verify_install, verify_live_adapters


def _load_task(path: str) -> TaskSpec:
    return TaskSpec.from_file(Path(path).expanduser().resolve())


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    task = _load_task(args.task)
    if args.trials and args.trials > 1:
        result = run_trials(task, settings, agent_override=args.agent, trials=args.trials, use_sandbox_server=args.sandbox_server)
        summary = result.to_dict()
        status = "PASS" if summary["pass_power_k"] else "FAIL"
        print(
            f"{status} {task.id} pass^{args.trials}={summary['pass_power_k']} "
            f"pass@1={summary['pass_at_1']:.3f} mean_score={summary['mean_score']:.3f}"
        )
        print(f"run_dir: {result.run_dir}")
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["pass_power_k"] else 2
    result = run_task(task, settings, agent_override=args.agent, use_sandbox_server=args.sandbox_server)
    status = "PASS" if result.score.passed else "FAIL"
    print(f"{status} {task.id} {result.score.points:.1f}/{result.score.max_points:.1f}")
    print(f"run_dir: {result.run_dir}")
    print(f"trace: {result.trace_path}")
    print(f"report: {result.report_path}")
    if args.json:
        print(json.dumps({"passed": result.score.passed, "run_dir": str(result.run_dir)}, indent=2))
    return 0 if result.score.passed else 2


def cmd_validate(args: argparse.Namespace) -> int:
    task = _load_task(args.task)
    fixture = task.resolve_fixture()
    if not fixture.exists():
        print(f"INVALID fixture does not exist: {fixture}", file=sys.stderr)
        return 2
    print(f"VALID {task.id}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    directory = Path(args.dir).resolve() if args.dir else settings.task_config_dir
    for path in sorted(directory.glob("*.json")):
        try:
            task = TaskSpec.from_file(path)
            print(f"{task.id}\t{path}\t{task.title}")
        except Exception as exc:
            print(f"INVALID\t{path}\t{exc}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for run_dir in args.run_dirs:
        score_path = Path(run_dir) / "score.json"
        if not score_path.exists():
            print(f"missing score: {score_path}", file=sys.stderr)
            return 2
        score = json.loads(score_path.read_text(encoding="utf-8"))
        rows.append((run_dir, score["passed"], score["points"], score["max_points"]))
    for run_dir, passed, points, max_points in rows:
        print(f"{'PASS' if passed else 'FAIL'}\t{points:.1f}/{max_points:.1f}\t{run_dir}")
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    env = load_environment(args.environment, settings=settings)
    env.setup()
    item = env.get_next_item()
    if item is None:
        print(f"Environment has no items: {args.environment}", file=sys.stderr)
        return 2
    result = env.rollout(
        item,
        EnvironmentOptions(agent=args.agent, max_minutes=args.max_minutes, backend=args.backend, image=args.image),
    )
    status = "PASS" if result.reward.passed else "FAIL"
    print(f"{status} {result.environment}/{result.item_id} {result.reward.value:.3f}/{result.reward.max_value:.3f}")
    print(f"run_dir: {result.run_dir}")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.reward.passed else 2


def cmd_evaluate(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    env = load_environment(args.environment, settings=settings)
    result = env.evaluate(
        EnvironmentOptions(
            max_items=args.max_items,
            agent=args.agent,
            max_minutes=args.max_minutes,
            resume=Path(args.resume).expanduser().resolve() if args.resume else None,
            backend=args.backend,
            image=args.image,
        )
    )
    summary = result.to_dict()
    status = "PASS" if summary["passed"] == summary["total"] else "FAIL"
    print(
        f"{status} {result.environment} pass_rate={summary['pass_rate']:.1%} "
        f"reward={summary['reward']:.3f}/{summary['max_reward']:.3f}"
    )
    print(f"run_dir: {result.run_dir}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] == summary["total"] else 2


def cmd_export_trajectories(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    target = export_trajectories(Path(args.run_dir), output=output)
    print(f"trajectories: {target}")
    return 0


def cmd_export_data(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    target = export_data(Path(args.run_dir), output=output, kind=args.format, compress=args.gzip)
    print(f"{args.format}: {target}")
    return 0


def cmd_export_dataset(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    manifest = write_dataset(Path(args.run_dir), output_dir=output, name=args.name)
    target = output if output else Path(args.run_dir).expanduser().resolve() / "dataset"
    print(f"dataset: {target}")
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_recompute_rewards(args: argparse.Namespace) -> int:
    result = recompute_rewards(Path(args.run_dir))
    status = "PASS" if result.get("matches_previous", True) else "FAIL"
    print(f"{status} recompute method={result.get('method')} recomputed={result.get('recomputed')}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("matches_previous", True) else 2


def cmd_improve_loop(args: argparse.Namespace) -> int:
    result = write_improvement_loop(Path(args.run_dir), output_dir=Path(args.output) if args.output else None)
    print(f"improvement-loop: {result['output_dir']}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    target = write_normalized_jsonl(Path(args.run_dir), output=output)
    print(f"normalized: {target}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    target = write_viewer(Path(args.run_dir), output=output)
    print(f"viewer: {target}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    if args.filter:
        rows = filtered_review_rows(Path(args.run_dir), args.filter, limit=args.limit)
        print(format_review_rows(rows))
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    output = Path(args.output) if args.output else None
    target = write_review(Path(args.run_dir), output=output)
    print(f"review: {target}")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    summary = run_suite(Path(args.suite), settings, agent_override=args.agent, workers=args.workers, resume=Path(args.resume) if args.resume else None)
    status = "PASS" if summary["passed"] == summary["total"] else "FAIL"
    print(f"{status} {summary['id']} pass_rate={summary['pass_rate']:.1%} mean_score={summary['mean_score']:.3f}")
    print(f"run_dir: {summary['run_dir']}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] == summary["total"] else 2


def cmd_baselines(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    result = run_baselines(Path(args.suite), settings, agents=agents, workers=args.workers, output=Path(args.output) if args.output else None)
    print(f"baselines: {result['run_dir']}")
    for row in result["rows"]:
        ci = row["pass_rate_ci"]
        print(f"{row['agent']}\tpass_rate={row['pass_rate']:.3f}\tci=[{ci['low']:.3f},{ci['high']:.3f}]\tmean_score={row['mean_score']:.3f}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_env_baselines(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    result = run_environment_baselines(
        args.environment,
        settings,
        agents=agents,
        max_items=args.max_items,
        trials=args.trials,
        backend=args.backend,
        image=args.image,
        max_minutes=args.max_minutes,
        output=Path(args.output) if args.output else None,
    )
    print(f"environment baselines: {result['run_dir']}")
    for row in result["rows"]:
        ci = row["pass_at_1_ci"]
        cost = "n/a" if row["cost_per_success_usd"] is None else f"${row['cost_per_success_usd']:.4f}"
        print(
            f"{row['agent']}\tpass@1={row['pass_at_1']:.3f}\tpass^{row['trials']}={row['pass_power_k']:.3f}\t"
            f"ci=[{ci['low']:.3f},{ci['high']:.3f}]\tcost_per_success={cost}"
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_calibrate_judge(args: argparse.Namespace) -> int:
    result = calibrate_judge_file(Path(args.labels), output=Path(args.output) if args.output else None)
    print(f"calibration: n={result['n']} kappa={result['kappa']:.3f}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["kappa"] >= args.min_kappa else 2


def cmd_make_calibration_set(args: argparse.Namespace) -> int:
    result = write_calibration_set(
        Path(args.run_dir),
        output=Path(args.output),
        sample_size=args.size,
        seed=args.seed,
    )
    print(f"calibration set: n={result['n']} (judge_passed={result['judge_passed']}, judge_failed={result['judge_failed']}) -> {result['output']}")
    print("Fill in human_passed for each row, then run: agenticevals calibrate-judge <file>")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_release_gate(args: argparse.Namespace) -> int:
    result = evaluate_release_gate(
        baselines_path=Path(args.baselines),
        calibration_path=Path(args.calibration) if args.calibration else None,
        min_agents=args.min_agents,
        min_kappa=args.min_kappa,
    )
    for check in result["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'}\t{check['name']}\t{check['detail']}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def cmd_new_task(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    path = scaffold_task(settings, task_id=args.id, kind=args.kind, force=args.force)
    print(f"task: {path}")
    return 0


def cmd_validate_quality(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    passed, issues = validate_task_quality(Path(args.task), settings, run=not args.no_run)
    for issue in issues:
        print(f"{'PASS' if issue.passed else 'FAIL'}\t{issue.name}\t{issue.detail}")
    if args.json:
        print(json.dumps({"passed": passed, "issues": [issue.to_dict() for issue in issues]}, indent=2, sort_keys=True))
    return 0 if passed else 2


def cmd_verify_install(args: argparse.Namespace) -> int:
    result = verify_install()
    for row in result["checks"]:
        print(f"{'PASS' if row['passed'] else 'FAIL'}\t{row['name']}\t{row['detail']}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def cmd_verify_adapters(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    result = verify_live_adapters(settings, task_path=Path(args.task) if args.task else None)
    for row in result["checks"]:
        print(f"{'PASS' if row['passed'] else 'FAIL'}\t{row['name']}\t{row['detail']}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def cmd_adapters(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    rows = adapter_status(settings)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    for row in rows:
        print(f"{row['name']}\t{row['kind']}\t{row['status']}")
    return 0


def cmd_sandbox_smoke(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PORT"] = str(args.port)
    env["AGENTICEVALS_SANDBOX_WORKSPACE"] = str(workspace)
    proc = subprocess.Popen([sys.executable, "-m", "agenticevals.sandbox.server"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"sandbox server exited before health check: {stderr.strip()}")
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("sandbox server did not become healthy")
        _post_json(base_url + "/write", {"path": "hello.txt", "content": "hello sandbox"})
        read = _post_json(base_url + "/read", {"path": "hello.txt"})
        page = workspace / "page.html"
        page.write_text("<!doctype html><title>Sandbox</title><main>browser action ready</main>", encoding="utf-8")
        goto = _post_json(base_url + "/browser/goto", {"url": page.as_uri(), "save_as": "artifacts/page.html"})
        check = _post_json(base_url + "/browser/check", {"url": page.as_uri(), "contains": "browser action ready"})
        ok = read.get("content") == "hello sandbox" and goto.get("status") == 200 and check.get("ok") is True
        print(f"{'PASS' if ok else 'FAIL'} sandbox-smoke workspace={workspace}")
        return 0 if ok else 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenticevals", description="Evaluate AI agent trajectories in controlled computer environments.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one agentic eval task")
    run.add_argument("task", help="Path to task JSON")
    run.add_argument("--agent", help="Override agent kind: scripted, command, codex, claude-code, openai, gemini, claude, http")
    run.add_argument("--trials", type=int, default=1, help="Run N independent trials and report pass^N")
    run.add_argument("--sandbox-server", action="store_true", help="Expose the workspace through the persistent sandbox HTTP interface")
    run.add_argument("--json", action="store_true", help="Print machine-readable summary")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="Validate one task config")
    validate.add_argument("task")
    validate.set_defaults(func=cmd_validate)

    list_cmd = sub.add_parser("list", help="List configured tasks")
    list_cmd.add_argument("--dir", help="Task directory; defaults to AGENTICEVALS_TASK_CONFIG_DIR")
    list_cmd.set_defaults(func=cmd_list)

    compare = sub.add_parser("compare", help="Compare completed run directories")
    compare.add_argument("run_dirs", nargs="+")
    compare.set_defaults(func=cmd_compare)

    rollout = sub.add_parser("rollout", help="Run one rollout from an environment class")
    rollout.add_argument("environment", help="Environment spec: module:ClassName")
    rollout.add_argument("--agent", default="scripted", help="Agent kind")
    rollout.add_argument("--max-minutes", type=int, default=20)
    rollout.add_argument("--backend", default="local", help="Execution backend: local, sandbox-http, or docker")
    rollout.add_argument("--image", help="Container image for docker backend")
    rollout.add_argument("--json", action="store_true")
    rollout.set_defaults(func=cmd_rollout)

    evaluate = sub.add_parser("evaluate", help="Evaluate an environment class")
    evaluate.add_argument("environment", help="Environment spec: module:ClassName")
    evaluate.add_argument("--agent", default="scripted", help="Agent kind")
    evaluate.add_argument("--max-items", type=int)
    evaluate.add_argument("--max-minutes", type=int, default=20)
    evaluate.add_argument("--resume", help="Existing eval run directory to continue")
    evaluate.add_argument("--backend", default="local", help="Execution backend: local, sandbox-http, or docker")
    evaluate.add_argument("--image", help="Container image for docker backend")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    export = sub.add_parser("export-trajectories", help="Export rollout records from a run directory as JSONL")
    export.add_argument("run_dir")
    export.add_argument("--output", "-o")
    export.set_defaults(func=cmd_export_trajectories)

    export_data_cmd = sub.add_parser("export-data", help="Export run data as trajectories, ShareGPT, actions, rewards, normalized, training, RL, or preference JSONL")
    export_data_cmd.add_argument("run_dir")
    export_data_cmd.add_argument("--format", choices=["trajectories", "sharegpt", "actions", "rewards", "normalized", "training", "rl", "preferences"], default="trajectories")
    export_data_cmd.add_argument("--output", "-o")
    export_data_cmd.add_argument("--gzip", action="store_true", help="Write gzip-compressed JSONL")
    export_data_cmd.set_defaults(func=cmd_export_data)

    dataset = sub.add_parser("export-dataset", help="Write RL data, preference data, hard negatives, manifest, and dataset card")
    dataset.add_argument("run_dir")
    dataset.add_argument("--output", "-o")
    dataset.add_argument("--name")
    dataset.add_argument("--json", action="store_true")
    dataset.set_defaults(func=cmd_export_dataset)

    recompute = sub.add_parser("recompute-rewards", help="Recompute task rewards from saved workspace artifacts when possible")
    recompute.add_argument("run_dir")
    recompute.add_argument("--json", action="store_true")
    recompute.set_defaults(func=cmd_recompute_rewards)

    improve = sub.add_parser("improve-loop", help="Mine failures into hard negatives and candidate regression evals")
    improve.add_argument("run_dir")
    improve.add_argument("--output", "-o")
    improve.add_argument("--json", action="store_true")
    improve.set_defaults(func=cmd_improve_loop)

    adapters = sub.add_parser("adapters", help="List built-in and external agent adapters")
    adapters.add_argument("--json", action="store_true")
    adapters.set_defaults(func=cmd_adapters)

    normalize = sub.add_parser("normalize", help="Write canonical normalized trace JSONL for a run directory")
    normalize.add_argument("run_dir")
    normalize.add_argument("--output", "-o")
    normalize.set_defaults(func=cmd_normalize)

    view = sub.add_parser("view", help="Write a self-contained HTML trajectory viewer for a run directory")
    view.add_argument("run_dir")
    view.add_argument("--output", "-o")
    view.set_defaults(func=cmd_view)

    review = sub.add_parser("review", help="Write a self-contained HTML suite review UI")
    review.add_argument("run_dir")
    review.add_argument("--output", "-o")
    review.add_argument("--filter", action="append", default=[], help="Print filtered run summaries instead of HTML; format key=value")
    review.add_argument("--limit", type=int, default=20)
    review.add_argument("--json", action="store_true")
    review.set_defaults(func=cmd_review)

    suite = sub.add_parser("suite", help="Run a JSON suite of task configs")
    suite.add_argument("suite")
    suite.add_argument("--agent", help="Override all suite task agents")
    suite.add_argument("--workers", type=int, default=1, help="Run suite tasks in parallel")
    suite.add_argument("--resume", help="Resume an existing suite run directory")
    suite.add_argument("--json", action="store_true")
    suite.set_defaults(func=cmd_suite)

    baselines = sub.add_parser("baselines", help="Run a suite for multiple agents and write baseline result artifacts")
    baselines.add_argument("suite")
    baselines.add_argument("--agents", required=True, help="Comma-separated agent kinds, e.g. openai,gemini,claude")
    baselines.add_argument("--workers", type=int, default=1)
    baselines.add_argument("--output", "-o")
    baselines.add_argument("--json", action="store_true")
    baselines.set_defaults(func=cmd_baselines)

    env_baselines = sub.add_parser("env-baselines", help="Run an environment for multiple agents and write baseline result artifacts")
    env_baselines.add_argument("environment", help="Environment spec: module:ClassName")
    env_baselines.add_argument("--agents", required=True, help="Comma-separated agent kinds, e.g. codex,claude-code,openai")
    env_baselines.add_argument("--max-items", type=int)
    env_baselines.add_argument("--trials", type=int, default=1, help="Independent attempts per item for pass^k")
    env_baselines.add_argument("--backend", default="local", help="Execution backend: local, sandbox-http, or docker")
    env_baselines.add_argument("--image", help="Container image for docker backend")
    env_baselines.add_argument("--max-minutes", type=int, default=20)
    env_baselines.add_argument("--output", "-o")
    env_baselines.add_argument("--json", action="store_true")
    env_baselines.set_defaults(func=cmd_env_baselines)

    calibrate = sub.add_parser("calibrate-judge", help="Compute judge agreement and Cohen kappa from labeled JSONL")
    calibrate.add_argument("labels")
    calibrate.add_argument("--output", "-o")
    calibrate.add_argument("--min-kappa", type=float, default=0.5)
    calibrate.add_argument("--json", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate_judge)

    make_calib = sub.add_parser("make-calibration-set", help="Sample llm_rubric judge decisions from runs into a balanced labeling template")
    make_calib.add_argument("run_dir", help="Run, suite, or trial directory to sample reward-details.json from")
    make_calib.add_argument("--output", "-o", required=True, help="Path to write the labeling JSONL")
    make_calib.add_argument("--size", type=int, default=100, help="Target sample size (balanced across judge verdicts)")
    make_calib.add_argument("--seed", type=int, default=0)
    make_calib.add_argument("--json", action="store_true")
    make_calib.set_defaults(func=cmd_make_calibration_set)

    gate = sub.add_parser("release-gate", help="Check v0.1 baseline and judge-calibration release criteria")
    gate.add_argument("--baselines", required=True, help="Path to baselines.json")
    gate.add_argument("--calibration", help="Path to calibration report JSON")
    gate.add_argument("--min-agents", type=int, default=3)
    gate.add_argument("--min-kappa", type=float, default=0.5)
    gate.add_argument("--json", action="store_true")
    gate.set_defaults(func=cmd_release_gate)

    new_task = sub.add_parser("new-task", help="Scaffold a new task fixture, hidden grader, and JSON config")
    new_task.add_argument("id", help="Task id")
    new_task.add_argument("--kind", default="hidden-grader", choices=["hidden-grader"])
    new_task.add_argument("--force", action="store_true", help="Overwrite an existing scaffold")
    new_task.set_defaults(func=cmd_new_task)

    quality = sub.add_parser("validate-task-quality", help="Run authoring quality checks for a task config")
    quality.add_argument("task")
    quality.add_argument("--no-run", action="store_true", help="Skip scripted/noop execution checks")
    quality.add_argument("--json", action="store_true")
    quality.set_defaults(func=cmd_validate_quality)

    verify_install_cmd = sub.add_parser("verify-install", help="Verify package import and CLI entrypoints")
    verify_install_cmd.add_argument("--json", action="store_true")
    verify_install_cmd.set_defaults(func=cmd_verify_install)

    verify_adapters = sub.add_parser("verify-adapters", help="Run live Codex and Claude Code adapter verification when CLIs are installed")
    verify_adapters.add_argument("--task", help="Task config to use for live adapter verification")
    verify_adapters.add_argument("--json", action="store_true")
    verify_adapters.set_defaults(func=cmd_verify_adapters)

    sandbox_smoke = sub.add_parser("sandbox-smoke", help="Start the local sandbox HTTP server and verify file operations")
    sandbox_smoke.add_argument("--port", type=int, default=18080)
    sandbox_smoke.add_argument("--workspace", default="/private/tmp/agenticevals-sandbox-smoke")
    sandbox_smoke.set_defaults(func=cmd_sandbox_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
