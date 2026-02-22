"""Click-based CLI for reels pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from reels.config import get_config
from reels.logging import setup_logging

console = Console()


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config YAML path")
@click.option("--work-dir", type=click.Path(), default="./work", help="Working directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--log-file", type=click.Path(), default=None, help="Log file path")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, work_dir: str, verbose: bool, log_file: str | None) -> None:
    """Reels Analyzer - Video structure extraction and template synthesis."""
    ctx.ensure_object(dict)

    cfg = get_config(Path(config_path) if config_path else None)
    setup_logging(
        level=cfg.get("pipeline", {}).get("log_level", "INFO"),
        log_file=Path(log_file) if log_file else None,
        verbose=verbose,
    )

    ctx.obj["config"] = cfg
    ctx.obj["work_dir"] = Path(work_dir)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("source")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--save-db", is_flag=True, help="Save template to DB")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--no-cache", is_flag=True, help="Disable cache")
@click.pass_context
def analyze(ctx: click.Context, source: str, output: str | None, save_db: bool, resume: bool, no_cache: bool) -> None:
    """Run full pipeline: ingest -> segment -> analyze -> synthesize."""
    from reels.pipeline import run_pipeline

    config = ctx.obj["config"]
    work_dir = Path(output) if output else ctx.obj["work_dir"]

    if no_cache:
        config.setdefault("pipeline", {})["cache_enabled"] = False

    console.print(f"[bold]Analyzing:[/bold] {source}")

    try:
        template = run_pipeline(
            source=source,
            work_dir=work_dir,
            config=config,
            save_to_db=save_db,
            resume=resume,
        )
        console.print(f"[green]Done![/green] Template: [bold]{template.template_id}[/bold]")
        console.print(f"  Shots: {template.shot_count}, Duration: {template.total_duration_sec:.1f}s")
        console.print(f"  Output: {work_dir / f'{template.template_id}.json'}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@cli.command()
@click.argument("source")
@click.pass_context
def ingest(ctx: click.Context, source: str) -> None:
    """Run ingest only."""
    from reels.ingest import ingest_video

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    console.print(f"[bold]Ingesting:[/bold] {source}")

    try:
        result = ingest_video(source, work_dir, config)
        console.print(f"[green]Done![/green] Normalized: {result.normalized_video}")
        console.print(f"  Duration: {result.metadata.duration_sec:.1f}s, Resolution: {result.metadata.resolution}")
        if result.audio_path:
            console.print(f"  Audio: {result.audio_path}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.pass_context
def segment(ctx: click.Context, video_path: str) -> None:
    """Run shot segmentation only."""
    from reels.ingest.probe import VideoProber
    from reels.segmentation import segment_video

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    console.print(f"[bold]Segmenting:[/bold] {video_path}")

    try:
        prober = VideoProber()
        metadata = prober.probe(Path(video_path))
        result = segment_video(Path(video_path), metadata, work_dir, config)

        console.print(f"[green]Done![/green] {result.total_shots} shots detected")
        for shot in result.shots:
            console.print(f"  Shot {shot.shot_id}: {shot.start_sec:.1f}s - {shot.end_sec:.1f}s ({shot.duration_sec:.1f}s)")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@cli.command("analyze-shots")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--shots-json", type=click.Path(exists=True), help="Existing segmentation result")
@click.option("--only", multiple=True, help="Run specific analyzers only")
@click.pass_context
def analyze_shots(ctx: click.Context, video_path: str, shots_json: str | None, only: tuple[str, ...]) -> None:
    """Run per-shot analysis."""
    from reels.analysis import (
        AnalysisContext,
        AnalysisRunner,
        CameraAnalyzer,
        PlaceAnalyzer,
        RhythmAnalyzer,
        SpeechAnalyzer,
        SubtitleAnalyzer,
    )
    from reels.ingest.probe import VideoProber
    from reels.models.shot import Shot

    config = ctx.obj["config"]
    work_dir = ctx.obj["work_dir"]

    console.print(f"[bold]Analyzing shots:[/bold] {video_path}")

    try:
        prober = VideoProber()
        metadata = prober.probe(Path(video_path))

        if shots_json:
            data = json.loads(Path(shots_json).read_text())
            shots = [Shot(**s) for s in (data.get("shots", data) if isinstance(data, dict) else data)]
        else:
            from reels.segmentation import segment_video
            seg_result = segment_video(Path(video_path), metadata, work_dir, config)
            shots = seg_result.shots

        ctx_analysis = AnalysisContext(
            video_path=Path(video_path),
            audio_path=None,
            work_dir=work_dir,
            metadata=metadata,
            config=config,
        )

        analyzer_map = {
            "place": PlaceAnalyzer,
            "camera": CameraAnalyzer,
            "subtitle": SubtitleAnalyzer,
            "speech": SpeechAnalyzer,
            "rhythm": RhythmAnalyzer,
        }

        runner = AnalysisRunner()
        selected = only if only else analyzer_map.keys()
        for name in selected:
            if name in analyzer_map:
                runner.register(analyzer_map[name](config))

        results = runner.run_all(shots, ctx_analysis)
        console.print(f"[green]Done![/green] Analyzed {len(shots)} shots with {len(results)} analyzers")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@cli.group()
def db() -> None:
    """Template database management."""


@db.command("list")
@click.option("--limit", default=20, help="Max results")
@click.pass_context
def db_list(ctx: click.Context, limit: int) -> None:
    """List stored templates."""
    from reels.db.repository import TemplateRepository

    config = ctx.obj["config"]
    db_path = config.get("db", {}).get("path", "./data/templates.db")

    repo = TemplateRepository(db_path)
    items = repo.list_all(limit=limit)
    repo.close()

    if not items:
        console.print("[yellow]No templates found.[/yellow]")
        return

    table = Table(title="Templates")
    table.add_column("ID", style="bold")
    table.add_column("Duration")
    table.add_column("Shots")
    table.add_column("Place")
    table.add_column("Created")

    for item in items:
        table.add_row(
            item["template_id"],
            f"{item['total_duration_sec']:.1f}s",
            str(item["shot_count"]),
            item.get("dominant_place", "-"),
            str(item.get("created_at", "-")),
        )

    console.print(table)


@db.command("search")
@click.option("--place", help="Filter by place label")
@click.option("--camera", help="Filter by camera type")
@click.option("--duration-min", type=float, default=0, help="Min duration (sec)")
@click.option("--duration-max", type=float, default=999, help="Max duration (sec)")
@click.pass_context
def db_search(ctx: click.Context, place: str | None, camera: str | None, duration_min: float, duration_max: float) -> None:
    """Search templates."""
    from reels.db.repository import TemplateRepository

    config = ctx.obj["config"]
    db_path = config.get("db", {}).get("path", "./data/templates.db")

    repo = TemplateRepository(db_path)

    if place:
        results = repo.search_by_place(place)
    elif camera:
        results = repo.search_by_camera_type(camera)
    else:
        results = repo.search_by_duration(min_sec=duration_min, max_sec=duration_max)

    repo.close()

    if not results:
        console.print("[yellow]No templates found.[/yellow]")
        return

    for t in results:
        console.print(f"[bold]{t.template_id}[/bold]: {t.shot_count} shots, {t.total_duration_sec:.1f}s")


@cli.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--name", help="숙소명")
@click.option("--region", help="지역")
@click.option("--target", type=click.Choice(["couple", "family", "solo", "friends"]), default="couple")
@click.option("--output", "-o", type=click.Path(), default="output/production", help="출력 디렉토리")
@click.option("--no-web", is_flag=True, help="웹 검증 비활성화")
@click.option("--team", is_flag=True, help="크리에이티브 팀 모드 (기획자+PD+작가+검수자)")
@click.option("--omc-team", is_flag=True, help="OMC 팀 에이전트 모드 (Claude Code 세션 기반 협업)")
@click.option("--i2v", is_flag=True, default=False, help="Run image-to-video conversion after render spec generation")
@click.pass_context
def produce(ctx: click.Context, images: tuple[str, ...], name: str | None, region: str | None, target: str, output: str, no_web: bool, team: bool, omc_team: bool, i2v: bool) -> None:
    """숙소 이미지로 쇼츠 스토리보드 생성."""
    import asyncio
    from pathlib import Path
    from reels.production.models import AccommodationInput, TargetAudience

    if not images:
        click.echo("Error: At least one image is required.", err=True)
        raise SystemExit(1)

    config = ctx.obj.get("config", {}) if ctx.obj else {}

    if no_web:
        config.setdefault("production", {}).setdefault("web_verification", {})["enabled"] = False

    input_data = AccommodationInput(
        name=name,
        region=region,
        target_audience=TargetAudience(target),
        images=[Path(img) for img in images],
    )

    output_dir = Path(output)

    if omc_team:
        console.print("[bold magenta]OMC Team Mode[/bold magenta] (Claude Code 세션 기반 협업)")
        console.print(f"  이미지: {len(images)}장")
        console.print(f"  타겟: {target}")
        console.print(f"  출력: {output_dir}")
        console.print()

        # Phase 0 실행
        from reels.production.omc_helpers.phase0_runner import (
            _build_input,
            run_phase0,
            save_phase0_outputs,
        )

        console.print("[bold]Phase 0 실행 중...[/bold] Feature 추출 + Claim 평가")
        phase0_input = _build_input(
            images=list(images),
            name=name,
            region=region,
            target=target,
        )
        try:
            verified, phase0_input = asyncio.run(
                run_phase0(phase0_input, config, no_web=no_web)
            )
        except RuntimeError as e:
            console.print(f"[red]Phase 0 실패:[/red] {e}")
            raise SystemExit(1)

        features_path, context_path = save_phase0_outputs(
            output_dir, verified, phase0_input
        )
        abs_output = output_dir.resolve()
        console.print(
            f"[green]Phase 0 완료:[/green] {len(verified)} features "
            f"→ {abs_output}"
        )
        console.print(f"  features.json: {features_path}")
        console.print(f"  context.json:  {context_path}")
        console.print()
        console.print("[bold cyan]Claude Code에서 다음 커맨드를 실행하세요:[/bold cyan]")
        console.print(f"  [green]/produce-omc --work-dir {abs_output}[/green]")
        return
    elif team:
        from reels.production.creative_team.team_agent import CreativeTeamAgent
        from reels.production.agent import ProductionAgent
        agent: CreativeTeamAgent | ProductionAgent = CreativeTeamAgent(config=config, enable_i2v=i2v)
        console.print("[bold cyan]Creative Team Mode[/bold cyan] (기획자+PD+작가+검수자)")
    else:
        from reels.production.agent import ProductionAgent
        agent = ProductionAgent(config=config, enable_i2v=i2v)

    result = asyncio.run(agent.produce(input_data, output_dir))

    if result.status == "complete":
        click.echo(f"Production complete: {result.project_id}")
        click.echo(f"  Features: {len(result.features)}")
        if result.storyboard:
            click.echo(f"  Shots: {len(result.storyboard.shots)}")
            click.echo(f"  Duration: {result.storyboard.total_duration_sec:.1f}s")
        click.echo(f"  Output: {output_dir}")
    else:
        click.echo(f"Production failed: {', '.join(result.errors)}", err=True)
        raise SystemExit(1)


@db.command("export")
@click.argument("template_id")
@click.argument("output_path", type=click.Path())
@click.pass_context
def db_export(ctx: click.Context, template_id: str, output_path: str) -> None:
    """Export template JSON."""
    from reels.db.repository import TemplateRepository

    config = ctx.obj["config"]
    db_path = config.get("db", {}).get("path", "./data/templates.db")

    repo = TemplateRepository(db_path)
    template = repo.get(template_id)
    repo.close()

    if template is None:
        console.print(f"[red]Template not found:[/red] {template_id}")
        raise SystemExit(1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.model_dump_json(indent=2))
    console.print(f"[green]Exported:[/green] {output_path}")
