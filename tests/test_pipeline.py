from pathlib import Path

import pysrt
import pytest

import src.pipeline as pipeline_module
from src.pipeline import PipelineOptions, PipelinePlan, build_jobs, run_pipeline


def _write_srt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pysrt.SubRipFile(
        items=[
            pysrt.SubRipItem(
                index=1,
                start=pysrt.SubRipTime(milliseconds=0),
                end=pysrt.SubRipTime(milliseconds=1_000),
                text=text,
            )
        ]
    ).save(str(path), encoding="utf-8")


def test_pipeline_batches_each_stage_and_keeps_source_paths_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Two source files and lightweight adapters recording stage order.
    first = tmp_path / "first.wav"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first-source")
    second.write_bytes(b"second-source")
    events: list[str] = []

    class FakeExtractor:
        def extract(
            self,
            input_path: str | Path,
            output_path: str | Path | None = None,
            force: bool = False,
        ) -> Path:
            source = Path(input_path)
            destination = Path(output_path or source.with_suffix(".wav"))
            events.append(f"extract:{source.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"audio")
            return destination

    class FakeTranscriber:
        def transcribe_many(self, tasks, language=None, quiet=False):
            events.append(f"transcribe:{len(tasks)}")
            for task in tasks:
                _write_srt(task.output_path, task.audio_path.stem)
            return []

    class FakeTranslator:
        def translate_many(self, tasks, options):
            events.append(f"translate:{len(tasks)}")
            for task in tasks:
                _write_srt(task.output_path, task.input_path.stem)
            return []

    def fake_clean(input_path: Path, output_path: Path | None = None) -> None:
        events.append(f"clean:{input_path.parent.parent.name}")
        _write_srt(output_path or input_path, input_path.stem)

    monkeypatch.setattr(pipeline_module, "AudioExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_module, "VLLMTranslator", FakeTranslator)
    monkeypatch.setattr(pipeline_module, "clean_srt", fake_clean)
    options = PipelineOptions(
        target_lang="Chinese",
        src_lang="en",
        keep_temp=True,
        force=True,
        translated_only=True,
    )
    jobs = build_jobs(
        PipelinePlan(
            input_paths=(first, second),
            input_root=tmp_path,
            output_dir=tmp_path / "output",
            options=options,
        )
    )

    # When: The batch pipeline processes both jobs.
    result = run_pipeline(jobs, options)

    # Then: Stages are global, heavyweight adapters run once, and sources survive.
    assert result.failures == ()
    assert events[:2] == ["extract:first.wav", "extract:second.mp3"]
    assert events[2] == "transcribe:2"
    assert events[3:5] == [
        "clean:.first.wav.simple-radeon-subs",
        "clean:.second.mp3.simple-radeon-subs",
    ]
    assert events[5] == "translate:2"
    assert first.read_bytes() == b"first-source"
    assert second.read_bytes() == b"second-source"


def test_pipeline_isolates_failed_extraction_and_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: One invalid source and one healthy source.
    broken = tmp_path / "broken.mp4"
    healthy = tmp_path / "healthy.mp4"
    broken.touch()
    healthy.touch()

    class FakeExtractor:
        def extract(
            self,
            input_path: str | Path,
            output_path: str | Path | None = None,
            force: bool = False,
        ) -> Path:
            source = Path(input_path)
            if source == broken:
                raise OSError("broken fixture")
            destination = Path(output_path or source.with_suffix(".wav"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"audio")
            return destination

    class FakeTranscriber:
        def transcribe_many(self, tasks, language=None, quiet=False):
            assert [task.audio_path.parent.name for task in tasks] == [
                ".healthy.mp4.simple-radeon-subs"
            ]
            for task in tasks:
                _write_srt(task.output_path, "healthy")
            return []

    class FakeTranslator:
        def translate_many(self, tasks, options):
            for task in tasks:
                _write_srt(task.output_path, "translated")
            return []

    monkeypatch.setattr(pipeline_module, "AudioExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_module, "VLLMTranslator", FakeTranslator)

    def fake_clean(input_path: Path, output_path: Path | None = None) -> None:
        _write_srt(output_path or input_path, "cleaned")

    monkeypatch.setattr(pipeline_module, "clean_srt", fake_clean)
    options = PipelineOptions(target_lang="Chinese", keep_temp=True)
    jobs = build_jobs(
        PipelinePlan(
            input_paths=(broken, healthy),
            input_root=tmp_path,
            output_dir=tmp_path / "output",
            options=options,
        )
    )

    # When: Extraction fails for only the first job.
    result = run_pipeline(jobs, options)

    # Then: The healthy movie completes and the batch reports one failure.
    assert len(result.failures) == 1
    assert result.failures[0].job.input_path == broken
    assert jobs[1].final_srt_path.is_file()
    assert not jobs[0].final_srt_path.exists()


def test_pipeline_creates_sidecar_before_starting_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A fresh job and an extractor that requires its destination parent.
    source = tmp_path / "movie.mp4"
    source.touch()
    parent_states: list[bool] = []

    class InspectingExtractor:
        def extract(
            self,
            input_path: str | Path,
            output_path: str | Path | None = None,
            force: bool = False,
        ) -> Path:
            destination = Path(output_path or Path(input_path).with_suffix(".wav"))
            parent_states.append(destination.parent.is_dir())
            raise OSError("stop after observing the extraction boundary")

    monkeypatch.setattr(pipeline_module, "AudioExtractor", InspectingExtractor)
    options = PipelineOptions()
    jobs = build_jobs(PipelinePlan((source,), tmp_path, None, options))

    # When: The extraction stage starts.
    result = run_pipeline(jobs, options)

    # Then: The managed sidecar already exists for FFmpeg's output.
    assert parent_states == [True]
    assert len(result.failures) == 1


def test_build_jobs_preserves_relative_output_paths_and_unique_sidecars(
    tmp_path: Path,
) -> None:
    # Given: Same-stem inputs in separate source directories.
    first = tmp_path / "a" / "movie.mp4"
    second = tmp_path / "b" / "movie.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    output_dir = tmp_path / "output"
    options = PipelineOptions(target_lang="Chinese")

    # When: Pipeline paths are built.
    jobs = build_jobs(
        PipelinePlan(
            input_paths=(first, second),
            input_root=tmp_path,
            output_dir=output_dir,
            options=options,
        )
    )

    # Then: Final paths preserve layout and sidecars stay beside each source.
    assert jobs[0].final_srt_path == output_dir / "a" / "movie.Chinese.srt"
    assert jobs[1].final_srt_path == output_dir / "b" / "movie.Chinese.srt"
    assert jobs[0].work_dir == first.parent / ".movie.mp4.simple-radeon-subs"
    assert jobs[1].work_dir == second.parent / ".movie.mp4.simple-radeon-subs"


def test_existing_final_skips_after_successful_sidecar_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A movie that can complete through lightweight pipeline adapters.
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"source")
    options = PipelineOptions(target_lang="Chinese")
    jobs = build_jobs(
        PipelinePlan(
            input_paths=(source,),
            input_root=tmp_path,
            output_dir=tmp_path / "output",
            options=options,
        )
    )

    class FakeExtractor:
        def extract(
            self,
            input_path: str | Path,
            output_path: str | Path | None = None,
            force: bool = False,
        ) -> Path:
            destination = Path(output_path or Path(input_path).with_suffix(".wav"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"audio")
            return destination

    class FakeTranscriber:
        def transcribe_many(self, tasks, language=None, quiet=False):
            for task in tasks:
                _write_srt(task.output_path, "raw")
            return []

    class FakeTranslator:
        def translate_many(self, tasks, options):
            for task in tasks:
                _write_srt(task.output_path, "translated")
            return []

    def fake_clean(input_path: Path, output_path: Path | None = None) -> None:
        _write_srt(output_path or input_path, "cleaned")

    monkeypatch.setattr(pipeline_module, "AudioExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_module, "VLLMTranslator", FakeTranslator)
    monkeypatch.setattr(pipeline_module, "clean_srt", fake_clean)

    first_result = run_pipeline(jobs, options)

    assert first_result.succeeded == jobs
    assert jobs[0].final_srt_path.is_file()
    assert not jobs[0].work_dir.exists()

    class ForbiddenExtractor:
        def __init__(self) -> None:
            raise AssertionError("skipped jobs must not initialize FFmpeg")

    monkeypatch.setattr(pipeline_module, "AudioExtractor", ForbiddenExtractor)
    monkeypatch.setattr(pipeline_module, "Transcriber", ForbiddenExtractor)
    monkeypatch.setattr(pipeline_module, "VLLMTranslator", ForbiddenExtractor)

    # When: The same input is processed again without force.
    result = run_pipeline(jobs, options)

    # Then: The final subtitle alone is enough to skip every pipeline stage.
    assert result.succeeded == ()
    assert result.failures == ()
    assert result.skipped == jobs
