from pathlib import Path

from src.pipeline import PipelineOptions, PipelinePlan, build_jobs


def test_vad_modes_use_separate_transcription_sidecars(tmp_path: Path) -> None:
    # Given: The same input can be transcribed with either fixed ASR profile.
    source = tmp_path / "movie.mkv"
    source.touch()

    # When: Jobs are built for the default and opt-in VAD modes.
    without_vad = build_jobs(
        PipelinePlan((source,), tmp_path, None, PipelineOptions())
    )[0]
    with_vad = build_jobs(
        PipelinePlan(
            (source,),
            tmp_path,
            None,
            PipelineOptions(enable_vad=True),
        )
    )[0]

    # Then: Switching modes cannot silently reuse incompatible raw subtitles.
    assert without_vad.raw_srt_path.parent.name == "asr-auto-no-vad"
    assert with_vad.raw_srt_path.parent.name == "asr-auto-vad"
