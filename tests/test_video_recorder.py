import json
import shutil
import subprocess

import numpy as np
import pytest

from tfp.visualization.renderer import VideoRecorder


def test_video_recorder_encodes_all_frames_at_constant_fps(tmp_path):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable")
    fps = 8
    recorder = VideoRecorder(tmp_path, fps=fps, terminal_hold_seconds=0.75)
    recorder.start_episode("episode", 109)
    source_frames = 17
    for i in range(source_frames):
        recorder.append(np.full((64, 96, 3), i * 7, dtype=np.uint8))
    recorder.finish_episode()

    path = tmp_path / "episode_seed109.mp4"
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,avg_frame_rate,duration", "-of", "json", str(path),
    ], text=True))
    stream = probe["streams"][0]
    expected = source_frames + round(fps * 0.75)
    assert int(stream["nb_read_frames"]) == expected
    assert stream["avg_frame_rate"] == f"{fps}/1"
    assert abs(float(stream["duration"]) - expected / fps) < 1e-6
