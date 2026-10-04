import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from lilylake.cli import main
from lilylake.config import Config
from lilylake.inference import predict_audio
from lilylake.models import EventModel
from lilylake.music.schema import Note, Part, Piece
from lilylake.rendering import synthesize
from lilylake.training import load_checkpoint, train


def test_cli_doctor_and_help(capsys):
    assert main(["doctor"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "cpu_count" in data and "torch" in data
    result = subprocess.run(
        [sys.executable, "-m", "lilylake", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "transcribe" in result.stdout


def make_dataset(directory):
    rows = []
    for i in range(2):
        p = Piece([Part("p", "grand_piano", [Note(60 + i, 0, 0.2)])])
        folder = directory / str(i)
        folder.mkdir()
        p.save(folder / "events.json")
        sf.write(folder / "audio.wav", synthesize(p, 8000), 8000)
        rows.append(
            {
                "id": str(i),
                "composition_id": str(i),
                "split": "train",
                "audio": f"{i}/audio.wav",
                "events": f"{i}/events.json",
            }
        )
    path = directory / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_training_resumes_same_as_uninterrupted(tmp_path):
    manifest = make_dataset(tmp_path)
    c = Config(
        sample_rate=8000,
        n_fft=512,
        hop=80,
        harmonics=2,
        width=8,
        depth=1,
        epochs=2,
        batch_size=2,
        threads=1,
        validation_interval=1,
    )
    train(manifest, tmp_path / "full", c, sanity_overfit=True)
    short = Config(**{**c.to_dict(), "epochs": 1})
    train(manifest, tmp_path / "resume", short, sanity_overfit=True)
    train(manifest, tmp_path / "resume", c, sanity_overfit=True, resume=tmp_path / "resume/last.pt")
    a = load_checkpoint(tmp_path / "full/last.pt")
    b = load_checkpoint(tmp_path / "resume/last.pt")
    assert a["step"] == b["step"] == 2 and a["epoch"] == b["epoch"] == 2
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_chunk_prediction_shape_and_boundary_continuity():
    c = Config(
        sample_rate=8000,
        n_fft=512,
        hop=80,
        harmonics=2,
        width=8,
        depth=1,
        chunk_seconds=1,
        threads=1,
    )
    torch.set_num_threads(1)
    model = EventModel(c)
    p = Piece([Part("p", "grand_piano", [Note(60, 0.8, 1.3)])])
    audio = synthesize(p, 8000).mean(axis=1)
    prediction = predict_audio(model, audio, c)
    assert prediction["frame"].shape == (len(audio) // 80 + 1, 2, 128)
    assert np.isfinite(prediction["frame"]).all()


def test_length_bucket_sampler_preserves_all_examples_and_seed():
    from lilylake.training import LengthBucketSampler

    class Dataset:
        def __len__(self):
            return 19

        def duration_hints(self):
            return [1, 8, 2, 6, 3, 4, 7, 1, 2, 12, 3, 4, 5, 2, 4, 10, 1, 3, 2]

    a = list(LengthBucketSampler(Dataset(), 4, torch.Generator().manual_seed(42)))
    b = list(LengthBucketSampler(Dataset(), 4, torch.Generator().manual_seed(42)))
    assert a == b and sorted(i for batch in a for i in batch) == list(range(19))
    assert all(1 <= len(batch) <= 4 for batch in a)


def test_training_rejects_concurrent_writer(tmp_path):
    from filelock import FileLock

    with FileLock(str(tmp_path / ".training.lock")):
        with pytest.raises(ValueError, match="already running"):
            train(None, tmp_path, Config(), online_count=1)


def test_chunked_inference_matches_whole_clip_with_global_normalization():
    from lilylake.audio import features

    c = Config(
        sample_rate=8000,
        n_fft=512,
        hop=80,
        harmonics=2,
        width=8,
        depth=2,
        chunk_seconds=1,
        threads=1,
    )
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = EventModel(c).eval()
    p = Piece([Part("p", "grand_piano", [Note(60, 0.1, 0.8), Note(67, 1.3, 2.3)])])
    audio = synthesize(p, 8000).mean(axis=1)
    with torch.inference_mode():
        whole = {k: v[0].sigmoid().numpy() for k, v in model(features(audio, c)[None]).items()}
    chunked = predict_audio(model, audio, c)
    for key in whole:
        np.testing.assert_allclose(chunked[key], whole[key], atol=2e-5, rtol=2e-5)
