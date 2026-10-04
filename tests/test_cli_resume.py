import json
import subprocess
import sys

import numpy as np
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
