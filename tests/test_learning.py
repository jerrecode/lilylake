import numpy as np
import pytest
import torch

from lilylake.audio import features
from lilylake.config import Config
from lilylake.evaluation import evaluate_notes, frame_metrics
from lilylake.inference import decode
from lilylake.models import EventModel, event_loss, targets
from lilylake.music.schema import Note, Part, Piece
from lilylake.training import load_checkpoint, save_checkpoint


def test_model_independent_instrument_pitch_axes_and_gradients():
    c = Config(width=16, depth=2, n_fft=1024, harmonics=4)
    x = features(np.sin(np.arange(8000) * 0.1).astype("float32"), c)
    model = EventModel(c)
    out = model(x.unsqueeze(0))
    assert out["frame"].shape == (1, x.shape[1], 2, 128)
    p = Piece(
        [Part("p", "grand_piano", [Note(60, 0.1, 0.3)]), Part("v", "violin", [Note(60, 0.1, 0.4)])]
    )
    y = targets(p, x.shape[1], c)
    assert y["frame"][:, 0, 60].sum() > 0 and y["frame"][:, 1, 60].sum() > 0
    loss, _ = event_loss(out, {k: v.unsqueeze(0) for k, v in y.items()}, model.pitch_mask)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder[0].weight.grad.abs().sum() > 0
    assert y["frame"][:, 0, :21].sum() == 0


def test_decoder_repeat_and_same_pitch_independence():
    c = Config(min_note_seconds=0.01)
    shape = (50, 2, 128)
    pred = {k: np.zeros(shape, dtype=np.float32) for k in ["frame", "onset", "offset", "velocity"]}
    pred["frame"][5:30, :, 60] = 0.9
    pred["onset"][5, :, 60] = 0.99
    pred["onset"][20, :, 60] = 0.99
    pred["offset"][30, :, 60] = 0.99
    pred["velocity"][:] = 0.7
    p = decode(pred, c)
    assert [len(part.notes) for part in p.parts] == [2, 2]
    assert p.parts[0].notes[0].offset == pytest.approx(0.2)
    assert p.parts[1].notes[1].onset == pytest.approx(0.2)
    assert p.parts[0].notes[-1].offset == pytest.approx(0.3)


def test_bipartite_evaluation_does_not_double_count():
    p = Piece([Part("p", "grand_piano", [Note(60, 0, 1)])])
    q = Piece([Part("p", "grand_piano", [Note(60, 0, 1), Note(60, 0.01, 1.01)])])
    m = evaluate_notes(p, q)
    assert m["onset"]["tp"] == 1 and m["onset"]["fp"] == 1
    assert m["onset"]["recall"] == 1 and m["onset"]["precision"] == 0.5
    wrong = Piece([Part("v", "violin", [Note(60, 0, 1)])])
    assert evaluate_notes(p, wrong)["onset"]["tp"] == 0
    assert evaluate_notes(Piece(), Piece())["onset"]["f1"] == 0


def test_frame_metrics_counts():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    m = frame_metrics(a, b)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_checkpoint_exact_optimizer_resume(tmp_path):
    c = Config(width=16, depth=2)
    m = EventModel(c)
    o = torch.optim.Adam(m.parameters())
    sum(x.sum() for x in m.parameters()).backward()
    o.step()
    scheduler = torch.optim.lr_scheduler.StepLR(o, 1, 0.9)
    o.step()
    scheduler.step()
    state = {
        "epoch": 3,
        "step": 7,
        "model": m.state_dict(),
        "optimizer": o.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": c.to_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    save_checkpoint(tmp_path / "last.pt", state)
    loaded = load_checkpoint(tmp_path / "last.pt")
    n = EventModel(c)
    n.load_state_dict(loaded["model"])
    assert loaded["epoch"] == 3 and loaded["step"] == 7
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), n.parameters()))
    assert not list(tmp_path.glob("*.tmp"))


def test_config_validation():
    with pytest.raises(ValueError):
        Config(hop=0)
    with pytest.raises(ValueError):
        Config(instruments=["made_up"])
    with pytest.raises(ValueError):
        Config(width=15)


def test_metric_report_checkpoint_uses_safe_builtin_scalars(tmp_path):
    p = Piece([Part("p", "grand_piano", [Note(60, 0, 1)])])
    report = evaluate_notes(p, p)
    report["frame"] = frame_metrics(np.array([1, 0]), np.array([1, 0]))
    save_checkpoint(tmp_path / "metrics.pt", {"validation": report})
    assert load_checkpoint(tmp_path / "metrics.pt")["validation"]["frame"]["f1"] == 1


def test_decoder_one_event_per_contiguous_onset_region():
    c = Config(min_note_seconds=0.01)
    shape = (60, 2, 128)
    pred = {k: np.zeros(shape, dtype=np.float32) for k in ["frame", "onset", "offset", "velocity"]}
    pred["frame"][5:50, 0, 60] = 0.9
    pred["onset"][5:16, 0, 60] = [0.6, 0.7, 0.8, 0.7, 0.75, 0.7, 0.65, 0.7, 0.65, 0.6, 0.55]
    pred["velocity"][:] = 0.7
    piece = decode(pred, c)
    assert len(piece.parts[0].notes) == 1
    assert piece.parts[0].notes[0].onset == pytest.approx(0.07)


def test_dataset_velocity_rmse_is_pooled_not_mean_of_piece_rmse():
    from lilylake.training import aggregate_timing

    errors = {"velocity_rmse": [(0.0, 1), (10.0, 1)], "velocity_mae": [(0.0, 1), (10.0, 1)]}
    result = aggregate_timing(errors)
    assert result["velocity_rmse"] == pytest.approx(np.sqrt(50))
    assert result["velocity_mae"] == 5
