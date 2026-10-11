import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cvsf.neural import PoolVerifier, pool_logits  # noqa: E402
from cvsf.packets import make_packet  # noqa: E402

pytestmark = pytest.mark.neural


def row():
    return dict(
        input_ids=[0, 5, 6, 7, 2],
        word_starts=[1, 2, 3],
        word_ends=[1, 2, 3],
        all_content_positions=[1, 2, 3],
        spans=[[0, 1], [1, 3]],
        targets=[1, 0],
    )


def test_marker_attention_is_directed_and_pool_predictions_reload(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = PoolVerifier.tiny(2).eval()
    packet = make_packet(row(), [0, 1], model.marker_ids)
    mask = packet["attention_mask"][0, 0]
    assert torch.all(mask[:5, 5:] < -1e30)
    assert torch.all(mask[5:, :5] == 0)
    assert torch.all(mask[5:7, 7:9] < -1e30)
    before = pool_logits(model, [row()])[0]
    model.save_pretrained(tmp_path, ["LOC", "PER"])
    after = pool_logits(PoolVerifier.from_pretrained(tmp_path), [row()])[0]
    np.testing.assert_array_equal(before, after)


def test_empty_span_inventory_has_correct_class_shape():
    model = PoolVerifier.tiny(2).eval()
    empty = dict(row(), spans=[], targets=[])
    assert pool_logits(model, [empty])[0].shape == (0, 3)
