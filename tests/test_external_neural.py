"""Reload checks for the separately installed external-model environment."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("span_marker")

from cvsf.external import load_model, normalized_gliner_text  # noqa: E402

pytestmark = pytest.mark.external


def test_spanmarker_reload_preserves_eager_attention_and_predictions(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel
    from span_marker import SpanMarkerModel

    torch.set_num_threads(2)
    torch.manual_seed(9)
    words = ["<s>", "<pad>", "</s>", "<unk>", "Ada", "visited", "Luma", "."]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=64,
    )
    source = tmp_path / "encoder"
    tokenizer.save_pretrained(source)
    config = XLMRobertaConfig(
        vocab_size=len(words),
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=40,
        max_position_embeddings=130,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
    )
    XLMRobertaModel(config, add_pooling_layer=False).save_pretrained(source)
    model = SpanMarkerModel.from_pretrained(
        str(source),
        labels=["O", "B-PER", "I-PER", "B-LOC", "I-LOC"],
        model_max_length=64,
        marker_max_length=8,
        entity_max_length=3,
        attn_implementation="eager",
    ).eval()
    examples = [["Ada", "visited", "Luma", "."], ["Luma", "visited", "Ada"]]

    def predict_with_logits(model):
        logits = []
        hook = model.classifier.register_forward_hook(
            lambda module, inputs, output: logits.append(output.detach().clone())
        )
        try:
            result = model.predict(examples, batch_size=2, show_progress_bar=False)
        finally:
            hook.remove()
        assert logits and all(value.numel() for value in logits)
        return result, logits

    before, before_logits = predict_with_logits(model)
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    restored = load_model("spanmarker", {}, {}, checkpoint=checkpoint, device="cpu").eval()
    assert model.encoder.config._attn_implementation == restored.encoder.config._attn_implementation == "eager"
    assert type(model.encoder.encoder.layer[0].attention.self) is type(restored.encoder.encoder.layer[0].attention.self)
    assert model.state_dict().keys() == restored.state_dict().keys()
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    after, after_logits = predict_with_logits(restored)
    assert after == before and len(before_logits) == len(after_logits)
    assert all(torch.equal(left, right) for left, right in zip(before_logits, after_logits))


def test_pinned_gliner_processor_keeps_original_coordinates():
    from gliner2.processor import SchemaTransformer

    for tokens in (["Ada", "Luma"], ["Ada", "."], ["كلام", "بىلەن"], [""], ["one", "two!"]):
        original = " ".join(tokens)
        normalized = normalized_gliner_text(tokens, SchemaTransformer._normalize_text)
        assert normalized[: len(original)] == original
