# Third-party resources

CVSF uses the libraries and models below. Their code and weights retain their original licenses.

| Resource | Project or model | License |
|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-style |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache 2.0 |
| scikit-learn | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause |
| XLM-RoBERTa | [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | MIT |
| SpanMarker | [tomaarsen/SpanMarkerNER](https://github.com/tomaarsen/SpanMarkerNER) | Apache 2.0 |
| GLiNER2 | [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) | Apache 2.0 |
| GLiNER2.5 multilingual weights | [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) | Apache 2.0 |

The CVSF code incorporates the authors' candidate-generation and entity-reranking routines from the earlier RELIC experiment artifact. Its MIT copyright notice is retained in `LICENSE`.

The contextual verifier follows the packed-marker input construction studied by Ye et al., *Packed Levitated Marker for Entity and Relation Extraction* (ACL 2022), while its POOL representation uses content-state pooling. See the manuscript for the relationship to prior NER and stacking methods.

The [data guide](docs/data.md) lists corpus sources, terms, and download instructions. Corpus texts and pretrained weights are retrieved from their original sources.
