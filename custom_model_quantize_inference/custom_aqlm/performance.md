| Model | AQLM config | Traning Config | Model Path | PPL | HellaSwag Acc | HellaSwag Acc Norm |
| --- | --- | --- | --- | --- | --- | --- |
| TinyLlama-v1.1 | baseline (fp16) | n/A | n/A | 7.7077 | 0.4643 | 0.6268 |
| TinyLlama-v1.1 | 8x8g8 | epochs: 5 | custom_model_quantize_inference/custom_aqlm/model/TinyLlama-v1.1-8x8g8 | 34.1107 | 0.2846 | 0.3106 |
| TinyLlama-1.1B-Chat-v1.0 (Official) | 1x16 (Group 8) | Official Reference | custom_aqlm/model/TinyLlama-v1.0-hf-aqlm | 10.7390 | TBD | TBD |
