# Introduction Of Code Oracle

The code oracle is a code complete plugin for vscode. 
This is based on Llama2 and the backend can be run on a single GPU server with pytorch. 

Actually we only developed this plugin for OrienLink Team self. 
So it is a little bit rough.

NEXT step for us is to realize the C++ Llama backend which can be easily run in a laptop. 

## Download

You need to download the model by your self with: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

### Model sizes

|  Model | Size |
|--------|----|
| 7B     | ~12.55GB  |
| 13B    | 24GB  |
| 34B    | 63GB  |


## Setup

Please follow the setup steps with : https://github.com/facebookresearch/codellama

In a conda env with PyTorch / CUDA available, clone the repo and run in the top-level directory:

```
pip install -e .
```

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 34B    | 4  |

All models support sequence lengths up to 100,000 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware and use-case.

Pretrained code models are: the Code Llama models `CodeLlama-7b`, `CodeLlama-13b`, `CodeLlama-34b` and the Code Llama - Python models 
`CodeLlama-7b-Python`, `CodeLlama-13b-Python`, `CodeLlama-34b-Python`.


### run the Code Infilling

The main backend code is in the file code_interpreter.py

RUN IT! 

```
torchrun --nproc_per_node 1 code_interpreter.py \
    --ckpt_dir CodeLlama-7b/ \
    --tokenizer_path CodeLlama-7b/tokenizer.model \
    --max_seq_len 192 --max_batch_size 4
```


## References

1. [Code Llama Research Paper](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
2. [Code Llama Blog Post](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

