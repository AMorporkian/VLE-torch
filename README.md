This is a Pytorch implementation of Variable Length Embeddings, as laid out in the paper [Variable Length Embeddings](https://arxiv.org/abs/2305.09967). 

I've tried to keep the code as readable and easy to tweak as possible. It uses wandb for logging by default. You can modify trainer.py to adjust the logging if you so with.

An example of the logging sample output:
![Sample](./logging_sample.png)

It's possible I've gotten some implementation detail wrong, but I've been through the paper many times and I feel it's likely largely correct. 