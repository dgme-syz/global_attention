# global_attention

> **A Bad Try**

> [!IMPORTANT]
> 我们的仓库使用 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 提供的微调支持，并且支持该实验提出的 `GA` 方法；使用 [peft](https://github.com/huggingface/peft) 提供的 `adapter` 并且制作了 `GA` adapter 从而能够在 LLMs 上添加我们所需要的模块 (📢 现在已经存在更方便的方案，可以参考:[BenjaminBossan/peft](https://github.com/BenjaminBossan/peft/tree/refactor-peft-method-registration))；最后，使用 [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) 完成新训练模型的评测


在 `global_attention` 文件夹下，运行 `./scripts/train.sh` 即可执行单卡训练。

</br>

使用如下命令完成在 `gsm_symbolic` 数据集的评测：
```bash
lm_eval --model hf \
  --model_args pretrained=xxx,peft=xxx,dtype="float16" \
  --tasks gsm_symbolic \
  --device cuda:0 \
  --batch_size 4 \
  --output_path ./result/
```

> Releases 中有一些我们训练好的 **peft adapter**，您可能需要在 `adapter_config.json` 中修改一些信息从而在您的机器上运行
