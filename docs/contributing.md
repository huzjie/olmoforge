# 贡献指南

1. 保持零依赖：新增代码优先用标准库。
2. 每个新模块配一个 examples/ 脚本和至少一个单测。
3. 训练类改动遵守「信号恒正」铁律。
4. 基准结果放进 docs/benchmarks.md 的对应行。

提交前：

```bash
python -m compileall olmoforge
python -m unittest discover -s tests -t .
python -m olmoforge doctor
```
