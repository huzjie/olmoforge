# 训练信号设计（踩坑复盘）

> 这条是最容易踩、也最容易写错的坑，值得单开一篇。

## 问题

用「对 +1 / 错 -1」更新 skill，当初始准确率 <50% 时，期望漂移为负，skill
被一路压到 0，训练看起来「越训越差」。

## 解法

让 delta 恒为正（对 +0.5 / 错 +0.2），skill 单调爬向 1.0。核心逻辑在
`backends/mock.py::MockBackend.train_step`。

## 为什么有效

skill 参与打分（`skill * is_correct + noise`），恒正的更新让 skill 单调上升，
确定性噪声不随迭代累积，准确率随之收敛。
