# olmoforge.backends.mock

`MockBackend` 是确定性可训练后端。打分 = `skill × is_correct + noise`。
训练 delta 恒正（对 +0.5 / 错 +0.2），skill 单调爬向 1.0。
