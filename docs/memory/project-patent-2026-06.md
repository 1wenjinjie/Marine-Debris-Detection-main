---
name: project-patent-2026-06
description: 凑数用发明专利（物理先验通道pipeline）已成稿；技术效果必须系统级、绝不claim单通道因果增益（Finding A否了）
metadata: 
  node_type: memory
  type: project
  originSessionId: 1472efed-bd62-4b27-8d0f-95803e24e6a8
---

2026-06-23：为**毕业/职称/项目结题凑数**目的，撰写一件中国发明专利《一种融合水体物理先验通道的海洋漂浮垃圾多光谱检测方法、装置及存储介质》。用途=拿到受理/授权状态，**不是护城河**，"弱但能下证"是正解，无人会无效它。

**成稿位置** `docs/patent/`：`申请文件_说明书_权利要求书_摘要_v1.md`（**v2**：9项权利要求[1-7方法+8装置+9介质]+说明书五段式+摘要）、`figures/fig1–fig5.png`（黑白线条图：1流程/2先验对照/3装置/4校准/5网络，生成脚本 `figures/make_patent_figures.py`，跑 `conda run -n marine python`，conda run 不支持多行-c）、`交底书_…md`（内部参考）。

**v2 删了 PKD 蒸馏全部内容**（原claim8/原图3/原实施例二）：因 PKD 在本项目是记录在案的**失败尝试**（`V2_TEACHER_TASKS_MATRIX.md`），且唯一真实现版 teacher 用 **GT派生oracle通道（泄漏真值）**（`pkd_trainer.py`/`v2_teacher_train.py`），"物理先验teacher"版从没跑过。删后更干净、无泄漏关联、无单一性问题。另删 claim3 的 FDI（需B6，不在6波段集），仅留 FAI/PI。

**两条红线（改稿务必守住）：**
1. **技术效果只能写系统级（整条pipeline端到端）**，绝不可写"加入物理先验通道使精度提升X"这种单通道因果断言——`7CH_BASELINE_CURRENT_AUDIT_2026-05-18.md` 的 **Finding A** 证明ch7置零只掉0.017 mAP50（≈0，因NDWI=f(B3,B8)已有波段的函数，冗余）。写了就和自己数据矛盾、可被无效。见 [[project-leak-attribution-2026-06]]。
2. 所有数据/实施例只能用**确定性NDWI-inverse先验**（physidx），严禁GT派生的oracle热力图（泄漏真值不可复现）。

**已知弱点（凑数可接受）：** 核心特征"光谱指数作CNN输入通道"在海漂垃圾文献里已是成熟做法（repo自己的`extension_experiments.md`都当常规扩展），新颖性/创造性弱、可能带病授权。退守路线：若权1被驳→退到权2(NDWI-inverse具体式)+"实例级边界框检测"这个真区别点（多数文献是像素分割）。

**真正的强牌不在这件专利**：分割线（U-Net实例recall~0.60 vs YOLO~0.15，4×）才有大效果，要单独立案，见 [[project-finding-d-segmentation]]。

**剩余=人的事（非文字）：** 职务发明归属（大概率申请人=学校，问知识产权办）、认哪个状态（受理/公开/授权，定要不要优先审查）、找代理机构。
