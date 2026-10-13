---
name: project-patent2-segmentation-2026-06
description: 案件2 交底书：分割→连通域实例化检测，docs/patent/案件2_分割实例化/，2026-06-24 定稿；强牌（Finding D），与案件1物理先验不重叠
metadata: 
  node_type: memory
  type: project
  originSessionId: 33bd61e7-ee40-4d38-886e-67136088ae66
---

第二件发明专利交底书（与 [[project-patent-2026-06]] 案件1 物理先验通道分属不同发明），2026-06-24 用 patent-disclosure-skill 挖掘+定稿。

**案件名**：一种基于多光谱语义分割与连通域实例化的海洋漂浮垃圾检测方法及系统
**落盘**：`docs/patent/案件2_分割实例化/`，最新定稿 `{案件名}_20260624204603.md`+`.docx`（6 图）。第五章已按"方法/流程"侧重强化为权利要求书式分层（独立权瘦=分割得概率图+连通域实例化输出实例框/计数；六从属=校准/聚合/阈值/不平衡训练/原生分辨率归一化/可选物理先验；+装置介质）。**附图**：用户嫌 mermaid 单薄，已 `figures/make_patent2_figures.py`（marine env python，需 scipy/torch）出 5 张 CNIPA 黑白真图（图1装置框图/图2分割网络/图3流程/图4分割实例化测试瓦片可视化=U-Net argmax 真实预测/图5召回柱状图），替换 mermaid。**图6** 由 `figures/make_calibration_fig.py` 真算：实例级 Platt 校准（val 279 实例 157TP 拟合 a=1.82/b=−1.28，test 155 实例 47TP→实例精度~30% 过分割），测试集 ECE 0.363→0.203（结果存 `figures/calibration_result.json`）。**实例级校准是本项目此前没跑过的、为本专利新算的真数据，非案件1 的 YOLO platt_fit.json。**`_draft.md` 为源；改稿后重跑 mermaid_render.py（无 mermaid 时仍出公式图+docx）。早期版本 200336/200924 保留。

**核心保护点**：语义分割得目标类像素概率图 → 阈值二值化 + 连通域标记 + 最小连通尺寸 \(n_{\min}\) 过滤 → 实例级框/计数；连通域内像素概率聚合为实例分 + Platt 单调校准 → 校准实例置信度。仅需像素级标注，区别于 Mask R-CNN（CN112990086B 建筑物实例分割，需实例掩膜、面向大目标实例分离）。

**效果（honest，系统级）**：v2split 测试集，主漂浮垃圾类实例 recall ≈0.59–0.62（3-seed）vs 同口径 YOLO ≈0.15，约 4×（数据源 `runs_v2_semseg/instance_recall_summary.json`，s0/s1 MIN=4=0.615）；MIN=1（含1–3px）recall 0.52–0.69。**坦承像素 precision 低（≈0.09 过分割），效果只报实例 recall，不claim像素precision**。这是 Finding D（[[project-finding-d-segmentation]]）正式专利化。

**查新**：领域 SOTA 即像素分割（MARIDA/MADOS/MariNeXt/AquaSense），纯 U-Net 高度公知；新颖性靠"分割→连通域实例化→校准实例置信度"组合 + 回收检测结构性漏检的亚像素目标。带病授权风险同案件1（资质用途可接受）。

**否决的候选**：U-Net 学习先验通道+K-fold 交叉预测——效果为负（unetprior Debris AP50 0.474<physidx 0.607），不作改进申请。

依赖环境：patent-disclosure-skill 装在 `.claude/skills/`；.docx 出图需 `PUPPETEER_EXECUTABLE_PATH=C:/Program Files/Google/Chrome/Application/chrome.exe`（puppeteer 自带 chrome 下载被墙，用系统 Chrome）。
