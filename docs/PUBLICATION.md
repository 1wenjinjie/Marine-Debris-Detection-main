# GitHub 与 Zenodo 公开发布清单

本轮只做本地、可逆的发布准备。尚未确认仓库 Public 状态，未提交、推送、上传、发布 release 或创建 DOI。本清单的未勾选项均需要实际核验，不能视为已完成。

## 1. 先解决公开阻断项

- [ ] 确认论文对应代码分支、实验版本和结果口径。v1 七通道含 GT 派生 oracle 输入，不能将其结果当作可部署六通道基线；保留 README 中的限制说明。
- [ ] 审核已跟踪文件及 Git 历史中的内部信息、绝对路径、凭据、未发表内容、专利材料和第三方文件。重点包括 `docs/memory/`、`docs/patent/`、`docs/paper/`、`docs/paper/literature/`、`docs/references/`；文献全文必须进行内容审核和权利确认后才能决定是否公开。
- [ ] 清理拟公开版本中的 `.codex_work/` 等内部工具状态、本机路径及配置。逐项区分可改成仓库相对路径的配置与需要用户本地填写的数据路径。此次文档和忽略规则更新没有完成仓库全部路径清理。
- [ ] 从实际生成结果的环境导出并审核依赖锁定记录，记录 Python、torch、torchvision、ultralytics、CUDA、驱动、硬件和关键配置；在匹配环境中完成复现，记录实际命令、数据版本、seed、结果及日志。当前 `requirements.txt` 并非完整训练环境锁文件。
- [ ] 由权利人确认代码许可，再添加 `LICENSE`。当前没有 `LICENSE`，不能声明 MIT。结合 [Ultralytics 官方许可](https://www.ultralytics.com/license)及[上游 AGPL-3.0 文本](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)核查本项目的许可义务；数据与权重单独确认权利和许可。
- [ ] 确认作者、作者顺序、ORCID、公开标题、版本号及关联论文。信息未确认前不生成虚构的 `CITATION.cff`、作者信息或 DOI。

`.gitignore` 只影响未跟踪文件，不会移除已跟踪文件，也不会清除提交历史；补忽略规则不能替代上述审核。可用 `git ls-files` 检查跟踪清单、`git log --all --name-only` 检查历史文件名，再检查实际内容。

建议保留本机原工作区，新建单独的发布目录，按人工审核的白名单复制最小可复现代码、配置和文档，再初始化新的公开仓库。这样可避免将原研究历史一起公开。新仓库仍需审核当前内容、第三方许可和运行所需文件；不要直接把原目录整体上传。公开前再次核验 GitHub 中的仓库可见性。

## 2. 代码：GitHub + Zenodo

- [ ] 在审核后的代码仓库补齐 README、已确认的 LICENSE、实际环境记录和可迁移的数据配置；确认未包含原始数据、模型权重、内部材料或未审核的文献全文。
- [ ] 由仓库管理者完成 GitHub 发布并核验其为 Public。普通 Git 仓库对单文件有 100 MiB 限制；权重、预测等大文件放在独立 Zenodo 产物记录。[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [ ] 登录正式 Zenodo，连接并授权 GitHub 账户；在 GitHub 集成页同步仓库列表，再启用目标仓库的归档开关。必须先启用，再创建拟归档的 GitHub release。[Zenodo 启用仓库说明](https://help.zenodo.org/docs/github/enable-repository/)
- [ ] 核对归档元数据和待发布文件，选定版本号及 release tag，再由管理者发布 GitHub release。
- [ ] 等待 Zenodo 处理，检查归档是否成功、元数据是否有错误、文件是否完整，打开实际生成的记录确认**该版本 DOI**。发布 release 本身不代表已成功归档。[Zenodo release 归档说明](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [ ] 固定论文使用的 release tag、commit SHA 和版本 DOI。版本 DOI 对应特定归档版本；概念 DOI 用于聚合版本，复现引用应明确具体版本。后续修改另发新版本。
- [ ] 在代码 README、Zenodo 元数据和论文可用性声明中填入已核验的代码 DOI，并与独立数据产物记录互链。

## 3. 数据产物：单独 Zenodo 记录

MARIDA、MADOS 原始数据不重新上传。核对并引用其原发布记录、版本、DOI 和许可，说明如何获取以及如何生成本研究派生产物；不要将原始数据装入代码归档或产物压缩包。

- [ ] 确认拟分享产物的分发权利和许可，尤其是预处理数据、标签衍生物和模型权重；原数据不再分发并不自动解决衍生产物的许可问题。
- [ ] 按论文实际实验选取模型权重、预测文件、统计结果、配置和场景划分 manifest。manifest 使用可迁移标识或相对路径，避免泄露本机目录；保留样本、场景、类别映射与原数据版本之间的对应关系。
- [ ] 给每个上传文件计算 SHA256，提供文件清单与校验值；记录样本数、各 split 数量、类别统计、通道顺序、归一化统计来源及数据格式。仅对确实计算过的结果作说明。
- [ ] 提供产物 README：用途、来源、处理步骤、脚本/参数、对应代码版本、seed、评估 split、阈值、图像尺寸、复用方式和限制；区分标准 AP 评估与逐实例诊断。
- [ ] 核对 Zenodo 标题、作者、ORCID、版本、描述、关键词、许可及关联论文/代码信息。未确认字段暂不编造。
- [ ] 创建并审核独立 Zenodo 记录，再上传经确认的产物。发布后核查实际 DOI、文件下载和校验值，并将代码版本 DOI 与产物 DOI 双向关联。

可在 PowerShell 中对审核后的产物逐个计算，例如 `Get-FileHash -Algorithm SHA256 -LiteralPath <文件路径>`；这些校验命令不代表数据已上传或通过科学验证。

## 4. 最后核验与人工发布

- [ ] 在新目录复现文档命令；训练、评估和诊断由调用方使用 `run_script` 执行，保留命令与真实输出，不在本轮直接启动 GPU 作业。
- [ ] 检查公开文件、Git 历史、许可、依赖、数据下载说明、代码 DOI、数据 DOI 和论文声明彼此一致。
- [ ] 获得已确认的元数据与发布授权后，由管理者完成提交、推送、Public 设置、release 和 Zenodo 发布；本轮不自动执行这些操作。
