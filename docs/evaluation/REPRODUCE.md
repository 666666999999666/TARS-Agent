# 复核公开结果与运行新实验

以下命令从**仓库根目录**运行，PowerShell 示例使用 Python 3.12、uv 和 Docker Linux 引擎。公开仓库不包含作者的密钥、原账本、世界状态或原始评分轨迹。前两节不需要模型凭证；完整新实验会产生 DeepSeek API 费用。

## 1. 零模型请求的结果复核

只复核公开摘要时，脚本仅使用 Python 标准库：

```powershell
python scripts/verify_eval_summary.py
```

也可显式传入文件：

```powershell
python scripts/verify_eval_summary.py docs/evaluation/results.json
```

它不写文件、不读取凭证或账本，不启动 Docker 或模型。它从脱敏逐题记录重新计算 TGC、SGC、配对数量和相关汇总，检查是否与声明一致；它不能替代根据世界状态重新运行官方 evaluator，也不能独立重做未公开轨迹的人工归因。

准备源码运行环境并查看固定清单：

```powershell
uv sync --locked --group qa --group security
uv run --no-sync python scripts/run_internship_evals.py deepseek-train-paired --batch-root build/evaluation-local/batch --output build/evaluation-local/batch/train
uv run --no-sync python scripts/run_internship_evals.py deepseek-dev-paired --batch-root build/evaluation-local/batch --output build/evaluation-local/batch/dev
uv run --no-sync python scripts/probe_deepseek.py
```

未加 `--execute` 时仅打印清单或接入检查计划，不创建模型请求。依赖安装本身会访问包源。`run_internship_evals.py` 是保留的历史脚本名，公开实验使用它现有的固定 A/B 模式。

## 2. 准备自己的 AppWorld 环境

```powershell
pwsh -File scripts/prepare_appworld.ps1
```

脚本下载固定官方源码和 0.2.0 数据，在 `build/appworld/` 保存新的本地文件并构建 Linux 镜像；需要网络、磁盘和 Docker，不调用模型。复用已有目录时会检查源码、标签、数据版本和 SQLModel 版本，遇到不匹配则停止，不覆盖旧数据。

环境固定官方源码 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`；兼容层只将 SQLModel 固定为 0.0.44。脚本保存镜像元数据、依赖清单和数据包 SHA-256，批次进一步绑定实际镜像身份。重新构建镜像可能产生不同 ID；不能在已开始的批次中换镜像后继续使用旧身份记录。

历史官方安装验证为 1840 项通过，属于原环境记录。新环境仍需自行验证，不能因为下载到同名镜像就继承该结果。本轮公开发布不重新调用模型，新的真实实验应先确认官方模型仍可用、计价与冻结策略相容。费用策略使用[实验记录](EXPERIMENT.md)中的历史高峰价格，并非自动更新的价格查询；价格变更时不要把旧策略当作当前账单保证。

## 3. 新用户只初始化一次自己的预算

本节**只适用于从未开始过该实验的新用户、新目录**。继续作者的实验或恢复你已开始的实验时，直接进入下一节的恢复说明，禁止用新目录、新账本或复制数据库重置额度。

在干净、已提交的源码检出中，确认本轮愿意承担最多 50 元本地估算额度。以下初始化不会调用模型；它创建新的请求账本和费用账本。费用账本目前固定 50 元，不是通用任意币种计费服务。

```powershell
$env:DEEPSEEK_REQUEST_BUDGET_PATH = Join-Path (Get-Location).Path 'build/evaluation-local/request-budget.sqlite3'
$env:DEEPSEEK_COST_BUDGET_PATH = Join-Path (Get-Location).Path 'build/evaluation-local/cost-budget.sqlite3'
$env:DEEPSEEK_REQUEST_LIMIT = '33120'
$tarsInitBudget = @'
import os
from pathlib import Path
from tars_agent.core.persistence.request_budget import RequestLedger
from tars_agent.core.persistence.cost_budget import CostLedger

request = Path(os.environ["DEEPSEEK_REQUEST_BUDGET_PATH"])
cost = Path(os.environ["DEEPSEEK_COST_BUDGET_PATH"])
assert request.is_absolute() and cost.is_absolute()
assert request.parent == cost.parent
# No exist_ok: an existing or interrupted experiment must not get a new budget.
request.parent.mkdir(parents=True, exist_ok=False)
assert RequestLedger(request, limit=33120).counts() == {"real": 0, "probe": 0}
CostLedger.initialize(cost, request_budget_path=request)
print("New request and 50-CNY cost budgets initialized; no model request sent.")
'@
uv run --no-sync python -c $tarsInitBudget
if ($LASTEXITCODE -ne 0) { throw '初始化未完成；检查原目录，不能删掉重建预算。' }
```

33120 是本次 train/dev 配对的理论 HTTP 次数上限：`(12 + 57) × 2 个版本 × 60 步 × 2 次 HTTP 尝试 × 2 次基础设施尝试`。实际同时受每题时间、步骤和费用预算限制；它不是保证可以执行 33120 次的付费授权，也不是每题额度。

密钥只放当前进程环境，输入过程隐藏，不写入命令历史或文件：

```powershell
$env:DEEPSEEK_API_KEY = [Net.NetworkCredential]::new('', (Read-Host 'DeepSeek API Key' -AsSecureString)).Password
```

专用加载器固定官方入口 `https://api.deepseek.com/anthropic`、模型 `deepseek-flash`、响应模型检查、8192 最大输出及相同重试政策，不需要作者的私密 `.env`。这些专用变量不能简单用普通 `tars eval run` 代替加载；下节脚本会明确调用同一专用加载器。

## 4. 明确执行或恢复

**下面命令会启动真实模型调用并产生费用。** 先保持工作树干净；train 完成后才能执行同一批次的 dev。两阶段共用同一配置、源码树、镜像、请求账本和费用账本。

```powershell
uv run --no-sync python scripts/run_internship_evals.py deepseek-train-paired --batch-root build/evaluation-local/batch --output build/evaluation-local/batch/train --execute
if ($LASTEXITCODE -ne 0) { throw 'train 未完成；先检查保存的结果，不启动 dev。' }
uv run --no-sync python scripts/run_internship_evals.py deepseek-dev-paired --batch-root build/evaluation-local/batch --output build/evaluation-local/batch/dev --execute
```

固定 train 12 题 × A/B 为 24 次运行，完整 dev 57 题 × A/B 为 114 次。普通任务失败仍保存和评分；脚本的批次完成退出码不能解释成全部任务答对。B 单版清单仅用于后续显式选择，不替代此次 A/B 对照，也没有在历史交付后再次付费执行。

你自己的批次中断时，保留原环境变量指向的账本、费用身份文件及所有批次文件。恢复原来的干净源码树、Python 环境、镜像和配置后，只对中断阶段的**原命令**追加 `--resume`；例如 train 中断：

```powershell
uv run --no-sync python scripts/run_internship_evals.py deepseek-train-paired --batch-root build/evaluation-local/batch --output build/evaluation-local/batch/train --execute --resume
```

dev 中断则在原 dev 命令追加 `--resume`。不重新执行初始化，不改输出目录，不提高已冻结请求上限；恢复不会刷新预算、补跑次数或覆盖已完成结果。费用记录、资源身份或保存状态不一致时停止排查，不删除身份文件、旧结果或未知费用预留来绕过检查。

历史作者批次的 `a7b0f68` 只作为来源标识，未随本次公开分支历史发布。上述命令在当前公开提交上创建**新的实验**，记录自己的冻结来源，不恢复作者的批次，也不保证得到同分。[公开核对记录](source-equivalence.json)列出 157 个相同运行时文件以及 MCP 解释器启动和依赖更新；公开版本不是原实验的完全相同源码与环境。

完成或暂停后清除当前进程的密钥变量：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY
```

保留本地结果和预算文件供继续核对，不把账本、原始轨迹或应用状态提交到 Git。

## 5. 查看自己生成的结果与工程检查

已有本地结果可由正式 CLI 只读输出报告，不发送模型请求：

```powershell
uv run --no-sync tars eval report build/evaluation-local/batch/dev/evidence/result.json --format md
```

公开候选的离线工程检查入口如下；审计会访问漏洞数据源，测试不发送模型请求，真实 Docker 和人工界面专项另行记录：

```powershell
npm --prefix web ci
uv run --no-sync python scripts/qa.py full
uv run --no-sync python scripts/check_public_content.py --history --refs HEAD
```

构建产物、测试日志和安装检查属于新候选自身证据，不覆盖[历史验证](VALIDATION.md)。完整模型实验、工程 QA 和公开摘要复算是三个不同检查，不能互相替代。
