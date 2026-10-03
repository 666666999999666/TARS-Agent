# 工程验证：合并后 main 与历史证据

本次公开版本保留运行时修复和评测实现，公开技术报告与脱敏摘要，不发布原始应用数据、模型轨迹、账本、凭证或本机路径。历史模型实验来自本地冻结源码 `a7b0f68a7aefd33b47eb93701bbdbd51a0a70378`；公开发布是独立提交历史，不让读者把这个本地提交号当作可检出的公开版本。

## 合并后 main

2026-10-03 核实 GitHub main 为 `917d0e957f6653719e8f50a5993b13c34619a661`，版本仍为 **0.8.0**。PR #1 已合并；PR head `31575d55` 的 [CI run 37124945865](https://github.com/666666999999666/TARS-Agent/actions/runs/37124945865) 与合并后 main 的 [CI run 37126210739](https://github.com/666666999999666/TARS-Agent/actions/runs/37126210739) 均成功，Windows、Ubuntu、Web 三组 job 均为 success。

main 的 Ubuntu job 实际执行了真实 Docker E2E、Core 崩溃回收与实例隔离、worker 路径边界、检出目录外的完整独立安装；Web job 执行 Core → FastAPI → SSE → React 的浏览器流程。这些检查补齐对应环境门槛，不代表新的真实模型成绩，也不覆盖 Linux AppWorld 调度器崩溃恢复。

本次收尾修复以 `917d0e9` 为基线。该提交的 CI 成功不能代替本次 patch 的 CI；历史标签保持原归属。

## 2026-10-03 本地收尾小修

本节对应 `917d0e9` 的公开源码加本地修改，不是新提交 CI 或完整 QA。没有安装依赖、调用模型、重跑 AppWorld 官方评分或续跑旧 `test_normal`。版本保持 0.8.0。

| 修改与复现 | 修复后的证据 | 限制 |
| --- | --- | --- |
| 显式 retry 在 SQLite 提交期间取消 caller，事务提交后 Run 留在 queued；关闭时未接管它，关闭后也仍接受 retry。真实 Database 回归旧代码 3 failed | 复用 Runtime 的 `_submissions`、完成回调、`shield` 与 shutdown；三个回归通过。相关 Runtime、取消、supervisor、SocketServer、V2 IPC 集合 **41 passed** | 显式 retry 会创建新 attempt，不新增 retry 幂等保证；不能回滚已有工具副作用 |
| 静默 SSE transport 下取消流，`Queue.get()` 仍 pending；旧代码 1 failed | finally 取消并等待 getter。普通查询另将不存在资源映射 404、Core 不可用映射 503；404 旧回归 2 failed。Web 与架构集合 **37 passed** | 保留 keepalive、overflow、cursor；不是新的前端浏览器验收 |
| POSIX 只发送 TERM 就退休记录，受控 Linux 子进程仍存活 | TERM 最多等 5 秒，未退出时重新核对身份再 KILL、最多再等 5 秒；无法确认退出就停止恢复。AppWorld 进程/恢复/worker 加 QA 门槛与公开摘要回归共 **67 passed、2 skipped** | 两个 skip 是要求 Linux `/proc` 的真实进程用例；完整项目环境仍需 Linux Python 3.12 |

现有 WSL Ubuntu 的真实受控子进程专项通过：正常 TERM 退出码 −15，忽略 TERM 后 KILL 退出码 −9。已有环境只有 Python 3.10，探针加载当前源码中仅依赖标准库的身份/退出函数原文，不使用模型或 AppWorld；不能算 Python 3.12 完整项目或真实 AppWorld 崩溃恢复全链通过。回归另外确认身份/命令变化、查询失败或强杀后仍存活时保留 running，不清理旧 world、不启动下一 attempt、不改 checkpoint。

本地 `qa.py quick` 的锁文件、全局 Ruff、严格 Mypy、协议、架构、文档、工作流与空白检查通过。CI 仅删除 quick 已执行的三个重复步骤，原检查要求保留。公开摘要离线复算通过，模型成绩仍属于 `a7b0f68`；`source-equivalence.json` 的 158 文件与指纹重新核对为 main `917d0e9`，没有改成当前 patch 的指纹。首次 QA 辅助回归因临时目录位于仓库内、Git 向上发现父仓库而失败；限定 Git 搜索边界后通过，未修改该测试或降低门槛。本轮未重跑 coverage/full、包安装、真实 Docker、浏览器或人工 TUI 验收。

## 公开发布准备的历史检查

公开候选的 QA、覆盖率、构建、安全审计、独立安装以及 PR/合并后 CI 必须分别记录最新提交结果。发布准备期间不将尚未完成的检查写成通过；下列历史数字不代替当前候选验收。

[源码对应记录](source-equivalence.json)绑定公开 main `917d0e9`，逐项核对 158 个运行时文件：156 个与历史实验源码相同，差异仅在 MCP 客户端与 AppWorld 的 Python 启动环境修复；依赖也有明确安全更新。该记录描述当时的对应关系，后续 retry、SSE 和 worker 退出小修不再与此指纹等价。历史模型成绩仍归属 `a7b0f68`，不能当作后续修复重新实测的结果。

本次不调用付费模型，不重跑完整 AppWorld，不把摘要复算写成新的官方评分。最终提交的远端结果见 [GitHub CI](https://github.com/666666999999666/TARS-Agent/actions/workflows/ci.yml)，合并要求最新候选全部门槛通过。

完成下面的 MCP 与依赖修复后，首个 PR 发布前的公开候选第二轮本地完整 QA **全部门槛通过**：1357 passed、4 skipped、11 deselected；行覆盖率 **83.94%**、分支 **69.36%**、综合 **80.74%**。四个跳过项为三个明确的 POSIX 用例和一个既有 Windows 符号链接环境限制。静态检查、类型检查、文档、覆盖率、构建、归档、Bandit、实际 HEAD 历史扫描、pip/npm 审计均通过。该轮属于原公开候选的本地证据，不能代替本轮 AppWorld 与跨平台测试辅助函数修复后的验收。

上述候选同一轮最终 wheel 与独立安装副本的 SHA256 完全一致。Windows 新建原生虚拟环境安装锁定依赖后，四个命令入口、数据库资源、打包 worker 及 Web smoke 通过，模型请求为零。这是本地安装 smoke，完整 Core/Docker 安装仍由下面的 Ubuntu CI 门槛验证。

首个 PR 发布前的公开候选首轮本地 QA 的测试为 **1355 passed、1 skipped、11 deselected**，覆盖率为行 83.88%、分支 69.33%、综合 80.69%。构建、Bandit、内容扫描和 npm 审计通过，但 pip-audit 在 PyJWT 2.14.0、urllib3 2.7.0 中发现四项漏洞，因此该轮完整 QA **失败**，原记录保留。

随后只更新两个锁定依赖版本：运行时 PyJWT 2.15.1、审计工具链 urllib3 2.8.0，并声明对应安全下限，其他锁定包版本不变；更新后独立依赖审计未发现已知漏洞。[PyJWT 官方修复说明](https://github.com/jpadilla/pyjwt/releases/tag/2.15.1)和 [urllib3 官方安全说明](https://github.com/urllib3/urllib3/releases/tag/2.8.0)给出修复范围。这属于发布时的安全维护，不能改写历史实验的依赖版本。

MCP 旧实现把虚拟环境 Python 的符号链接转换为基础解释器路径，可能丢失虚拟环境依赖。修复保留实际启动入口，并继续使用解析后的目标进行现有路径检查。新增测试在旧实现上得到 2 failed；修复后与原 MCP 单元和集成测试合计 20 passed。三个 POSIX 真实虚拟环境/符号链接用例在 Windows 明确跳过，须由 Ubuntu CI 实际运行；没有通过给集成测试额外注入依赖路径来隐藏缺陷。

首个 PR 的 [Ubuntu Python CI](https://github.com/666666999999666/TARS-Agent/actions/runs/37041400253)（run `37041400253`、head `111b0f7`）**失败**：5 failed、1342 passed、4 skipped、11 deselected、10 errors。归档日志显示，多个 Core/daemon 子进程从系统目录加载 `typing_extensions`，报 `ImportError: cannot import name 'Sentinel'`；其测试辅助函数在 POSIX 也无条件选择并解析 `sys._base_executable`，绕过了虚拟环境入口。原 CI 附件已在本地归档，未作为公开原始日志提交。归档未覆盖每项失败的完整子进程 stderr，因此尚未逐项确认所有失败都来自同一根因。

发布准备修复让测试辅助函数在 POSIX 保留 `sys.executable`，继续使用当前虚拟环境；Windows 保留已有的基础解释器与原生进程句柄方案。AppWorld 的 `python_environment()` 也修复同类问题：POSIX 保留虚拟环境 Python 入口，Windows 仍解析基础解释器路径。这两处属于发布时的启动环境修复，没有重跑付费模型或改写原 Windows AppWorld 成绩。该候选及合并后 main 已取得上节列出的完整 CI 成功；旧 Ubuntu 失败仍归属 `111b0f7`。

首个 PR 发布前，本机 Docker Desktop 因旧套接字无法访问而启动失败，该轮本地不能新增真实 Docker 或完整 Core 安装通过结论。PR 的 Ubuntu CI 必须真实执行崩溃恢复、多实例隔离、路径边界，以及检出目录外的锁定依赖独立安装；检查四个入口、数据库、Core、worker、Web、进程端口收尾和零模型请求，不能以跳过代替通过。原验证器保持不变。

首个 PR Ubuntu 失败后，测试启动辅助程序的回归在旧实现上得到 4 failed；修复后相关 MCP/IPC 测试为 20 passed、4 个 POSIX 用例在 Windows 跳过。AppWorld 平台启动回归先复现 1 failed，修复后与 worker、恢复及对照测试合计 43 passed、1 个 POSIX 用例跳过。两组均为离线检查；最新提交仍必须取得自己的完整 CI 结果。

## 历史冻结候选的完整检查

| 检查 | 历史已确认结果 | 范围 |
| --- | --- | --- |
| 完整 QA 第三轮 | 1325 passed、1 skipped、11 deselected；17 个既定门槛通过 | 源码 `a7b0f68`，不是本次公开候选新测数字 |
| 覆盖率 | 行 83.8716964%、分支 69.2761394%、综合 80.6713303% | 总体及变更覆盖率门槛通过 |
| 静态、构建与安全 | Ruff、Mypy、锁文件、协议/架构/文档/工作流、空白、包构建与归档、Bandit、公开扫描、pip/npm 审计通过 | 当时的候选、工具和依赖环境 |
| 独立安装第五轮 | wheel 安装，四个入口、数据库资源、Core 启停、打包 worker、Web 健康和收尾通过 | Windows 独立环境；原验证器未改 |
| 安装独立复核 | 16 个环境锚点未变，Core/Web 进程和端口已关闭，模型请求 0 次 | 不代表模型任务效果 |

1 个 skip 是既有 Windows 符号链接用例；11 个 deselected 是离线命令明确排除的用例，不能写成所有测试均已执行。日志有两个故意构造非法模型字段的测试警告。安装验证使用新建原生 CPython 虚拟环境，以区分 Windows uv 启动器与实际服务进程；主环境及旧安装环境未修改。

第一轮 DeepSeek QA 的测试为 1318 passed、1 skipped，但 pip-audit 报告 PyJWT 2.13.0 的十项漏洞，完整 QA 因此失败。随后仅更新该依赖至 2.14.0 并声明安全下限；第二轮通过。费用校验发布及部分用量报告修复后，第三轮与第五轮独立安装再次通过。旧失败记录保留，不把“测试通过”改写成该轮完整 QA 通过。

## 运行时专项

以下专项是先前已执行的记录，本次发布不重复付费模型检查，也不把历史人工界面观察改写成当前 CI 截图。

| 能力 | 设计与实际验证 | 限制 |
| --- | --- | --- |
| Core 崩溃恢复 | 同一数据目录 OS 锁；容器创建前持久化意图、执行前确认完整身份；真实 A 重启回收旧容器，B 实例身份与心跳正常 | 不承诺撤销已发生的外部修改；无可信归属的旧容器不自动认领 |
| Docker 路径边界 | 真实 worker 正常读取通过、越界拒绝、外部哨兵未改变或泄露，确认清理完成 | CLI 权限拒绝不能代替此 worker 验证 |
| Skill 拒绝 | 不存在、非法或加载失败时，在保存消息、创建 Run、模型调用前拒绝 | 提交途中断线的结果未知仍需单独查询 |
| 完整审批参数 | 保留摘要，`v` 打开可滚动参数；查看不批准，过期请求不可批准 | 参数变更需要新的审批 |
| CLI 正式入口 | 正常、工具失败、取消、恢复、拒绝审批、越界六类，各三轮均通过 | 保留原 harness 失败记录及修复版本归属 |
| TUI 实际界面 | ConPTY/xterm 深色 99×27、浅色 83×24，检查首尾、滚动与文字归属；80×24 有布局自动检查 | 未覆盖原生 Windows Terminal 所有尺寸 |

Core 启动时先完成旧资源恢复再接受连接；`preferred` 同样不能绕过该流程。运行期间允许的宿主回退仍须单独审批。未知归属、查询失败、记录损坏或清理未确认时停止启动。恢复后的旧任务标为 `interrupted`，不重放工具。公开 RPC 和 Run 状态类型保持兼容。

## 费用账本故障与修复

第一批 train `001` 保存 5/24 次运行后暂停，另有一次基础设施失败：数据库已更新，费用校验文件未完成发布。数据库当时有 98 条 HTTP 尝试预留，已确认用量对应估算 0.525641760 元，未知预留 2.162688000 元。

停止调用并确认没有在途操作后，先归档数据库、旧校验文件、待发布文件及只读检查结果。核对待发布文件与数据库记录、摘要和原请求账本一致后，只发布该文件；数据库字节、预算 ID、50 元上限和金额未改变，没有退款或重置预算。

公共实现对 `PermissionError` 增加最多六次、累计等待 0.75 秒的校验文件发布重试，不额外预留费用或调用模型；持续 I/O 故障仍拒绝继续调用。回归包含真实 Windows 文件占用。原故障未记录 OSError 子类型，不能断言某个外部程序就是实际根因。

同时修复 `confirmed_usage`：已取得完整响应后，后续调用失败不再清空此前已确认 Token；完整 `usage` 与 `usage_complete` 仍保留未知。回归使用真实 SDK 和离线 HTTP 替身模拟后续费用发布故障，旧原始结果不事后改写。

修复后新建批次 `002`，沿用原累计请求及费用账本，完整执行 train 24 次和 dev 114 次；新批次无基础设施补跑。两批分开保存，没有选择最好样本或合并评分，旧未知预留仍在。完整费用见[结果报告](RESULTS.md)。

## AppWorld 与模型接入证据

AppWorld 固定官方源码 `42b5bcf`、数据 0.2.0，独立 Linux 兼容镜像仅固定 SQLModel 0.0.44；当时官方安装验证 1840 项通过，未改官方任务和评分器。管理端口绑定 loopback，独立 bridge 的出站网络未禁用，与文件工具的 `network=none` 沙箱不同。

DeepSeek 真实接入共五次 HTTP 尝试：四次完整响应形成读、写、读回和最终回答，实际返回模型均为 `deepseek-flash` 且包含思考块；第五次在首个文本 token 后取消，无重发。取消阶段未收到完整响应，实际模型和用量保留未知。中文 JSON 在文件中正确保留，最终回答未复述中文值，因此不宣称完成了中文最终答复验证。

内部历史 48 次为单 Agent 24/24、多 Agent 23/24；完整 dev 配对 A 20/57、B 54/57。官方分数、运行状态、负面案例和成本分别见[结果报告](RESULTS.md)。旧 `test_normal` 的 63/168 题仍未完成，没有完整 TGC/SGC。

完整项目环境中，Linux AppWorld 调度器崩溃后旧 worker 退出与恢复 attempt 的全链验证仍未完成。上文受控 Linux 进程探针、Windows 真实恢复和 Linux 文件沙箱恢复均不能代替这项 AppWorld 调度器证据。本轮也没有新增长对话压缩实测或生产环境验收。
