# 本地打包与源码恢复检查

`final_delivery.py` 从干净的 main 提交导出源码 ZIP 和 Git bundle，可附带一个指向同一提交的稳定标签。它不会修改源仓库分支或标签，也不会推送远程。

## 使用方法

在项目根目录运行，输出位置必须是仓库外的新空目录：

```powershell
uv run --no-sync python scripts/final_delivery.py --main-ref main --stable-tag baseline-v1 --output ../TARS-Agent-export --preflight
uv run --no-sync python scripts/final_delivery.py --main-ref main --stable-tag baseline-v1 --output ../TARS-Agent-export
```

没有稳定标签时省略 `--stable-tag`，导出结果会明确标记为候选快照。可用 `--acceptance-summary <文件>` 附带经过脱敏的实际验收 JSON；缺省时清单标明未提供，不能把源码恢复检查当作运行验收。

## 检查内容

- main 必须是当前干净工作区的准确提交；仓库历史完整，不依赖 Git alternates，不包含待处理 stash。
- 使用当前仓库的初始化提交验证历史可达性和差异，不引用其他项目的提交号。
- 检查所选分支的文件和可达历史，拒绝私有运行文件及未处理的敏感内容告警。
- ZIP 的路径和内容逐项对照 Git blob；bundle 只导出 main 和明确指定的标签。
- 实际克隆 bundle，核对提交、标签、文件、fsck 和公共内容检查。
- 输出源码 ZIP、bundle、`verification.json`、`handoff-manifest.json` 和 SHA256 校验文件。已有输出不会被覆盖。

安装包另用 `uv run --no-sync python scripts/qa.py build` 构建，输出在 `build/qa/dist`。构建成功后应按[验收方法](../docs/baseline/ACCEPTANCE.md)检查安装和实际使用。
