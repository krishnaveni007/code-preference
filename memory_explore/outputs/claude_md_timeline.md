# CLAUDE.md 在仓库提交序列里出现的位置

对每个仓库，把数据集里该仓库的所有提交按时间排序，再定位第一次动 CLAUDE.md 的提交在第几位。

两种情况必须分开看：

- **观测到诞生**（首个事件是 `A`）：文件在观测期内被创建，所以“之前有几次提交”是真实的、确实没有 CLAUDE.md 的提交数。共 **23** 个仓库。
- **早于观测期**（首个事件是 `M`/`D`）：观测开始时文件已存在，“之前有几次提交”没有意义 —— 只是没看到它诞生。共 **48** 个仓库。

**贯穿性限制**：数据集只有这些仓库约 7.3% 的提交、约 2.5 个月的窗口，所以“观测序列里的位置”不等于仓库真实历史里的位置。

## 观测到诞生的仓库

`位置` = 第一次出现在观测序列的第几个提交；`之前/之后` = 该位置前后的提交数；
`动过的提交` = 观测期内一共有几个提交动过它；`分布` = 这些提交在序列中的位次。

| 仓库 | 观测提交数 | 位置 | 之前 | 之后 | 动过的提交 | 诞生日 | 首次归属 | 分布 |
|---|---|---|---|---|---|---|---|---|
| osabiohq/osabio | 971 | 99 | 98 | 872 | 13 | 2026-03-04 | agent_only | 99,102,106,114,221,241,575,840,841,849,927,930,964 |
| marcus-sa/brain | 850 | 99 | 98 | 751 | 12 | 2026-03-04 | agent_only | 99,102,106,114,221,241,719,720,728,806,809,843 |
| timelabs-npo/rhea-project | 375 | 7 | 6 | 368 | 4 | 2026-02-14 | agent_only | 7,8,22,27 |
| obsessiondb/rudel | 362 | 54 | 53 | 308 | 32 | 2026-02-19 | agent_only | 54,55,60,66,78,79,82,87,92,93,97,100,104,105,… |
| wanshicheng/duckdb-data-agent | 339 | 173 | 172 | 166 | 1 | 2026-02-25 | agent_only | 173 |
| cyyeh/duckdb-data-agent | 315 | 170 | 169 | 145 | 1 | 2026-02-25 | agent_only | 170 |
| oddessentials/ado-git-repo-insights | 309 | 251 | 250 | 58 | 2 | 2026-04-04 | agent_only | 251,259 |
| BIDEquity/outbid-dirigent | 72 | 40 | 39 | 32 | 1 | 2026-04-02 | human_only | 40 |
| serg-alexv/rhea-project | 46 | 7 | 6 | 39 | 4 | 2026-02-14 | human_only | 7,8,22,27 |
| blittle/pressy | 41 | 1 | 0 | 40 | 1 | 2026-02-11 | human_only | 1 |
| chuhemiao/portfolio | 37 | 6 | 5 | 31 | 2 | 2026-03-11 | agent_only | 6,21 |
| vfaraji89/tokalator | 34 | 10 | 9 | 24 | 2 | 2026-03-21 | agent_only | 10,32 |
| bids-standard/bids-utils | 24 | 7 | 6 | 17 | 1 | 2026-04-04 | agent_only | 7 |
| moven0831/anonbook | 21 | 2 | 1 | 19 | 3 | 2026-03-10 | agent_only | 2,3,14 |
| kgcrom/agent-foundry | 12 | 2 | 1 | 10 | 2 | 2026-02-21 | human_only | 2,3 |
| pambrose/srcref | 9 | 1 | 0 | 8 | 3 | 2026-02-11 | mixed | 1,2,9 |
| Tiryoh/entireio-local-viewer | 8 | 4 | 3 | 4 | 1 | 2026-03-09 | mixed | 4 |
| PanthroCorp-Limited/openclaw-skills | 7 | 1 | 0 | 6 | 1 | 2026-03-29 | agent_only | 1 |
| vaayne/agent-kit | 6 | 3 | 2 | 3 | 1 | 2026-03-13 | human_only | 3 |
| snhryt-neo/dotfiles | 3 | 2 | 1 | 1 | 1 | 2026-02-14 | agent_only | 2 |
| sintezcs/jetcodesync | 2 | 2 | 1 | 0 | 1 | 2026-02-23 | human_only | 2 |
| nuttycc/LuminTime | 2 | 2 | 1 | 0 | 1 | 2026-02-14 | agent_only | 2 |
| jfmoe/react-native-live-text-view | 1 | 1 | 0 | 0 | 1 | 2026-02-17 | agent_only | 1 |

## 文件早于观测期的仓库

这些只能看到修改，看不到诞生。

| 仓库 | 观测提交数 | 首次动它的位置 | 动过的提交 | 首个事件 | 日期 |
|---|---|---|---|---|---|
| entireio/cli | 1068 | 23 | 26 | M | 2026-01-06 |
| hutusi/amytis | 162 | 30 | 6 | M | 2026-03-04 |
| Nagi-ovo/gemini-voyager | 113 | 1 | 5 | M | 2026-02-16 |
| 135yshr/documents | 98 | 10 | 6 | M | 2026-03-03 |
| desplega-ai/agent-swarm | 88 | 2 | 5 | M | 2026-03-05 |
| melagiri/code-insights | 83 | 13 | 7 | M | 2026-03-02 |
| gregszero/open-fang | 79 | 1 | 20 | M | 2026-02-16 |
| tpmjs/tpmjs | 76 | 28 | 5 | M | 2026-02-12 |
| PackmindHub/context-evaluator | 72 | 42 | 2 | M | 2026-02-20 |
| ClusterCockpit/cc-backend | 68 | 34 | 1 | M | 2026-03-16 |
| pc035860/cee | 68 | 5 | 15 | M | 2026-03-07 |
| henryph24/neuralips26 | 65 | 51 | 1 | M | 2026-04-10 |
| jdsingh122918/forge | 54 | 6 | 2 | M | 2026-03-01 |
| pablo-health/pablo-companion | 49 | 44 | 1 | M | 2026-03-09 |
| matthsena/reef-coder | 41 | 16 | 3 | M | 2026-02-18 |
| LitMc/gc-playground | 41 | 10 | 18 | M | 2026-02-24 |
| andrefurt/safo | 40 | 32 | 2 | M | 2026-02-12 |
| tslateman/reck | 40 | 6 | 1 | M | 2026-03-08 |
| ckeith26/cameronkeithgolf | 39 | 17 | 2 | M | 2026-02-27 |
| camkeith/cameronkeithgolf | 39 | 17 | 2 | M | 2026-02-27 |
| CPS-IT/quality-tools | 38 | 17 | 1 | M | 2026-02-28 |
| schmalle/secman | 34 | 4 | 3 | M | 2026-02-26 |
| ravencloak-org/ravencloak | 34 | 29 | 1 | M | 2026-03-14 |
| jukellam/dispersal-draft | 33 | 1 | 4 | M | 2026-02-10 |
| yorrick/claude-code-plugins | 24 | 13 | 2 | M | 2026-03-17 |
| pc035860/agent-tail | 22 | 2 | 8 | M | 2026-03-07 |
| vaayne/anna | 22 | 5 | 2 | M | 2026-03-13 |
| bravegeek/ai-workshop | 21 | 14 | 1 | M | 2026-04-06 |
| tslateman/duet | 20 | 5 | 8 | M | 2026-02-20 |
| ASRagab/optimize-anything | 20 | 6 | 2 | M | 2026-02-25 |
| ujuc/agent-stuff | 17 | 1 | 6 | M | 2026-02-11 |
| mhavelock/white-hat-label | 17 | 1 | 4 | M | 2026-04-06 |
| moltis-org/moltis | 17 | 10 | 2 | M | 2026-03-11 |
| Whiteknight07/AiTutor | 12 | 7 | 1 | M | 2026-04-14 |
| punt-labs/langlearn-tts | 12 | 11 | 2 | M | 2026-03-09 |
| ujuc/dotrc | 11 | 2 | 2 | D | 2026-02-16 |
| AndreaPT1/mkdownEditor | 8 | 1 | 2 | D | 2026-03-29 |
| tvararu/tuicraft | 8 | 5 | 2 | M | 2026-02-23 |
| eneakllomollari/mpad | 4 | 3 | 1 | M | 2026-04-12 |
| sparkling/claude-flow-patch | 4 | 1 | 4 | M | 2026-02-18 |
| roo-oliv/monodreams | 3 | 1 | 2 | M | 2026-02-15 |
| Lightprotocol/light-protocol | 3 | 2 | 1 | M | 2026-02-17 |
| punt-labs/koch-trainer-swift | 3 | 1 | 1 | M | 2026-02-12 |
| 135yshr/savanna-vet-go | 3 | 1 | 1 | M | 2026-03-12 |
| nsega/mcp-todoist | 2 | 2 | 1 | M | 2026-04-19 |
| partio-io/cli | 2 | 1 | 1 | M | 2026-02-27 |
| libfunc/rapira | 1 | 1 | 1 | T | 2026-02-21 |
| gagan114662/moltbot | 1 | 1 | 1 | D | 2026-02-15 |

## 汇总（只统计观测到诞生的那组）

| 指标 | 值 |
|---|---|
| 仓库数 | 23 |
| 诞生前有提交、之后也有提交（真正“出现在中间”） | 17 |
| 诞生就在观测序列第 1 位（之前没有提交） | 4 |
| 诞生后再没有提交 | 3 |
| 诞生前提交数 中位/p90/最大 | 5 / 169 / 250 |
| 诞生后提交数 中位/p90/最大 | 24 / 368 / 872 |
| 动过它的提交数 中位/最大 | 1 / 32 |
| 诞生后再也没被改过的仓库 | 12 |
