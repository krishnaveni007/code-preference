# CLAUDE.md 里哪些小标题是关于 preference 的

统计对象：**74** 个内容不重复的 CLAUDE.md，共 **1092** 个标题。
其中 **29** 个是 `/init` 生成的样板标题（`# CLAUDE.md` 等），已剔除。

归入 preference 的：**304**（占非样板标题的 29%）。未归类：**759**。

匹配按**词边界**进行，不是子串 —— 否则 `pr` 会匹配到 `project`。每个被归类的标题都在下面列出，请自行核对。

## 各类数量

| 类别 | 标题数 | 出现在几个文件 |
|---|---|---|
| 硬性禁止 / 必须做 | 33 | 20 |
| 代码风格 / 命名 / 格式 | 29 | 22 |
| 测试要求 | 56 | 26 |
| Git / 提交 / 分支 / PR | 34 | 18 |
| 流程 / 检查清单 / 质量门 | 50 | 25 |
| 设计原则 / 模式偏好 | 36 | 28 |
| 对 agent 的行为指令 | 12 | 11 |
| 笼统的约定 / 规则 / 坑 | 54 | 32 |

## 硬性禁止 / 必须做

共 33 个标题，29 个不同，出现在 20 个文件。

| 次数 | 标题 |
|---|---|
| 2 | do not |
| 2 | do |
| 2 | core rules |
| 2 | required environment variables |
| 1 | ⛔ hard stop — before you do anything |
| 1 | every session — no exceptions |
| 1 | hard rules |
| 1 | important: do not import css in react components |
| 1 | why you must clean before building |
| 1 | required flow path |
| 1 | golden rule |
| 1 | legacy modules (do not extend or modify) |
| 1 | legacy scripts (do not extend) |
| 1 | before every commit (required) |
| 1 | force unwrapping — never use |
| 1 | implicitly unwrapped optionals — never use |
| 1 | ⚠️ critical: always use worktrees |
| 1 | before creating a new defect, always search first |
| 1 | ⚠️ protected files — do not modify without explicit instruction |
| 1 | paradedb setup (required for v2+ migrations) |
| 1 | critical formatting requirements |
| 1 | critical rules — read these first |
| 1 | development philosophy (critical) |
| 1 | template cleanup required |
| 1 | frontmatter (required) |
| 1 | quality gates (mandatory) |
| 1 | releases (must follow) |
| 1 | required package declaration |
| 1 | pre-commit gate: tox must pass |

## 代码风格 / 命名 / 格式

共 29 个标题，25 个不同，出现在 22 个文件。

| 次数 | 标题 |
|---|---|
| 5 | code style |
| 1 | frontend style changes |
| 1 | filename format |
| 1 | result file naming convention |
| 1 | linting and formatting |
| 1 | go code style |
| 1 | swiftformat & swiftlint compliance |
| 1 | branch naming convention |
| 1 | comment hygiene |
| 1 | layout & style guide |
| 1 | rust style and idioms |
| 1 | format and lint |
| 1 | linting |
| 1 | css tokens (defined in web/public/css/style.css) |
| 1 | php cs fixer (code style) |
| 1 | typoscript lint |
| 1 | message format |
| 1 | testing/lint/format/hooks status (current repo) |
| 1 | lint |
| 1 | formatting rules |
| 1 | naming conventions |
| 1 | wire format |
| 1 | commit format |
| 1 | lint ルール |
| 1 | file format |

## 测试要求

共 56 个标题，41 个不同，出现在 26 个文件。

| 次数 | 标题 |
|---|---|
| 14 | testing |
| 3 | build & test |
| 1 | evaluator coverage areas |
| 1 | test (90-99% savings) |
| 1 | running tests |
| 1 | running integration tests |
| 1 | running all tests (ci) |
| 1 | running e2e canary tests (vogon agent) |
| 1 | running e2e tests (only when explicitly requested) |
| 1 | test parallelization |
| 1 | git in tests |
| 1 | testing requirements |
| 1 | coverage status |
| 1 | static views excluded from coverage |
| 1 | test file patterns |
| 1 | step 6: update docs and test |
| 1 | fixtures |
| 1 | adding tests for a new defect |
| 1 | unit tests |
| 1 | e2e testing with docker |
| 1 | entrypoint integration testing |
| 1 | mcp tool testing (streamable http) |
| 1 | ui testing |
| 1 | test programme |
| 1 | test infrastructure |
| 1 | e2e tests (web ui) |
| 1 | testing patterns |
| 1 | unit tests (program-libs/) |
| 1 | integration tests (program-tests/) |
| 1 | sdk tests (sdk-tests/) |
| 1 | javascript/typescript tests |
| 1 | go prover tests |
| 1 | forester tests |
| 1 | test organization principles |
| 1 | test assertion pattern |
| 1 | testing conventions |
| 1 | testing notes |
| 1 | run a single test file |
| 1 | running & testing |
| 1 | build & test commands |
| 1 | testing requirements for /workflow (stategraph engine) |

## Git / 提交 / 分支 / PR

共 34 个标题，30 个不同，出现在 18 个文件。

| 次数 | 标题 |
|---|---|
| 3 | git workflow |
| 2 | release workflow |
| 2 | changelog |
| 1 | pre-commit hook |
| 1 | git (59-80% savings) |
| 1 | git operations |
| 1 | commit trailers |
| 1 | commit message convention |
| 1 | parallel development with worktrees |
| 1 | worktree location |
| 1 | creating a worktree |
| 1 | working in a worktree |
| 1 | listing worktrees |
| 1 | removing a worktree |
| 1 | syncing beads across branches |
| 1 | pull request titles |
| 1 | entity merge (asset) |
| 1 | manual release (cli) |
| 1 | pr descriptions |
| 1 | git hooks (lefthook) |
| 1 | git identity |
| 1 | pr/faq |
| 1 | ブランチ運用と pr ワークフロー |
| 1 | pr の作業フロー |
| 1 | commits |
| 1 | git worktree support |
| 1 | release |
| 1 | version bump & release |
| 1 | git |
| 1 | git hooks |

## 流程 / 检查清单 / 质量门

共 50 个标题，38 个不同，出现在 25 个文件。

| 次数 | 标题 |
|---|---|
| 5 | development workflow |
| 4 | code quality |
| 2 | workflow |
| 2 | pre-pr checklist |
| 2 | quality standards |
| 2 | quality gates |
| 2 | ci |
| 1 | before you start: read the spec, actually read it |
| 1 | production build pipeline: css generation |
| 1 | production build pipeline |
| 1 | build pipeline stages |
| 1 | correct build workflow |
| 1 | implementation checklist |
| 1 | rtk commands by workflow |
| 1 | orchestrator shell scripts (race vm workflow) |
| 1 | race vm workflow |
| 1 | z specification workflow |
| 1 | workflow by scenario |
| 1 | full checklist |
| 1 | pre-push checklist |
| 1 | your checklist |
| 1 | claude's checklist |
| 1 | woodpecker ci |
| 1 | pipeline overview |
| 1 | code quality checklist |
| 1 | github actions (ci) |
| 1 | debugging workflow |
| 1 | beads workflow |
| 1 | agent team workflow |
| 1 | definition of done |
| 1 | code quality standards |
| 1 | common workflows |
| 1 | code review flow |
| 1 | implementation workflow |
| 1 | rendering pipeline |
| 1 | validation workflow for changes |
| 1 | validation checklist (every change) |
| 1 | review |

## 设计原则 / 模式偏好

共 36 个标题，29 个不同，出现在 28 个文件。

| 次数 | 标题 |
|---|---|
| 3 | design principles |
| 3 | behavioral guidelines |
| 2 | key design decisions |
| 2 | best practices |
| 2 | key patterns |
| 1 | voice principles |
| 1 | color philosophy |
| 1 | worker-sharding pattern |
| 1 | code patterns |
| 1 | implementation guidelines |
| 1 | spec-code alignment principles |
| 1 | key design patterns |
| 1 | technical development approach |
| 1 | common patterns |
| 1 | duplicate prevention pattern |
| 1 | oauth robustness pattern |
| 1 | memory optimization pattern |
| 1 | server-injected data (gon pattern) |
| 1 | communication guidelines |
| 1 | core philosophy |
| 1 | friction & pattern taxonomy |
| 1 | key patterns & conventions |
| 1 | rego policy patterns |
| 1 | development paradigm |
| 1 | guidelines |
| 1 | design philosophy |
| 1 | development guidelines |
| 1 | best practice |
| 1 | ado-git-repo-insights development guidelines |

## 对 agent 的行为指令

共 12 个标题，10 个不同，出现在 11 个文件。

| 次数 | 标题 |
|---|---|
| 3 | work rules |
| 1 | imi — agent instruction manual |
| 1 | imi — ai voice guide |
| 1 | multi-session behavior |
| 1 | beads redirect behavior |
| 1 | background task behavior |
| 1 | agent instructions |
| 1 | how to learn the codebase (for agents) |
| 1 | role behavior |
| 1 | platform and behavior constraints |

## 笼统的约定 / 规则 / 坑

共 54 个标题，41 个不同，出现在 32 个文件。

| 次数 | 标题 |
|---|---|
| 6 | rules |
| 4 | key conventions |
| 3 | gotchas |
| 3 | conventions |
| 2 | standards |
| 1 | datasets, backbones, conventions |
| 1 | seeds and statistical conventions |
| 1 | mse convention |
| 1 | conventions and gotchas |
| 1 | dependencies (requirements.txt) |
| 1 | technical constraints |
| 1 | swift 6 gotchas |
| 1 | xcodegen gotchas |
| 1 | appkit gotchas |
| 1 | xcuitest gotchas |
| 1 | github issue policy |
| 1 | bun rules |
| 1 | adding a new auction rule |
| 1 | css conventions |
| 1 | standards compliance |
| 1 | api namespace convention |
| 1 | architecture rules |
| 1 | the rust decision rule |
| 1 | ffi boundary rule |
| 1 | swift / xcode conventions (mac/) |
| 1 | rust conventions (core/) |
| 1 | standard path detection |
| 1 | implementation completion standards |
| 1 | coding standards |
| 1 | standards references |
| 1 | performance rules |
| 1 | protocol gotchas |
| 1 | known gotchas |
| 1 | key constraints |
| 1 | constraints |
| 1 | regorus wasm policy evaluation |
| 1 | queried rules |
| 1 | language policy |
| 1 | coding conventions |
| 1 | clickup task resolution (shared convention for skills) |
| 1 | requirements |

## 同时命中多类的标题（已归入第一类，此处仅供核对）

| 次数 | 命中的类别组合 |
|---|---|
| 5 | 硬性禁止 / 必须做 + 笼统的约定 / 规则 / 坑 |
| 5 | 设计原则 / 模式偏好 + 对 agent 的行为指令 |
| 5 | Git / 提交 / 分支 / PR + 流程 / 检查清单 / 质量门 |
| 4 | 测试要求 + 设计原则 / 模式偏好 |
| 4 | 对 agent 的行为指令 + 笼统的约定 / 规则 / 坑 |
| 3 | 代码风格 / 命名 / 格式 + 笼统的约定 / 规则 / 坑 |
| 3 | 流程 / 检查清单 / 质量门 + 笼统的约定 / 规则 / 坑 |
| 2 | 测试要求 + 笼统的约定 / 规则 / 坑 |
| 2 | 硬性禁止 / 必须做 + Git / 提交 / 分支 / PR |
| 2 | 设计原则 / 模式偏好 + 笼统的约定 / 规则 / 坑 |
| 1 | 测试要求 + 流程 / 检查清单 / 质量门 |
| 1 | 测试要求 + Git / 提交 / 分支 / PR |
| 1 | 硬性禁止 / 必须做 + Git / 提交 / 分支 / PR + 流程 / 检查清单 / 质量门 |
| 1 | 代码风格 / 命名 / 格式 + Git / 提交 / 分支 / PR + 笼统的约定 / 规则 / 坑 |
| 1 | Git / 提交 / 分支 / PR + 笼统的约定 / 规则 / 坑 |
| 1 | 代码风格 / 命名 / 格式 + 设计原则 / 模式偏好 |
| 1 | 硬性禁止 / 必须做 + 代码风格 / 命名 / 格式 + 笼统的约定 / 规则 / 坑 |
| 1 | 硬性禁止 / 必须做 + 设计原则 / 模式偏好 |
| 1 | 代码风格 / 命名 / 格式 + 测试要求 |
| 1 | 代码风格 / 命名 / 格式 + Git / 提交 / 分支 / PR |
| 1 | 硬性禁止 / 必须做 + 流程 / 检查清单 / 质量门 |
| 1 | 测试要求 + 流程 / 检查清单 / 质量门 + 笼统的约定 / 规则 / 坑 |

## 每个文件的 preference 标题数

| 文件 | preference 标题数 |
|---|---|
| [punt-labs~koch-trainer-swift__CLAUDE.md](punt-labs~koch-trainer-swift__CLAUDE.md) | 25 |
| [PackmindHub~context-evaluator__CLAUDE.md](PackmindHub~context-evaluator__CLAUDE.md) | 23 |
| [entireio~cli__CLAUDE.md](entireio~cli__CLAUDE.md) | 15 |
| [punt-labs~langlearn-tts__CLAUDE.md](punt-labs~langlearn-tts__CLAUDE.md) | 13 |
| [Lightprotocol~light-protocol__CLAUDE.md](Lightprotocol~light-protocol__CLAUDE.md) | 12 |
| [pablo-health~pablo-companion__CLAUDE.md](pablo-health~pablo-companion__CLAUDE.md) | 12 |
| [henryph24~neuralips26__CLAUDE.md](henryph24~neuralips26__CLAUDE.md) | 11 |
| [sparkling~claude-flow-patch__CLAUDE.md](sparkling~claude-flow-patch__CLAUDE.md) | 10 |
| [moltis-org~moltis__CLAUDE.md](moltis-org~moltis__CLAUDE.md) | 10 |
| [desplega-ai~agent-swarm__CLAUDE.md](desplega-ai~agent-swarm__CLAUDE.md) | 9 |
| [mhavelock~white-hat-label__CLAUDE.md](mhavelock~white-hat-label__CLAUDE.md) | 9 |
| [CPS-IT~quality-tools__CLAUDE.md](CPS-IT~quality-tools__CLAUDE.md) | 8 |
| [AndreaPT1~mkdownEditor__CLAUDE.md](AndreaPT1~mkdownEditor__CLAUDE.md) | 7 |
| [pc035860~cee__CLAUDE.md](pc035860~cee__CLAUDE.md) | 6 |
| [schmalle~secman__CLAUDE.md](schmalle~secman__CLAUDE.md) | 6 |
| [andrefurt~safo__CLAUDE.md](andrefurt~safo__CLAUDE.md) | 5 |
| [ravencloak-org~ravencloak__CLAUDE.md](ravencloak-org~ravencloak__CLAUDE.md) | 4 |
| [tpmjs~tpmjs__CLAUDE.md](tpmjs~tpmjs__CLAUDE.md) | 4 |
| [ClusterCockpit~cc-backend__CLAUDE.md](ClusterCockpit~cc-backend__CLAUDE.md) | 4 |
| [eneakllomollari~mpad__CLAUDE.md](eneakllomollari~mpad__CLAUDE.md) | 4 |
| [tvararu~tuicraft__CLAUDE.md](tvararu~tuicraft__CLAUDE.md) | 4 |
| [moven0831~anonbook__CLAUDE.md](moven0831~anonbook__CLAUDE.md) | 4 |
| [sintezcs~jetcodesync__CLAUDE.md](sintezcs~jetcodesync__CLAUDE.md) | 4 |
| [tslateman~duet__CLAUDE.md](tslateman~duet__CLAUDE.md) | 4 |
| [Nagi-ovo~gemini-voyager__CLAUDE.md](Nagi-ovo~gemini-voyager__CLAUDE.md) | 4 |
| [marcus-sa~brain__app~src~server~policy~CLAUDE.md](marcus-sa~brain__app~src~server~policy~CLAUDE.md) | 4 |
| [yorrick~claude-code-plugins__CLAUDE.md](yorrick~claude-code-plugins__CLAUDE.md) | 4 |
| [obsessiondb~rudel__CLAUDE.md](obsessiondb~rudel__CLAUDE.md) | 3 |
| [jukellam~dispersal-draft__CLAUDE.md](jukellam~dispersal-draft__CLAUDE.md) | 3 |
| [gregszero~open-fang__CLAUDE.md](gregszero~open-fang__CLAUDE.md) | 3 |
| [gregszero~open-fang__workspace~CLAUDE.md](gregszero~open-fang__workspace~CLAUDE.md) | 3 |
| [melagiri~code-insights__CLAUDE.md](melagiri~code-insights__CLAUDE.md) | 3 |
| [partio-io~cli__CLAUDE.md](partio-io~cli__CLAUDE.md) | 3 |
| [roo-oliv~monodreams__.claude~CLAUDE.md](roo-oliv~monodreams__.claude~CLAUDE.md) | 3 |
| [nuttycc~LuminTime__CLAUDE.md](nuttycc~LuminTime__CLAUDE.md) | 3 |
| [yorrick~claude-code-plugins__dev-loop~CLAUDE.md](yorrick~claude-code-plugins__dev-loop~CLAUDE.md) | 3 |
| [135yshr~documents__CLAUDE.md](135yshr~documents__CLAUDE.md) | 3 |
| [nsega~mcp-todoist__CLAUDE.md](nsega~mcp-todoist__CLAUDE.md) | 3 |
| [LitMc~gc-playground__CLAUDE.md](LitMc~gc-playground__CLAUDE.md) | 2 |
| [Whiteknight07~AiTutor__CLAUDE.md](Whiteknight07~AiTutor__CLAUDE.md) | 2 |
| [ASRagab~optimize-anything__CLAUDE.md](ASRagab~optimize-anything__CLAUDE.md) | 2 |
| [vfaraji89~tokalator__CLAUDE.md](vfaraji89~tokalator__CLAUDE.md) | 2 |
| [vaayne~anna__CLAUDE.md](vaayne~anna__CLAUDE.md) | 2 |
| [pambrose~srcref__CLAUDE.md](pambrose~srcref__CLAUDE.md) | 2 |
| [jfmoe~react-native-live-text-view__CLAUDE.md](jfmoe~react-native-live-text-view__CLAUDE.md) | 2 |
| [serg-alexv~rhea-project__CLAUDE.md](serg-alexv~rhea-project__CLAUDE.md) | 2 |
| [marcus-sa~brain__CLAUDE.md](marcus-sa~brain__CLAUDE.md) | 2 |
| [135yshr~savanna-vet-go__CLAUDE.md](135yshr~savanna-vet-go__CLAUDE.md) | 2 |
| [ujuc~agent-stuff__claude~CLAUDE.md](ujuc~agent-stuff__claude~CLAUDE.md) | 2 |
| [ujuc~dotrc__CLAUDE.md](ujuc~dotrc__CLAUDE.md) | 2 |
| [BIDEquity~outbid-dirigent__input~lbx~CLAUDE.md](BIDEquity~outbid-dirigent__input~lbx~CLAUDE.md) | 2 |
| [oddessentials~ado-git-repo-insights__CLAUDE.md](oddessentials~ado-git-repo-insights__CLAUDE.md) | 2 |
| [bids-standard~bids-utils__CLAUDE.md](bids-standard~bids-utils__CLAUDE.md) | 2 |
| [ujuc~agent-stuff__CLAUDE.md](ujuc~agent-stuff__CLAUDE.md) | 2 |
| [ujuc~agent-stuff__claude~skills~CLAUDE.md](ujuc~agent-stuff__claude~skills~CLAUDE.md) | 2 |
| [vaayne~agent-kit__skills~pi-delegate~CLAUDE.md](vaayne~agent-kit__skills~pi-delegate~CLAUDE.md) | 2 |
| [pc035860~agent-tail__CLAUDE.md](pc035860~agent-tail__CLAUDE.md) | 1 |
| [hutusi~amytis__CLAUDE.md](hutusi~amytis__CLAUDE.md) | 1 |
| [libfunc~rapira__CLAUDE.md](libfunc~rapira__CLAUDE.md) | 1 |
| [bravegeek~ai-workshop__CLAUDE.md](bravegeek~ai-workshop__CLAUDE.md) | 1 |
| [PanthroCorp-Limited~openclaw-skills__google-workspace~CLAUDE.md](PanthroCorp-Limited~openclaw-skills__google-workspace~CLAUDE.md) | 1 |
| [camkeith~cameronkeithgolf__CLAUDE.md](camkeith~cameronkeithgolf__CLAUDE.md) | 1 |
| [kgcrom~agent-foundry__CLAUDE.md](kgcrom~agent-foundry__CLAUDE.md) | 1 |
| [jdsingh122918~forge__CLAUDE.md](jdsingh122918~forge__CLAUDE.md) | 1 |
| [PanthroCorp-Limited~openclaw-skills__zoho-mail~CLAUDE.md](PanthroCorp-Limited~openclaw-skills__zoho-mail~CLAUDE.md) | 1 |
| [tslateman~reck__CLAUDE.md](tslateman~reck__CLAUDE.md) | 1 |

完全没有 preference 标题的文件（8 个）：

- [chuhemiao~portfolio__CLAUDE.md](chuhemiao~portfolio__CLAUDE.md)
- [matthsena~reef-coder__CLAUDE.md](matthsena~reef-coder__CLAUDE.md)
- [snhryt-neo~dotfiles__CLAUDE.md](snhryt-neo~dotfiles__CLAUDE.md)
- [cyyeh~duckdb-data-agent__.claude~CLAUDE.md](cyyeh~duckdb-data-agent__.claude~CLAUDE.md)
- [marcus-sa~brain__app~src~server~CLAUDE.md](marcus-sa~brain__app~src~server~CLAUDE.md)
- [blittle~pressy__CLAUDE.md](blittle~pressy__CLAUDE.md)
- [marcus-sa~brain__app~src~client~CLAUDE.md](marcus-sa~brain__app~src~client~CLAUDE.md)
- [Tiryoh~entireio-local-viewer__CLAUDE.md](Tiryoh~entireio-local-viewer__CLAUDE.md)

## 未归类的标题（759 个，前 120）

这些多是项目专有内容，或需要读正文才能判断。

| 次数 | 标题 |
|---|---|
| 29 | architecture |
| 17 | project overview |
| 16 | commands |
| 11 | project structure |
| 8 | tech stack |
| 6 | key files |
| 6 | build commands |
| 5 | overview |
| 5 | dependencies |
| 4 | common commands |
| 4 | development |
| 4 | environment variables |
| 4 | documentation |
| 3 | directory structure |
| 3 | api endpoints |
| 3 | database migrations |
| 3 | quick reference |
| 3 | skills |
| 3 | setup |
| 3 | references |
| 2 | architecture overview |
| 2 | core flow |
| 2 | error handling |
| 2 | build & run |
| 2 | essential commands |
| 2 | packages |
| 2 | local development |
| 2 | authentication |
| 2 | implementation notes |
| 2 | stack |
| 2 | active technologies |
| 2 | recent changes |
| 2 | build & run commands |
| 2 | issue tracking |
| 2 | data flow |
| 2 | building |
| 2 | code generation |
| 2 | key concepts |
| 2 | hooks |
| 2 | recent activity |
| 2 | environment |
| 2 | コマンド |
| 2 | technical stack |
| 2 | development commands |
| 1 | mode routing |
| 1 | quick command reference |
| 1 | if imi is not installed |
| 1 | imi — ops mode |
| 1 | understanding the system you're working with |
| 1 | the commands you have in ops mode |
| 1 | how to actually engage in this mode |
| 1 | common scenarios |
| 1 | imi — plan mode |
| 1 | your commands |
| 1 | imi goal |
| 1 | imi task |
| 1 | imi log |
| 1 | imi decide |
| 1 | one goal, or just one task? |
| 1 | assess complexity before you write anything |
| 1 | discovery: understanding before you write |
| 1 | what a rich description actually looks like |
| 1 | imi — execute mode |
| 1 | your imi commands |
| 1 | hankweave and entire |
| 1 | the one thing that will make you fail |
| 1 | writing your completion summary |
| 1 | done gate (non-negotiable) |
| 1 | logging decisions and observations mid-task |
| 1 | execution flow |
| 1 | when things break |
| 1 | tool choice |
| 1 | completion summary structure (imi complete) |
| 1 | observation entries (imi log) |
| 1 | blocker entries (when you can't complete) |
| 1 | lesson entries (imi lesson) |
| 1 | bad vs good examples |
| 1 | a) completion summaries (imi complete) |
| 1 | b) log entries (imi log) |
| 1 | c) lesson entries (imi lesson) |
| 1 | after every code modification |
| 1 | css build architecture |
| 1 | development mode |
| 1 | production mode |
| 1 | why build:binaries alone is broken |
| 1 | dual execution modes |
| 1 | typescript path aliases |
| 1 | ai providers |
| 1 | claude code (default) |
| 1 | opencode |
| 1 | cursor agent |
| 1 | openai codex |
| 1 | evaluation engine |
| 1 | evaluator configuration |
| 1 | file filtering strategies |
| 1 | file path resolution for multiple agents.md files |
| 1 | configuration options flow |
| 1 | example: evaluatorfilter option |
| 1 | common mistake |
| 1 | temporary data cleanup |
| 1 | automatic cleanup after successful evaluation |
| 1 | preserving debug output |
| 1 | implementation details |
| 1 | runtime prompt debugging |
| 1 | always-on persistent logging |
| 1 | use cases |
| 1 | manual cleanup |
| 1 | implementation |
| 1 | job status flow |
| 1 | deployment |
| 1 | remote api server access |
| 1 | option 1: using cli flag (recommended) |
| 1 | option 2: using environment variables |
| 1 | option 3: using npm scripts (development) |
| 1 | verification |
| 1 | security considerations |
| 1 | common issues |
| 1 | frontend color system |
| 1 | centralized color palette |
| 1 | css classes to use |
