Cross-Embodiment Continual PPO — 开发文档（Toy Project）
 版本：v1.2（在 v1.1 基础上：T3 加 action remap 逼真冲突、KL 明确闭式高斯、网络统一分离 actor/critic、加 seeds 统计口径、§7.4 区分「真无冲突 vs bug」） 一句话定位：用一个 轻量 2D toy environment 验证 持续强化学习（Continual RL） 的算法骨架与评估流程，为后续接入 Vincent 的 Isaac Sim 真实抓取环境打地基。
 v1.2 改动速览（review 后采纳）：① §3.1/§3.3 T3 不只 mask 第三指，还对 active 指做 action remap/符号翻转——这是 forgetting 信号的真正来源；② §3.2 注明 descriptor on/off 的 ablation 是 toy-specific，不保证在 AE pipeline 复现；③ §3.5 标注 reward-hacking 防御①②是真 3D 环境教训、别在 2D toy cargo-cult；④ §4.1 统一为分离 actor/critic（不共享 trunk），KL 只 snapshot actor；⑤ §4.2 强制闭式高斯 KL（含 std，非 MSE-on-means），并弱化「KL 方向写反就崩」的过度断言；⑥ §6.1 明确 Fine-tune 的 LR anneal / optimizer 口径需与 KL-distill 一致；⑦ 新增 §6.3 seeds 统计要求（3–5 seeds，报 mean±std）；⑧ §7.4 把「Fine-tune 不遗忘」拆成「真无冲突」与「实现 bug」两路，避免 debug 错方向。
 
0. 这份文档怎么用

每一节末尾若涉及分工，会标注 🤖 可交给 Codex / 🙋 必须本人把关。

1. 项目背景（Why）
1.1 项目大背景
所属方向：Robot Learning → Dexterous Manipulation → Cross-Embodiment / Continual RL。 组里项目目标：训练一个能适配 多种手型（hand morphology） 的灵巧抓取策略（cross-embodiment dexterous grasping），底层用 PPO，在 Isaac Sim / Isaac Lab 上做。
本人负责的子方向：Continual Learning（持续学习），且持续学习发生在 morphology / embodiment 维度（不是物体维度）——即让策略从一种手顺序迁移到另一种手，且不遗忘旧手。
1.2 这个 Toy Project 要补的「缺失证据」
团队（Vincent）当前在 Slack 上展示的结果是：

做了 base + 3-finger 的 Wuji 手变体（USD 资产）。

用 joint / multi-task training：把 base 和 3-finger 同时放进一个 policy 训练。

成功率约 0.93（base）/ 0.91（3-finger），只是 "slight drop"。

关键判断：这个结果不能证明 fine-tuning 会 forgetting。 因为它是「同时训练」，不是「先训 base → 再只训 3-finger → 回测 base」。
 所以本 Toy Project 的第一价值，就是构造并量化这个缺失实验： Does sequential PPO fine-tuning forget previous morphologies? 这是把「练手项目」升级成「补上项目证据」的核心。
 
1.3 与团队 pipeline 的接口约束（来自 Slack）

团队用 autoencoder（AE） 把 observation 从 63D 压到 15D，目的有二：降维 + 给出 unified input（不同手型映射到同一表示空间）。

因此本人的 continual learning 策略，输入应是 AE 之后的统一低维表示，而不是原始高维观测。

待确认问题（需问 Vincent）：AE 在 continual learning 阶段是否冻结（frozen）？ 本文档默认「冻结」——只在 policy 层做持续学习，初期不动 AE。

2. 研究问题与设计哲学（What）
2.1 核心 Research Question
 Can a single PPO policy continually adapt to new hand morphologies while preserving performance on previous morphologies?
 
拆成两个可量化的子目标（持续学习的标准双指标）：

Stability（稳定性 / 抗遗忘）：学新 morphology 后，旧 morphology 的 success rate 不大幅下降（forgetting 小）。

Plasticity（可塑性 / 正向迁移）：迁移到新 morphology 时，比从头训（from scratch）学得更快或更好（forward transfer 为正）。

2.2 第一版方法选型（已 firm）

算法：PPO（on-policy），与团队主线一致。

抗遗忘方法（anti-forgetting）：Policy Distillation / KL Regularization，而不是 replay。 

理由：replay 存 transition 会破坏 PPO 的 on-policy 假设；而 KL 蒸馏只需在旧任务的 state 上约束新旧策略输出分布的距离，不参与 advantage 估计，对 on-policy 主损失无污染。



网络：单一 shared MLP，第一版不做 adapter / multi-head。

3. Toy Environment 设计
3.1 任务与形态序列
任务永远是「grasp cube」，变化的只有 hand morphology：
   Task
 Morphology
 mask

    T1
 base 3-finger hand：正常 finger length / joint limits / action mapping
 [1,1,1]

  T3
 2-finger hand：finger 3 inactive + 剩余两指 action remapping（见下方「⚠️ 制造真冲突」）
 [1,1,0]

  T4（可选扩展）
 2-finger + 改 joint limit / action sign
 [1,1,0]（scalar 变化）


 设计变更说明（v1.1）：原 T2（3-finger variant：仅改 finger length / joint limit）与 T1 的 morphology gap 太小，遗忘信号弱、不具说服力，已删除。第一版只做 T1 → T3 两段闭环即可。
 ⚠️ 制造真冲突（v1.1 关键修正）：forgetting 的本质是「同一个 observation，T1 与 T3 的最优 action 不同」。如果 T3 只是把 finger 3 mask 掉（action 忽略、不进 reward），那 T3 几乎是 T1 的子集——策略学到「只用 1、2 指」即可，根本不需要 overwrite 任何旧知识，于是 Fine-tune 可能真的几乎不遗忘，整个实验前提落空。因此 T3 不能只「失活第三指」，必须让剩余两指的 action→effect 映射相对 T1 发生改变（最简单：对 active 指做 action 符号翻转或线性 remap），使「无 descriptor」版的 T1/T3 在重叠 observation 上要求冲突的 action。这才是逼出真 forgetting 的机制——本质上把原 T4 的「action sign」手段提前到 T1→T3 这一步。
 注：在「有 descriptor」版里，mask/scalar 给了策略区分两形态的信息，所以冲突可被 condition 掉 → 遗忘理论上更轻；这正是 §6.2 受控变量想验证的对比。但「无 descriptor」版必须真有冲突，否则连 baseline 的病症都不存在。
 副作用提醒：序列变短为 2 个 task，评估矩阵相应缩小（见 §6/§7）。严格说这是一个「两任务 forgetting demo」，不是完整 continual 序列——plasticity / forward-transfer 只有单个数字，写作时不要 oversell。若第一版结果好、想要更长的序列曲线，再启用 T4（已预留口子，无需重构）。
 
3.2 Observation：固定 15D（对齐团队 unified input）
[
  object_x, object_y,                       # 2  物体位置
  palm_x, palm_y,                           # 2  手掌位置
  fingertip_1_x, fingertip_1_y,             # 2  指尖1
  fingertip_2_x, fingertip_2_y,             # 2  指尖2
  fingertip_3_x, fingertip_3_y,             # 2  指尖3
  mask_1, mask_2, mask_3,                   # 3  形态掩码（手指是否存在）
  morphology_scalar_1, morphology_scalar_2  # 2  形态标量（如指长/关节限位）
]
= 15D

2-finger：mask = [1,1,0]；3-finger：mask = [1,1,1]。

设计要求（重要）：obs 要支持「morphology 描述符可开关」。即能一键切换： 

无 morphology 信息版：去掉 mask + scalar（其余补零/固定），策略「看不到」形态切换 → 遗忘最严重，是病症最明显的 baseline。

有 unified morphology 描述符版：完整 15D，策略能区分形态 → 遗忘理论上更轻。

这正好把「without vs with morphology descriptor」做成一个受控变量，无需另起炉灶。

⚠️ 适用范围提醒（toy 专属）：这个「关掉 mask+scalar」的 ablation 只在手工拼的 15D toy obs 里干净成立。团队真实的 15D 来自 63D→15D 的 autoencoder，是纠缠表示，无法「只关掉某几维 morphology 信息」。所以「加了 descriptor 就几乎不遗忘」这一结论是 toy-specific 的，不保证在 AE pipeline 里复现——写作/汇报时要注明，避免给团队留下错误预期（§11 已把「AE 是否 frozen / 15D 语义是否跨手稳定」列为待确认）。



3.3 Action：固定 3D
[action_f1, action_f2, action_f3]

inactive finger 的 action 被 mask 掉（不参与环境物理 / 不计入奖励）。

T3 的 active 指 action remapping（配合 §3.1 的「制造真冲突」）：T3 下保留的 f1/f2 的 action→指尖位移映射相对 T1 改变（最简单实现：`effect = sign * action`，T1 用 +1、T3 用 −1，或更一般的固定线性 remap）。这一步是 forgetting 信号的来源，必须在环境里实现、且 morphology 描述符的 scalar 维度应能编码这个 remap（让「有 descriptor」版可区分，「无 descriptor」版被迫冲突）。

3.4 Reward / Success 设计（吸收 Slack 实测踩坑经验）
第一版仍是 toy 级别，但 reward 不能只写「指尖接近 + 稳定接触」——团队在 Slack 已经踩过具体的 reward hacking 坑，必须把教训直接编码进 reward 结构，否则会重蹈覆辙。
结构化 reward（v1.1）：
reward =
    w1 * lateral_approach(palm → object)                          # ① 参考点用 palm，不用 proximal joint
  + w2 * lift_height   IF lateral_approach > threshold ELSE 0      # ② height 必须 gated behind lateral（乘 gate，不是相加）
  + w3 * multi_finger_contact_bonus(active fingertips → object)    # ③ 只计 active 手指（被 mask 的不计）
  - w4 * action_smoothness_penalty                                 # ④ 可选，权重要小，防抖动/防消极不动
Success metric：二值（成功 1 / 失败 0），判定为「active 指尖与 cube 形成稳定接触且 cube 被抬起超过阈值」。Continual World 用的就是「dense reward + 二值 success」这套口径，照搬即可。
3.5 已知 Reward Hacking 模式与对策（来自 Slack，务必编码进环境）
   Hacking 模式（实测发生过）
 根因
 对策（写进 reward）

    Agent 钻到桌子下面 hack height reward
 lateral reward 与 height reward 分开线性相加，agent 发现不接近 cube 也能拿 height 分
 height reward 必须 gated behind lateral：只有当 lateral_approach > threshold 时 height 项才解锁（乘 gate），否则为 0。不是简单相加。

  出现 weird grasp（用奇怪姿势让某关节贴近物体，而非真抓）
 用 middle finger proximal joint 当 reward 的距离/接近参考点
 接近/对齐这类粗定位改用 palm region / palm center 当参考点；指尖只用于 contact bonus 这类细粒度项

  2-finger 任务有一项永远拿不到的奖励，污染对比
 contact bonus 把被 mask 的 finger 3 也算进去了
 multi-finger contact bonus 只统计 active fingers（与 mask 一致）


 这三条是团队真金白银踩出来的经验，写明后既能让 Codex 实现时直接避开，也能向组里展示「我吸收了你们的 reward 调试经验」。
 ⚠️ 别 cargo-cult：①②的 hacking 模式（钻桌子刷 height、用怪姿势贴近）依赖真 3D 环境里的桌子 + 重力 + 几何捷径。2D toy 若没建这些结构，这些失败模式根本不会出现——所以保留「gated lift / palm 参考点 / active-only contact」这套 reward 结构是因为它是好习惯、且能无缝迁移到第二阶段真环境，而不是因为 toy 里真会发生 ①②。不要为了「复现 hacking」反而给 toy 加桌子/重力把环境搞复杂。③（active-only contact）则是 toy 里真实需要的——它直接关系 T1/T3 对比是否被污染，务必实现。
 
🤖 可交给 Codex：按上面精确公式与对策实现 reward（含 gate 条件、palm 参考点、active-only contact）。 🙋 必须本人把关：

确认 gate 写成「乘法/条件解锁」而非「线性相加」（Codex 极易写成相加，坑就回来了）。

确认接近项参考点是 palm 而非任何 proximal joint。

调权重 w1~w4 使任务可学（w4 不能过大，否则 agent 为「不动」而消极）。

🤖 可交给 Codex：按上面精确 spec 实现 gym 环境（observation/action/reward/mask 逻辑、morphology 描述符开关）。 🙋 必须本人把关：reward shaping 是否合理（能不能学出来）、success 阈值定义、以及确认「无 morphology 版 / 有 morphology 版」切换正确。
4. 网络与算法实现
4.1 网络结构
15D obs ─┬─ actor MLP（2~3 层） → continuous action distribution（Gaussian，state-independent log_std 作 nn.Parameter）
         └─ critic MLP（2~3 层，独立权重） → state value V(s)
统一用分离的 actor / critic（不共享 trunk）。理由：(1) CleanRL ppo_continuous_action.py 与 CrossDex module.py（见 module.py:117-138）都是分离结构，照搬最省事；(2) 分离网络的遗忘动力学更好推理；(3) KL 蒸馏只需 snapshot actor（value 不参与蒸馏），分离结构下「冻结旧 actor」干净利落，不会牵连 critic。
 —（原文档曾写「shared MLP + policy/value head」共享 trunk，已统一为分离，避免实现时两套说法打架。）
基线脚手架：CleanRL ppo_continuous_action.py（单文件、易读易改，适合往里塞 KL 项；其 log_std 也是 state-independent 的全局 nn.Parameter，与上面一致）。
 注意：CleanRL 是「单任务单文件」哲学，没有任务序列调度概念。task sequence 的调度器需要在它外面自己包一层。
 
4.2 KL Distillation 的正确实现（核心，最易写错）
总损失：
L_total = L_PPO(new task) + λ · KL( π_old(·|s_old) || π_new(·|s_old) )
实现要点（务必逐条核对）：

必须存一份冻结的旧策略 π_old（学完 T1 时 snapshot 旧 actor 的权重——只需 actor，value 不参与蒸馏），不是只存 state。 

⚠️ 原始直觉「只存 state buffer 不存 transition」只对了一半：state 用作 KL 的评估点，但还需要 π_old 来产生 target 分布。漏了 snapshot 旧 actor，KL 项就没有 target、等于空。



存一批旧任务的代表性 states s_old：用作 KL 的评估点。第一版可均匀采样 T1（及后续旧 task）的若干 states；其分布代表性会影响蒸馏效果（可调点）。设计选择：在「旧任务 states」上算 KL（更针对性地防旧任务上的遗忘），而非在当前 T3 rollout states 上算——两者都可，第一版用前者。

⚠️ 必须用闭式高斯 KL（含 std），不是 MSE-on-means。最常见的隐藏 bug 是把蒸馏写成 `MSE(mean_old, mean_new)` 还叫它「KL」。本项目 actor 是对角高斯（state-independent log_std），两个高斯有解析 KL，必须用完整公式（同时约束 mean 和 std），否则 std 不受约束、蒸馏不完整。

KL 方向：默认用 KL(π_old || π_new)（forward KL，把旧策略当 target）。⚠️ 措辞修正：这不是「写反就崩」的正确性地雷——两个高斯的 KL 正反向都有闭式、都能算、文献里都有人用，方向反了只是行为略偏、不会报错也不会发散。把 forward 当默认即可，无需当成 must-get-right 的硬约束。真正会「不报错但错」的是上面的 MSE-冒充-KL 和下面的 detach。

梯度：π_old 必须 detach（停止梯度），只更新 π_new。

on-policy 安全性：KL 项不进入 advantage 估计，只作为附加正则项，不破坏 PPO 主损失的 on-policy 性质。

🤖 可交给 Codex：CleanRL 改造（加闭式高斯 KL 项到 loss）、π_old(actor) snapshot 的存取、s_old buffer 的采样、λ 作为超参接入。 🙋 必须本人把关：逐条 review（Codex 最易错的三处：① 把 KL 写成 MSE-on-means、② 漏 detach、③ target 来源；KL 方向反而不必过度纠结）；训练时打印 KL 值确认其非零、随 λ 变化、且包含 std 项的贡献。
5. Pipeline（完整数据流）
morphology 序列 [T1, T3]（第一版；T4 可选追加）
  │
  ├─ 训 T1（base 3-finger）  → snapshot policy_T1 → 在 T1 上评估（记 baseline 分数）
  │
  ├─ load policy_T1 → 训 T3（2-finger，开启 KL distillation，target=policy_T1）
  │                        → snapshot policy_T3
  │                        → 回测 T1（看掉多少 = forgetting）
  │                        → 测 T3（看学得如何 = plasticity）
  │
  └─（可选）load policy_T3 → 训 T4 → 回测 T1、T3、测 T4
  │
  └─ 汇总成评估矩阵 → 计算 CRL 指标 → 出图
五个部件：① 环境序列 ② PPO agent ③ checkpoint 存取（命脉）④ 抗遗忘机制（KL distill）⑤ 评估与指标。
🤖 可交给 Codex：调度器主循环骨架（for morphology in sequence: train→snapshot→load→eval）、checkpoint 存取代码、结果矩阵汇总与画图、config / 命令行参数（如 --method kl_distill --seq 3 --use_morphology 1）。 🙋 必须本人把关：

验证 checkpoint 真能续训：load 回来后 loss/性能是接续的、不是从头开始（亲自看曲线确认；Codex 写的加载常「能跑但没真正恢复状态」）。

实验设计决策：每个 task 训多少步、λ 取值、形态顺序、训练预算。

6. 实验组（Experiment Matrix）
6.1 方法（method）维度
   方法
 说明
 作用

    Single-task PPO
 每个 morphology 单独训
 各形态独立性能参照

  Multi-task / Joint PPO
 所有形态一起训（= Vincent 当前做法）
 性能上界（upper bound），对应 0.93/0.91

  Sequential Fine-tune PPO
 顺序训，不加任何保护
 遗忘下界，要补的缺失实验
 实现口径：继承 T1 的 actor+critic 权重继续训 T3；但每个 task 重启 LR anneal schedule（frac 从 1.0 重新计），Adam moment 是否继承统一选一种并固定（建议继承，更贴近「真·fine-tune」）。这点必须和 KL-distill 完全一致，否则两者差异里混入了 optimizer 设置的噪声。注：FAME 的 continue 分支（test_main.py:285-292）连 optimizer state 一起拷，那是 SAC；PPO 这里简化为「同 net、同/重置 optimizer、换 env」即可。

  Sequential PPO + KL Distillation
 顺序训 + KL 抗遗忘
 本项目主方法

  Reset / From-scratch（建议补）
 每个新形态从头训，不迁移
 plasticity 下界，证明顺序学习有正向迁移


6.2 形态信息（morphology descriptor）维度

without morphology descriptor（去 mask+scalar）

with unified morphology descriptor（完整 15D）

 第一版可不必全做完整 2×2；但建议至少覆盖： Fine-tune / KL-distill × with descriptor，外加 Multi-task 上界 + Reset 下界。 一个有 insight 的附加问题：若「光加 mask 就几乎不遗忘」，则 distillation 的边际价值需重新评估——这本身就是有价值的结论。

6.3 统计有效性（seeds，不可省）
每个 (method × descriptor) 组合至少跑 3–5 个 seeds，评估矩阵里所有数字报 mean ± std。这是硬要求：整个项目的卖点是「量化 forgetting」，而 RL 单 seed 方差极大——单 seed 的「Fine-tune 掉了 / KL 没掉」在统计上等于没说。要拿给 Soofiyan / Yuheng 看，必须有跨 seed 的均值与误差带。曲线图也应是 seed 聚合（mean ± std/ci 阴影带），而非单条。
 🤖 可交给 Codex：seed 循环、结果按 (method, descriptor, seed, task) 落 CSV、聚合画带误差带的曲线（口径参考 Continual World）。 🙋 必须本人定：seed 数量、训练预算是否够每个 seed 都收敛。

🙋 必须本人把关：定性 sanity check（见 §7.2）。
7. 验收标准（Acceptance Criteria）
7.1 交付物

一个可运行的 toy env + CleanRL-based PPO + KL distillation 代码骨架。

一张 评估矩阵（见下）+ 对应曲线图。

一段简短结论：sequential fine-tune 是否遗忘、KL distillation 是否缓解。

7.2 核心评估矩阵（要做出来的表）
第一版（T1 → T3 两段）。每个 cell = 在该列任务上的 success rate；关键看「学完 T3 后回测 T1」掉了多少。
   Method
 After T1: T1
 After T3: T1（回测=forgetting）
 After T3: T3

    Fine-tune（顺序，无保护）
 high
 ↓↓ 预期明显下降
 high

  KL-distill（本项目主方法）
 high
 less drop（预期）
 high

  Multi-task（= Vincent 当前做法，上界）
 —
 high
 high

  Reset / from-scratch（plasticity 下界）
 high
 （从头，无迁移）
 high


 核心对比就一列：After T3 的 T1 列——Fine-tune 该掉、KL-distill 该掉得少。 若启用 T4，矩阵按相同逻辑向右扩展（After T4: T1 / T3 / T4）。
 
7.3 项目「成立」判据

Stability：Fine-tune 的旧任务 success rate 明显下降，而 KL-distill 下降更少 → 抗遗忘有效。

Plasticity：Sequential（迁移）比 Reset（从头）在新任务上收敛更快 / 更高 → 有正向迁移。

两者同时成立，项目即成立。

7.4 定性 sanity check（🙋 只有本人能判断）
正确实现下，结果应该呈现：Fine-tune 遗忘 > KL-distill 遗忘；Multi-task 最高；Reset 在新任务起点最低。 若出现「KL-distill 毫无改善」「Multi-task 不是最高」等反常 → 大概率是实现 bug，需排查（常见：checkpoint 未真正续训、KL 写成 MSE-on-means、KL 项为空、success 评估口径不一致）。

⚠️ 但「Fine-tune 居然不遗忘」要分两种情况，不能无条件当 bug：
 (a) 实现 bug：checkpoint 没真续训（每个 task 其实从头训，自然不"遗忘"）、success 评估口径不一致——这类要排查。
 (b) 任务本身没冲突（真问题，不是 bug）：如果 T3 只 mask 掉第三指、没做 §3.1 的 action remap，那 T1/T3 在重叠 obs 上最优 action 其实一致，Fine-tune 本就不该遗忘——这是 toy 设计没逼出冲突，不是代码错。
 区分方法：先验证 checkpoint 真续训（看 load 后曲线接续、§9 步骤 3），再确认「无 descriptor」版里 T1/T3 确有 action 级冲突。两者都对却仍不遗忘，就回去改 toy（加大 remap / 符号翻转），而不是去 debug 训练代码。把「无遗忘」一律归为 bug 会让你在错误的地方耗时间。
8. 参考公开 Repo（各取所需，不要照搬）
   Repo
 定位
 借鉴什么
 不借鉴什么

    CrossDex（PKU, ICLR 2025）
 Cross-embodiment 灵巧抓取，zero-shot（IsaacGym）
 「统一中间动作表示 + retargeting」让一个策略适配多手的设计思想；observation/reward 构成参考
 它与 continual learning 无关、且是 IsaacGym 非 Isaac Sim → 代码不可直接用，仅借设计。本阶段几乎用不到，留到接 Vincent 时参考。

  FAME（ICLR 2026）
 持续 RL 算法（Fast/Meta 双学习器）
 Atari 部分是 PPO + Continual RL，是本阶段最贴的 PPO+CRL 代码参照；其 baseline 清单（Reset/Finetune/PackNet/ProgressiveNet/CompoNet）与 CRL 指标计算脚本（process_results.py）直接照搬
 第一版不要复现 FAME 本身的双学习器算法（过复杂）；把它当「项目结构 + 评估流程范本」

  MetaWorld（Farama）
 机械臂操作 benchmark（多用 SAC）
 现成的连续控制操作任务、success metric 口径；若想要更真实的中间过渡环境可用它
 它本身是 multi-task/meta-RL，不是 CRL；任务序列要自己按 CW 来组

  Continual World（awarelab）
 基于 MetaWorld 的 CRL benchmark（CW10/CW20，SAC 实现）
 最值钱：已提供 single / multi-task / sequential 三种实验脚本（run_single/run_mt/run_cl）+ 7 种算法实现 + 标准 CRL 指标与画图，直接照搬其评估框架与 baseline 组织方式
 源码 2021 后未维护、用的是有问题的 MetaWorld v1 → 当「设计与指标参考」，不要直接 fork 来跑


8.1 选型小结

算法骨架 → CleanRL ppo_continuous_action.py（自己改）。

PPO + CRL 怎么搭 → 参考 FAME 的 Atari 部分。

CRL 评估框架 / baseline 组织 / 指标 → 参考 Continual World 的 run_single/run_mt/run_cl 与结果脚本。

更真实的操作任务（可选过渡） → MetaWorld。

跨形态架构设计（留到第二阶段接 Vincent） → CrossDex。

 方法论佐证：已有工作（Replay-enhanced Continual RL, TMLR 2023）在 SAC 中结合 policy distillation（KL 正则）做抗遗忘，实验在 Continual World 上——与本项目「PPO + KL distillation」同源，可在写作时引用。
 
9. 开发顺序与分工总表
推荐分块开发，每块跑通再下一块：
   步骤
 内容
 主责

    1
 环境搭建（CleanRL 跑通单任务 PPO；如怕踩坑，先用更轻的离散环境验证骨架）
 🙋 本人盯（依赖易踩坑，报错需判断）

  2
 toy env 实现（15D obs / 3D action / mask / morphology 开关 / T3 的 active-finger action remap）
 🤖 Codex 写 + 🙋 本人验设计（关键：确认「无 descriptor」版 T1/T3 在重叠 obs 上确有 action 级冲突——这是后续遗忘信号的前提，见 §3.1/§7.4(b)）

  3
 checkpoint 存取 + 验证续训真生效
 🤖 Codex 写 + 🙋 本人必看曲线确认

  4
 调度器主循环（train→snapshot→load→eval over sequence）
 🤖 Codex

  5
 评估指标（对照 Continual World 口径，算 forgetting / forward transfer）
 🤖 Codex 写 + 🙋 本人核对指标定义

  6
 KL distillation（含 π_old(actor) snapshot、s_old buffer、λ、闭式高斯 KL）
 🤖 Codex 写 + 🙋 本人 review §4.2（重点：闭式 KL 含 std 非 MSE、detach、target 来源）

  7
 baseline 全跑（single / multi / finetune / KL / reset）× ≥3 seeds + 出带误差带的矩阵图
 🤖 Codex 跑+画 + 🙋 本人做 §7.4 sanity check（含「真无冲突 vs bug」判别）


一句话原则：Codex 写「代码」，本人做「实验与判断」。最危险的是把「验证 checkpoint 续训」「判断遗忘曲线对不对」这种需要科研直觉的验证也丢给 Codex。
10. 时间估算（非全职、兼顾其他事）
   阶段
 估时

    环境跑通（最易超时，早开始）
 1–2 天

  单任务 PPO + checkpoint 续训验证
 1 天

  调度器主循环 + 评估指标
 1–2 天

  KL distillation + baseline 对比 + 出图（×≥3 seeds）
 1–2 天

  合计
 约 5–7 个工作日



顺利最快 3–4 天；卡 MuJoCo 依赖可能拖到 8–9 天 → 环境那步千万早开始。
 ⚠️ 真正的隐藏卡点往往不是依赖，而是「调出真遗忘信号」：toy env 要同时满足「可学」且「无 descriptor 版 T1/T3 真冲突」，reward 权重与 action remap 幅度多半要迭代几轮。多 seed 也会拉长 wall-clock（建议 toy 训练量压到单 seed 几分钟级，让 ≥3 seeds × 5 方法 × 2 descriptor 仍可接受）。把这块时间预留出来，别只盯依赖。

降险技巧：先用最轻的离散环境（如 FAME 的 MinAtar 风格）把整条 CRL 链路跑通，验证骨架后再换到 toy grasping / MetaWorld。

11. 动手前要向团队确认的问题（建议发 Slack / 问 Vincent）

AE 在 continual learning 阶段是否 frozen？（影响是否只在 policy 层做持续学习）

63→15 的 AE 具体编码什么？输出的 15D 语义是否稳定（不同手型间一致）？

现用 RL 库（rl_games / rsl_rl / 其他）与 PPO config 位置 —— 为第二阶段接入做准备。

checkpoint 如何保存 + 如何从已有 policy 继续训练（第二阶段命脉）。

「跨手」走 CrossDex 的 eigengrasp+retargeting 统一动作空间，还是走当前 grasp-prior 路线？（值得与 Soofiyan/Yuheng 对齐）

文档结束。建议先按 §9 步骤 1 动手，同时把 §11 的问题发出去并行推进。