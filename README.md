# tachikoma_lab ― タチコマを強化学習で前に歩かせる

![完成版：平地をまっすぐ歩く](docs/note/videos/13_terrain_v4_flat.gif)

譲り受けた『攻殻機動隊』のタチコマの 3D モデルを使い、NVIDIA Isaac Lab と強化学習（PPO）で歩行を学習させたプロジェクトです。

タチコマは脚の付け根が横を向いた構造のため、**横歩きはすぐに習得** しました。一方で、**前に歩く** のは難しく、
報酬設計 → 学習 → 評価 → 原因分析 → 修正 を何度も繰り返しました。このリポジトリには、その全過程のコードと記録を置いています。

- 「Physical AI 応用1」最終課題（テーマ：AI×ロボティクスによる創造的タスクの実現）
- 詳しいレポート：[`docs/submission/report.md`](docs/submission/report.md)
- 試行錯誤の全記録：[`docs/note/journal.md`](docs/note/journal.md)

> [!NOTE]
> **タチコマの 3D モデル（URDF・メッシュ・USD）は譲り受けたもののため、公開していません**（`assets/` は `.gitignore` 済み）。
> コードを動かすには、ご自身のロボットモデルを `assets/usd/tachikoma/tachikoma.usda` に置き、
> `source/tachikoma_lab/tachikoma_lab/assets/tachikoma.py` の関節名・リンク名を合わせてください。

## 完成版の歩行

| 15 mm の凸凹 | 1 cm の段差 |
|---|---|
| ![凸凹](docs/note/videos/14_terrain_v4_rough15mm.gif) | ![段差](docs/note/videos/15_terrain_v4_step10mm.gif) |

評価条件：前進 0.08 m/s の指令、64 体 × 10 秒（地形の高さは見せない「盲目」の方策）

| 項目 | 地形学習前（stable_v1） | 曲がる癖が出た版（terrain_v3） | **完成版（terrain_v4）** |
|---|---|---|---|
| 平地：10 秒の前進 | 0.80 m | 0.45 m | **0.77 m** |
| 平地：向きのずれ | 4.2° | −71° | **−4.4°** |
| 15 mm 凸凹：転倒 | 54 / 64 体 | 0 / 64 体 | **0 / 64 体** |
| 15 mm 凸凹：前進 | ― | 0.31 m | **0.45 m** |
| 1 cm 段差：1 段以上上れた | 61%（3 体転倒） | 22% | **72%（転倒 0）** |

## 試行錯誤の流れ

| 段階 | ねらい | 何が起きたか |
|---|---|---|
| v1 | 前に歩く | 前を +Y 軸として報酬を定義し直し、位置制御に変えて前進できた。ただし 1 秒に 4 歩の小刻み歩き |
| safe | 安全に歩く | 罰を強くしすぎて「わざと転んで終わらせる」を学習 → 生存報酬を増やして解決 |
| stable | 重心を考える | 支持多角形と重心の距離（安定余裕）を報酬に追加 |
| lift | 足を上げる | 足をゆっくり動かして罰を避ける「すり抜け」 |
| terrain v1〜v3 | 地形で鍛える | 転倒は 0 になったが、歩きながら右へ曲がる癖が出た。報酬の重みを 3 倍にしても直らず、原因は報酬の **幅（std）** だった |
| terrain v4 | まっすぐ歩く | 「この方角を向け」という指令を加えて、癖を解消。最後まで学習させると「歩かない」方策に崩れたため、途中の最良モデルを採用 |

## 動かし方

動作確認環境：Windows 11、Isaac Sim 5 + Isaac Lab 3.x（`..\IsaacLab` に配置）、RTX 5090

```powershell
# インストール（初回のみ）
..\IsaacLab\_isaac_sim\python.bat -m pip install -e source\tachikoma_lab

# 学習（各段階は前の段階のモデルから --resume で再開）
#   --min_std：探索ノイズの下限（std がマイナスになって学習が止まる問題への対策）
..\IsaacLab\_isaac_sim\python.bat scripts\rsl_rl\train.py --task Tachikoma-Walk-Terrain-v4 --headless `
    --run_name terrain_v4 --resume --load_run <stable_v1 の実行フォルダ> --checkpoint model_4997.pt `
    --max_iterations 3000 --min_std 0.01

# 評価（--terrain flat|step|rough、--video で追従カメラの動画を保存）
..\IsaacLab\_isaac_sim\python.bat scripts\rsl_rl\eval_forward.py --task Tachikoma-Walk-Terrain-v4 --headless `
    --num_envs 64 --cmd_vx 0.08 --terrain rough --terrain_height 0.015 `
    --checkpoint models\terrain_v4_model_7300.pt
```

`isaaclab.bat -p` ではなく、`_isaac_sim\python.bat` で起動してください。
W&B にログを送る場合は、`.env` に `WANDB_API_KEY=...` を書きます（`.env` はコミットされません）。

### タスク一覧

| タスク名 | 段階 | 内容 |
|---|---|---|
| `Tachikoma-Walk-v0` | v1 | 機体基準の速度追従（前 = +Y）、位置制御 |
| `Tachikoma-Walk-Safe-v0` | safe | 遅い指令、3 本支持・足すべりの罰、生存報酬の強化 |
| `Tachikoma-Walk-Stable-v0` | stable | 立ち姿勢の調整、安定余裕の報酬 |
| `Tachikoma-Walk-Lift-v0` | lift | 足先の高さの報酬 |
| `Tachikoma-Walk-Terrain-v0` | terrain v1 | 小型機向けに縮小した地形と、距離によるカリキュラム |
| `Tachikoma-Walk-Terrain-v2` | terrain v2 | 向きの報酬の重み 1.5、転倒で降格 |
| `Tachikoma-Walk-Terrain-v3` | terrain v3 | 向きの報酬の std 0.15、速度追従によるカリキュラム |
| `Tachikoma-Walk-Terrain-v4` | terrain v4 | v3 に方角の指令（heading command）を追加。完成版は 7300 回目のモデル |

## 主なファイル

| ファイル | 内容 |
|---|---|
| `source/.../tachikoma_lab/tachikoma_lab_env_cfg.py` | 環境・報酬・指令・終了条件（全段階） |
| `source/.../tachikoma_lab/mdp/rewards.py` | 独自の報酬（安定余裕、足の高さ、足すべり、機体基準の速度追従） |
| `source/.../tachikoma_lab/mdp/curriculums.py` | 地形カリキュラム（距離版・速度追従版） |
| `source/.../tachikoma_lab/terrains.py` | 小型機向けの地形と評価コース |
| `source/.../tachikoma_lab/agents/rsl_rl_ppo_cfg.py` | PPO のハイパーパラメータ |
| `source/tachikoma_lab/tachikoma_lab/assets/` | ロボットの定義（`tachikoma.py`）と接地センサの修正（`spawn.py`） |
| `scripts/rsl_rl/eval_forward.py` | 条件を固定した評価と動画の撮影 |
| `scripts/note/` | GIF・提出動画の作成 |
| `models/terrain_v4_model_7300.pt` | 完成版の学習済み方策 |

`source/...` は `source/tachikoma_lab/tachikoma_lab/tasks/manager_based/tachikoma_lab` の略です。

## AI の使い方

- **強化学習（PPO, RSL-RL）**：4096 体を GPU 上で並列に動かして、歩行の方策を学習しました。
- **生成 AI（Claude Code）**：報酬設計の相談、コードの実装、学習結果の分析、動画とレポートの作成を共同で行いました。

## 注意

- 「タチコマ」は士郎正宗『攻殻機動隊』に登場するキャラクターです。本リポジトリは個人の学習目的のファン作品で、権利者とは関係ありません。
- Isaac Lab のプロジェクトテンプレートをもとにしています（BSD-3-Clause）。
