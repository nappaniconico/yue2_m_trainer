# YuE2 LoRA Trainer

実音源を使ってYuE2のAR（自己回帰）モデルにLoRAを学習し、ComfyUI向けに書き出すためのツールです。Kawaii Future Bass向けの設定例を同梱しており、音源・スタイル記述・トリガーを変更して別のデータセットにも利用できます。

Mothersuperiorのreal-audio tokenizerと対応するNAR（非自己回帰）LoRAを組み合わせ、音源のトークン化から学習、検証、アダプターの書き出しまでを扱います。

## 主な機能

- **v4 / v5 / v8 / v9のpair切替**：tokenizer headと対応NAR LoRAをTOMLで選択。
- **3つの学習方式**：`cot=off`、ABC＋semantic、ABC-only。
- **regularization**：YuE2生成曲のsemantic tokenとABC譜面を用いた学習。
- **共通条件での検証**：同じデータ分割・評価入力・checkpoint stepで学習方式を比較。
- **ComfyUI向け出力**：標準ComfyUIとComfyUI-FL-YuE2の両形式に対応。
- **再現性の管理**：モデルのrevision・SHA-256固定、トークンキャッシュ、学習再開。

学習対象はAR LoRAです。NARは選択したpairの学習済みアダプターを使用します。独自データでのNAR追加学習には対応していません。

## 必要環境

| 項目 | 構成 |
|---|---|
| OS | Linux / WSL2 |
| Python | 3.12 |
| GPU | BF16に対応したNVIDIA GPU |
| PyTorch | 2.10.0 / CUDA 12.8 wheel |
| パッケージ管理 | `uv` |

必要なGPUメモリは系列長などの設定に依存します。モデル、特徴量キャッシュ、複数のcheckpointを保存するためのディスク容量も確保してください。

以降のコマンドはリポジトリのルートで実行します。

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDAの確認結果が`True`であることを確認してください。

## クイックスタート

まずはABC譜面を必要としない`cot=off`で、トークン化から学習までを実行できます。

### 1. 設定と音源を用意する

[v9_off.toml](configs/examples/v9_off.toml)を開き、次の項目をデータセットに合わせて編集します。TOML内の相対パスは、**そのTOMLファイルがあるディレクトリ**を基準に解決されます。

```toml
[dataset]
audio_directory = "../../data/audio"
trigger = "my_style"
default_style = "bright electronic music, sparkling synths, punchy drums"

[assets]
pair = "v9"

[train]
output_directory = "../../outputs/my_style_v9"
```

上記は変更する項目の抜粋です。設定ファイル内の他の項目は残してください。音源は指定ディレクトリの直下に配置します。対応形式はWAV・FLAC・MP3で、ファイル名のstemは重複できません。

```text
data/audio/
├── track01.wav
├── track02.flac
└── track03.wav
```

### 2. 曲ごとの説明と歌詞を用意する

```bash
uv run yue2-lora init-sidecars \
  --config configs/examples/v9_off.toml --instrumental
```

音源と同じ場所に、次のファイルが作成されます。既存ファイルは上書きしません。

| ファイル | 内容 |
|---|---|
| `track01.caption.txt` | 音色、楽器、テンポ感、構成などの曲固有の説明 |
| `track01.lyrics.txt` | 歌詞。インストゥルメンタルは空ファイル |
| `track01.song.txt` | 任意の曲ID。同じ曲の別編集版には同じIDを指定 |

captionを曲に合わせて編集してください。歌のある音源では、lyricsにレビュー済みの歌詞を記入します。歌詞テンプレートを作成する場合は`--instrumental`の代わりに`--no-instrumental`を指定します。

train/validationは曲ID単位で分割します。`.song.txt`を省略した場合はファイル名のstemを曲IDとして扱います。学習にはtrain・validationの両方が必要です。

### 3. モデル取得・トークン化・学習

```bash
uv run yue2-lora download --config configs/examples/v9_off.toml
uv run yue2-lora prepare --config configs/examples/v9_off.toml
uv run yue2-lora train --config configs/examples/v9_off.toml --offline
```

`prepare`は音源からMERT特徴量とsemantic tokenを作成し、キャッシュに保存します。入力音源自体は変更しません。学習では実音源と生成曲のregularizerを混ぜ、AR LoRAを更新します。

中断した学習は、同じ設定・データ・モデルで再開できます。

```bash
uv run yue2-lora train \
  --config configs/examples/v9_off.toml --resume --offline
```

### tokenizer / NAR pairの切替

`assets.pair`には`v4`、`v5`、`v8`、`v9`を指定できます。同梱の設定例は`v9`です。headとNARは必ず対応する組から取得します。

pairを変更した場合は`download`と`prepare`を再実行し、学習先の`output_directory`も変更してください。MERT特徴量は再利用でき、semantic tokenはheadのhash別に保存されます。異なるデータやheadで作成したmanifestを学習に流用することはできません。

## ABC譜面を使った学習と比較

ABCはメロディーやコードをテキストで表す記譜形式です。ABCを教師に加えることで、semantic token生成と同じARモデルの譜面生成部分も学習できます。

比較には[v9_comparison.toml](configs/examples/v9_comparison.toml)を使用します。`audio_directory`、`trigger`、`default_style`をクイックスタートと同じデータセットに合わせて編集してください。

### 1. 音源のABC譜面を用意する

音源と同じstemの`.abc`を配置します。

```text
data/audio/
├── track01.wav
├── track01.caption.txt
├── track01.lyrics.txt
└── track01.abc
```

SheetSage2を使用する場合は、学習環境とは別のPython環境を用意します。モデルの利用条件・アクセス手順は[SheetSage2の配布ページ](https://huggingface.co/m-a-p/SheetSage2)を参照してください。

<details>
<summary>SheetSage2用環境のセットアップ例</summary>

```bash
uv venv --python 3.11 .venv-sheetsage2
uv pip install --python .venv-sheetsage2/bin/python huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 \
  --local-dir models/SheetSage2
uv pip install --python .venv-sheetsage2/bin/python torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sheetsage2/bin/python \
  -r models/SheetSage2/requirements.txt
```

アクセス制限のあるモデルを取得する場合は、事前にアクセス承認とHugging Faceへのログインを済ませてください。

</details>

クイックスタートで取得したMERTを再利用して一括転写します。

```bash
.venv-sheetsage2/bin/python scripts/transcribe_abc.py \
  --input-directory data/audio \
  --model models/SheetSage2 \
  --base-model models/MERT-v2-FullSong \
  --offline
```

転写結果は学習前に確認してください。内容のある既存ABCは保護されます。再転写する場合は`--overwrite`を指定します。3方式の比較には、通常出力のメロディー＋コード譜面を使用します。

### 2. ABC regularizerを作成する

semantic regularizer packにはABCが含まれないため、minted corpusの譜面を別途取得して結合します。以下のコマンドはテキストだけを取得し、音声やlatentをダウンロードしません。

```bash
uv run hf download Mothersuperior/yue2-minted-corpus --repo-type dataset \
  --revision 5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35 \
  --include 'tracks/*/score.abc' 'tracks/*/request.json' \
  --local-dir cache/minted-corpus

uv run yue2-lora build-abc-regularizer \
  --config configs/examples/v9_comparison.toml \
  --tracks cache/minted-corpus/tracks \
  --output cache/minted_abc.json

uv run yue2-lora prepare --config configs/examples/v9_comparison.toml
```

曲IDで結合し、元のsemantic packの`minted` / `minted_val`分割を維持します。譜面とrequestが揃い、`cot=full/melody`の曲が対象です。重複IDや同じ譜面のtrain/validation間混入はエラーになります。

ABC packの出力先を変えた場合は、TOMLの`train.abc_regularizer`も変更してください。

### 3. 3方式を実行する

```bash
uv run yue2-lora compare-train \
  --config configs/examples/v9_comparison.toml --offline
```

| run | 実音源の教師 | regularizerの教師（既定） |
|---|---|---|
| `cot_off` | semanticのみ、cot=off | semanticのみ |
| `abc_semantic` | ABC＋semantic、cot=full | semanticとABCを1:1 |
| `abc_only` | ABCのみ、cot=full | ABCのみ |

全runで実音源とregularizerの比率、rank、seed、学習率schedule、step数、保存間隔、分割、pair、系列長の上限を共通化します。教師系列の長さは方式によって異なるため、計算量は同一ではありません。

`abc_regularizer_fraction`はregularizer内のABCの割合です。既定の`generated_fraction=0.5`、`abc_regularizer_fraction=0.5`では、ABC＋semantic runのサンプリング比率は実音源50%、semantic regularizer 25%、ABC regularizer 25%になります。ABC-onlyではABCの割合を1.0にし、semantic lossを計算しません。共通評価にはsemantic tokenを使うため、ABC-onlyでもトークン化が必要です。

比較設定の`output_directory`配下に3つのrunと`comparison.json`を保存します。既定の出力先は`outputs/kfb_v9_comparison/`です。

```bash
uv run yue2-lora compare-train \
  --config configs/examples/v9_comparison.toml --resume --offline
```

再開時はcheckpointのあるrunを再開し、checkpointのないrunは最初から実行します。

## 評価結果の読み方

比較では、学習方式にかかわらずstep 0と各checkpointで共通の課題を評価します。指標はcross entropy（CE）で、低いほど正解の教師系列を予測できていることを表します。

| 指標 | 評価内容 |
|---|---|
| `real_audio_validation` | 実曲のsemantic token。cot=off |
| `real_abc_validation` | 実曲のABC。cot=full |
| `real_conditioned_audio_validation` | 正解ABCを条件にした実曲のsemantic token |
| `minted_validation` | held-out生成曲のsemantic token |
| `minted_abc_validation` | held-out生成曲のABC |

実曲は全validation曲、生成曲は各packのvalidation ID順先頭6曲を固定して使用し、曲ごとの平均CEを等重みで集計します。`real_validation`は`real_audio_validation`の互換名です。ABCを用意しない単独学習では、ABC関連の評価は行いません。

`metrics.jsonl`に指標を、`validation.json`に評価入力のhash・条件・署名を記録します。方式ごとに学習の目的が異なるため、学習中の`loss`をそのまま方式間の順位付けに使うことはできません。

結果を再集計する場合は、各runの出力先を指定します。

```bash
uv run yue2-lora compare \
  --runs outputs/kfb_v9_comparison/cot_off \
         outputs/kfb_v9_comparison/abc_semantic \
         outputs/kfb_v9_comparison/abc_only \
  --output outputs/kfb_v9_comparison/comparison.json
```

評価条件の署名が異なるrunは比較を拒否し、全runに共通するstepだけを集計します。

これらは正解系列を入力するteacher forcingでの評価です。最終的な音質や作曲品質は、同じstyle・lyrics・seed群・生成長・CFG・sampler・VAE・NAR強度で生成し、試聴して判断してください。

## ComfyUIで使用

学習時に、AR LoRAと対応NAR companionを`output_directory/adapters/`へ書き出します。

```bash
uv run yue2-lora install-comfyui \
  --config configs/examples/v9_off.toml \
  --comfyui /path/to/ComfyUI \
  --step 800
```

`--step`には保存済みのstepを指定します。省略時、または`--step 0`は最新のcheckpointを選びます。配置先は`ComfyUI/models/loras/YuE2/<output_directory名>/`です。

### 標準ComfyUI

[Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2)のモデルを用意し、2つの`Load LoRA`ノードでARとNARを読み込みます。

| アダプター | 作用する側 | `strength_model` | `strength_clip` |
|---|---|---|---|
| `step-000800.comfyui.safetensors` | AR / CLIP | `0` | `0.7〜1.0`を目安に調整 |
| `nar_lora_joint_v9.comfyui.safetensors` | NAR / MODEL | `1.0` | `0` |

ファイル名は選択したstepとpairに置き換えてください。生成時のstyleには、学習で設定したトリガーも含めます。

### ComfyUI-FL-YuE2

`FL YuE2 · Load LoRA`で`step-*.fl_yue2.safetensors`を選択します。対応NAR companionはARのmetadataに記録されています。

比較runをインストールする場合は、TOMLの`output_directory`を選んだ子runに合わせ、`cot`・`objective`・regularizer設定もそのrunに合わせてください。ABC-onlyでは`abc_regularizer_fraction=1.0`が必要です。設定とcheckpointのpairが異なる場合、インストールはエラーになります。

## 制約

- 本リポジトリによるGPU実学習・生成音質のベンチマーク結果は未掲載です。CPUテストの通過は音質の保証ではありません。
- real-audio tokenizerの出力は推定されたsemantic tokenです。原音の完全な再現を保証するものではありません。
- 独自のNAR acoustic LoRA学習は未実装です。追加学習の必要性は、まず対応pairでのheld-out音源の再構成とAR学習結果を評価して判断してください。
- 旧v4実装のresume checkpointは現在の学習署名と互換ではありません。移行時は新しい出力ディレクトリを使用してください。

## 開発・テスト

```bash
uv run pytest -q
uv run ruff check src tests
```

テストではpair選択、NAR変換、ABC-onlyの教師範囲・勾配、regularizer分割、CPU代替モデルでの学習ループ、繰り返しvalidation、比較条件の照合を確認しています。

## 参照モデル・ライセンス

モデル資産の固定値は[assets.py](src/yue2_lora/assets.py)と[pairs.json](src/yue2_lora/pairs.json)で管理しています。v5/v8/v9も、名前に`v4`を含む同じMothersuperiorリポジトリから取得します。

- [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- [YuE2公式実装](https://github.com/multimodal-art-projection/YuE)：依存commit `92a73cc7`
- [Mothersuperior tokenizer / NAR pairs](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/blob/e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5/README.md)
- [Mothersuperior minted corpus](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus)
- [ComfyUI-FL-YuE2](https://github.com/filliptm/ComfyUI-FL-YuE2)

YuE2および関連する配布weights・派生アダプターには、CC BY-NC 4.0を含む配布元のライセンス条件が適用されます。学習音源は必要な利用権を持つものを使用してください。第三者のソフトウェア・モデルに関する情報は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
