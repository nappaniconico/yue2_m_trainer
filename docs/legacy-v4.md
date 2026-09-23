# 旧v4手順（履歴用）

以下はリファクタリング前の記録です。現行の設定・実行方法は[README](../README.md)を参照してください。

# YuE2 Kawaii Future Bass LoRA trainer

実音源を YuE2 の semantic token に変換し、YuE2-3B の AR 部分へジャンル LoRA を学習するコードです。このリポジトリの `YouTube/*.wav`（23曲、48 kHz stereo）をそのまま Kawaii Future Bass データセットとして使える設定を同梱しています。音響tokenだけを学ぶ `cot="off"` 版に加え、SheetSage2のABCと音響tokenを同時に教師にする版もあります。

学習後は次の2形式を同時に保存します。

- `*.comfyui.safetensors`: 現行の標準 ComfyUI YuE2 向け。AR と NAR を2つの通常の `Load LoRA` ノードで読みます。
- `*.fl_yue2.safetensors`: [ComfyUI-FL-YuE2](https://github.com/filliptm/ComfyUI-FL-YuE2) 向け。専用 `FL YuE2 · Load LoRA` ノードで読みます。

## 根拠にした公開仕様

- [YuE2-3B 公式モデルカード](https://huggingface.co/m-a-p/YuE2-3B): YuE2 は AR–NAR Mixture-of-Transformers、semantic token 25 Hz、48 kHz stereo 出力です。公式推論は BF16 対応 NVIDIA GPU と Python 3.10+ を対象にしています。
- [YuE2 公式リポジトリ](https://github.com/multimodal-art-projection/YuE): `cot="off"` の style/lyrics prefix、AR/NAR モジュール構造、公式推論コードに合わせています。
- [Mothersuperior real-audio tokenizer v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4): MERT-v2-FullSong layer 20 → 32,768 semantic codes の tokenizer head、rank-32 NAR companion、50/50 regularizer、rank-64 AR LoRA、`1e-4`、最大約1,500 stepという公開レシピを採用しています。
- [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2): 標準 ComfyUI 用の単一checkpointと配置方法に合わせ、ARの分離Q/K/Vとgate/upをComfyUIの結合projectionへ数学的に等価なblock-rank LoRAとして変換します。

指定モデルの元スクリプトは固定の `/workspace/...` パスを前提にしていました。この実装では全パスを TOML 化し、ハッシュ固定ダウンロード、キャッシュ、検証分割、resume、標準ComfyUI用変換を追加しています。

## 必要環境

- Linux / WSL2、Python 3.12
- NVIDIA GPU（BF16対応）。このマシンの RTX 5090 32 GB を想定
- 空きディスクはモデル・キャッシュ・複数checkpoint込みで少なくとも約30 GB推奨
- PyTorch 2.10 + CUDA 12.8

セットアップ:

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`pyproject.toml` は公式と tokenizer v4 の組み合わせに合わせ、YuE2の指定commit、Torch 2.10、Transformers 4.57.6を固定しています。RTX 50系用にCUDA 12.8 indexを使用します。

## 1. データセットsidecarを用意

まずcaptionと空のlyricsファイルを作ります。既存ファイルは上書きしません。

```bash
uv run yue2-lora init-sidecars --config configs/kawaii_future_bass.toml --instrumental
```

各音源に次のファイルができます。

```text
YouTube/000.wav
YouTube/000.caption.txt
YouTube/000.lyrics.txt
```

学習前に必ず全ファイルを確認してください。

- `*.caption.txt`: 曲固有の音色、楽器、テンポ感、展開を追記します。アーティスト名や曲名だけではなく、音響的な説明にします。
- `*.lyrics.txt`: instrumentalなら空のまま。歌がある曲は `[Verse]`、`[Chorus]` 等を含む完全な歌詞へ置き換えます。現在の学習コードはlyric cursorを使わないため、聞き取れない声ネタはcaption側で `chopped vocal textures` と記述してlyricsは空でも構いません。
- 同じ原曲の別編集版が複数ある場合は同じ内容の `*.song.txt` を作ると、train/validationをまたいで漏洩しません。

トリガーは `kfbass_v1` です。生成時のstyleにも必ず入れます。設定は [kawaii_future_bass.toml](configs/kawaii_future_bass.toml) で変更できます。

## 2. モデルを取得してtokenize

```bash
uv run yue2-lora download --config configs/kawaii_future_bass.toml
uv run yue2-lora prepare --config configs/kawaii_future_bass.toml
```

取得物はcommit/hashを固定して検証します。

- `m-a-p/YuE2-3B`
- `m-a-p/MERT-v2-FullSong`
- `tokenizer_head_joint_v4.pt`
- `nar_lora_joint_v4.pt`
- `minted_regularizer_pack.pt`

`prepare` は24 kHz monoのMERT layer-20特徴を25 Hzへ補間し、track単位instance normalization後、512-frame・50% overlapでsemantic tokenを推定します。音源自体は変更しません。再実行時はcontent-addressed cacheを再利用します。

## 3. 学習

```bash
uv run yue2-lora train --config configs/kawaii_future_bass.toml
```

中断後は同じ設定で:

```bash
uv run yue2-lora train --config configs/kawaii_future_bass.toml --resume --offline
```

既定値はrank 64、1,500 steps、LR `1e-4`、gradient accumulation 2、real/mintedを50/50です。全28層のAR `self_attn.{q,k,v,o}_proj` と `mlp.{gate,up,down}_proj` のみ学習し、base weightは凍結します。`outputs/kawaii_future_bass/metrics.jsonl` の以下を比較してください。

- `real_validation`: Kawaii Future Bass検証曲のloss。下がるのが期待値
- `minted_validation`: YuE2本来のtoken文法の保持指標。大きく悪化するなら過学習

配布元は1,500 step超で暗記が増えたと明記しています。最後を自動採用せず、600/800/1000/1200/1400/1500の固定seed生成を聴き比べて選んでください。配布元の例では800 stepが採用されていますが、このデータでも同じとは限りません。

### ABC作譜も学習する版

公式SheetSage2で各曲を転写し、出力された `score.abc` を音源と同じstemの `.abc` として置きます。既定の `cot="full"` はメロディーとコードを学ぶため、SheetSage2の通常出力を使います。

```text
YouTube/000.wav
YouTube/000.caption.txt
YouTube/000.lyrics.txt
YouTube/000.abc
```

ABCの空テンプレートだけを先に作る場合は次を実行します。既存のcaption/lyricsは上書きしません。

```bash
uv run yue2-lora init-sidecars --config configs/kawaii_future_bass_abc.toml --instrumental
```

SheetSage2は学習コードとは対応Python/Torchが異なるため、公式手順どおり専用環境を作ります。リポジトリへのアクセス許可とHugging Faceへのログインが必要です。

```bash
uv venv --python 3.11 .venv-sheetsage2
uv pip install --python .venv-sheetsage2/bin/python huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 --local-dir models/SheetSage2
uv pip install --python .venv-sheetsage2/bin/python torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sheetsage2/bin/python -r models/SheetSage2/requirements.txt
```

公式例はCUDA 12.6ですが、このマシンのRTX 5090向けに、PyTorch公式が配布している同じ2.8.0のCUDA 12.8 wheelを指定しています。

通常のメロディー＋コードABCを全曲一括出力します。配置済みの空テンプレートは置換しますが、内容のあるABCは保護します。

```bash
.venv-sheetsage2/bin/python scripts/transcribe_abc.py \
  --input-directory YouTube \
  --model models/SheetSage2 \
  --base-model models/MERT-v2-FullSong \
  --offline
```

`cot="melody"` 用のコード記号なしABCにする場合は `--melody-only` を追加します。確認済みABCも再転写する場合だけ `--overwrite` を指定してください。このリポジトリが既に持つ、SheetSage2公式と同じrevisionのMERT-v2親モデルを `--base-model` で再利用しています。

全23曲の `.abc` をSheetSage2出力へ置き換えて目視・試聴確認した後、ABC用設定でprepareとtrainを実行します。既存の音響token cacheは再利用されます。

```bash
uv run yue2-lora prepare --config configs/kawaii_future_bass_abc.toml --offline
uv run yue2-lora train --config configs/kawaii_future_bass_abc.toml --offline
```

このモードの教師系列は次の2区間です。YuE2が自動挿入する `MUSIC_START` 自体は教師から除外します。

```text
条件 → ABC_START → [ABC本文 + ABC_END] → MUSIC_START → [semantic tokens + MUSIC_END]
                       ABC loss                         audio loss
```

`abc_loss_weight` と `audio_loss_weight` は区間ごとの平均cross entropyを合成する重みです。既定の `1.0 : 1.0` では、token数の短いABC区間も音響区間と同じ比重を持ちます。`metrics.jsonl` には次も記録されます。

- `abc_loss`: そのstepで実曲が選ばれた場合のABC学習loss
- `audio_loss`: semantic audio token学習loss
- `real_abc_validation`: held-out実曲のABC loss
- `real_audio_validation`: held-out実曲のaudio loss

minted regularizerにはABCがないため、従来どおり `cot="off"` のaudio lossだけを掛けます。ABC版は `outputs/kawaii_future_bass_abc` へ別runとして保存され、従来runを上書き・resumeしません。SheetSage2の `melody_only=True` 出力を使う場合は、ABC用設定の `cot = "melody"` に変更してください。

## 4. ComfyUIで使用

任意checkpointをComfyUIへコピーします（`--step 0` は最新）。

```bash
uv run yue2-lora install-comfyui \
  --config configs/kawaii_future_bass.toml \
  --comfyui /path/to/ComfyUI \
  --step 800
```

### 標準ComfyUI

1. [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) の `yue2_3b_bf16.safetensors` と `sheetsage2_bf16.safetensors` をモデルカード記載の場所へ置きます。
2. Checkpoint loaderの `MODEL` と `CLIP` を1個目の `Load LoRA` へ接続し、`step-000800.comfyui.safetensors` を選択します。`strength_model=0`、`strength_clip=0.7〜1.0`。
3. その出力を2個目の `Load LoRA` へ接続し、`nar_lora_joint_v4.comfyui.safetensors` を選択します。`strength_model=1.0`、`strength_clip=0`。
4. 以後は標準の `YuE2 Generate ABC` → `YuE2 Generate Music` → sampler → VAE decodeへ接続します。
5. style例: `kfbass_v1, Kawaii Future Bass, sparkling synths, playful melody, punchy sidechained drums, colorful energetic drop`

ARはsemantic token生成側（CLIP）、NAR companionは音響latent生成側（MODEL）へ作用するため、片方だけでは意図した実音源tokenizerの音になりません。

### ComfyUI-FL-YuE2

`FL YuE2 · Load LoRA` で `kawaii_future_bass/step-000800.fl_yue2.safetensors` を選択してください。同じフォルダの `nar_lora_joint_v4.fl_yue2.safetensors` はmetadataから自動的に併用されます。ABC版を学習した場合は、代わりに `kawaii_future_bass_abc` 側のcheckpointを選びます。

## 安全性・ライセンス

YuE2-3B、tokenizer由来weights、学習LoRAは **CC BY-NC 4.0（非商用）** の制約を引き継ぎます。学習音源についても、複製・学習・生成に必要な権利を自分が持つものだけを使用してください。YouTube由来であること自体は利用許諾を意味しません。

実装上の既知の制約:

- v4 tokenizerのモデルカード上のtop-1 exact matchは16.1%で、token IDの完全再現を保証しません。音の近いcodeが多いという前提の研究的ツールです。
- このコードはARジャンルLoRAのみを学習します。ABC版でもABC text tokenとsemantic audio tokenを出す共有ARだけが対象です。tokenizer headとNARを手元音源へ再学習するoptional joint stageは、過学習・破壊的劣化のリスクが高いため含めていません。
- 初回の品質判断には必ず複数checkpoint、同一prompt、同一seedでの試聴が必要です。
