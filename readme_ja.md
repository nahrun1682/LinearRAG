# **LinearRAG: 大規模コーパス向け線形グラフ検索拡張生成**  

> 効率的な GraphRAG を実現する、関係非依存のグラフ構築手法です。グラフ構築時の LLM トークンコストを排除し、GraphRAG をこれまで以上に高速かつ効率的にします。

<p align="center">
  <a href="https://arxiv.org/abs/2510.10114" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2506.08938">
  </a>
  <a href="https://huggingface.co/datasets/Zly0523/linear-rag/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=flat-square" alt="HuggingFace">
  </a>
  <a href="https://github.com/LuyaoZhuang/linear-rag" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
</p>

---
## 🎉 **ニュース**
- **[2026-04-07]** RAG の忠実性に関する **[ProbeRAG](https://github.com/LinfengGao/ProbeRAG.git)** が ACL'26 に採択されました。
- **[2026-04-07]** 信頼性の高いエージェント型検索に関する **[BAPO](https://github.com/Liushiyu-0709/BAPO-Reliable-Search.git)** が ACL'26 に採択されました。
- **[2026-04-07]** 信頼性の高い法的推論に関する **[LegalGraphRAG](https://github.com/XMUDeepLIT/LegalGraphRAG.git)** が ACL'26 に採択されました。
- **[2026-04-07]** GraphRAG 攻撃モデルである **[LogicPoison](https://github.com/Jord8061/logicPoison.git)** が ACL'26 に採択されました。
- **[2026-01-26]** 効率的な GraphRAG のための **[LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)** が ICLR’26 に採択されました。
- **[2026-01-26]** **[GraphRAG Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)** が ICLR’26 に採択されました。
- **[2025-10-27]** 効率的な GraphRAG のための関係非依存グラフ構築手法 **[LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)** を公開しました。
- **[2025-06-06]** GraphRAG モデル評価用の **[GraphRAG Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git)** を公開しました。
- **[2025-01-21]** **[GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG)** を公開しました。

---

## 🚀 **ハイライト**
- ✅ **文脈保持**: 軽量なエンティティ認識とセマンティックリンクに基づく関係非依存グラフ構築により、包括的な文脈理解を実現します。 
- ✅ **複雑推論**: セマンティックブリッジによる深い検索を可能にし、明示的な関係グラフなしで、1 回の検索パスでマルチホップ推論を実現します。
- ✅ **高いスケーラビリティ**: LLM トークン消費ゼロ、高速な処理速度、線形の時間・空間計算量を実現します。
  
<p align="center">
  <img src="figure/main_figure.png" width="95%" alt="Framework Overview">
</p>

---

## 🛠️ **使い方**

### 1️⃣ 依存関係のインストール  

**ステップ 1: Python パッケージをインストール**

```bash
pip install -r requirements.txt
(Python 3.9 を推奨)
```

**ステップ 2: Spacy の言語モデルをダウンロード**

```bash
python -m spacy download en_core_web_trf
```

> **注記:** `medical` データセットでは、科学・生物医学向けの Spacy モデルを別途インストールする必要があります。
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
```

**ステップ 3: OpenAI API キーを設定**

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="your-base-url-here"
```

**ステップ 4: データセットをダウンロード**

HuggingFace からデータセットをダウンロードし、`dataset/` フォルダに配置してください。

```bash
git clone https://huggingface.co/datasets/Zly0523/linear-rag
cp -r linear-rag/* dataset/
```

**ステップ 5: 埋め込みモデルを準備**

以下の場所に埋め込みモデルが存在することを確認してください。

```
model/all-mpnet-base-v2/
```


### 2️⃣ クイックスタート例

```bash
SPACY_MODEL="en_core_web_trf"
EMBEDDING_MODEL="model/all-mpnet-base-v2"
DATASET_NAME="2wikimultihop"
LLM_MODEL="gpt-4o-mini"
MAX_WORKERS=16

python run.py \
    --spacy_model ${SPACY_MODEL} \
    --embedding_model ${EMBEDDING_MODEL} \
    --dataset_name ${DATASET_NAME} \
    --llm_model ${LLM_MODEL} \
    --max_workers ${MAX_WORKERS} 
    # --use_vectorized_retrieval # オプション: 強力な GPU が利用可能な場合は、GPU 高速化のためベクトル化された行列ベース検索を使用します。そうでない場合は BFS 反復を使用します。
```

## 🎯 **性能**

<div align="center">
<img src="figure/generation_results.png" alt="framework" width="1000">

**エンドツーエンド性能の主要結果**
</div>
<div align="center">
<img src="figure/efficiency_result.png" alt="framework" width="1000">

**効率と性能の比較**
</div>


## 📬 Citation

この研究が役に立った場合は、以下の引用をご検討ください。
```bibtex
@article{zhuang2025linearrag,
  title={LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora},
  author={Zhuang, Luyao and Chen, Shengyuan and Xiao, Yilin and Zhou, Huachi and Zhang, Yujing and Chen, Hao and Zhang, Qinggang and Huang, Xiao},
  journal={arXiv preprint arXiv:2510.10114},
  year={2025}
}
``` 
