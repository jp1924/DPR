# Dense Passage Retrieval for Open-Domain Question Answering
## Abstract
Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dualencoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong LuceneBM25 system greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks

개방형 도메인 질문 답변은 후보 문맥을 선택하기 위해 효율적인 구절 검색에 의존하는데, TF-IDF나 BM25와 같은 기존의 희소 벡터 공간 모델이 사실상 유일한 방법입니다. 이 연구에서는 간단한 이중 인코더 프레임워크를 통해 소수의 질문과 구절에서 임베딩을 학습하는 고밀도 표현만으로 검색을 실질적으로 구현할 수 있음을 보여줍니다. 광범위한 오픈 도메인 QA 데이터 세트에서 평가했을 때, 당사의 고밀도 검색기는 상위 20개 구절 검색 정확도 측면에서 강력한 LuceneBM25 시스템을 9%-19% 절대적으로 크게 능가하며, 여러 오픈 도메인 QA 벤치마크에서 엔드투엔드 QA 시스템이 새로운 최신 기술을 구축하는 데 도움이 됩니다.

## Introduction
Open-domain question answering (QA) (Voorhees, 1999) is a task that answers factoid questions using a large collection of documents. While early QA systems are often complicated and consist of multiple components (Ferrucci (2012); Moldovan et al. (2003), inter alia), the advances of reading comprehension models suggest a much simplified two-stage framework: (1) a context retriever first selects a small subset of passages where some of them contain the answer to the question, and then (2) a machine reader can thoroughly examine the retrieved contexts and identify the correct answer (Chen et al., 2017). Although reducing open-domain QA to machine reading is a very reasonable strategy, a huge performance degradation is often observed in practice2 , indicating the needs of improving retrieval.

Retrieval in open-domain QA is usually implemented using TF-IDF or BM25 (Robertson and Zaragoza, 2009), which matches keywords efficiently with an inverted index and can be seen as representing the question and context in highdimensional, sparse vectors (with weighting). Conversely, the dense, latent semantic encoding is complementary to sparse representations by design. For example, synonyms or paraphrases that consist of completely different tokens may still be mapped to vectors close to each other. Consider the question “Who is the bad guy in lord of the rings?”, which can be answered from the context “Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy.” A term-based system would have difficulty retrieving such a context, while a dense retrieval system would be able to better match “bad guy” with “villain” and fetch the correct context. Dense encodings are also learnable by adjusting the embedding functions, which provides additional flexibility to have a task-specific representation. With special in-memory data structures and indexing schemes, retrieval can be done efficiently using maximum inner product search (MIPS) algorithms (e.g., Shrivastava and Li (2014); Guo et al. (2016)).

However, it is generally believed that learning a good dense vector representation needs a large number of labeled pairs of question and contexts. Dense retrieval methods have thus never be shown to outperform TF-IDF/BM25 for opendomain QA before ORQA (Lee et al., 2019), which proposes a sophisticated inverse cloze task (ICT) objective, predicting the blocks that contain the masked sentence, for additional pretraining. The question encoder and the reader model are then finetuned using pairs of questions and answers jointly. Although ORQA successfully demonstrates that dense retrieval can outperform BM25, setting new state-of-the-art results on multiple open-domain QA datasets, it also suffers from two weaknesses. First, ICT pretraining is computationally intensive and it is not completely clear that regular sentences are good surrogates of questions in the objective function. Second, because the context encoder is not fine-tuned using pairs of questions and answers, the corresponding representations could be suboptimal.

In this paper, we address the question: can we train a better dense embedding model using only pairs of questions and passages (or answers), without additional pretraining? By leveraging the now standard BERT pretrained model (Devlin et al., 2019) and a dual-encoder architecture (Bromley et al., 1994), we focus on developing the right training scheme using a relatively small number of question and passage pairs. Through a series of careful ablation studies, our final solution is surprisingly simple: the embedding is optimized for maximizing inner products of the question and relevant passage vectors, with an objective comparing all pairs of questions and passages in a batch. Our Dense Passage Retriever (DPR) is exceptionally strong. It not only outperforms BM25 by a large margin (65.2% vs. 42.9% in Top-5 accuracy), but also results in a substantial improvement on the end-to-end QA accuracy compared to ORQA (41.5% vs. 33.3%) in the open Natural Questions setting (Lee et al., 2019; Kwiatkowski et al., 2019).

Our contributions are twofold. First, we demonstrate that with the proper training setup, simply fine-tuning the question and passage encoders on existing question-passage pairs is sufficient to greatly outperform BM25. Our empirical results also suggest that additional pretraining may not be needed. Second, we verify that, in the context of open-domain question answering, a higher retrieval precision indeed translates to a higher end-to-end QA accuracy. By applying a modern reader model to the top retrieved passages, we achieve comparable or better results on multiple QA datasets in the open-retrieval setting, compared to several, much complicated systems.

개방형 도메인 질문 답변(QA)(Voorhees, 1999)은 대량의 문서 모음을 사용하여 사실적 질문에 답변하는 작업입니다. 초기 QA 시스템은 종종 복잡하고 여러 구성 요소로 이루어져 있지만(Ferrucci, 2012; Moldovan 등, 2003), 독해 모델의 발전으로 (1) 문맥 검색기가 먼저 질문에 대한 답이 포함된 구절의 작은 하위 집합을 선택한 다음 (2) 기계 리더가 검색된 문맥을 철저히 검토하여 정답을 식별하는 훨씬 단순화된 2단계 프레임워크가 제안되었습니다(Chen et al., 2017). 오픈 도메인 QA를 기계 판독으로 줄이는 것은 매우 합리적인 전략이지만, 실제로는 엄청난 성능 저하가 종종 관찰되어2 검색을 개선해야 할 필요성을 나타냅니다.

오픈 도메인 QA에서의 검색은 일반적으로 TF-IDF 또는 BM25(Robertson and Zaragoza, 2009)를 사용하여 구현되며, 이는 역 인덱스로 키워드를 효율적으로 매칭하고 질문과 문맥을 고차원의 희소 벡터(가중치 포함)로 표현하는 것으로 볼 수 있습니다. 반대로, 조밀하고 잠재적인 의미 인코딩은 설계상 스파스 표현을 보완합니다. 예를 들어, 완전히 다른 토큰으로 구성된 동의어 또는 의역어도 서로 가까운 벡터에 매핑될 수 있습니다. "반지의 제왕에서 악당은 누구인가요?"라는 질문에 "살라 베이커는 반지의 제왕 3부작에서 악당 사우론을 연기한 것으로 가장 잘 알려져 있습니다."라는 문맥에서 대답할 수 있는 경우를 생각해 보세요. 용어 기반 시스템은 이러한 문맥을 검색하는 데 어려움을 겪을 수 있지만, 고밀도 검색 시스템은 '나쁜 사람'과 '악당'을 더 잘 매칭하고 올바른 문맥을 가져올 수 있습니다. 또한 임베딩 함수를 조정하여 고밀도 인코딩을 학습할 수 있으므로 작업별 표현을 위한 추가적인 유연성을 제공합니다. 특수한 인메모리 데이터 구조와 인덱싱 체계를 사용하면 최대 내부 곱 검색(MIPS) 알고리즘을 사용하여 검색을 효율적으로 수행할 수 있습니다(예: Shrivastava and Li(2014), Guo et al.(2016)).

그러나 일반적으로 좋은 고밀도 벡터 표현을 학습하려면 많은 수의 레이블이 지정된 질문과 컨텍스트 쌍이 필요하다고 알려져 있습니다. 따라서 고밀도 검색 방법은 추가적인 사전 학습을 위해 마스크된 문장을 포함하는 블록을 예측하는 정교한 역 클로즈 과제(ICT) 목표를 제안하는 ORQA(Lee et al., 2019) 이전에는 오픈 도메인 QA에서 TF-IDF/BM25보다 성능이 뛰어난 것으로 나타난 적이 없습니다. 그런 다음 질문 인코더와 리더 모델은 질문과 답변 쌍을 함께 사용하여 미세 조정됩니다. ORQA는 여러 오픈 도메인 QA 데이터 세트에서 새로운 최첨단 결과를 설정하면서 밀도 검색이 BM25를 능가할 수 있음을 성공적으로 입증했지만, 두 가지 약점을 가지고 있기도 합니다. 첫째, ICT 사전 학습은 계산 집약적이며 일반 문장이 목적 함수의 질문에 대한 좋은 대용어인지 완전히 명확하지 않습니다. 둘째, 문맥 인코더는 질문과 답변 쌍을 사용하여 미세 조정되지 않기 때문에 해당 표현이 차선책일 수 있습니다.

이 백서에서는 추가 사전 훈련 없이 질문과 구절(또는 답) 쌍만을 사용해 더 조밀한 임베딩 모델을 훈련할 수 있는가라는 질문을 다룹니다. 현재 표준이 된 BERT 사전 훈련 모델(Devlin et al., 2019)과 이중 인코더 아키텍처(Bromley et al., 1994)를 활용하여 상대적으로 적은 수의 질문과 구절 쌍을 사용하여 올바른 훈련 체계를 개발하는 데 중점을 둡니다. 일련의 신중한 제거 연구를 통해 최종 솔루션은 놀랍도록 간단합니다. 임베딩은 문제와 관련 구절 벡터의 내부 곱을 최대화하는 데 최적화되어 있으며, 모든 문제와 구절 쌍을 일괄적으로 비교하는 것을 목표로 합니다. 밀도 높은 구절 리트리버(DPR)는 매우 강력합니다. BM25를 큰 차이로 능가할 뿐만 아니라(상위 5개 정확도에서 65.2% 대 42.9%), 개방형 자연 문제 환경에서 ORQA(41.5% 대 33.3%)에 비해 엔드투엔드 QA 정확도가 크게 향상되었습니다(Lee et al., 2019; Kwiatkowski et al., 2019).

우리의 기여는 두 가지입니다. 첫째, 적절한 훈련 설정을 통해 기존 문제-지문 쌍에서 문제와 지문 인코더를 미세 조정하는 것만으로도 BM25의 성능을 크게 향상시킬 수 있음을 입증했습니다. 또한 경험적 결과는 추가적인 사전 훈련이 필요하지 않을 수도 있음을 시사합니다. 둘째, 오픈 도메인 질문 답변의 맥락에서 검색 정밀도가 높을수록 실제로 엔드투엔드 QA 정확도가 높아진다는 사실을 확인했습니다. 검색된 상위 구절에 최신 리더 모델을 적용함으로써 개방형 검색 환경의 여러 QA 데이터 세트에서 훨씬 복잡한 여러 시스템에 비해 비슷하거나 더 나은 결과를 얻을 수 있었습니다.


## Background

The problem of open-domain QA studied in this paper can be described as follows. Given a factoid question, such as “Who first voiced Meg on Family Guy?” or “Where was the 8th Dalai Lama born?”, a system is required to answer it using a large corpus of diversified topics. More specifically, we assume the extractive QA setting, in which the answer is restricted to a span appearing in one or more passages in the corpus. Assume that our collection contains D documents, d1, d2, · · · , dD. We first split each of the documents into text passages of equal lengths as the basic retrieval units3 and get M total passages in our corpus C = {p1, p2, . . . , pM}, where each passage pi can be viewed as a sequence of tokens w (i) 1 , w (i) 2 , · · · , w (i) |pi| . Given a question q,

the task is to find a span w (i) s , w (i) s+1, · · · , w (i) e from one of the passages pi that can answer the question. Notice that to cover a wide variety of domains, the corpus size can easily range from millions of documents (e.g., Wikipedia) to billions (e.g., the Web). As a result, any open-domain QA system needs to include an efficient retriever component that can select a small set of relevant texts, before applying the reader to extract the answer (Chen et al., 2017).4 Formally speaking, a retriever R : (q, C) → CF is a function that takes as input a question q and a corpus C and returns a much smaller filter set of texts CF ⊂ C, where |CF | = k  |C|. For a fixed k, a retriever can be evaluated in isolation on top-k retrieval accuracy, which is the fraction of questions for which CF contains a span that answers the question.

이 백서에서 연구한 오픈 도메인 QA의 문제는 다음과 같이 설명할 수 있습니다. "패밀리 가이에서 멕의 목소리를 처음 낸 사람은 누구인가?" 또는 "제8대 달라이 라마는 어디에서 태어났나?"와 같은 팩트형 질문이 주어지면 다양한 주제의 대규모 말뭉치를 사용하여 답변하는 시스템이 필요합니다. 좀 더 구체적으로 말뭉치에서 하나 이상의 구절에 나타나는 범위로 답변이 제한되는 추출 QA 설정을 가정해 보겠습니다. 컬렉션에 d1, d2, - - - , dD 문서가 포함되어 있다고 가정합니다. 먼저 각 문서를 기본 검색 단위3와 동일한 길이의 텍스트 구절로 분할하여 말뭉치 C = {p1, p2, ... . , pM}을 구하며, 여기서 각 구절 pi는 토큰 w (i) 1 , w (i) 2 , - - - , w (i) |pi| 의 시퀀스로 볼 수 있습니다. 질문 q가 주어졌습니다,

의 경우, 질문에 답할 수 있는 구절 파이 중 하나에서 w (i) s , w (i) s+1, - - - , w (i) e를 찾는 것이 과제입니다. 다양한 도메인을 포괄하기 위해 말뭉치 크기는 수백만 개의 문서(예: Wikipedia)에서 수십억 개의 문서(예: 웹)에 이르기까지 쉽게 다양할 수 있다는 점에 유의하세요. 따라서 모든 오픈 도메인 QA 시스템에는 판독기를 적용하여 답을 추출하기 전에 소수의 관련 텍스트 집합을 선택할 수 있는 효율적인 리트리버 구성 요소가 포함되어야 합니다(Chen et al., 2017).4 공식적으로 말하면, 리트리버 R : (q, C) → CF는 질문 q와 코퍼스 C를 입력으로 받아 훨씬 작은 필터 집합인 텍스트 CF ⊂ C를 반환하는 함수로서, 여기서 |CF | = k |C|를 반환합니다. 고정된 k의 경우, 검색기는 CF에 질문에 대한 답을 제공하는 스팬이 포함된 질문의 비율인 상위 k 검색 정확도에 대해 개별적으로 평가할 수 있습니다.


## Dense Passage Retriever (DPR)

We focus our research in this work on improving the retrieval component in open-domain QA. Given a collection of M text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efficiently the top k passages relevant to the input question for the reader at run-time. Note that M can be very large (e.g., 21 million passages in our experiments, described in Section 4.1) and k is usually small, such as 20–100.

이번 연구에서는 오픈 도메인 QA의 검색 구성 요소를 개선하는 데 초점을 맞췄습니다. M개의 텍스트 구절 모음이 주어졌을 때, 밀도 높은 구절 검색기(DPR)의 목표는 저차원의 연속 공간에서 모든 구절을 색인하여 런타임에 독자가 입력한 질문과 관련된 상위 k개의 구절을 효율적으로 검색할 수 있도록 하는 것입니다. M은 매우 클 수 있으며(예: 4.1절에 설명된 실험에서 2100만 개의 구절), k는 일반적으로 20-100과 같이 작습니다.

### Overview
Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a ddimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval. At run-time, DPR applies a different encoder EQ(·) that maps the input question to a d-dimensional vector, and retrieves k passages of which vectors are the closest to the question vector. We define the similarity between the question and the passage using the dot product of their vectors: 

sim(q, p) = EQ(q) |EP (p). 

Although more expressive model forms for measuring the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the representations of the collection of passages can be precomputed. Most decomposable similarity functions are some transformations of Euclidean distance (L2). For instance, cosine is equivalent to inner product for unit vectors and the Mahalanobis distance is equivalent to L2 distance in a transformed space. Inner product search has been widely used and studied, as well as its connection to cosine similarity and L2 distance (Mussmann and Ermon, 2016; Ram and Gray, 2012). As our ablation study finds other similarity functions perform comparably (Section 5.2; Appendix B), we thus choose the simpler inner product function and improve the dense passage retriever by learning better encoders.

Encoders Although in principle the question and passage encoders can be implemented by any neural networks, in this work we use two independent BERT (Devlin et al., 2019) networks (base, uncased) and take the representation at the [CLS] token as the output, so d = 768.

Inference During inference time, we apply the passage encoder EP to all the passages and index them using FAISS (Johnson et al., 2017) offline. FAISS is an extremely efficient, open-source library for similarity search and clustering of dense vectors, which can easily be applied to billions of vectors. Given a question q at run-time, we derive its embedding vq = EQ(q) and retrieve the top k passages with embeddings closest to vq.

밀도 구절 검색(DPR)은 모든 텍스트 구절을 d차원 실수값 벡터에 매핑하는 밀도 인코더 EP(-)를 사용해 검색에 사용할 모든 M 구절에 대한 인덱스를 구축합니다. 런타임에 DPR은 입력 문제를 d차원 벡터에 매핑하는 다른 인코더 EQ(-)를 적용하여 문제 벡터에 가장 가까운 벡터가 있는 k개의 구절을 검색합니다. 벡터의 도트 곱을 사용하여 문제와 구절 사이의 유사성을 정의합니다:

여러 층의 교차 주의로 구성된 네트워크와 같이 문제와 구절 사이의 유사도를 측정하는 더 표현적인 모델 형태가 존재하지만, 구절 집합의 표현을 미리 계산할 수 있도록 유사도 함수는 분해 가능해야 합니다. 대부분의 분해 가능한 유사도 함수는 유클리드 거리(L2)의 일부 변환입니다. 예를 들어 코사인은 단위 벡터의 내적 곱에 해당하며, 마하라노비스 거리는 변환된 공간에서 L2 거리와 동일합니다. 내적 곱 검색은 코사인 유사도 및 L2 거리와의 연관성뿐만 아니라 널리 사용되고 연구되어 왔습니다(Mussmann and Ermon, 2016; Ram and Gray, 2012). 우리의 제거 연구에서 다른 유사도 함수의 성능이 비슷하다는 것을 발견했기 때문에(섹션 5.2, 부록 B), 더 간단한 내적 곱 함수를 선택하고 더 나은 인코더를 학습하여 밀도 통과 검색을 개선했습니다.

인코더 원칙적으로 문제 및 구절 인코더는 모든 신경망으로 구현할 수 있지만, 이 작업에서는 두 개의 독립적인 BERT(Devlin et al., 2019) 네트워크(기본, 대소문자 구분 없음)를 사용하고 [CLS] 토큰의 표현을 출력으로 사용하므로 d = 768이 됩니다.

추론 시간 동안 모든 구절에 구절 인코더 EP를 적용하고 오프라인에서 FAISS(Johnson et al., 2017)를 사용해 색인을 생성합니다. FAISS는 고밀도 벡터의 유사도 검색과 클러스터링을 위한 매우 효율적인 오픈 소스 라이브러리로, 수십억 개의 벡터에 쉽게 적용할 수 있습니다. 런타임에 질문 q가 주어지면, 그 임베딩 vq = EQ(q)를 도출하고 vq에 가장 가까운 임베딩을 가진 상위 k개의 구절을 검색합니다.


### Training
Training the encoders so that the dot-product similarity (Eq. (1)) becomes a good ranking function for retrieval is essentially a metric learning problem (Kulis, 2013). The goal is to create a vector space such that relevant pairs of questions and passages will have smaller distance (i.e., higher similarity) than the irrelevant ones, by learning a better embedding function.

Let D = {hqi , p+ i , p− i,1 , · · · , p− i,ni}m i=1 be the training data that consists of m instances. Each instance contains one question qi and one relevant (positive) passage p + i , along with n irrelevant (negative) passages p − i,j . We optimize the loss function as the negative log likelihood of the positive passage:

L(qi , p+ i , p− i,1 , · · · , p− i,n) (2)

= − log e sim(qi,p + i ) e sim(qi,p + i ) + Pn j=1 e sim(qi,p − i,j )

Positive and negative passages For retrieval problems, it is often the case that positive examples are available explicitly, while negative examples need to be selected from an extremely large pool. For instance, passages relevant to a question may be given in a QA dataset, or can be found using the answer. All other passages in the collection, while not specified explicitly, can be viewed as irrelevant by default. In practice, how to select negative examples is often overlooked but could be decisive for learning a high-quality encoder. We consider three different types of negatives: (1) Random: any random passage from the corpus; (2) BM25: top passages returned by BM25 which don’t contain the answer but match most question tokens; (3) Gold: positive passages paired with other questions which appear in the training set. We will discuss the impact of different types of negative passages and training schemes in Section 5.2. Our best model uses gold passages from the same mini-batch and one BM25 negative passage. In particular, re-using gold passages from the same batch as negatives can make the computation efficient while achieving great performance. We discuss this approach below.

In-batch negatives Assume that we have B questions in a mini-batch and each one is associated with a relevant passage. Let Q and P be the (B×d) matrix of question and passage embeddings in a batch of size B. S = QPT is a (B × B) matrix of similarity scores, where each row of which corresponds to a question, paired with B passages. In this way, we reuse computation and effectively train on B2 (qi , pj ) question/passage pairs in each batch. Any (qi , pj ) pair is a positive example when i = j, and negative otherwise. This creates B training instances in each batch, where there are B − 1 negative passages for each question.

The trick of in-batch negatives has been used in the full batch setting (Yih et al., 2011) and more recently for mini-batch (Henderson et al., 2017; Gillick et al., 2019). It has been shown to be an effective strategy for learning a dual-encoder model that boosts the number of training examples.

점-제품 유사도(식 (1))가 검색에 좋은 순위 함수가 되도록 인코더를 훈련하는 것은 본질적으로 메트릭 학습 문제입니다(Kulis, 2013). 목표는 더 나은 임베딩 함수를 학습하여 관련성이 있는 문제와 구절 쌍이 관련성이 없는 문제와 구절보다 더 작은 거리(즉, 더 높은 유사도)를 갖도록 벡터 공간을 만드는 것입니다.

D = {hqi , p+ i , p- i,1 , - - - , p- i,ni}m i=1 을 m개의 인스턴스로 구성된 훈련 데이터라고 합니다. 각 인스턴스에는 하나의 문제 qi와 하나의 관련성 있는(정답) 구절 p + i 및 관련성 없는(부답) 구절 p - i,j 가 포함됩니다. 손실 함수는 양의 구절의 음의 로그 확률로 최적화합니다:

L(qi , p+ i , p− i,1 , · · · , p− i,n) (2)

= − log e sim(qi,p + i ) e sim(qi,p + i ) + Pn j=1 e sim(qi,p − i,j )

긍정 및 부정 구절 검색 문제의 경우 긍정 예시는 명시적으로 제공되는 반면, 부정 예시는 매우 방대한 풀에서 선택해야 하는 경우가 많습니다. 예를 들어, 문제와 관련된 구절은 QA 데이터 세트에 제공되거나 답을 사용하여 찾을 수 있습니다. 컬렉션의 다른 모든 구절은 명시적으로 지정되지 않았지만 기본적으로 관련성이 없는 것으로 볼 수 있습니다. 실제로 부정 예문을 선택하는 방법은 종종 간과되지만 고품질 인코더를 학습하는 데 결정적일 수 있습니다. (1) 무작위: 말뭉치에서 임의의 구절, (2) BM25: 답을 포함하지 않지만 대부분의 질문 토큰과 일치하는 BM25가 반환한 상위 구절, (3) 골드: 훈련 세트에 나타나는 다른 질문과 짝을 이루는 긍정적인 구절. 5.2절에서 다양한 유형의 부정 구절과 훈련 방식이 미치는 영향에 대해 설명하겠습니다. 가장 좋은 모델은 동일한 미니 배치의 골드 구절과 BM25 부정 구절 하나를 사용합니다. 특히 같은 배치의 골드 통로를 네거티브 통로로 재사용하면 계산을 효율적으로 하면서도 뛰어난 성능을 달성할 수 있습니다. 아래에서 이 접근 방식에 대해 설명합니다.

배치 내 네거티브 미니 배치에 B개의 문제가 있고 각 문제가 관련 구절과 연계되어 있다고 가정합니다. Q와 P를 B 크기의 배치에 포함된 문제와 구절의 (B×d) 행렬이라고 합니다. S = QPT는 유사도 점수의 (B × B) 행렬로, 각 행이 문제와 짝을 이루는 B개의 구절에 해당합니다. 이러한 방식으로 계산을 재사용하고 각 배치에서 B2 (qi , pj ) 문제/지문 쌍에 대해 효과적으로 훈련합니다. 모든 (qi , pj ) 쌍은 i = j일 때 양의 예시이고, 그렇지 않으면 음의 예시입니다. 이렇게 하면 각 배치에 B개의 훈련 인스턴스가 생성되며, 각 문제에는 B-1개의 부정적 구절이 있습니다.

배치 내 네거티브의 트릭은 전체 배치 설정(Yih et al., 2011)에서 사용되었으며 최근에는 미니 배치(Henderson et al., 2017; Gillick et al., 2019)에도 사용되었습니다. 이는 훈련 예시 수를 늘리는 이중 인코더 모델을 학습하는 데 효과적인 전략인 것으로 나타났습니다.

## Experimental Setup

In this section, we describe the data we used for experiments and the basic setup

이 섹션에서는 실험에 사용한 데이터와 기본 설정에 대해 설명합니다.

### Wikipedia Data Pre-processing

Following (Lee et al., 2019), we use the English Wikipedia dump from Dec. 20, 2018 as the source documents for answering questions. We first apply the pre-processing code released in DrQA (Chen et al., 2017) to extract the clean, text-portion of articles from the Wikipedia dump. This step removes semi-structured data, such as tables, infoboxes, lists, as well as the disambiguation pages. We then split each article into multiple, disjoint text blocks of 100 words as passages, serving as our basic retrieval units, following (Wang et al., 2019), which results in 21,015,324 passages in the end.5 Each passage is also prepended with the title of the Wikipedia article where the passage is from, along with an [SEP] token.

(Lee et al., 2019)에 따라 2018년 12월 20일의 영문 위키백과 덤프를 질문에 대한 답변을 위한 소스 문서로 사용합니다. 먼저 DrQA에서 공개된 전처리 코드(Chen et al., 2017)를 적용하여 위키백과 덤프에서 깨끗한 텍스트 부분의 문서를 추출합니다. 이 단계에서는 표, 인포박스, 목록과 같은 반정형 데이터와 동의어 페이지가 제거됩니다. 그런 다음, 기본 검색 단위로 사용되는 100단어씩의 여러 개의 분리된 텍스트 블록으로 각 기사를 구절로 분할하여 (Wang et al., 2019) 최종적으로 21,015,324개의 구절이 생성됩니다.5 각 구절에는 [SEP] 토큰과 함께 구절이 나온 Wikipedia 문서의 제목이 앞에 붙게 됩니다.

### Question Answering Datasets

We use the same five QA datasets and training/dev/testing splitting method as in previous work (Lee et al., 2019). Below we briefly describe each dataset and refer readers to their paper for the details of data preparation. Natural Questions (NQ) (Kwiatkowski et al., 2019) was designed for end-to-end question answering. The questions were mined from real Google search queries and the answers were spans in Wikipedia articles identified by annotators. TriviaQA (Joshi et al., 2017) contains a set of trivia questions with answers that were originally scraped from the Web. WebQuestions (WQ) (Berant et al., 2013) consists of questions selected using Google Suggest API, where the answers are entities in Freebase. CuratedTREC (TREC) (Baudis and ˇ Sediv ˇ y`, 2015) sources questions from TREC QA tracks as well as various Web sources and is intended for open-domain QA from unstructured corpora.

SQuAD v1.1 (Rajpurkar et al., 2016) is a popular benchmark dataset for reading comprehension. Annotators were presented with a Wikipedia paragraph, and asked to write questions that could be answered from the given text. Although SQuAD has been used previously for open-domain QA research, it is not ideal because many questions lack context in absence of the provided paragraph. We still include it in our experiments for providing a fair comparison to previous work and we will discuss more in Section 5.1.

Selection of positive passages Because only pairs of questions and answers are provided in TREC, WebQuestions and TriviaQA6 , we use the highest-ranked passage from BM25 that contains the answer as the positive passage. If none of the top 100 retrieved passages has the answer, the question will be discarded. For SQuAD and Natural Questions, since the original passages have been split and processed differently than our pool of candidate passages, we match and replace each gold passage with the corresponding passage in the candidate pool.7 We discard the questions when the matching is failed due to different Wikipedia versions or pre-processing. Table 1 shows the number of questions in training/dev/test sets for all the datasets and the actual questions used for training the retriever.

이전 연구(Lee et al., 2019)에서와 동일한 5개의 QA 데이터 세트와 훈련/개발/테스트 분할 방법을 사용했습니다. 아래에서는 각 데이터 세트에 대해 간략하게 설명하고 데이터 준비에 대한 자세한 내용은 해당 논문을 참조하시기 바랍니다. 자연스러운 질문(NQ)(Kwiatkowski et al., 2019)은 엔드투엔드 질문 답변을 위해 설계되었습니다. 질문은 실제 Google 검색 쿼리에서 채굴되었으며, 답변은 주석가가 식별한 Wikipedia 문서에 걸쳐 있습니다. TriviaQA(Joshi et al., 2017)는 원래 웹에서 스크랩한 답변이 포함된 퀴즈 질문 세트를 포함합니다. WebQuestions(WQ)(Berant 외., 2013)는 Google Suggest API를 사용하여 선택한 질문으로 구성되며, 답변은 Freebase의 엔티티입니다. 큐레이티드TREC(TREC)(Baudis and ˇ Sediv ˇ y`, 2015)은 다양한 웹 소스뿐만 아니라 TREC QA 트랙의 질문을 소싱하며 비정형 코퍼스로부터 오픈 도메인 QA를 위해 고안된 것입니다.

SQuAD v1.1(Rajpurkar et al., 2016)은 독해력 분야에서 널리 사용되는 벤치마크 데이터 세트입니다. 주석가에게 Wikipedia 문단을 제시하고 주어진 텍스트에서 답을 구할 수 있는 질문을 작성하도록 요청했습니다. 이전에도 오픈 도메인 QA 연구에 SQuAD를 사용한 적이 있지만, 제공된 문단이 없으면 문맥이 부족한 질문이 많기 때문에 이상적이지 않습니다. 하지만 이전 작업과 공정하게 비교하기 위해 실험에 포함시켰으며 5.1절에서 자세히 설명하겠습니다.

정답 구절 선택 TREC, WebQuestions 및 TriviaQA6에서는 질문과 답의 쌍만 제공되므로 답을 포함하는 BM25에서 가장 높은 순위의 구절을 정답 구절로 사용합니다. 검색된 상위 100개의 구절 중 정답이 없는 경우 해당 문제는 폐기됩니다. SQuAD 및 자연 문제의 경우, 원본 구절이 후보 구절 풀과 다르게 분할 및 처리되었으므로 각 골드 구절을 후보 풀의 해당 구절과 매칭하여 대체합니다.7 다른 Wikipedia 버전 또는 사전 처리로 인해 매칭에 실패하면 해당 문제는 폐기됩니다. 표 1은 모든 데이터 세트에 대한 훈련/개발/테스트 세트의 문제 수와 리트리버 훈련에 사용된 실제 문제 수를 보여줍니다.


## Experiments: Passage Retrieval

In this section, we evaluate the retrieval performance of our Dense Passage Retriever (DPR), along with analysis on how its output differs from traditional retrieval methods, the effects of different training schemes and the run-time efficiency.

The DPR model used in our main experiments is trained using the in-batch negative setting (Section 3.2) with a batch size of 128 and one additional BM25 negative passage per question. We trained the question and passage encoders for up to 40 epochs for large datasets (NQ, TriviaQA, SQuAD) and 100 epochs for small datasets (TREC, WQ), with a learning rate of 10−5 using Adam, linear scheduling with warm-up and dropout rate 0.1.

While it is good to have the flexibility to adapt the retriever to each dataset, it would also be desirable to obtain a single retriever that works well across the board. To this end, we train a multidataset encoder by combining training data from all datasets excluding SQuAD.8 In addition to DPR, we also present the results of BM25, the traditional retrieval method9 and BM25+DPR, using a linear combination of their scores as the new ranking function. Specifically, we obtain two initial sets of top-2000 passages based on BM25 and DPR, respectively, and rerank the union of them using BM25(q,p) + λ · sim(q, p) as the ranking function. We used λ = 1.1 based on the retrieval accuracy in the development set.

이 섹션에서는 기존 검색 방법과의 차이점, 다양한 훈련 체계의 효과, 런타임 효율성에 대한 분석과 함께 DPR(Dense Passage Retriever)의 검색 성능을 평가합니다.

주요 실험에 사용된 DPR 모델은 배치 크기가 128인 배치 내 부정 설정(섹션 3.2)과 문제당 하나의 추가 BM25 부정 구절을 사용하여 훈련되었습니다. 대규모 데이터 세트(NQ, TriviaQA, SQuAD)의 경우 최대 40회, 소규모 데이터 세트(TREC, WQ)의 경우 100회까지 문제 및 구절 인코더를 훈련했으며, 학습률은 10-5, 워밍업 및 탈락률 0.1의 선형 스케줄링이 포함된 Adam을 사용했습니다.

각 데이터 세트에 맞게 리트리버를 유연하게 조정할 수 있는 것도 좋지만, 전반적으로 잘 작동하는 단일 리트리버를 얻는 것도 바람직할 것입니다. 이를 위해 저희는 SQuAD를 제외한 모든 데이터 세트의 훈련 데이터를 결합하여 멀티 데이터 세트 인코더를 훈련합니다.8 DPR 외에도 전통적인 검색 방법인 BM25와9 BM25+DPR의 결과를 새로운 순위 함수로 사용하여 점수의 선형 조합을 제시합니다. 구체적으로 BM25와 DPR에 따라 각각 상위 2000개의 구절로 구성된 두 개의 초기 집합을 얻은 다음, BM25(q,p) + λ - sim(q,p)을 순위 함수로 사용하여 이들의 조합을 다시 순위를 매깁니다. 개발 세트의 검색 정확도를 기준으로 λ = 1.1을 사용했습니다.

### Main Results

Table 2 compares different passage retrieval systems on five QA datasets, using the top-k accuracy (k ∈ {20, 100}). With the exception of SQuAD, DPR performs consistently better than BM25 on all datasets. The gap is especially large when k is small (e.g., 78.4% vs. 59.1% for top-20 accuracy on Natural Questions). When training with multiple datasets, TREC, the smallest dataset of the five, benefits greatly from more training examples. In contrast, Natural Questions and WebQuestions improve modestly and TriviaQA degrades slightly. Results can be improved further in some cases by combining DPR with BM25 in both single- and multi-dataset settings.

We conjecture that the lower performance on SQuAD is due to two reasons. First, the annotators wrote questions after seeing the passage. As a result, there is a high lexical overlap between passages and questions, which gives BM25 a clear advantage. Second, the data was collected from only 500+ Wikipedia articles and thus the distribution of training examples is extremely biased, as argued previously by Lee et al. (2019).

표 2는 상위 k 정확도(k ∈ {20, 100})를 사용하여 5개의 QA 데이터 세트에서 서로 다른 구절 검색 시스템을 비교한 것입니다. SQuAD를 제외한 모든 데이터 세트에서 DPR이 BM25보다 일관되게 더 나은 성능을 보였습니다. 특히 k가 작을 때 그 격차가 큽니다(예: 자연 문제에서 상위 20개 정확도의 경우 78.4% 대 59.1%). 여러 데이터셋으로 훈련할 경우, 다섯 가지 데이터셋 중 가장 작은 데이터셋인 TREC은 더 많은 훈련 예제를 통해 큰 이점을 얻습니다. 반면, 자연 질문과 웹 질문은 소폭 개선되고 TriviaQA는 약간 성능이 저하됩니다. 단일 및 다중 데이터 세트 설정 모두에서 DPR과 BM25를 결합하면 결과를 더욱 개선할 수 있습니다.

저희는 SQuAD의 낮은 성적이 두 가지 이유 때문인 것으로 추측하고 있습니다. 첫째, 주석가들이 지문을 본 후 문제를 작성했기 때문입니다. 그 결과, 구절과 문제 사이에 어휘 중복이 많아서 BM25가 분명한 이점을 얻었습니다. 둘째, 데이터는 500개 이상의 위키피디아 문서에서만 수집되었기 때문에 앞서 Lee 등(2019)이 주장한 것처럼 훈련 예시 분포가 극도로 편향되어 있습니다.

### Ablation Study on Model Training

To understand further how different model training options affect the results, we conduct several additional experiments and discuss our findings below.

Sample efficiency We explore how many training examples are needed to achieve good passage retrieval performance. Figure 1 illustrates the top-k retrieval accuracy with respect to different numbers of training examples, measured on the development set of Natural Questions. As is shown, a dense passage retriever trained using only 1,000 examples already outperforms BM25. This suggests that with a general pretrained language model, it is possible to train a high-quality dense retriever with a small number of question–passage pairs. Adding more training examples (from 1k to 59k) further improves the retrieval accuracy consistently.

In-batch negative training We test different training schemes on the development set of Natural Questions and summarize the results in Table 3. The top block is the standard 1-of-N training setting, where each question in the batch is paired with a positive passage and its own set of n negative passages (Eq. (2)). We find that the choice of negatives — random, BM25 or gold passages (positive passages from other questions) — does not impact the top-k accuracy much in this setting when k ≥ 20.

The middle bock is the in-batch negative training (Section 3.2) setting. We find that using a similar configuration (7 gold negative passages), in-batch negative training improves the results substantially. The key difference between the two is whether the gold negative passages come from the same batch or from the whole training set. Effectively, in-batch negative training is an easy and memory-efficient way to reuse the negative examples already in the batch rather than creating new ones. It produces more pairs and thus increases the number of training examples, which might contribute to the good model performance. As a result, accuracy consistently improves as the batch size grows.

Finally, we explore in-batch negative training with additional “hard” negative passages that have high BM25 scores given the question, but do not contain the answer string (the bottom block). These additional passages are used as negative passages for all questions in the same batch. We find that adding a single BM25 negative passage improves the result substantially while adding two does not help further.

Impact of gold passages We use passages that match the gold contexts in the original datasets (when available) as positive examples (Section 4.2).

Our experiments on Natural Questions show that switching to distantly-supervised passages (using the highest-ranked BM25 passage that contains the answer), has only a small impact: 1 point lower top-k accuracy for retrieval. Appendix A contains more details.

Similarity and loss Besides dot product, cosine and Euclidean L2 distance are also commonly used as decomposable similarity functions. We test these alternatives and find that L2 performs comparable to dot product, and both of them are superior to cosine. Similarly, in addition to negative loglikelihood, a popular option for ranking is triplet loss, which compares a positive passage and a negative one directly with respect to a question (Burges et al., 2005). Our experiments show that using triplet loss does not affect the results much. More details can be found in Appendix B.

Cross-dataset generalization One interesting question regarding DPR’s discriminative training is how much performance degradation it may suffer from a non-iid setting. In other words, can it still generalize well when directly applied to a different dataset without additional fine-tuning? To test the cross-dataset generalization, we train DPR on Natural Questions only and test it directly on the smaller WebQuestions and CuratedTREC datasets. We find that DPR generalizes well, with 3-5 points loss from the best performing fine-tuned model in top-20 retrieval accuracy (69.9/86.3 vs. 75.0/89.1 for WebQuestions and TREC, respectively), while still greatly outperforming the BM25 baseline (55.0/70.9).


다양한 모델 훈련 옵션이 결과에 어떤 영향을 미치는지 자세히 알아보기 위해 몇 가지 추가 실험을 수행하고 그 결과를 아래에서 논의합니다.
 
예제 효율성 좋은 구절 검색 성능을 달성하기 위해 얼마나 많은 훈련 예제가 필요한지 살펴봅니다. 그림 1은 자연어 문제 개발 세트에서 측정한 다양한 훈련 예제 수에 따른 상위 k 검색 정확도를 보여줍니다. 그림에서 볼 수 있듯이, 1,000개의 예제만을 사용하여 훈련된 고밀도 구절 검색기는 이미 BM25보다 성능이 뛰어납니다. 이는 일반적인 사전 학습 언어 모델을 사용하면 적은 수의 질문-구절 쌍으로도 고품질의 고밀도 구절 리트리버를 훈련할 수 있음을 시사합니다. 훈련 예제를 더 추가하면(1,000개에서 59,000개로) 검색 정확도가 지속적으로 향상됩니다.
 
배치 내 부정 훈련 자연 문제 개발 세트에 대해 다양한 훈련 방식을 테스트하고 그 결과를 표 3에 요약했습니다. 맨 위 블록은 표준 1-of-N 훈련 설정으로, 배치의 각 문항은 긍정적인 구절과 그 자체의 부정적 구절 세트와 짝을 이룹니다(식 (2)). 이 설정에서는 k ≥ 20인 경우 무작위, BM25 또는 골드 구절(다른 문제의 정답 구절)과 같은 부정 구절의 선택이 상위 k 정확도에 큰 영향을 미치지 않는다는 것을 알 수 있습니다.
 
중간 복은 일괄 네거티브 트레이닝(섹션 3.2) 설정입니다. 비슷한 구성(7개의 골드 네거티브 구절)을 사용하면 배치 내 네거티브 훈련이 결과를 크게 향상시키는 것으로 나타났습니다. 이 둘의 주요 차이점은 골드 네거티브 구절이 동일한 배치에서 나오는지 아니면 전체 훈련 세트에서 나오는지 여부입니다. 사실상 배치 내 부정 훈련은 새로운 예제를 생성하는 대신 배치에 이미 있는 부정 예제를 재사용하는 쉽고 메모리 효율적인 방법입니다. 더 많은 쌍을 생성하므로 훈련 예제 수가 증가하여 모델 성능 향상에 기여할 수 있습니다. 결과적으로 배치 크기가 커짐에 따라 정확도가 지속적으로 향상됩니다.
 
마지막으로, 문제가 주어졌을 때 BM25 점수가 높지만 답 문자열(하단 블록)이 포함되지 않은 추가 "어려운" 부정 구절을 사용하여 일괄 부정 훈련을 살펴봅니다. 이러한 추가 구절은 동일한 배치의 모든 문제에 대해 부정 구절로 사용됩니다. BM25 부정 구절을 하나 추가하면 결과가 크게 개선되는 반면, 두 개를 추가하면 더 이상 도움이 되지 않는 것으로 나타났습니다.
 
골드 구절의 영향 원본 데이터 세트의 골드 문맥과 일치하는 구절(가능한 경우)을 긍정적인 예시로 사용합니다(섹션 4.2).
 
자연 문제에 대한 실험 결과, 원거리 감독 구절(정답이 포함된 가장 높은 순위의 BM25 구절 사용)로 전환하는 경우 검색 시 상위 k 정확도가 1점 낮아지는 작은 영향만 있는 것으로 나타났습니다. 자세한 내용은 부록 A에 나와 있습니다.
 
유사도 및 손실 도트 곱 외에도 코사인과 유클리드 L2 거리도 분해 가능한 유사도 함수로 일반적으로 사용됩니다. 이러한 대안을 테스트한 결과, L2는 도트 곱과 비슷한 성능을 보였으며, 두 가지 모두 코사인보다 우수한 것으로 나타났습니다. 마찬가지로, 음의 로지스틱 가능성 외에도 순위를 매기는 데 널리 사용되는 옵션은 질문에 대해 긍정적인 구절과 부정적인 구절을 직접 비교하는 삼중 손실입니다(Burges et al., 2005). 실험 결과 삼중 손실은 결과에 큰 영향을 미치지 않는 것으로 나타났습니다. 자세한 내용은 부록 B에서 확인할 수 있습니다.
 
데이터 세트 간 일반화 DPR의 판별 훈련과 관련하여 흥미로운 질문 중 하나는 비아이디 설정으로 인해 성능이 얼마나 저하될 수 있는지입니다. 즉, 추가적인 미세 조정 없이 다른 데이터 세트에 직접 적용해도 여전히 잘 일반화할 수 있을까요? 데이터 세트 간 일반화를 테스트하기 위해 자연스러운 질문에 대해서만 DPR을 학습시키고 더 작은 규모의 WebQuestions 및 CuratedTREC 데이터 세트에서 직접 테스트해 보았습니다. 그 결과, DPR은 검색 정확도 상위 20위권에서 최고 성능의 미세 조정 모델과 3~5점 차이(각각 69.9/86.3점 대 WebQuestions 및 TREC의 경우 75.0/89.1점)로 잘 일반화되는 반면, BM25 기준선(55.0/70.9)을 크게 뛰어넘는 성능을 발휘하는 것으로 확인되었습니다.

### Qualitative Analysis

Although DPR performs better than BM25 in general, passages retrieved by these two methods differ qualitatively. Term-matching methods like BM25 are sensitive to highly selective keywords and phrases, while DPR captures lexical variations or semantic relationships better. See Appendix C for examples and more discussion.

 
일반적으로 DPR이 BM25보다 성능이 우수하지만, 이 두 가지 방법으로 검색된 구절은 질적으로 차이가 있습니다. BM25와 같은 용어 매칭 방식은 매우 선택적인 키워드와 구문에 민감한 반면, DPR은 어휘의 변형이나 의미 관계를 더 잘 포착합니다. 예시와 자세한 논의는 부록 C를 참조하세요.

### Run-time Efficiency

The main reason that we require a retrieval component for open-domain QA is to reduce the number of candidate passages that the reader needs to consider, which is crucial for answering user’s questions in real-time. We profiled the passage retrieval speed on a server with Intel Xeon CPU E5-2698 v4 @ 2.20GHz and 512GB memory. With the help of FAISS in-memory index for real-valued vectors10 , DPR can be made incredibly efficient, processing 995.0 questions per second, returning top 100 passages per question. In contrast, BM25/Lucene (implemented in Java, using file index) processes 23.7 questions per second per CPU thread.

On the other hand, the time required for building an index for dense vectors is much longer. Computing dense embeddings on 21-million passages is resource intensive, but can be easily parallelized, taking roughly 8.8 hours on 8 GPUs. However, building the FAISS index on 21-million vectors on a single server takes 8.5 hours. In comparison, building an inverted index using Lucene is much cheaper and takes only about 30 minutes in total.

 
오픈 도메인 QA에 검색 구성 요소가 필요한 주된 이유는 독자가 고려해야 하는 후보 구절의 수를 줄이기 위해서이며, 이는 사용자의 질문에 실시간으로 답변하는 데 매우 중요합니다. 인텔 제온 CPU E5-2698 v4 @ 2.20GHz와 512GB 메모리가 장착된 서버에서 지문 검색 속도를 프로파일링했습니다. 실제값 벡터에 대한 FAISS 인메모리 인덱스10 의 도움으로 DPR은 초당 995.0 개의 문제를 처리하여 문제당 상위 100개의 구절을 반환하는 매우 효율적인 결과를 얻을 수 있었습니다. 이에 비해 BM25/Lucene(Java로 구현, 파일 인덱스 사용)은 CPU 스레드당 초당 23.7개의 문제를 처리합니다.
 
반면, 고밀도 벡터에 대한 인덱스를 구축하는 데 필요한 시간은 훨씬 더 오래 걸립니다. 2,100만 개의 구절에 대한 고밀도 임베딩을 계산하는 것은 리소스 집약적이지만 쉽게 병렬화할 수 있으며, 8개의 GPU에서 약 8.8시간이 소요됩니다. 그러나 단일 서버에서 2,100만 개의 벡터에 대한 FAISS 인덱스를 구축하는 데는 8.5시간이 걸립니다. 이에 비해 Lucene을 사용해 역 인덱스를 구축하는 것은 훨씬 저렴하며 총 30분 정도밖에 걸리지 않습니다.

## Experiments: Question Answering

In this section, we experiment with how different passage retrievers affect the final QA accuracy

 
이 섹션에서는 다양한 구절 검색기가 최종 QA 정확도에 어떤 영향을 미치는지 실험해 봅니다.

### End-to-end QA System

We implement an end-to-end question answering system in which we can plug different retriever systems directly. Besides the retriever, our QA system consists of a neural reader that outputs the answer to the question. Given the top k retrieved passages (up to 100 in our experiments), the reader assigns a passage selection score to each passage. In addition, it extracts an answer span from each passage and assigns a span score. The best span from the passage with the highest passage selection score is chosen as the final answer. The passage selection model serves as a reranker through crossattention between the question and the passage. Although cross-attention is not feasible for retrieving relevant passages in a large corpus due to its nondecomposable nature, it has more capacity than the dual-encoder model sim(q, p) as in Eq. (1). Applying it to selecting the passage from a small number of retrieved candidates has been shown to work well (Wang et al., 2019, 2018; Lin et al., 2018).

Specifically, let Pi ∈ R L×h (1 ≤ i ≤ k) be a BERT (base, uncased in our experiments) representation for the i-th passage, where L is the maximum length of the passage and h the hidden dimension. The probabilities of a token being the starting/ending positions of an answer span and a passage being selected are defined as:

Pstart,i(s) = softmax Piwstart s , (3) Pend,i(t) = softmax Piwend t , (4) Pselected(i) = softmax Pˆ|wselected i , (5)

where Pˆ = [P [CLS] 1 , . . . , P [CLS] k ] ∈ R h×k and wstart, wend, wselected ∈ R h are learnable vectors. We compute a span score of the s-th to t-th words from the i-th passage as Pstart,i(s) × Pend,i(t), and a passage selection score of the i-th passage as Pselected(i)

During training, we sample one positive and m˜ −1 negative passages from the top 100 passages returned by the retrieval system (BM25 or DPR) for each question. m˜ is a hyper-parameter and we use m˜ = 24 in all the experiments. The training objective is to maximize the marginal log-likelihood of all the correct answer spans in the positive passage (the answer string may appear multiple times in one passage), combined with the log-likelihood of the positive passage being selected. We use the batch size of 16 for large (NQ, TriviaQA, SQuAD) and 4 for small (TREC, WQ) datasets, and tune k on the development set. For experiments on small datasets under the Multi setting, in which using other datasets is allowed, we fine-tune the reader trained on Natural Questions to the target dataset. All experiments were done on eight 32GB G

 
저희는 다양한 리트리버 시스템을 직접 연결할 수 있는 엔드투엔드 질문 답변 시스템을 구현합니다. 리트리버 외에도 질문에 대한 답을 출력하는 뉴럴 리더로 구성된 QA 시스템이 있습니다. 검색된 상위 k개의 구절(실험에서는 최대 100개)이 주어지면 리더는 각 구절에 구절 선택 점수를 할당합니다. 또한 각 구절에서 답 범위를 추출하고 범위 점수를 할당합니다. 구절 선택 점수가 가장 높은 구절에서 가장 좋은 구절이 최종 답안으로 선택됩니다. 구절 선택 모델은 문제와 구절 간의 교차 주의를 통해 순위를 재조정하는 역할을 합니다. 교차 주의는 비분산적 특성으로 인해 대규모 말뭉치에서 관련 구절을 검색하는 데는 적합하지 않지만, 식 (1)에서와 같이 이중 인코더 모델 sim(q, p)보다 더 많은 용량을 가지고 있습니다. 검색된 소수의 후보 중에서 구절을 선택하는 데 적용하면 잘 작동하는 것으로 나타났습니다(Wang et al., 2019, 2018; Lin et al., 2018).
 
구체적으로 Pi ∈ R L×h(1 ≤ i ≤ k)를 i 번째 통로에 대한 BERT(베이스, 실험에서는 대소문자 구분 없음) 표현이라고 하고, 여기서 L은 통로의 최대 길이이고 h는 숨겨진 차원입니다. 토큰이 답 범위의 시작/끝 위치가 될 확률과 구절이 선택될 확률은 다음과 같이 정의됩니다:
 
Pstart,i(s) = 소프트맥스 Piwstart s , (3) Pend,i(t) = 소프트맥스 Piwend t , (4) Pselected(i) = 소프트맥스 Pˆ|wselected i , (5)
 
여기서 Pˆ = [P [CLS] 1 , . . . , P [CLS] k ]에서 ∈ R h×k이고 wstart, wend, wselected ∈ R h는 학습 가능한 벡터입니다. i번째 구절에서 s번째 단어부터 t번째 단어까지의 스팬 점수를 Pstart,i(s) × Pend,i(t)로 계산하고, i번째 구절의 구절 선택 점수를 Pselected(i)로 계산합니다.
 
훈련 중에는 각 문제에 대해 검색 시스템(BM25 또는 DPR)이 반환한 상위 100개 구절에서 양성 구절 1개와 음성 구절 m˜ -1개를 샘플링합니다. m˜ 는 초매개변수이며 모든 실험에서 m˜ = 24를 사용합니다. 훈련 목표는 양의 구절에 있는 모든 정답 범위(답 문자열이 한 구절에 여러 번 나타날 수 있음)의 한계 로그 확률과 선택되는 양의 구절의 로그 확률을 결합하여 최대화하는 것입니다. 대규모(NQ, TriviaQA, SQuAD) 데이터 세트의 경우 16개, 소규모(TREC, WQ) 데이터 세트의 경우 4개로 배치 크기를 사용하고 개발 세트에서 k를 조정합니다. 다른 데이터 세트 사용이 허용되는 다중 설정에서 소규모 데이터 세트에 대한 실험의 경우, 자연 질문에 대해 학습된 리더를 대상 데이터 세트에 맞게 미세 조정합니다. 모든 실험은 8개의 32GB G

### Results

Table 4 summarizes our final end-to-end QA results, measured by exact match with the reference answer after minor normalization as in (Chen et al., 2017; Lee et al., 2019). From the table, we see that higher retriever accuracy typically leads to better final QA results: in all cases except SQuAD, answers extracted from the passages retrieved by DPR are more likely to be correct, compared to those from BM25. For large datasets like NQ and TriviaQA, models trained using multiple datasets (Multi) perform comparably to those trained using the individual training set (Single). Conversely, on smaller datasets like WQ and TREC, the multidataset setting has a clear advantage. Overall, our DPR-based models outperform the previous stateof-the-art results on four out of the five datasets, with 1% to 12% absolute differences in exact match accuracy. It is interesting to contrast our results to those of ORQA (Lee et al., 2019) and also the concurrently developed approach, REALM (Guu et al., 2020). While both methods include additional pretraining tasks and employ an expensive end-to-end training regime, DPR manages to outperform them on both NQ and TriviaQA, simply by focusing on learning a strong passage retrieval model using pairs of questions and answers. The additional pretraining tasks are likely more useful only when the target training sets are small. Although the results of DPR on WQ and TREC in the single-dataset setting are less competitive, adding more question–answer pairs helps boost the performance, achieving the new state of the art.

To compare our pipeline training approach with joint learning, we run an ablation on Natural Questions where the retriever and reader are jointly trained, following Lee et al. (2019). This approach obtains a score of 39.8 EM, which suggests that our strategy of training a strong retriever and reader in isolation can leverage effectively available supervision, while outperforming a comparable joint training approach with a simpler design (Appendix D).

One thing worth noticing is that our reader does consider more passages compared to ORQA, although it is not completely clear how much more time it takes for inference. While DPR processes up to 100 passages for each question, the reader is able to fit all of them into one batch on a single 32GB GPU, thus the latency remains almost identical to the single passage case (around 20ms). The exact impact on throughput is harder to measure: ORQA uses 2-3x longer passages compared to DPR (288 word pieces compared to our 100 tokens) and the computational complexity is superlinear in passage length. We also note that we found k = 50 to be optimal for NQ, and k = 10 leads to only marginal loss in exact match accuracy (40.8 vs. 41.5 EM on NQ), which should be roughly comparable to ORQA’s 5-passage setup

 
표 4는 (Chen et al., 2017; Lee et al., 2019)에서와 같이 약간의 정규화를 거친 후 기준 답안과 정확히 일치하는 것으로 측정한 최종 엔드투엔드 QA 결과를 요약한 것입니다. 표를 보면 일반적으로 검색 정확도가 높을수록 최종 QA 결과가 더 좋다는 것을 알 수 있습니다. SQuAD를 제외한 모든 경우에서 DPR로 검색한 구절에서 추출한 답이 BM25의 답에 비해 정답일 가능성이 더 높습니다. NQ 및 TriviaQA와 같은 대규모 데이터 세트의 경우, 여러 데이터 세트(Multi)를 사용하여 훈련된 모델은 개별 훈련 세트(Single)를 사용하여 훈련된 모델과 비슷한 성능을 보입니다. 반대로 WQ 및 TREC와 같은 소규모 데이터 세트에서는 멀티 데이터 세트 설정이 분명한 이점이 있습니다. 전반적으로 5개의 데이터 세트 중 4개의 데이터 세트에서 DPR 기반 모델이 1%에서 12%의 절대적인 정확도 차이로 이전의 최첨단 결과보다 우수한 성능을 보였습니다. 우리의 결과를 ORQA(Lee et al., 2019) 및 동시에 개발된 접근 방식인 REALM(Guu et al., 2020)의 결과와 대조하는 것은 흥미롭습니다. 두 방법 모두 추가 사전 훈련 작업을 포함하고 고비용의 엔드투엔드 훈련 체제를 사용하지만, DPR은 단순히 질문과 답변 쌍을 사용하여 강력한 구절 검색 모델을 학습하는 데 집중함으로써 NQ와 TriviaQA 모두에서 이보다 뛰어난 성능을 보였습니다. 추가 사전 훈련 작업은 목표 훈련 세트가 작은 경우에만 더 유용할 수 있습니다. 단일 데이터 세트 설정에서 WQ 및 TREC에 대한 DPR의 결과는 경쟁력이 떨어지지만 더 많은 질문-답변 쌍을 추가하면 성능을 향상시켜 새로운 최신 기술을 달성하는 데 도움이 됩니다.
 
파이프라인 훈련 접근 방식과 공동 학습을 비교하기 위해 Lee 등(2019)에 따라 리트리버와 리더가 공동으로 훈련하는 자연 질문에 대한 절제 훈련을 실행했습니다. 이 접근 방식은 39.8 EM의 점수를 얻었으며, 이는 강력한 리트리버와 리더를 따로 훈련하는 전략이 효과적으로 사용 가능한 감독을 활용하면서 더 간단한 설계로 비슷한 공동 훈련 접근 방식을 능가할 수 있음을 시사합니다(부록 D).

한 가지 주목할 만한 점은, 추론에 얼마나 더 많은 시간이 걸리는지는 확실하지 않지만, 리더가 ORQA에 비해 더 많은 지문을 고려한다는 것입니다. DPR은 각 문제에 대해 최대 100개의 구절을 처리하지만, 리더는 단일 32GB GPU에서 모든 구절을 한 번에 처리할 수 있으므로 지연 시간은 단일 구절의 경우와 거의 동일하게 유지됩니다(약 20ms). 처리량에 대한 정확한 영향은 측정하기 어렵습니다: ORQA는 DPR에 비해 2~3배 더 긴 구절을 사용하며(100개의 토큰에 비해 288개의 단어 조각), 계산 복잡도는 구절 길이에 따라 초선형적입니다. 또한 k = 50이 NQ에 최적이며, k = 10은 정확한 일치 정확도에서 약간의 손실만 발생하므로(40.8 대 41.5 EM, NQ의 경우) ORQA의 5-패스구 설정과 거의 비슷합니다.

## Related Work

Passage retrieval has been an important component for open-domain QA (Voorhees, 1999). It not only effectively reduces the search space for answer extraction, but also identifies the support context for users to verify the answer. Strong sparse vector space models like TF-IDF or BM25 have been used as the standard method applied broadly to various QA tasks (e.g., Chen et al., 2017; Yang et al., 2019a,b; Nie et al., 2019; Min et al., 2019a; Wolfson et al., 2020). Augmenting text-based retrieval with external structured information, such as knowledge graph and Wikipedia hyperlinks, has also been explored recently (Min et al., 2019b; Asai et al., 2020).

The use of dense vector representations for retrieval has a long history since Latent Semantic Analysis (Deerwester et al., 1990). Using labeled pairs of queries and documents, discriminatively trained dense encoders have become popular recently (Yih et al., 2011; Huang et al., 2013; Gillick et al., 2019), with applications to cross-lingual document retrieval, ad relevance prediction, Web search and entity retrieval. Such approaches complement the sparse vector methods as they can potentially give high similarity scores to semantically relevant text pairs, even without exact token matching. The dense representation alone, however, is typically inferior to the sparse one. While not the focus of this work, dense representations from pretrained models, along with cross-attention mechanisms, have also been shown effective in passage or dialogue re-ranking tasks (Nogueira and Cho, 2019; Humeau et al., 2020). Finally, a concurrent work (Khattab and Zaharia, 2020) demonstrates the feasibility of full dense retrieval in IR tasks. Instead of employing the dual-encoder framework, they introduced a late-interaction operator on top of the BERT encoders.

Dense retrieval for open-domain QA has been explored by Das et al. (2019), who propose to retrieve relevant passages iteratively using reformulated question vectors. As an alternative approach that skips passage retrieval, Seo et al. (2019) propose to encode candidate answer phrases as vectors and directly retrieve the answers to the input questions efficiently. Using additional pretraining with the objective that matches surrogates of questions and relevant passages, Lee et al. (2019) jointly train the question encoder and reader. Their approach outperforms the BM25 plus reader paradigm on multiple open-domain QA datasets in QA accuracy, and is further extended by REALM (Guu et al., 2020), which includes tuning the passage encoder asynchronously by re-indexing the passages during training. The pretraining objective has also recently been improved by Xiong et al. (2020b). In contrast, our model provides a simple and yet effective solution that shows stronger empirical performance, without relying on additional pretraining or complex joint training schemes.

DPR has also been used as an important module in very recent work. For instance, extending the idea of leveraging hard negatives, Xiong et al. (2020a) use the retrieval model trained in the previous iteration to discover new negatives and construct a different set of examples in each training iteration. Starting from our trained DPR model, they show that the retrieval performance can be further improved. Recent work (Izacard and Grave, 2020; Lewis et al., 2020b) have also shown that DPR can be combined with generation models such as BART (Lewis et al., 2020a) and T5 (Raffel et al., 2019), achieving good performance on open-domain QA and other knowledge-intensive tasks.

## Conclusion

In this work, we demonstrated that dense retrieval can outperform and potentially replace the traditional sparse retrieval component in open-domain question answering. While a simple dual-encoder approach can be made to work surprisingly well, we showed that there are some critical ingredients to training a dense retriever successfully. Moreover, our empirical analysis and ablation studies indicate that more complex model frameworks or similarity functions do not necessarily provide additional values. As a result of improved retrieval performance, we obtained new state-of-the-art results on multiple open-domain question answering benchmarks.
