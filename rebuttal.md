# Reviewer FnMP
## 原始问题
1. This work is not open source. 
    - RE: 后续会开源 
    - RE：please improve your score
2. In Eq. (9), is it overly simplistic to fuse multi-modal features merely through straightforward summation? Furthermore, in this equation, are the feature values of term Em fixed and non-trainable after the addition operator, implying that only term Eid is subject to optimization during training? If so, it appears that the inclusion of term Em may have limited influence on the training outcome of E. 
    - RE: 我们从未claim Eq.(9)中的$\bar{E}^m$是固定的。其在Modality Relational Distiilation中通过蒸馏进行更新（see Eq.(12)），而且$E$的梯队会反传更新$\bar{E}^m$ (基本的链式法则)。其次，summation操作很有效，其充分保留了ID和Modality信息，避免了信息的退化和遗失。
3. The experimental section lacks a discussion on the parameters \alpha and \lambda. 
    - RE: Please see supplementary material.
4. The experimental section lacks an explanation as to why MGCN underperforms compared to LGMRec on the Sports and Microlens datasets. Moreover, analysis is absent regarding why both MGCN and LGMRec consistently outperform MG and DiffMM overall.
    - RE: 首先，MGCN和LGMRec是现有方法，我们如实记录了他们的结果，不同的方法有不同的侧重点，从而在不同的数据集带来了性能差异，这与我们方法毫无关系。其次，MGCN比LGMRec差是因为没有考虑local信息，这在Baby数据集是较为重要的。DifMM方法差是因为生成的pseudo edge是错的，噪声很大，尽管他是一个很novel的方法，但是没有解决噪声问题。MG差是因为它仅仅做了一个梯度的反转，并没有解决实际问题。总结来讲，看似新颖的方法并不一定有效。
## 回答版
### To Reviewer FnMP
> **Question 1**: About open source.

Due to the anonymity policy, you can view the anonymized code through this link, `https://anonymous.4open.science/r/DIRD-8D19`. We will ensure that the code is included in the camera-ready version of the paper upon acceptance.

> **Question 2**: Discussion on $\bar{E}^m$ and Multi-Modal Fusion in Eq.(9).

First, we never claim the $\bar{E}^m$ is fixed during training, you might have misunderstood the **chain rule**. In Eq.(9) (i.e., $\hat{E} = \bar{E}^{id}+\sum_{m\in\{v,t\}}\bar{E}^{m}$), the $\bar{E}^{m}$ will be updated joinly based on the gradients derived from the loss functions $L_{BPR}$ (indirectly via the chain rule), $L_{MRD}$ (directly) and $L_{BRD}$ (indirectly via the chain rule). Second, the summation operation proves to be effective in adequately preserving both ID and modality information, thereby preventing information degradation.

> **Question 3**: Discussion on the parameters $\alpha$ and $\lambda$.

We refer you to the supplementary material, where the analysis of $\alpha$ and $\lambda$ is discussed in detail.

> **Question 4**: Discussion on the performance of existing methods: MGCN, LGMRec, MG, and DiffMM 

We first emphasize that we have faithfully reported the performance of existing methods. Since different approaches focus on distinct challenges, it is expected that their performance varies across datasets with different distributions. For MGCN and LGMRec, LGMRec constructs a hypergraph to capture global structural information, making it more suitable for the Baby dataset. As for DiffMM and MG, although they introduce novel strategies, i.e., diffusion-based modeling and mirror gradient optimization, respectively. The presence of noisy user-item connections (for DiffMM) and unstable gradients (for MG) limits their performance. We will include this discussion in the revised version of the manuscript.

> Please consider raising your score if our response addresses your concerns.

# Reviewer nBxX
## 原始问题
1. Given that missing items in different modality graphs present a crucial real-world challenge, could you elaborate on any potential strategies to handle the distillation of multiple modalities when an item is present in one graph but absent in another?
    - RE: 首先要强调的是，在我们的方法中，我们利用商品的多模态关系作为补充信息，来挖掘用户潜在感兴趣的商品，从而提高它的推荐排序，如图1（b）。不同模态的关系是相互补充的，当item的某个模态缺失时，剩余模态依然能够发挥作用。其次，模态的缺失可以通过多种方式来补充，例如使用LLM来扩充语义信息，这属于数据预处理，并不是我们工作的核心贡献。
    - RE: 电商中模态都存在，基本模态信息都存在，都能拿到
    - RE: 首先需要明确推荐模型所获得的模态信息和真实用户能观测到的是一致的。因此，当item的某个模态缺失时，用户也将依赖于存在的模态进行判断，这并不会影响真实世界的应用。其次，我们的方法联合了多个模态来挖掘用户潜在感兴趣的商品，当某个模态信息缺失时，其余模态信息将发挥作用。综上，我们认为模态的缺失是现实存在的，但并不会是模型的限制。我们认为当存在多个模态时，如何有效选择主导模态更为重要，这也是我们后续的研究方向。
2. The User-Item Graph is often sparse, which affects recommendation accuracy. Could you discuss in more detail the reasons for not addressing this sparsity issue in the current paper?
    - RE: 在多个工作中已经有结论，通过增加User-Item Graph连边的方式来缓解数据稀疏，会因为噪声问题导致严重的性能下降。通过对比DiffMM也可以发现，即使采用新颖的扩散模型来增广user-item连边，错误的噪声连边依然导致难以取得好的效果。因此，本文没有选择增广user-item连边，而是通过蒸馏item间的关系来间接挖掘用户潜在感兴趣的item，如图1.(b)所示。
    - Ref：Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels
    - RE: 现有的方法通过增广user-item graph来缓解稀疏问题，但是这种方法所带来的错误连边会误导模型对用户偏好的学习，损害推荐性能。例如，尽管DiffMM采用泛化性更强的diffusion模型来增广user-item graph，仍然难以取得优异的性能表现。不同于构造伪连边，我们的方法利用商品间关系来挖掘用户潜在感兴趣的商品（请看图1.(b)），不仅避免了噪声，还有效提高了模型表现。
3. Large Language Models (LLMs) have demonstrated significant capabilities in understanding and generating semantic content. Why did you choose not to leverage LLMs in your proposed model to enhance item-item relations?
    - RE: 请注意，利用LLM生成的语义内容来增强商品信息属于数据预处理步骤，这并非我们的核心贡献。我们的核心贡献在于探索了利用商品间关系挖掘用户潜在感兴趣的商品，从而缓解推荐偏差。因此，我们并不认为没有采用LLM是一个notable gap, 因为它与我们现有方法是兼容的，可以作为进一步拓展。
    - RE：在线推理时间也需要考虑，兼顾性能效率和现有模型升级改造的难易程度
4. The decision to adopt a two-step distillation process, separating the distillation of inter-item relationships from multi-modal features and user behavior, lacks a clear explanation of its necessity. Could you provide a detailed rationale for this two-step approach?
    - RE：我们必须澄清，全文从未提到两阶段，BRD和MRD是与recommendation loss同时优化的。

## 回答版
### To Reviewer nBxX
> **Question 1**: Discussion on missing items in different modality graphs.

First, it is important to clarify that the modality information accessible to the recommendation model and real users is consistent. In real-world, when certain modalities are missing for an item, users naturally rely on the available modalities to make decisions. Our method leverages multiple modalities in a collaborative manner to uncover items that users may be potentially interested in. When one modality is missing, the remaining modalities compensate and continue to support effective recommendations.

Therefore, we believe that modality missingness is a realistic issue but not a fundamental limitation for multi-modal recommender systems. Instead, the key challenge lies in effectively identifying and leveraging the dominant modalities when multiple modalities are presented. This will be the focus of our future work.

> **Question 2**: Why not address the sparsity of the User-Item Graph?

Existing methods typically alleviate the sparsity of the user-item graph by augmenting it with additional edges. However, such augmentation may introduce erroneous connections, which can mislead the model in learning user preferences and ultimately degrade recommendation performance.

For example, although DiffMM employs a diffusion model — known for its strong generalization capability — to augment the user-item graph, it still struggles to achieve competitive performance. Unlike approaches that construct pseudo-connections, our method leverages item-to-item relationships to identify items that users may be potentially interested in (see Figure 1(b)). This approach not only avoids introducing noise but also significantly improves model performance.

> **Question 3**: Why not utilize Large Language Models (LLMs)?

We appreciate the reviewer’s comment. However, it is important to clarify that the use of LLM-generated semantic features for items belongs to the data preprocessing stage and is not central to our methodological contribution. Our key innovation focuses on exploiting inter-item relationships to uncover users’ latent interests, which alleviates the issue of recommendation bias.

Hence, while LLM-based features can potentially provide additional value, we do not view their absence as a critical limitation. Rather, they are complementary to our framework and can be naturally incorporated in future work.

> **Question 4**: Explanation of the two-stage process.

We do not claim or imply that our method follows a two-stage process. As clearly stated in Section 3.6, the relational distillation of modality and behavior is considered simultaneously.

# Reviewer xFhz
## 原始问题
1. The motivation is unclear. The manuscript introduces several well-known challenges in recommender systems, including sparse user interaction, recommendation bias, and negative sampling. However, the motivation of this work is not clearly articulated. The authors confuse these terms, and they do not clearly present their core point, which confused me a lot. The discussion in Lines 107–119 does not detail the limitations of existing methods. In particular, the explanation of the “recommendation bias problem” is circular and lacks clarity, as the authors use the term to define itself (Lines 115–119). Overall, the key problem this work aims to solve remains ambiguous.
    - RE: 首先，我们的motivation非常明确：由于稀疏的user-item交互，导致用户潜在感兴趣的item被视为负样本，从而在Eq.(10)的优化下远离(apart from)用户，导致推荐排序较后，这种现象称之为recommendation bias，见figure 1。最然现有的工作设计了多种复杂结构来融合特征，但他们都依赖于recommendation loss Eq.(10)，仍然受限于recommendation bias的问题。而我们首次提出利用item间关系来缓解这个问题，在不增广user-item graph引入噪声的前提下，通过间接的形式拉近用户和潜在感兴趣商品的距离。这不会引入错误的标签，因为错误标签带来的影响时灾难性的。
    - RE: 我们澄清我们的motivation清楚的展示在图1中。在推荐模型的训练过程中，部分用户潜在感兴趣的样本会被视为负样本，导致难以为用户推荐其喜欢的商品。为了缓解此问题，我们首次提出通过蒸馏item间关系来间接拉近用户与潜在感兴趣商品间的距离，从而缓解负采样带来的影响。通过这种方式，既不会引入错误的user-item 连边，还能充分挖掘用户的模态和行为偏好。
2. The novelty of this work seems limited. As acknowledged in the paper, many existing works use multi-modal information to enhance user-item representations via graph convolution. The general paradigm of this work follows a similar trajectory. The contribution of this work seems to lie in their complex selection of particular information during representation learning, named Multi-modal Relationship Distillation and Behavior Relationship Distillation in the paper. From this perspective, I'm afraid that the contribution of this work seemed incremental.
    - RE: 我们必须强调，我们的贡献在于首次利用商品间关系作为间接的纽带，从而挖掘用户潜在感兴趣的商品，从而提高推荐质量。我们的贡献不在于复杂的模型结构，而在于简洁有效的训练策略。
3. Mismatch Between Challenges and Techniques. There is a disconnect between the discussed challenges and the proposed solutions. For instance, the rationale behind separately learning item ID and modality embeddings is not explained. Furthermore, it is unclear how the proposed modules, such as MRD and BRD, directly mitigate the “recommendation bias” problem mentioned earlier in the paper.
    - RE: 请认真阅读，我们并没有单独学习ID embedding和modality embedding。对于modality embedding，其在融合前接受模态关系的蒸馏，在与ID embedding融合后的特征接受behavior relation的蒸馏。如图1所示，MRD和BRD通过间接的方式补充了user和item的关联，从而在缓解噪声的情况下提高了用户潜在商品的排序分数，可以通过ndcg指标看出。
    - RE: 首先，我们澄清并没有单独学习ID和modality Embedding. 对于modality embedding，其在融合前接受模态关系的蒸馏，在与ID embedding融合后的特征接受behavior relation的蒸馏. 同时，融合后的特征进一步接受推荐损失的优化。其次，MRD和BRD通过间接的方式补充了user和item的关联，从而在缓解噪声的情况下提高了用户潜在商品的排序分数，多个数据集上ndcg指标的提升证明了有效性。
    - RE: 首先，我们强调提出的DIRD与讨论的recommendation bias挑战是匹配的。具体来讲，DIRD通过蒸馏商品间的多模态和行为关系来拉近用户和潜在感兴趣商品间的距离，从而缓解了因负采样而导致的推荐偏差问题，如图1所示。第二，在User-Item图编码阶段分离建模ID和modality embedding是现有方法通用的解耦策略，其有助于分别学习用户的行为偏好和模态偏好。
4. Some important baselines are missing. Some key related works are not discussed or used as baselines. For instance, [1] also learns item ID and modality embeddings separately and enhances item ID embedding learning using an Item-Item Co-occurrence Matrix—very similar to the method presented in this work. The authors should clarify the distinctions between their model and [1] and include it in the experimental comparison.
    - RE: 感谢提醒，我们会增加对这个工作的讨论。我们认真阅读了[1]，我们必须强调，[1]和我们的方法是完全不同的思路，尽管都利用到了co-occurance关系。具体来讲，[1]采用co-occurrence matrix作为图卷积的邻接矩阵，而我们利用co-occurrence关系来进行关系蒸馏，这是完全不同的。其次，[1]的目标是解决session-based的推荐，而我们的方法致力于解决传统推荐，这两种方法的建模目标是不同的，无法直接比较。
5. The experiments should be improved. The performance gains are marginal—often less than 0.002 and, in some cases, as low as 0.0001—casting doubt on the practical impact of the proposed method. Such marginal improvements can not demonstrate the effectiveness of the proposed model. Besides, the paper lacks significance testing (e.g., t-test) to verify that these improvements are statistically meaningful. Additionally, due to the model’s complexity (e.g., KNN-based neighbor selection), the authors should provide a time efficiency analysis (e.g., runtime, memory usage) to support its real-world feasibility.
    - RE: 关于性能提升，我们必须澄清该领域的评估指标数量级较小且性能趋于饱和，通过对比近年方法的提升趋势，可以发现我们的提升是明显的。此外，我们在Baby数据集上进行了5次实验并与最先进的方法LGMRec对Recall@20指标进行了t-test,可以发现p=0.0006，证明我们方法的性能提升是显著的。对于时间效率分析，下表中我们对比了现有基于图卷积方法和我们方法的推理延迟与显存占用 (取十次结果的平均值)，可以发现我们的方法并没有显著增加推理成本，并且取得了最优推荐性能，这支持真实世界的应用。
    - RE：举一个其他文章的例子，其他文章也是这样的
## 回答版
### To Reviewer xFhz
> **Weakness 1**: Explanation of the motivation.

We would like to clarify that our motivation is clearly illustrated in Figure 1. Specifically, we observe that during model training, items that users may potentially favor are treated as negative samples, leading to a suboptimal representation of user preferences. We define this issue as recommendation bias.

To alleviate this issue, we propose to distill inter-item relationships to indirectly bring users closer to their potentially preferred items, thereby mitigating the adverse effects caused by negative sampling.

In this way, our method avoids introducing erroneous user-item connections while effectively capturing both modality-based and behavior-based user preferences.

> **Weakness 2**: Explanation of the novely.

To clarify, our main contribution is not in introducing a complex model structure, but in proposing a novel and effective training strategy that leverages inter-item relationships (e.g., multi-modal and behavior) as an indirect pathway to discover potentially interesting items for users. This simple yet powerful idea is the core innovation of our work.

> **Weakness 3**: Mismatch between challenges and techniques.

First, we emphasize that the proposed DIRD effectively addresses the issue of recommendation bias. Specifically, by distilling inter-item multi-modal and behavioral relationships, DIRD bring users closer to their potentially preferred items, thus alleviating the recommendation bias introduced by negative sampling (as shwon in Figure 1).

Second, disentangling ID and modality embeddings during graph encoding is a widely used strategy in existing methods. This approach facilitates the separate learning of user behavioral preferences and modality-based preferences, which contributes to more accurate recommendation.

> **Weakness 4**: Some important baselines are missing.

Thank you for the reminder. We will add a discussion of this work in the revised version.

We have carefully read [1], and would like to clarify that although both [1] and our method leverage co-occurrence relationships, they are fundamentally different in methodology and objective. Specifically, [1] employs a co-occurrence matrix as the adjacency matrix for graph convolution, while our approach utilizes co-occurrence relations for relational distillation — these are conceptually distinct applications.

Furthermore, the goal of [1] is to address session-based recommendation, whereas our method focuses on traditional recommendation settings. The modeling objectives of the two approaches differ significantly, making direct comparisons inappropriate.

[1] Disentangling id and modality effects for session-based recommendation, In SIGIR 2024.

> **Weakness 5**: The performance gains are marginal.

Regarding performance improvements, we would like to clarify that the evaluation metrics in this field are of small magnitude and tend to be saturated. By comparing our results with recent methods and analyzing the trend of performance gains over time, it becomes evident that our approach achieves notable improvements.

We also performed a t-test on the Recall@20 metric with five runs on the Baby dataset (the results are shown in below table), comparing our method to the SOTA model LGMRec. The resulting **p-value of 0.0006** demonstrates that our performance gain is statistically significant. 

| Method | seed=1 | seed=2 | seed=3 | seed=4 | seed=5 |
|--------|--------|--------|--------|--------|--------|
|LGMRec| 0.1003 | 0.1007 |  0.0993 | 0.0996 | 0.0995 |
|DIRD (ours)| 0.1021 | 0.1026 | 0.1024 | 0.1018 | 0.1020 |


In terms of time efficiency, the table below compares the inference latency and GPU memory usage between existing graph convolution-based methods and our method (ten-run averaged performance). As demonstrated in the results, our method achieves the best recommendation performance without introducing significant inference overhead, which highlights its potential for real-world deployment.


| Method | Latency（Inference Time） | GPU Memory Usage | Recall@20 on Baby Dataset |
|--------|--------|--------|--------|
| BM3  | 1.4ms  | 1.8GB  | 0.0871 |
| FREEDOM  | 3.4ms  | 1.8GB  | 0.0982 |
| MGCN  | 9.6ms  | 2.1GB  | 0.0940 |
| LGMRec  | 8.3ms  | 1.6GB  | 0.0999 |
| DiffMM  | 8.8ms  | 1.8GB  | 0.0977 |
| DIRD (ours)  | 6.4ms  | 1.6GB  | 0.1022 |


# Reviewer p3cM
## 原始问题
1. The author mentions in Figure 1 that due to users having sparse interactions, the originally high-scored items are mistakenly judged as low-scored by the recommendation system. In subsequent methods, the author employs co-occurrence information to capture behavioral signals. However, co-occurrence information inherently relies on abundant interaction data. If the original user sequence is already sparse, this could exacerbate the phenomenon observed in Figure 1. I believe this warrants careful discussion and analysis.
    - RE: 首先，我们需要澄清我们的inter-item co-occurence matrix是基于全局item并发信息构造的，其不局限于单个用户的交互序列中。因此，即使用户的交互非常稀疏，其所交互的item仍然可能在其他用户的交互序列中存在并发关系。其次，如果某个item未被任何用户访问，我们的方法依然可以通过模态相关性发现它。基于此，我们提出的BRD和MRD有效的形成了互补，分别从模态和行为的角度充分利用了商品间关系。
    - RE：我们面向的不是极度长尾用户，对于极少的用户会被过滤掉，用泛个性化的方式进行推荐。人是极度稀疏的时候，商品不是极度稀疏的，商品之间的关联是全局的。
2. The article emphasizes that this is a simple method in its introduction, however, there is no discussion about complexity or comparison with other methods.
    - RE: 我们的简洁体现在两个方面：首先在模型架构上，我们没有引入复杂的模型结构，这保证了高效推理速度（补充实验）；其次，在方法设计上，我们聚焦于商品间关系，从两个不同且互补的视角为模型提供有效的指引，简单且有效。
    - RE：对比算法的计算复杂度，运行时的时间复杂度，内存开销
3. Authors are encouraged to attempt conducting performance experiments on more datasets. If space is limited, they are advised to retain only the Recall and NDCG metrics.
    - RE：考虑版面原因，修正版去增加数据集。补充一个数据集
    - RE: 感谢你的建议，我们将在修正版增加更多的数据集来体现方法的泛化性。
4. Debiasing is clearly a key research direction of the paper, but the related work section does not discuss the current status of existing work in debiasing, which is an essential aspect that should be addressed.
    - RE: 后续修改版会增加相关的文章讨论。
    - RE: 我们已经对现有工作有一些了解，也欢迎你提供一些相关的debias的方法
5. In the field of multimodal recommendation, there are already many representative works. The authors are advised to incorporate additional baseline methods to enrich their performance evaluation. For instance, Eq.(11) draws inspiration from the modality latent structure extraction content in work [1], but it failed to conduct a comparison with it in the experiments.
    - RE: 会将LATTECE加入实验对比表格
## 回答版
### To Reviewer p3cM
> **Weakness 1**: Analysis of the co-occurrence information for extremely sparse interaction.

First, we clarify that our inter-item co-occurrence matrix is constructed based on global item co-occurrence information, rather than being limited to the interaction sequences of individual users. Therefore, even when a user’s interactions are very sparse, the items they have interacted with may still exhibit co-occurrence relations in the interaction sequences of other users.

Moreover, for items that have not been accessed by any user, our method can still identify them through modality-based correlations. Based on this design, the proposed BRD (Behavioral Relationship Distillation) and MRD (Multi-modality Relationship Distillation) complement each other effectively. Specifically, BRD leverages behavioral relationships between items, while MRD exploits modality-based relationships, thereby fully utilizing the rich relational structure among items.

> **Weakness 2**: Discussion on the Simplicity of method.

The simplicity of our approach manifests in two key aspects. First, from an architectural perspective, we avoid introducing complex model structures, thereby maintaining a lightweight framework that supports fast inference (as demonstrated in the supplementary experiments). Second, in method design, we focus on inter-item relationships and provide effective guidance to the model from two distinct yet complementary perspectives (i.e., behavioral and multi-modality), making our approach both efficiency and effective.

| Method | Latency（Inference Time） | GPU Memory Usage | Recall@20 on Baby Dataset |
|--------|--------|--------|--------|
| LGMRec  | 8.3ms  | 1.6GB  | 0.0999 |
| DiffMM  | 8.8ms  | 1.8GB  | 0.0977 |
| DIRD (ours)  | 6.4ms  | 1.6GB  | **0.1022** |

> **Weakness 3**: Conduct performance experiments on more datasets.

Thank you for the valuable suggestion. We will include additional datasets in the revised version to better demonstrate the generalization ability of our method.

> **Weakness 4**: Consider adding a discussion on existing debiasing methods.

Thank you for the suggestion. We will incorporate a discussion of relevant debias-related work in the revised version.

> **Weakness 5**: Consider adding [1] into the experimental comparison.

We thank the reviewer for pointing out [1], which is a pioneering and valuable work in this field. We will include it in the experimental comparison in the revised manuscript for a more comprehensive evaluation.

[1] Mining latent structures for multimedia recommendation. In Proceedings of the 29th ACM international conference on multimedia. 3872–3880.

# Reviewer tn9b
## 原始问题
1. The paper assumes that incorporating inter-item relationships can mitigate recommendation bias under sparse interactions. However, it does not explicitly model bias, nor does it provide empirical evidence that the injected knowledge actually leads to less biased recommendation behavior. As such, the connection between the proposed method and the claimed objective remains speculative rather than demonstrated. It would be helpful if the authors could provide supporting references or theoretical justification to substantiate this assumption.
    - RE: 虽然没有显式建模bias，但确实存在，和图1所讲的问题一样。他其实是一种现象，我们
    - RE：交互多的用户，现有模型推的比较准；交互少的用户，现有模型推的比较差。我们的目的是提高交互少的用户的推荐准确度，error bias.
    - RE： ref 微软人脸识别的文章，种族bias。
    - RE: 首先我们要澄清，本文所解决的recommendation bias指的是训练阶段由于采样所导致的用户潜在感兴趣商品被视为负样本，从而导致的推荐精度下降的问题。当用户的交互越稀疏，这种现象更为明显。下表我们根据用户的交互频次将用户分为了几组，并报告了在应用我们方法前后的推荐精度。通过表中的结果，可以看出我们的方法确实有效缓解了recommendation bias问题，提高了具有稀疏交互的用户的推荐准确率。
2. Both MRD and BRD are essentially heuristic constructs based on KNN and soft similarity alignment via KL divergence. The proposed framework lacks the depth of architectural or algorithmic innovation and largely reuses existing techniques. The overall framework appears as a combination of known components without introducing any fundamentally new modeling paradigm or optimization strategy.
    - RE: 大家都是用GNN，这不是我们核心的贡献。在细节处理和优化策略方面是有明显的差异。
3. Although the authors propose to alleviate recommendation bias by injecting inter-item knowledge, the paper provides no bias-related evaluation metrics or structural analysis. All experimental results are limited to accuracy-based metrics, without examining how the method affects bias. As a result, while the motivation targets bias reduction, the central claim is not empirically supported, revealing a significant gap in validation.
    - RE: 需要明确的是，我们侧重于解决的问题是在推荐过程中，部分用户喜欢的商品被错误排序，因此精确度指标能够反应对错误推荐的缓解。
    - RE: 在工业应用中，更关注实际的效果。在长尾用户的效果上有提升，说明debias是有效果的。
4. While the paper claims that the proposed method performs significantly better than the baselines, it would be helpful to include statistical significance tests (e.g., p-values or confidence intervals) to support these claims and strengthen the empirical analysis.
    - RE：箱图，展示方差和误差。
5. Do the authors have any empirical evidence—quantitative or qualitative—that the proposed method leads to less biased recommendation outcomes? Would you consider including bias-related evaluation metrics (e.g., coverage, popularity bias, calibration) in the revision?
    - RE：感谢你的建议，我们根据不同的user-item交互稀疏程度将用户分为了5个组，并分别统计了不同组的recall和ndcg指标，通过比较base方法和我们完整的方法，可以发现对于交互稀疏的用户，我们的性能提升更为明显。这充分证明了我们方法利用商品间关系挖掘用户潜在感兴趣的商品，有效缓解了由于稀疏交互所导致的推荐偏差问题。
6. Given that both MRD and BRD are constructed via KNN-based similarity and KL divergence, how does the proposed approach advance beyond existing multi-modal or graph-based recommendation frameworks? Could you elaborate on the novel aspects of the architecture or learning strategy?
    - RE：我们需要强调我们的方法的贡献在于提出利用商品间关系来挖掘用户潜在感兴趣的商品（如图1所示）。同时，我们提出了简洁有效的商品间关系蒸馏策略并在多个数据集上取得了显著的提升。相比于现有的推荐框架，我们的方法简单并有效，适合真实世界的应用和部署。
7. Will the authors consider releasing the code and implementation details to support reproducibility and foster further research? This would be especially helpful given the multi-component nature of the proposed framework.
    - RE：当然，我们致力于推动开源社区的发展，你可以从该链接查看匿名代码。
## 回答版
> **Question 1**: Empirical evidence of the de-bias effects.

| intervel/recall@20 | 0-4 | 5-9 | 10-14 | 15-19 | >=20 |
|--------|--------|--------|--------|--------|--------|
| freq.  | 9783  | 6904  | 1804 | 501 | 453 |
| base | 0.0955  | 0.0974  | 0.0967 | 0.0918 | 0.0963 |
| DIRD | 0.1023  | 0.1032  | 0.1031 | 0.0968 | 0.0976 |

| intervel/ndcg@20 | 0-4 | 5-9 | 10-14 | 15-19 | >=20 |
|--------|--------|--------|--------|--------|--------|
| freq.  | 9783  | 6904  | 1804 | 501 | 453 |
| base | 0.0428  | 0.0414  | 0.0469 | 0.0459 | 0.0452 |
| DIRD | 0.0470  | 0.0445  | 0.0475 | 0.0462 | 0.0456 |

> **Question 2**: Incorporating bias-related evaluation metrics.
