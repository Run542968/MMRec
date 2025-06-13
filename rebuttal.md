# Reviewer FnMP
1. This work is not open source. 
    - RE: 后续会开源 
    - RE：please improve your score
2. In Eq. (9), is it overly simplistic to fuse multi-modal features merely through straightforward summation? Furthermore, in this equation, are the feature values of term Em fixed and non-trainable after the addition operator, implying that only term Eid is subject to optimization during training? If so, it appears that the inclusion of term Em may have limited influence on the training outcome of E. 
    - RE: 我们从未claim Eq.(9)中的$\bar{E}^m$是固定的。其在Modality Relational Distiilation中通过蒸馏进行更新（see Eq.(12)），而且$E$的梯队会反传更新$\bar{E}^m$ (基本的链式法则)。其次，summation操作很有效，其充分保留了ID和Modality信息，避免了信息的退化和遗失。
3. The experimental section lacks a discussion on the parameters \alpha and \lambda. 
    - RE: Please see supplementary material.
4. The experimental section lacks an explanation as to why MGCN underperforms compared to LGMRec on the Sports and Microlens datasets. Moreover, analysis is absent regarding why both MGCN and LGMRec consistently outperform MG and DiffMM overall.
    - RE: 首先，MGCN和LGMRec是现有方法，我们如实记录了他们的结果，不同的方法有不同的侧重点，从而在不同的数据集带来了性能差异，这与我们方法毫无关系。其次，MGCN比LGMRec差是因为没有考虑local信息，这在Baby数据集是较为重要的。DifMM方法差是因为生成的pseudo edge是错的，噪声很大，尽管他是一个很novel的方法，但是没有解决噪声问题。MG差是因为它仅仅做了一个梯度的反转，并没有解决实际问题。总结来讲，看似新颖的方法并不一定有效。

# Reviewer nBxX
1. Given that missing items in different modality graphs present a crucial real-world challenge, could you elaborate on any potential strategies to handle the distillation of multiple modalities when an item is present in one graph but absent in another?
    - RE: 首先要强调的是，在我们的方法中，我们利用商品的多模态关系作为补充信息，来挖掘用户潜在感兴趣的商品，从而提高它的推荐排序，如图1（b）。不同模态的关系是相互补充的，当item的某个模态缺失时，剩余模态依然能够发挥作用。其次，模态的缺失可以通过多种方式来补充，例如使用LLM来扩充语义信息，这属于数据预处理，并不是我们工作的核心贡献。
    - RE: 电商中模态都存在，基本模态信息都存在，都能拿到
2. The User-Item Graph is often sparse, which affects recommendation accuracy. Could you discuss in more detail the reasons for not addressing this sparsity issue in the current paper?
    - RE: 在多个工作中已经有结论，通过增加User-Item Graph连边的方式来缓解数据稀疏，会因为噪声问题导致严重的性能下降。通过对比DiffMM也可以发现，即使采用新颖的扩散模型来增广user-item连边，错误的噪声连边依然导致难以取得好的效果。因此，本文没有选择增广user-item连边，而是通过蒸馏item间的关系来间接挖掘用户潜在感兴趣的item，如图1.(b)所示。
    - Ref：Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels
3. Large Language Models (LLMs) have demonstrated significant capabilities in understanding and generating semantic content. Why did you choose not to leverage LLMs in your proposed model to enhance item-item relations?
    - RE: 我们必须强调，LLM生成的语义内容确实有助于增强商品的原始信息，但这属于一种数据增广，并不是我们方法的核心贡献。我们的核心贡献是利用商品间的关系来挖掘用户潜在感兴趣的商品。
    - RE：在线推理时间也需要考虑，兼顾性能效率和现有模型升级改造的难易程度
4. The decision to adopt a two-step distillation process, separating the distillation of inter-item relationships from multi-modal features and user behavior, lacks a clear explanation of its necessity. Could you provide a detailed rationale for this two-step approach?
    - RE：我们必须澄清，全文从未提到两阶段，BRD和MRD是与recommendation loss同时优化的。

# Reviewer xFhz
1. The motivation is unclear. The manuscript introduces several well-known challenges in recommender systems, including sparse user interaction, recommendation bias, and negative sampling. However, the motivation of this work is not clearly articulated. The authors confuse these terms, and they do not clearly present their core point, which confused me a lot. The discussion in Lines 107–119 does not detail the limitations of existing methods. In particular, the explanation of the “recommendation bias problem” is circular and lacks clarity, as the authors use the term to define itself (Lines 115–119). Overall, the key problem this work aims to solve remains ambiguous.
    - RE: 首先，我们的motivation非常明确：由于稀疏的user-item交互，导致用户潜在感兴趣的item被视为负样本，从而在Eq.(10)的优化下远离(apart from)用户，导致推荐排序较后，这种现象称之为recommendation bias，见figure 1。最然现有的工作设计了多种复杂结构来融合特征，但他们都依赖于recommendation loss Eq.(10)，仍然受限于recommendation bias的问题。而我们首次提出利用item间关系来缓解这个问题，在不增广user-item graph引入噪声的前提下，通过间接的形式拉近用户和潜在感兴趣商品的距离。这不会引入错误的标签，因为错误标签带来的影响时灾难性的。
2. The novelty of this work seems limited. As acknowledged in the paper, many existing works use multi-modal information to enhance user-item representations via graph convolution. The general paradigm of this work follows a similar trajectory. The contribution of this work seems to lie in their complex selection of particular information during representation learning, named Multi-modal Relationship Distillation and Behavior Relationship Distillation in the paper. From this perspective, I'm afraid that the contribution of this work seemed incremental.
    - RE: 我们必须强调，我们的贡献在于首次利用商品间关系作为间接的纽带，从而挖掘用户潜在感兴趣的商品，从而提高推荐质量。我们的贡献不在于复杂的模型结构，而在于简洁有效的训练策略。
3. Mismatch Between Challenges and Techniques. There is a disconnect between the discussed challenges and the proposed solutions. For instance, the rationale behind separately learning item ID and modality embeddings is not explained. Furthermore, it is unclear how the proposed modules, such as MRD and BRD, directly mitigate the “recommendation bias” problem mentioned earlier in the paper.
    - RE: 请认真阅读，我们并没有单独学习ID embedding和modality embedding。对于modality embedding，其在融合前接受模态关系的蒸馏，在与ID embedding融合后的特征接受behavior relation的蒸馏。如图1所示，MRD和BRD通过间接的方式补充了user和item的关联，从而在缓解噪声的情况下提高了用户潜在商品的排序分数，可以通过ndcg指标看出。
4. Some important baselines are missing. Some key related works are not discussed or used as baselines. For instance, [1] also learns item ID and modality embeddings separately and enhances item ID embedding learning using an Item-Item Co-occurrence Matrix—very similar to the method presented in this work. The authors should clarify the distinctions between their model and [1] and include it in the experimental comparison.
    - RE: 我们会增加对这个工作的讨论。我们认真阅读了[1]，我们必须强调[1]采用co-occurrence matrix作为图卷积的邻接矩阵，而我们利用co-occurrence关系来进行关系蒸馏，这是完全不同的思路。
5. The experiments should be improved. The performance gains are marginal—often less than 0.002 and, in some cases, as low as 0.0001—casting doubt on the practical impact of the proposed method. Such marginal improvements can not demonstrate the effectiveness of the proposed model. Besides, the paper lacks significance testing (e.g., t-test) to verify that these improvements are statistically meaningful. Additionally, due to the model’s complexity (e.g., KNN-based neighbor selection), the authors should provide a time efficiency analysis (e.g., runtime, memory usage) to support its real-world feasibility.
    - RE: 关于性能提升，该领域已经趋近饱和，近几年的提升本来就很小，你应该关注领域的指标提升趋势。这里的KNN仅用来剥离子图，采用top-k计算即可，并不会带来额外的内存占用。
    - RE：举一个其他文章的例子，其他文章也是这样的


# Reviewer p3cM
1. The author mentions in Figure 1 that due to users having sparse interactions, the originally high-scored items are mistakenly judged as low-scored by the recommendation system. In subsequent methods, the author employs co-occurrence information to capture behavioral signals. However, co-occurrence information inherently relies on abundant interaction data. If the original user sequence is already sparse, this could exacerbate the phenomenon observed in Figure 1. I believe this warrants careful discussion and analysis.
    - RE: 你的分析是现实且合理的，正是因为存在这种extremely sparse user sequence, 所以我们需要引入多模态的信息进行补充，通过联合利用模态和行为信息来辅助探索用户潜在感兴趣的商品。从而通过间接的方式缓解用户交互稀疏的问题。
    - RE：我们面向的不是极度长尾用户，对于极少的用户会被过滤掉，用泛个性化的方式进行推荐。人是极度稀疏的时候，商品不是极度稀疏的，商品之间的关联是全局的。
2. The article emphasizes that this is a simple method in its introduction, however, there is no discussion about complexity or comparison with other methods.
    - RE: 我们的简洁体现在两个方面：首先在模型架构上，我们没有引入复杂的模型结构，这保证了高效推理速度（补充实验）；其次，在方法设计上，我们聚焦于商品间关系，从两个不同且互补的视角为模型提供有效的指引，简单且有效。
    - RE：对比算法的计算复杂度，运行时的时间复杂度，内存开销
3. Authors are encouraged to attempt conducting performance experiments on more datasets. If space is limited, they are advised to retain only the Recall and NDCG metrics.
    - RE：考虑版面原因，修正版去增加数据集。补充一个数据集
4. Debiasing is clearly a key research direction of the paper, but the related work section does not discuss the current status of existing work in debiasing, which is an essential aspect that should be addressed.
    - RE: 后续修改版会增加相关的文章讨论。
    - RE: 我们已经对现有工作有一些了解，也欢迎你提供一些相关的debias的方法
5. In the field of multimodal recommendation, there are already many representative works. The authors are advised to incorporate additional baseline methods to enrich their performance evaluation. For instance, Eq.(11) draws inspiration from the modality latent structure extraction content in work [1], but it failed to conduct a comparison with it in the experiments.
    - RE: 会将LATTECE加入实验对比表格

# Reviewer tn9b
1. The paper assumes that incorporating inter-item relationships can mitigate recommendation bias under sparse interactions. However, it does not explicitly model bias, nor does it provide empirical evidence that the injected knowledge actually leads to less biased recommendation behavior. As such, the connection between the proposed method and the claimed objective remains speculative rather than demonstrated. It would be helpful if the authors could provide supporting references or theoretical justification to substantiate this assumption.
    - RE: 虽然没有显式建模bias，但确实存在，和图1所讲的问题一样。他其实是一种现象，我们
    - RE：交互多的用户，现有模型推的比较准；交互少的用户，现有模型推的比较差。我们的目的是提高交互少的用户的推荐准确度，error bias.
    - RE： ref 微软人脸识别的文章，种族bias。
2. Both MRD and BRD are essentially heuristic constructs based on KNN and soft similarity alignment via KL divergence. The proposed framework lacks the depth of architectural or algorithmic innovation and largely reuses existing techniques. The overall framework appears as a combination of known components without introducing any fundamentally new modeling paradigm or optimization strategy.
    - RE: 大家都是用GNN，这不是我们核心的贡献。在细节处理和优化策略方面是有明显的差异。
3. Although the authors propose to alleviate recommendation bias by injecting inter-item knowledge, the paper provides no bias-related evaluation metrics or structural analysis. All experimental results are limited to accuracy-based metrics, without examining how the method affects bias. As a result, while the motivation targets bias reduction, the central claim is not empirically supported, revealing a significant gap in validation.
    - RE: 需要明确的是，我们侧重于解决的问题是在推荐过程中，部分用户喜欢的商品被错误排序，因此精确度指标能够反应对错误推荐的缓解。
    - RE: 在工业应用中，更关注实际的效果。在长尾用户的效果上有提升，说明debias是有效果的。
4. While the paper claims that the proposed method performs significantly better than the baselines, it would be helpful to include statistical significance tests (e.g., p-values or confidence intervals) to support these claims and strengthen the empirical analysis.
    - RE：箱图，展示方差和误差。
5. Do the authors have any empirical evidence—quantitative or qualitative—that the proposed method leads to less biased recommendation outcomes? Would you consider including bias-related evaluation metrics (e.g., coverage, popularity bias, calibration) in the revision?
    - RE：
6. Given that both MRD and BRD are constructed via KNN-based similarity and KL divergence, how does the proposed approach advance beyond existing multi-modal or graph-based recommendation frameworks? Could you elaborate on the novel aspects of the architecture or learning strategy?
    - RE：
7. Will the authors consider releasing the code and implementation details to support reproducibility and foster further research? This would be especially helpful given the multi-component nature of the proposed framework.
    - RE：