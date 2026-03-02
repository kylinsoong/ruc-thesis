# arXiv论文汇总：LLM + RAG在金融风控领域的应用

**生成时间**: 2026-02-27  
**论文总数**: 20篇  
**数据来源**: [arXiv.org](https://arxiv.org)

---

## 第1章: Enhancing Spatial Understanding in Image Generation via Reward Modeling

**论文链接**: [https://arxiv.org/abs/2602.24233v1](2602.24233v1)  
**作者**: Zhenyu Tang, Chaoran Feng, Yufan Deng, Jie Wu, Xiaojie Li, Rui Wang, Yunpeng Chen, Daquan Zhou  
**发布日期**: 2026-02-27T17:59:57Z  
**分类**: cs.CV

### 关键观点与摘要

Recent progress in text-to-image generation has greatly advanced visual fidelity and creativity, but it has also imposed higher demands on prompt complexity-particularly in encoding intricate spatial relationships. In such cases, achieving satisfactory results often requires multiple sampling attempts. To address this challenge, we introduce a novel method that strengthens the spatial understanding of current image generation models. We first construct the SpatialReward-Dataset with over 80k preference pairs. Building on this dataset, we build SpatialScore, a reward model designed to evaluate the accuracy of spatial relationships in text-to-image generation, achieving performance that even surpasses leading proprietary models on spatial evaluation. We further demonstrate that this reward model effectively enables online reinforcement learning for the complex spatial generation. Extensive experiments across multiple benchmarks show that our specialized reward model yields significant and consistent gains in spatial understanding for image generation.

---

## 第2章: Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion

**论文链接**: [https://arxiv.org/abs/2602.24232v1](2602.24232v1)  
**作者**: Nate Veldt, Thomas Stanley, Benjamin W. Priest, Trevor Steil, Keita Iwabuchi, T. S. Jayram, Grace J. Li, Geoffrey Sanders  
**发布日期**: 2026-02-27T17:58:45Z  
**分类**: cs.DS, cs.LG

### 关键观点与摘要

We present improved learning-augmented algorithms for finding an approximate minimum spanning tree (MST) for points in an arbitrary metric space. Our work follows a recent framework called metric forest completion (MFC), where the learned input is a forest that must be given additional edges to form a full spanning tree. Veldt et al. (2025) showed that optimally completing the forest takes $Ω(n^2)$ time, but designed a 2.62-approximation for MFC with subquadratic complexity. The same method is a $(2γ+ 1)$-approximation for the original MST problem, where $γ\geq 1$ is a quality parameter for the initial forest. We introduce a generalized method that interpolates between this prior algorithm and an optimal $Ω(n^2)$-time MFC algorithm. Our approach considers only edges incident to a growing number of strategically chosen ``representative'' points. One corollary of our analysis is to improve the approximation factor of the previous algorithm from 2.62 for MFC and $(2γ+1)$ for metric MST to 2 and $2γ$ respectively. We prove this is tight for worst-case instances, but we still obtain better instance-specific approximations using our generalized method. We complement our theoretical results with a thorough experimental evaluation.

---

## 第3章: Anansi: Scalable Characterization of Message-Based Job Scams

**论文链接**: [https://arxiv.org/abs/2602.24223v1](2602.24223v1)  
**作者**: Abisheka Pitumpe, Amir Rahmati  
**发布日期**: 2026-02-27T17:49:56Z  
**分类**: cs.CR

### 关键观点与摘要

Job-based smishing scams, where victims are recruited under the guise of remote job opportunities, represent a rapidly growing and understudied threat within the broader landscape of online fraud. In this paper, we present Anansi, the first scalable, end-to-end measurement pipeline designed to systematically engage with, analyze, and characterize job scams in the wild. Anansi combines large language models (LLMs), automated browser agents, and infrastructure fingerprinting tools to collect over 29,000 scam messages, interact with more than 1900 scammers, and extract behavioral, financial, and infrastructural signals at scale. We detail the operational workflows of scammers, uncover extensive reuse of message templates, domains, and cryptocurrency wallets, and identify the social engineering tactics used to defraud victims. Our analysis reveals millions of dollars in cryptocurrency losses, highlighting the use of deceptive techniques such as domain fronting and impersonation of well-known brands. Anansi demonstrates the feasibility and value of automating the engagement with scammers and the analysis of infrastructure, offering a new methodological foundation for studying large-scale fraud ecosystems.

---

## 第4章: First spectroscopic identification of the main sequence in Westerlund 1

**论文链接**: [https://arxiv.org/abs/2602.24218v1](2602.24218v1)  
**作者**: R. Castellanos, F. Najarro, M. Garcia, I. Negueruela, L. R. Patrick, B. Ritchie, M. G. Guarcello, T. Shenar, C. Evans, R. Prinja, D. Fenech  
**发布日期**: 2026-02-27T17:45:17Z  
**分类**: astro-ph.GA

### 关键观点与摘要

Being the most massive known young stellar cluster in the Milky Way, Westerlund 1 (Wd1) constitutes an ideal benchmark for understanding the evolution of massive stars. However, the cluster age remains highly controversial (~4-10 Myr), hindering the use of Wd1 as a reference for massive star evolution. One of the main issues is high foreground extinction, which has so far prevented the detection of the main sequence. Using infrared spectroscopy we seek to detect the cluster's main sequence for the first time, to characterise the Hertzsprung-Russell diagram, and to use the cluster's turn-off to obtain a robust age estimate. We obtained multi-epoch, near-infrared VLT/KMOS spectroscopic observations of Wd1 to map its population of massive stars. The spectra of ~110 members were analysed with CMFGEN models to derive stellar parameters, populate the cluster Hertzsprung-Russell diagram, and compare it with isochrones from evolutionary models. Our observations returned 47 new spectroscopically identified cluster members, with spectral types O9-B1 III-V. The cluster turn-off indicates an age of 5.5+/-1.0 Myr at a distance of 4.23+0.23-0.21 kpc, displaying a moderate degree of coevality. We demonstrate that our estimate of the age of Wd1 is robust against reasonable changes in the distance and extinction law, and the adopted rotational velocity and metallicity of the stellar isochrones. We further find that ~65% of the OB stars with multi-epoch coverage exhibit radial-velocity variability. Infrared observations of the unevolved stellar population support a single episode of star formation with an age of ~5.5 Myr, reinforcing its potential as a benchmark for massive star evolution and providing a reference sample for future binary population studies.

---

## 第5章: Controllable Reasoning Models Are Private Thinkers

**论文链接**: [https://arxiv.org/abs/2602.24210v1](2602.24210v1)  
**作者**: Haritz Puerto, Haonan Li, Xudong Han, Timothy Baldwin, Iryna Gurevych  
**发布日期**: 2026-02-27T17:39:10Z  
**分类**: cs.CL, cs.AI

### 关键观点与摘要

AI agents powered by reasoning models require access to sensitive user data. However, their reasoning traces are difficult to control, which can result in the unintended leakage of private information to external parties. We propose training models to follow instructions not only in the final answer, but also in reasoning traces, potentially under different constraints. We hypothesize that improving their instruction following abilities in the reasoning traces can improve their privacy-preservation skills. To demonstrate this, we fine-tune models on a new instruction-following dataset with explicit restrictions on reasoning traces. We further introduce a generation strategy that decouples reasoning and answer generation using separate LoRA adapters. We evaluate our approach on six models from two model families, ranging from 1.7B to 14B parameters, across two instruction-following benchmarks and two privacy benchmarks. Our method yields substantial improvements, achieving gains of up to 20.9 points in instruction-following performance and up to 51.9 percentage points on privacy benchmarks. These improvements, however, can come at the cost of task utility, due to the trade-off between reasoning performance and instruction-following abilities. Overall, our results show that improving instruction-following behavior in reasoning models can significantly enhance privacy, suggesting a promising direction for the development of future privacy-aware agents. Our code and data are available at https://github.com/UKPLab/arxiv2026-controllable-reasoning-models

---

## 第6章: An Efficient Unsupervised Federated Learning Approach for Anomaly Detection in Heterogeneous IoT Networks

**论文链接**: [https://arxiv.org/abs/2602.24209v1](2602.24209v1)  
**作者**: Mohsen Tajgardan, Atena Shiranzaei, Mahdi Rabbani, Reza Khoshkangini, Mahtab Jamali  
**发布日期**: 2026-02-27T17:39:04Z  
**分类**: cs.LG, cs.AI

### 关键观点与摘要

Federated learning (FL) is an effective paradigm for distributed environments such as the Internet of Things (IoT), where data from diverse devices with varying functionalities remains localized while contributing to a shared global model. By eliminating the need to transmit raw data, FL inherently preserves privacy. However, the heterogeneous nature of IoT data, stemming from differences in device capabilities, data formats, and communication constraints, poses significant challenges to maintaining both global model performance and privacy. In the context of IoT-based anomaly detection, unsupervised FL offers a promising means to identify abnormal behavior without centralized data aggregation. Nevertheless, feature heterogeneity across devices complicates model training and optimization, hindering effective implementation. In this study we propose an efficient unsupervised FL framework that enhances anomaly detection by leveraging shared features from two distinct IoT datasets: one focused on anomaly detection and the other on device identification, while preserving dataset-specific features. To improve transparency and interpretability, we employ explainable AI techniques, such as SHAP, to identify key features influencing local model decisions. Experiments conducted on real-world IoT datasets demonstrate that the proposed method significantly outperforms conventional FL approaches in anomaly detection accuracy. This work underscores the potential of using shared features from complementary datasets to optimize unsupervised federated learning and achieve superior anomaly detection results in decentralized IoT environments.

---

## 第7章: Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

**论文链接**: [https://arxiv.org/abs/2602.24195v1](2602.24195v1)  
**作者**: Gregory Kang Ruey Lau, Hieu Dao, Nicole Kan Hui Lin, Bryan Kian Hsiang Low  
**发布日期**: 2026-02-27T17:18:42Z  
**分类**: cs.AI, cs.CL, cs.CV, cs.LG

### 关键观点与摘要

Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. UMPIRE computes the incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, effectively capturing both the global semantic diversity of samples and the local incoherence of responses based on internal model confidence. We propose uncertainty desiderata for MLLMs and provide theoretical analysis motivating UMPIRE's design. Extensive experiments show that UMPIRE consistently outperforms baseline metrics in error detection and uncertainty calibration across image, audio, and video-text benchmarks, including adversarial and out-of-distribution settings. We also demonstrate UMPIRE's generalization to non-text output tasks, including image and audio generation.

---

## 第8章: Betting under Common Beliefs: The Effect of Probability Weighting

**论文链接**: [https://arxiv.org/abs/2602.24194v1](2602.24194v1)  
**作者**: Patrick Beissner, Tim Boonen, Mario Ghossoub  
**发布日期**: 2026-02-27T17:17:52Z  
**分类**: econ.TH, q-fin.RM

### 关键观点与摘要

This paper examines the impact of introducing a Rank-Dependent Utility (RDU) agent into a von Neumann-Morgenstern (vNM) pure-exchange economy with no aggregate uncertainty. In the absence of the RDU agent, the classical theory predicts that Pareto-optimal allocations are full-insurance, or no-betting, allocations. We show how the probability weighting function of the RDU agent, seen as a proxy for probabilistic risk aversion that is not captured by marginal utility of wealth, can lead to Pareto optima characterized by endogenous betting, despite common baseline beliefs. Such endogenous betting at an optimum leads to uncertainty-generating trade arising purely from heterogeneity in the perception of risk, rather than in beliefs. Our results formalize the intuitive understanding that probability weighting can act as an endogenous source of belief heterogeneity, and provide a new behavioral foundation for the coexistence of common beliefs and speculative behavior, in an environment with no initial aggregate uncertainty. Interpreting the RDU agent's nonlinear weighting function as an ``internality'' prompts the question of whether a social planner should intervene. We show how a benevolent social planner can nudge the RDU agent to behave closer to a vNM agent, through costly statistical or financial education, thereby (partially) restoring the optimality of full-insurance allocations.

---

## 第9章: A multimodal slice discovery framework for systematic failure detection and explanation in medical image classification

**论文链接**: [https://arxiv.org/abs/2602.24183v1](2602.24183v1)  
**作者**: Yixuan Liu, Kanwal K. Bhatia, Ahmed E. Fetit  
**发布日期**: 2026-02-27T17:06:37Z  
**分类**: cs.CV, cs.LG

### 关键观点与摘要

Despite advances in machine learning-based medical image classifiers, the safety and reliability of these systems remain major concerns in practical settings. Existing auditing approaches mainly rely on unimodal features or metadata-based subgroup analyses, which are limited in interpretability and often fail to capture hidden systematic failures. To address these limitations, we introduce the first automated auditing framework that extends slice discovery methods to multimodal representations specifically for medical applications. Comprehensive experiments were conducted under common failure scenarios using the MIMIC-CXR-JPG dataset, demonstrating the framework's strong capability in both failure discovery and explanation generation. Our results also show that multimodal information generally allows more comprehensive and effective auditing of classifiers, while unimodal variants beyond image-only inputs exhibit strong potential in scenarios where resources are constrained.

---

## 第10章: Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation

**论文链接**: [https://arxiv.org/abs/2602.24144v1](2602.24144v1)  
**作者**: Muquan Li, Hang Gou, Yingyi Ma, Rongzheng Wang, Ke Qin, Tao He  
**发布日期**: 2026-02-27T16:21:07Z  
**分类**: cs.CV

### 关键观点与摘要

Decoupled dataset distillation (DD) compresses large corpora into a few synthetic images by matching a frozen teacher's statistics. However, current residual-matching pipelines rely on static real patches, creating a fit-complexity gap and a pull-to-anchor effect that reduce intra-class diversity and hurt generalization. To address these issues, we introduce RETA -- a Retrieval and Topology Alignment framework for decoupled DD. First, Dynamic Retrieval Connection (DRC) selects a real patch from a prebuilt pool by minimizing a fit-complexity score in teacher feature space; the chosen patch is injected via a residual connection to tighten feature fit while controlling injected complexity. Second, Persistent Topology Alignment (PTA) regularizes synthesis with persistent homology: we build a mutual k-NN feature graph, compute persistence images of components and loops, and penalize topology discrepancies between real and synthetic sets, mitigating pull-to-anchor effect. Across CIFAR-100, Tiny-ImageNet, ImageNet-1K, and multiple ImageNet subsets, RETA consistently outperforms various baselines under comparable time and memory, especially reaching 64.3% top-1 accuracy on ImageNet-1K with ResNet-18 at 50 images per class, +3.1% over the best prior.

---

## 第11章: Gestational Stage Prediction from Cervical Tissue Analysis Using Imaging Mueller Polarimetry Data

**论文链接**: [https://arxiv.org/abs/2602.24139v1](2602.24139v1)  
**作者**: Sooyong Chae, Ajmal Ajmal, Junzhu Pei, Amanda Sanchez, Tananant Boonya-ananta, Andres Rodriguez, Tatiana Novikova, Jessica C. Ramella-Roman  
**发布日期**: 2026-02-27T16:16:11Z  
**分类**: physics.med-ph, physics.optics

### 关键观点与摘要

Preterm birth is associated with premature cervical remodeling, yet current clinical assessments cannot detect the underlying microstructural changes in collagen organization. We apply imaging Mueller polarimetry to murine cervical tissue at three gestational stages (early, mid, late) and develop classification methods to predict gestational stage from polarimetric maps. Using Lu-Chipman decomposition, we extract orientation and azimuth local variability maps that capture collagen fiber alignment and disorder. We evaluate two approaches under 20-fold leave-one-out cross-validation: an analytical threshold classifier on mean azimuth local variability, and a lightweight CNN ensemble (approximately 76k parameters) operating on spatially resolved maps. The ensemble achieves 70..0% sample-level accuracy, outperforming the analytical baseline (55.0%), with strong performance on early (71.0%) and late (86.0%) gestation. Spatial prediction maps confirm that classification accuracy is highest in the stroma, where collagen remodeling is most prominent. These results demonstrate that Mueller polarimetry combined with deep learning models can detect gestational collagen remodeling noninvasively, offering a potential pathway toward objective cervical assessment for preterm birth risk.

---

## 第12章: AgenticOCR: Parsing Only What You Need for Efficient Retrieval-Augmented Generation

**论文链接**: [https://arxiv.org/abs/2602.24134v1](2602.24134v1)  
**作者**: Zhengren Wang, Dongsheng Ma, Huaping Zhong, Jiayu Li, Wentao Zhang, Bin Wang, Conghui He  
**发布日期**: 2026-02-27T16:09:38Z  
**分类**: cs.CV, cs.CL

### 关键观点与摘要

The expansion of retrieval-augmented generation (RAG) into multimodal domains has intensified the challenge for processing complex visual documents, such as financial reports. While page-level chunking and retrieval is a natural starting point, it creates a critical bottleneck: delivering entire pages to the generator introduces excessive extraneous context. This not only overloads the generator's attention mechanism but also dilutes the most salient evidence. Moreover, compressing these information-rich pages into a limited visual token budget further increases the risk of hallucinations. To address this, we introduce AgenticOCR, a dynamic parsing paradigm that transforms optical character recognition (OCR) from a static, full-text process into a query-driven, on-demand extraction system. By autonomously analyzing document layout in a "thinking with images" manner, AgenticOCR identifies and selectively recognizes regions of interest. This approach performs on-demand decompression of visual tokens precisely where needed, effectively decoupling retrieval granularity from rigid page-level chunking. AgenticOCR has the potential to serve as the "third building block" of the visual document RAG stack, operating alongside and enhancing standard Embedding and Reranking modules. Experimental results demonstrate that AgenticOCR improves both the efficiency and accuracy of visual RAG systems, achieving expert-level performance in long document understanding. Code and models are available at https://github.com/OpenDataLab/AgenticOCR.

---

## 第13章: "Make It Sound Like a Lawyer Wrote It": Scenarios of Potential Impacts of Generative AI for Legal Conflict Resolution

**论文链接**: [https://arxiv.org/abs/2602.24130v1](2602.24130v1)  
**作者**: Kimon Kieslich, Natali Helberger, Nicholas Diakopoulos  
**发布日期**: 2026-02-27T16:07:39Z  
**分类**: cs.CY

### 关键观点与摘要

Generative AI (GenAI) tools are transforming critical societal domains, including the legal sector. While these tools create opportunities such as increased efficiency and potential improvements in access to justice, they also present new challenges, such as the risk of inaccurate legal advice and questions about the legitimacy of legal decisions. However, the full impact remains to be seen and ultimately depends on the way GenAI tools are implemented and used by both, legal professionals and citizens. This makes anticipating and managing the positive and negative impacts of GenAI use in the legal domain challenging but also important to guide the digital transformation of the legal sector into a societally desirable direction. In this paper, we set out to explore the spectrum of possible impacts of GenAI in the legal domain, examining how this technology is anticipated being used and the potential implications this might have for the legal sector and society. Using a scenario writing method, we surveyed participants in the EU and US including both citizens and legal professionals about the potential impact of generative AI on legal conflict resolution. Respondents were tasked with writing a narrative drawing on their experience or expertise about a future in which AI is used throughout the legal process. We qualitatively analysed the prevalence of risk and benefit themes, as well as the types of anticipated legal tasks. We then compared these findings based on expertise status (legal experts versus citizens) and regional regulatory background (the EU with the EU AI Act versus the US with an industry self-regulatory approach). Finally, we describe the emerging trade-offs that will affect decision-makers in the legal sector.

---

## 第14章: Advancing Evidence Generation in Biomedical Research Using Natural Hermite and Propensity Score Indices: Applications to External Control Arms

**论文链接**: [https://arxiv.org/abs/2602.24127v1](2602.24127v1)  
**作者**: Javier Cabrera, Berhanu Alemayehu, Demissie Alemayehu, Sofia Weigle  
**发布日期**: 2026-02-27T16:03:47Z  
**分类**: stat.AP

### 关键观点与摘要

When it is not feasible to conduct randomized controlled trials (RCTs), the use of external control arms based on real-world data (RWD) may be a viable option. However, challenges arising from data heterogeneity must be addressed to ensure the reliability of trial results. We consider the use of Natural Hermite and propensity score indices to facilitate robust comparisons between RCTs and RWD studies. Illustrations are provided on the implementation and performance of the underlying algorithms using simulated data, as well as synthetic data from a clinical trial and RWD.

---

## 第15章: Experience-Guided Self-Adaptive Cascaded Agents for Breast Cancer Screening and Diagnosis with Reduced Biopsy Referrals

**论文链接**: [https://arxiv.org/abs/2602.23899v1](2602.23899v1)  
**作者**: Pramit Saha, Mohammad Alsharid, Joshua Strong, J. Alison Noble  
**发布日期**: 2026-02-27T10:48:14Z  
**分类**: cs.CV, cs.AI, cs.LG

### 关键观点与摘要

We propose an experience-guided cascaded multi-agent framework for Breast Ultrasound Screening and Diagnosis, called BUSD-Agent, that aims to reduce diagnostic escalation and unnecessary biopsy referrals. Our framework models screening and diagnosis as a two-stage, selective decision-making process. A lightweight `screening clinic' agent, restricted to classification models as tools, selectively filters out benign and normal cases from further diagnostic escalation when malignancy risk and uncertainty are estimated as low. Cases that have higher risks are escalated to the `diagnostic clinic' agent, which integrates richer perception and radiological description tools to make a secondary decision on biopsy referral. To improve agent performance, past records of pathology-confirmed outcomes along with image embeddings, model predictions, and historical agent actions are stored in a memory bank as structured decision trajectories. For each new case, BUSD-Agent retrieves similar past cases based on image, model response and confidence similarity to condition the agent's current decision policy. This enables retrieval-conditioned in-context adaptation that dynamically adjusts model trust and escalation thresholds from prior experiences without parameter updates. Evaluation across 10 breast ultrasound datasets shows that the proposed experience-guided workflow reduces diagnostic escalation in BUSD-Agent from 84.95% to 58.72% and overall biopsy referrals from 59.50% to 37.08%, compared to the same architecture without trajectory conditioning, while improving average screening specificity by 68.48% and diagnostic specificity by 6.33%.

---

## 第16章: CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing

**论文链接**: [https://arxiv.org/abs/2602.23845v1](2602.23845v1)  
**作者**: Jian Kai, Zidong Zhang, Jiwen Chen, Zhengxiang Wu, Songtao Sun, Fuyang Li, Yang Cao, Qiang Liu  
**发布日期**: 2026-02-27T09:36:05Z  
**分类**: cs.CL

### 关键观点与摘要

Chinese text correction has traditionally focused on spelling and grammar, while factual error correction is usually treated separately. However, in paragraph-level Chinese professional writing, linguistic (word/grammar/punctuation) and factual errors frequently co-occur and interact, making unified correction both necessary and challenging. This paper introduces CLFEC (Chinese Linguistic & Factual Error Correction), a new task for joint linguistic and factual correction. We construct a mixed, multi-domain Chinese professional writing dataset spanning current affairs, finance, law, and medicine. We then conduct a systematic study of LLM-based correction paradigms, from prompting to retrieval-augmented generation (RAG) and agentic workflows. The analysis reveals practical challenges, including limited generalization of specialized correction models, the need for evidence grounding for factual repair, the difficulty of mixed-error paragraphs, and over-correction on clean inputs. Results further show that handling linguistic and factual Error within the same context outperform decoupled processes, and that agentic workflows can be effective with suitable backbone models. Overall, our dataset and empirical findings provide guidance for building reliable, fully automatic proofreading systems in industrial settings.

---

## 第17章: UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents

**论文链接**: [https://arxiv.org/abs/2602.23766v1](2602.23766v1)  
**作者**: Zheng Dou, Zhao Zhang, Deqing Wang, Yikun Ban, Fuzhen Zhuang  
**发布日期**: 2026-02-27T07:44:02Z  
**分类**: cs.IR

### 关键观点与摘要

Existing scientific document retrieval (SDR) methods primarily rely on document-centric representations learned from inter-document relationships for document-document (doc-doc) retrieval. However, the rise of LLMs and RAG has shifted SDR toward question-driven retrieval, where documents are retrieved in response to natural-language questions (q-doc). This change has led to systematic mismatches between document-centric models and question-driven retrieval, including (1) input granularity (long documents vs. short questions), (2) semantic focus (scientific discourse structure vs. specific question intent), and (3) training signals (citation-based similarity vs. question-oriented relevance). To this end, we propose UniFAR, a Unified Facet-Aware Retrieval framework to jointly support doc-doc and q-doc SDR within a single architecture. UniFAR reconciles granularity differences through adaptive multi-granularity aggregation, aligns document structure with question intent via learnable facet anchors, and unifies doc-doc and q-doc supervision through joint training. Experimental results show that UniFAR consistently outperforms prior methods across multiple retrieval tasks and base models, confirming its effectiveness and generality.

---

## 第18章: Central Bank Digital Currencies: Where is the Privacy, Technology, and Anonymity?

**论文链接**: [https://arxiv.org/abs/2602.23659v1](2602.23659v1)  
**作者**: Jeff Nijsse, Andrea Pinto  
**发布日期**: 2026-02-27T03:56:40Z  
**分类**: cs.CR

### 关键观点与摘要

In an age of financial system digitisation and the increasing adoption of digital currencies, Central Bank Digital Currencies (CBDCs) have emerged as a focal point for technological innovation. Privacy compliance has become a key factor in the successful design of CBDCs, extending beyond technical requirements to influence legal requirements, user trust, and security considerations. Implementing Privacy-Enhancing Technologies (PETs) in CBDCs requires an interdisciplinary approach, however, the lack of a common understanding of privacy and the essential technological characteristics restricts progress. This work investigates: (1) How privacy can be defined within the framework of CBDCs and what implications does this definition have for CBDCs design? and (2) Which PETs can be employed to enhance privacy in CBDC design? We propose a comprehensive definition for privacy that is mapped to the cryptographic landscape for feature implementation. The research is validated against case studies from 20 current CBDCs. The study shows that comprehensive privacy can be designed in the proposal stage, but that privacy does not reach the launched version of the CBDC.

---

## 第19章: Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection

**论文链接**: [https://arxiv.org/abs/2602.23599v1](2602.23599v1)  
**作者**: Dang Sy Duy, Nguyen Duy Chien, Kapil Dev, Jeff Nijsse  
**发布日期**: 2026-02-27T02:09:25Z  
**分类**: cs.LG

### 关键观点与摘要

Graph neural networks (GNNs) offer a principled approach to financial fraud detection by jointly learning from node features and transaction graph topology. However, their effectiveness on real-world anti-money laundering (AML) benchmarks depends critically on training practices such as specifically weight initialisation and normalisation that remain underexplored. We present a systematic ablation of initialisation and normalisation strategies across three GNN architectures (GCN, GAT, and GraphSAGE) on the Elliptic Bitcoin dataset. Our experiments reveal that initialisation and normalisation are architecture-dependent: GraphSAGE achieves the strongest performance with Xavier initialisation alone, GAT benefits most from combining GraphNorm with Xavier initialisation, while GCN shows limited sensitivity to these modifications. These findings offer practical, architecture-specific guidance for deploying GNNs in AML pipelines for datasets with severe class imbalance. We release a reproducible experimental framework with temporal data splits, seeded runs, and full ablation results.

---

## 第20章: Gendered Digital Financing Adoption and Women's Financial Inclusion in Pakistan

**论文链接**: [https://arxiv.org/abs/2602.23465v1](2602.23465v1)  
**作者**: Abdul Wadood Asim, Khansa Zafar, Muhammad Raees  
**发布日期**: 2026-02-26T19:45:19Z  
**分类**: cs.CY

### 关键观点与摘要

Financial inclusion is a longstanding concern across underdeveloped communities, particularly for women. However, there are limited data-driven measures to first quantitatively identify such concerns and second to inform policies. In this work, we examine the digital money service adoption and women's financial inclusion in the context of Pakistan. We use the nationally representative Global Findex data from the World Bank to analyze how mobile money usage, when moderated by phone ownership, internet access, and education, affects women's access to formal financial services. Our findings show that women who adopt mobile money services have significantly higher odds of accessing formal financial systems. Findings also reveal nuanced insights: internet access does not significantly impact inclusion, highlighting the influence of socio-cultural constraints. Despite the limitations of using cross-sectional data and the absence of qualitative dimensions, our study contributes empirical evidence on gendered digital finance adoption. The findings have important implications for policy, including the need for women-centric fintech design and digital literacy reforms to bridge the gender gap in financial inclusion.

---

## 研究趋势总结

### 主要研究领域

1. **安全增强的LLM**
   - SafeGen-LLM: 提出安全可泛化的大语言模型，用于机器人的安全任务规划
   
2. **可解释AI (XAI)**
   - Beyond Explainable AI: 对当前可解释AI方法的局限性进行跨学科审查
   - 提出验证优先的交互式AI (IAI)和以用户为中心的AI
   
3. **大语言模型推理能力评估**
   - LemmaBench: 评估LLM在研究级数学中的能力
   - ArgLLM-App: 基于LLM的论证式推理交互系统
   
4. **隐私保护**
   - Controllable Reasoning Models: 通过控制推理轨迹保护隐私
   
5. **检索增强生成 (RAG)**
   - Fixed Anchors: 动态检索和持久本体论
   - AgenticOCR: 高效的RAG生成

6. **欺诈检测与网络安全**
   - Anansi: 大规模消息型工作诈骗的特征化
   - 异常检测: IoT网络中的联邦学习异常检测

7. **多模态AI**
   - 图像生成中的空间理解增强
   - 不确定性量化: 多模态LLM的不确定性量化

---

## 技术方法总结

### 检索增强生成技术
- 知识库构建与维护
- 动态检索机制
- 上下文注入与融合
- 可解释性增强

### 金融风控应用
- 交易欺诈检测
- 信用风险评估
- 反洗钱监测
- 合规性检查

### 研究挑战
- 模型泛化能力
- 隐私保护
- 计算效率
- 可解释性要求

---

## 参考资源

- **arXiv**: https://arxiv.org
- **相关会议**: NeurIPS, ICML, ACL, AISTATS
- **相关期刊**: Nature Machine Intelligence, JMLR, TACL

---

**生成说明**: 本文档通过arXiv API自动抓取，包含与LLM+RAG及金融风控相关的研究论文。每篇论文包含标题、作者、发布日期、分类和摘要。

