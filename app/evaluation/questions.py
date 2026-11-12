
QUESTIONS = [
    # ============================================================
    # ATTENTION IS ALL YOU NEED
    # ============================================================
    {
        "id": "ATT-01",
        "paper_id": "attention_is_all_you_need",
        "question": "How does scaled dot-product attention calculate attention weights?",
        "expected_section": "3.2.1",
        "expected_pages": [4],
        "answerable": True,
        "expected_evidence": (
            "Query-key dot products are scaled by the square root of the "
            "key dimension (sqrt(d_k)) and passed through softmax to "
            "produce attention weights."
        )
    },
    {
        "id": "ATT-02",
        "paper_id": "attention_is_all_you_need",
        "question": "Why does the Transformer use multi-head attention instead of a single attention operation?",
        "expected_section": "3.2.2",
        "expected_pages": [4, 5],
        "answerable": True,
        "expected_evidence": (
            "Multiple attention heads allow the model to attend to "
            "information from different representation subspaces and "
            "different positions simultaneously."
        )
    },
    {
        "id": "ATT-03",
        "paper_id": "attention_is_all_you_need",
        "question": "What is the role of positional encoding in the Transformer?",
        "expected_section": "3.5",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "Positional encodings provide information about token positions "
            "because the Transformer architecture has neither recurrence "
            "nor convolution."
        )
    },
    {
        "id": "ATT-04",
        "paper_id": "attention_is_all_you_need",
        "question": "What is the difference between the encoder and decoder stacks?",
        "expected_section": "3.1",
        "expected_pages": [3],
        "answerable": True,
        "expected_evidence": (
            "The encoder uses self-attention and feed-forward layers. "
            "The decoder additionally uses masked self-attention and "
            "encoder-decoder attention over encoder outputs."
        )
    },
    {
        "id": "ATT-05",
        "paper_id": "attention_is_all_you_need",
        "question": "Which optimizer and learning-rate schedule were used to train the Transformer?",
        "expected_section": "5.3",
        "expected_pages": [7],
        "answerable": True,
        "expected_evidence": (
            "The paper uses the Adam optimizer with a learning-rate "
            "schedule involving warmup followed by inverse-square-root "
            "decay."
        )
    },
    {
        "id": "ATT-06",
        "paper_id": "attention_is_all_you_need",
        "question": "How does the Transformer prevent future positions from being accessed during decoder self-attention?",
        "expected_section": "3.2.3",
        "expected_pages": [5],
        "answerable": True,
        "expected_evidence": (
            "Masking prevents decoder self-attention positions from "
            "attending to subsequent positions, preserving autoregressive "
            "generation."
        )
    },

    # ============================================================
    # MOSHI
    # ============================================================
    {
        "id": "MOS-01",
        "paper_id": "moshi",
        "question": "What is the purpose of inner monologue in Moshi?",
        "expected_section": "3.4.4",
        "expected_pages": [15, 16],
        "answerable": True,
        "expected_evidence": (
            "A textual representation of Moshi's own speech provides "
            "linguistic scaffolding that improves the quality and "
            "consistency of generated speech."
        )
    },
    {
        "id": "MOS-02",
        "paper_id": "moshi",
        "question": "What is the role of the Helium text language model in Moshi?",
        "expected_section": "3.1",
        "expected_pages": [7],
        "answerable": True,
        "expected_evidence": (
            "Moshi is built on top of Helium, a text language model "
            "trained on high-quality text data to provide strong "
            "reasoning abilities."
        )
    },
    {
        "id": "MOS-03",
        "paper_id": "moshi",
        "question": "How does Moshi tokenize audio?",
        "expected_section": "3.3",
        "expected_pages": [9, 10, 11, 12],
        "answerable": True,
        "expected_evidence": (
            "Mimi is a neural audio codec with a discrete bottleneck. "
            "It combines semantic information, which correlates with "
            "linguistic content, and acoustic information, which captures "
            "fine audio details and supports high-quality reconstruction. "
            "Distillation transfers non-causal semantic information into "
            "tokens produced by a causal model, enabling streaming "
            "semantic-acoustic tokenization."
        )
    },
    {
        "id": "MOS-04",
        "paper_id": "moshi",
        "question": "Why does Moshi use hierarchical autoregressive modeling?",
        "expected_section": "3.4.1",
        "expected_pages": [12, 13, 14],
        "answerable": True,
        "expected_evidence": (
            "Directly modeling multiple audio codebook streams as one "
            "flattened sequence creates a large prediction burden and "
            "high computational cost. The RQ-Transformer addresses this "
            "using a Temporal Transformer to produce temporal context "
            "and a Depth Transformer to predict multiple sub-sequences "
            "at each timestep."
        )
    },
    {
        "id": "MOS-05",
        "paper_id": "moshi",
        "question": "What are the main stages of Moshi's training process?",
        "expected_section": "4.4",
        "expected_pages": [20, 21],
        "answerable": True,
        "expected_evidence": (
            "The paper describes successive training stages, including "
            "Helium text-model pretraining, Moshi audio pretraining, "
            "multi-stream post-training, finetuning on real multi-stream "
            "data, and final instruction finetuning with synthetic "
            "instruction data. The answer should distinguish the stages "
            "and their associated data or objectives."
        )
    },
    {
        "id": "MOS-06",
        "paper_id": "moshi",
        "question": "What safety analyses are reported in the Moshi paper?",
        "expected_section": "6",
        "expected_pages": [35, 36, 37, 38, 39, 40, 41],
        "answerable": True,
        "expected_evidence": (
            "The paper reports analyses of toxicity, regurgitation, "
            "system voice consistency, and identification of generated "
            "content through watermarking."
        )
    },

    # ============================================================
    # DYNAMIC CHUNKING
    # ============================================================
    {
        "id": "DYN-01",
        "paper_id": "dynamic_chunking",
        "question": "How does dynamic chunking decide when to create a boundary?",
        "expected_section": "2.2.1",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "The routing module compares adjacent projected encoder "
            "representations using cosine similarity. Lower similarity "
            "indicates a potential contextual or semantic change and "
            "produces a higher boundary probability."
        )
    },
    {
        "id": "DYN-02",
        "paper_id": "dynamic_chunking",
        "question": "How does the H-Net routing module detect chunk boundaries?",
        "expected_section": "2.2.1",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "The routing module projects adjacent encoder representations "
            "into query and key vectors, then computes a cosine-similarity-"
            "based boundary probability from those vectors."
        )
    },
    {
        "id": "DYN-03",
        "paper_id": "dynamic_chunking",
        "question": "How are boundary probabilities calculated from adjacent encoder representations?",
        "expected_section": "2.2.1",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "The boundary probability is p_t = 0.5 * "
            "(1 - cosine_similarity(q_t, k_(t-1))), where q_t and "
            "k_(t-1) are the projected query and key representations."
        )
    },
    {
        "id": "DYN-04",
        "paper_id": "dynamic_chunking",
        "question": "What threshold is used to convert a boundary probability into a boundary indicator?",
        "expected_section": "2.2.1",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "The boundary indicator is one when the boundary probability "
            "is greater than or equal to 0.5, and zero otherwise."
        )
    },
    {
        "id": "DYN-05",
        "paper_id": "dynamic_chunking",
        "question": "How does the downsampler use boundary indicators?",
        "expected_section": "2.2.1",
        "expected_pages": [6],
        "answerable": True,
        "expected_evidence": (
            "The downsampler retains encoder representations whose "
            "boundary indicator is one and discards representations "
            "whose indicator is zero."
        )
    },
    {
        "id": "DYN-06",
        "paper_id": "dynamic_chunking",
        "question": "What is the purpose of the ratio loss in Dynamic Chunking?",
        "expected_section": "2.2.3",
        "expected_pages": [8],
        "answerable": True,
        "expected_evidence": (
            "Ratio loss regulates the compression ratio of the chunking "
            "process by encouraging the learned chunking behavior "
            "toward a target ratio."
        )
    },

    # ============================================================
    # CROSS-PAPER QUESTION
    # ============================================================
    {
        "id": "CROSS-01",
        "paper_id": None,
        "question": "How do the Transformer and H-Net differ in how they handle sequence structure?",
        "expected_section": None,
        "expected_pages": None,
        "answerable": True,
        "expected_evidence": (
            "The Transformer uses attention over token positions, whereas "
            "H-Net uses learned dynamic chunking and hierarchical "
            "sequence representations. The comparison must be grounded "
            "in evidence from both papers."
        )
    },

    # ============================================================
    # UNANSWERABLE QUESTION
    # ============================================================
    {
        "id": "UNANS-01",
        "paper_id": "moshi",
        "question": "What is the exact commercial API price in US dollars per million tokens for using Moshi?",
        "expected_section": None,
        "expected_pages": None,
        "answerable": False,
        "expected_evidence": (
            "The paper does not specify a commercial API price. "
            "The system must refuse to provide an unsupported figure "
            "rather than inventing a price."
        )
    }
]
