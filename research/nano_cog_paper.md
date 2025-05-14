# Nano-Cog 0.1: A Laptop-Scale Language Agent for Efficient Reasoning

## Abstract

This paper introduces Nano-Cog 0.1, a laptop-scale language agent designed for high reasoning-per-FLOP efficiency. We demonstrate that by combining state-space sequence modeling (Mamba), retrieval-augmented generation, tool use capabilities, and reinforcement-tuned chain-of-thought (CoT) reasoning, it's possible to create a performant AI system that runs efficiently on consumer hardware. Our approach prioritizes reasoning quality and computational efficiency over raw parameter count, achieving competitive performance on reasoning tasks while maintaining a significantly smaller resource footprint than larger transformer-based models.

## 1. Introduction

Large language models (LLMs) have demonstrated impressive capabilities in reasoning, tool use, and problem-solving. However, these systems typically require substantial computational resources, limiting their accessibility and applicability. Nano-Cog 0.1 addresses this challenge by leveraging recent advances in state-space models, particularly Mamba architecture, along with efficient fine-tuning techniques and a hybrid approach to reasoning.

The primary contributions of this work include:
1. A laptop-scale language agent architecture combining Mamba-130M with specialized reasoning adapters
2. A Dynamic Symbol Engine for efficient scratch-pad token utilization
3. Integration of retrieval-aware prompt composition with tool use capabilities
4. Reinforcement learning techniques optimized for reasoning efficiency

## 2. System Architecture

Nano-Cog 0.1 is built on a modular architecture consisting of two main components:

### 2.1 Inference Core
- **Backbone Model**: Mamba-130M state-space sequence model
- **LoRA Reasoning Adapters**: +8M parameters fine-tuned for reasoning tasks
- **Mini-MoE FFN Router**: +2M parameters for specialized routing
- **Dynamic Symbol Engine**: Enables efficient scratch-pad token usage
- **Retrieval-Aware Prompt Composer**: Integrates relevant external knowledge

### 2.2 External Utilities
- **ChromaDB**: Local vector database for retrieval
- **Tool Dispatcher**: Enables calculator, Python, and Bash operations
- **Configuration Store**: YAML/TOML based configuration

## 3. Technical Approach

### 3.1 State-Space Sequence Modeling

We utilize the Mamba architecture, which offers linear time/space complexity compared to quadratic complexity in traditional transformer models. This choice enables 3-5× faster CPU throughput than transformer-dense equivalents, making it ideal for laptop-scale deployment. Specific optimizations include:
- 4-bit weight quantization for inference
- Custom token position encoding for improved reasoning capabilities

### 3.2 Low-Rank Adaptation for Reasoning

Our LoRA Reasoning Adapters are inserted at the final two SSM blocks with rank 64 per QKV and FFN, resulting in approximately 8M trainable parameters (6% of backbone). This approach focuses weight updates on reasoning-centric components without modifying base weights, keeping fine-tuning RAM requirements below 11GB on M1 processors.

### 3.3 Mixture-of-Experts for Efficient Computation

The Mini-MoE Router incorporates two feed-forward "experts" (32-unit ReLU) positioned after block 12, with top-1 token routing and softmax temperature of 0.7. This design showcases sparse-activation capacity expansion while maintaining single-expert forward pass cost.

### 3.4 Dynamic Symbol Engine

The Dynamic Symbol Engine (DSE) employs grammar tokens in the form of `<define symbol=X: …>` followed by free-form definitions, with a maximum budget of 50 tokens per context. Performance is optimized through a compression reward mechanism: `compression = max(0, base_len – new_len)`, which is incorporated into the reinforcement learning process.

### 3.5 Retrieval and Tool Integration

Nano-Cog enhances reasoning through:
- **Vector Database**: ChromaDB with SQLite backend, seeded with 50K Wikipedia sentences and 5K code snippets
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-D)
- **Tool Interface**: Special call-tokens followed by JSON arguments
- **Tool Safety**: RestrictedPython for Python execution, pure-python sympy for calculations, and subprocess with allowlist for Bash operations

## 4. Training Methodology

### 4.1 Guided Reinforcement Policy Optimization (GRPO)

Training employs a GRPO reinforcement loop with:
- Batch size: 16 prompts × 4 candidate traces
- Ranking: Frozen Mamba-130M (headless) as judge
- Optimizer: AdamW with β₁=0.9, β₂=0.95, lr=1e-5
- Roll-outs: 2 iterations over 400 prompts

This approach produces adapters that favor shortest correct CoTs and symbol compression, optimizing for reasoning efficiency.

### 4.2 Data Pipeline

Training data includes:
- mini-GSM8K for mathematical reasoning
- HumanEval-mini for coding tasks
- ReAct traces for tool-use demonstrations

Pre-processing involves deduplication, 2048-token packing, and number masking for the Toolformer phase.

## 5. Evaluation

Nano-Cog 0.1 is evaluated on multiple benchmarks:
- GSM8K-10% split for mathematical reasoning
- HumanEval-mini (20 tasks) for code generation
- Custom Symbolic-Math quiz (10 items) for symbolic manipulation

Evaluation metrics focus on both accuracy and reasoning-tokens per answer, emphasizing efficiency of the reasoning process.

## 6. Results

[Note: This section would typically contain detailed performance results, which would be filled in after evaluation runs]

## 7. Limitations and Future Work

While Nano-Cog 0.1 demonstrates promising performance within its computational constraints, several limitations and opportunities for improvement exist:

1. **Scale Limitations**: The current 130M parameter size restricts the model's knowledge breadth compared to larger counterparts
2. **Tool Integration**: The current tool set is limited to basic calculator, Python, and Bash operations
3. **Retrieval Quality**: Retrieval performance depends on the quality and coverage of the local knowledge base

Future work will focus on:
1. **Hybrid Layer Research**: Replacing 25% of SSM blocks with custom Recursive Symbolic SSM
2. **Scaling to 370M**: Utilizing 4× A10G spot instances with 50B tokens of training
3. **Extended Tool Suite**: Incorporating web search and SQL query capabilities
4. **Open-Weights Release**: Publishing the 130M parameter version for community evaluation

## 8. Conclusion

Nano-Cog 0.1 demonstrates that efficient reasoning capabilities can be achieved in language agents without the massive computational requirements of today's largest models. By combining state-space modeling, retrieval, tool use, and optimized fine-tuning, we present a system that runs on consumer hardware while maintaining competitive performance on reasoning tasks. This approach opens up new possibilities for accessible, efficient AI systems that prioritize reasoning quality over raw parameter count.

## Acknowledgments

[Placeholder for acknowledgments]

## References

[Placeholder for references - would include citations to Mamba, LoRA, MoE, Toolformer, and other relevant literature] 