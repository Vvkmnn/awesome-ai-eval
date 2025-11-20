# Awesome AI Eval [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of tools, methods & platforms for evaluating AI quality in real applications.

<img src="./assets/robot-shades.svg" align="right" width="150" alt="Awesome AI Eval robot logo" />

A curated list of tools, frameworks, benchmarks, and observability platforms for evaluating LLMs, RAG pipelines, and autonomous agents to minimize hallucinations & evaluate practical performance in real production environments.

---

## Contents

- [Tools](#tools)
  - [Evaluators and Test Harnesses](#evaluators-and-test-harnesses)
  - [RAG and Retrieval](#rag-and-retrieval)
  - [Prompt Evaluation & Safety](#prompt-evaluation--safety)
  - [Datasets and Methodology](#datasets-and-methodology)
- [Platforms](#platforms)
  - [Open Source Platforms](#open-source-platforms)
  - [Hosted Platforms](#hosted-platforms)
  - [Cloud Platforms](#cloud-platforms)
- [Benchmarks](#benchmarks)
  - [General](#general)
  - [Domain](#domain)
  - [Agent](#agent)
  - [Safety](#safety)
- [Leaderboards](#leaderboards)
- [Resources](#resources)
  - [Guides & Training](#guides--training)
  - [Examples](#examples)
  - [Related Collections](#related-collections)
- [Licensing](#licensing)

---


## Tools

### Evaluators and Test Harnesses

#### Core Frameworks

- [ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) - Unified pipeline for classic metrics plus GPT-assisted scoring across public datasets.
- [DeepEval](https://github.com/confident-ai/deepeval) - Python unit-test style metrics for hallucination, relevance, toxicity, and bias.
- [Hugging Face lighteval](https://github.com/huggingface/lighteval) - Toolkit powering HF leaderboards with 1k+ tasks and pluggable metrics.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Safety Institute framework for scripted eval plans, tool calls, and model-graded rubrics.
- [MLflow Evaluators](https://github.com/mlflow/mlflow) - Eval API that logs LLM scores next to classic experiment tracking runs.
- [OpenAI Evals](https://github.com/openai/evals) - Reference harness plus registry spanning reasoning, extraction, and safety evals.
- [OpenCompass](https://github.com/open-compass/opencompass) - Research harness with CascadeEvaluator, CompassRank syncing, and LLM-as-judge utilities.
- [Prompt Flow](https://github.com/microsoft/promptflow) - Flow builder with built-in evaluation DAGs, dataset runners, and CI hooks.
- [Promptfoo](https://github.com/promptfoo/promptfoo) - Local-first CLI and dashboard for evaluating prompts, RAG flows, and agents.
- [Ragas](https://github.com/explodinggradients/ragas) - Evaluation library that grades answers, context, and grounding with pluggable scorers.
- [TruLens](https://github.com/truera/trulens) - Feedback function framework for chains and agents with customizable judge models.
- [W&B Weave Evaluations](https://wandb.ai/site/evaluations/) - Managed evaluation orchestrator with dataset versioning and dashboards.
- [ZenML](https://github.com/zenml-io/zenml) - Pipeline framework that bakes evaluation steps and guardrail metrics into LLM workflows.

#### Application and Agent Harnesses

- [Braintrust](https://www.braintrust.dev/) - Hosted evaluation workspace with CI-style regression tests and agent sandboxes.
- [LangSmith](https://smith.langchain.com/) - Hosted tracing plus datasets, batched evals, and regression gating for LangChain apps.
- [W&B Prompt Registry](https://docs.wandb.ai/weave/guides/core-types/evaluations) - Prompt evaluation templates with reproducible scoring and reviews.

### RAG and Retrieval

#### RAG Frameworks

- [EvalScope RAG](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) - Guides and templates that extend Ragas-style metrics with domain rubrics.
- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/stable/understanding/evaluating/index.html) - Modules for replaying queries, scoring retrievers, and comparing query engines.
- [Open RAG Eval](https://github.com/vectara/open-rag-eval) - Vectara harness with pluggable datasets for comparing retrievers and prompts.
- [RAGEval](https://github.com/OpenBMB/RAGEval) - Framework that auto-generates corpora, questions, and RAG rubrics for completeness.
- [R-Eval](https://github.com/THU-KEG/R-Eval) - Toolkit for robust RAG scoring aligned with the Evaluation of RAG survey taxonomy.

#### Retrieval Benchmarks

- [BEIR](https://github.com/beir-cellar/beir) - Benchmark suite covering dense, sparse, and hybrid retrieval tasks.
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) - Late-interaction dense retriever with evaluation scripts for IR datasets.
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Embeddings benchmark measuring retrieval, reranking, and similarity quality.

#### RAG Datasets and Surveys

- [Awesome-RAG-Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) - Curated catalog of RAG evaluation metrics, datasets, and leaderboards.
- [RAG Evaluation Survey](https://arxiv.org/abs/2405.07437) - Comprehensive paper covering metrics, judgments, and open problems for RAG.
- [RAGTruth](https://github.com/zhengzangw/RAGTruth) - Human-annotated dataset for measuring hallucinations and faithfulness in RAG answers.

### Prompt Evaluation & Safety

- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) - Automated instruction-following evaluator with length-controlled LLM judge scoring.
- [ChainForge](https://github.com/ianarawjo/ChainForge) - Visual IDE for comparing prompts, sampling models, and scoring batches with rubrics.
- [Guardrails AI](https://github.com/ShreyaR/guardrails) - Declarative validation framework that enforces schemas, correction chains, and judgments.
- [Lakera Guard](https://www.lakera.ai/guard) - Hosted prompt security platform with red-team datasets for jailbreak and injection testing.
- [PromptBench](https://github.com/microsoft/promptbench) - Benchmark suite for adversarial prompt stress tests across diverse tasks.
- [Red Teaming Handbook](https://learn.microsoft.com/en-us/security/) - Microsoft playbook for adversarial prompt testing and mitigation patterns.

### Datasets and Methodology

- [Deepchecks Evaluation Playbook](https://www.deepchecks.com/llm-evaluation/best-tools/) - Survey of evaluation metrics, failure modes, and platform comparisons.
- [HELM](https://crfm.stanford.edu/helm/latest/) - Holistic Evaluation of Language Models methodology emphasizing multi-criteria scoring.
- [Instruction-Following Evaluation (IFEval)](https://github.com/google-research/google-research/tree/master/instruction_following_eval) - Constraint-verification prompts for automatically checking instruction compliance.
- [OpenAI Cookbook Evals](https://github.com/openai/openai-cookbook/tree/main/examples/evals) - Practical notebooks showing how to build custom evals.
- [Safety Evaluation Guides](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/safety-evaluations) - Cloud vendor recipes for testing quality, safety, and risk.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) - EvalGen workflow aligning LLM judges with human rubrics via mixed-initiative criteria design.
- [ZenML Evaluation Playbook](https://www.zenml.io/blog/the-evaluation-playbook-making-llms-production-ready) - Playbook for embedding eval gates into pipelines and deployments.

---

## Platforms

### Open Source Platforms

- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - OpenTelemetry-native observability and evaluation toolkit for RAG, LLMs, and agents.
- [Langfuse](https://github.com/langfuse/langfuse) - Open-source LLM engineering platform providing tracing, eval dashboards, and prompt analytics.
- [OpenLIT](https://github.com/openlit/openlit) - Telemetry instrumentation for LLM apps with built-in quality metrics and guardrail hooks.
- [OpenLLMetry](https://github.com/traceloop/openllmetry) - OpenTelemetry instrumentation for LLM traces that feed any backend or custom eval logic.
- [Opik](https://github.com/comet-ml/opik) - Self-hostable evaluation and observability hub with datasets, scoring jobs, and interactive traces.
- [UpTrain](https://github.com/uptrain-ai/uptrain) - OSS/hosted evaluation suite with 20+ checks, RCA tooling, and LlamaIndex integrations.
- [VoltAgent](https://github.com/VoltAgent/voltagent) - TypeScript agent framework paired with VoltOps for trace inspection and regression testing.
- [Zeno](https://zenoml.com/) - Data-centric evaluation UI for slicing failures, comparing prompts, and debugging retrieval quality.
- [traceAI](https://github.com/future-agi/traceAI) - Open-source multi-modal tracing and diagnostics framework for LLM, RAG, and agent workflows built on OpenTelemetry.

### Hosted Platforms

- [Confident AI](https://www.confident-ai.com/) - DeepEval-backed platform for scheduled eval suites, guardrails, and production monitors.
- [Datadog AI Observability](https://www.datadoghq.com/product/ai-observability/) - Datadog module capturing LLM traces, metrics, and safety signals.
- [Deepchecks LLM Evaluation](https://www.deepchecks.com/solutions/llm-evaluation/) - Managed eval suites with dataset versioning, dashboards, and alerting.
- [Galileo GenAI](https://www.rungalileo.io/product/genai-studio) - Evaluation and data-curation studio with labeling, slicing, and issue triage.
- [Humanloop](https://humanloop.com/) - Production prompt management with human-in-the-loop evals and annotation queues.
- [Maxim AI](https://www.getmaxim.ai/) - Evaluation and observability platform focusing on agent simulations and monitoring.
- [PostHog LLM Observability](https://posthog.com/blog/llm-observability) - Product analytics toolkit extended to track custom LLM events and metrics.
- [Future AGI](https://futureagi.com/) - Multi-modal evaluation, simulation, and optimization platform for reliable AI systems across software and hardware.


### Cloud Platforms

- [Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/) - Managed service for scoring foundation models and RAG pipelines.
- [Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) - Safety layer that evaluates prompts and responses for policy compliance.
- [Azure AI Foundry Evaluations](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluate-generative-ai-app) - Evaluation flows and risk reports wired into Prompt Flow projects.
- [Vertex AI Generative AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) - Adaptive rubric-based evaluation for Google and third-party models.

---

## Benchmarks

### General

- [AGIEval](https://github.com/ruixiangcui/AGIEval) - Human-centric standardized exams spanning entrance tests, legal, and math scenarios.
- [BIG-bench](https://github.com/google/BIG-bench) - Collaborative benchmark probing reasoning, commonsense, and long-tail tasks.
- [CommonGen-Eval](https://github.com/allenai/CommonGen-Eval) - GPT-4 judged CommonGen-lite suite for constrained commonsense text generation.
- [DyVal](https://arxiv.org/abs/2309.17167) - Dynamic reasoning benchmark that varies difficulty and graph structure to stress models.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Standard harness for scoring autoregressive models on dozens of tasks.
- [LLM-Uncertainty-Bench](https://github.com/smartyfh/LLM-Uncertainty-Bench) - Adds uncertainty-aware scoring across QA, RC, inference, dialog, and summarization.
- [LLMBar](https://github.com/princeton-nlp/LLMBar) - Meta-eval testing whether LLM judges can spot instruction-following failures.
- [LV-Eval](https://github.com/infinigence/LVEval) - Long-context suite with five length tiers up to 256K tokens and distraction controls.
- [MMLU](https://github.com/hendrycks/test) - Massive multitask language understanding benchmark for academic and professional subjects.
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) - Harder 10-choice extension focused on reasoning-rich, low-leakage questions.
- [PertEval](https://github.com/aigc-apps/PertEval) - Knowledge-invariant perturbations to debias multiple-choice accuracy inflation.

### Domain

- [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval) - Chinese financial QA and reasoning benchmark across regulation, accounting, and markets.
- [LAiW](https://github.com/Dai-shen/LAiW) - Legal benchmark covering retrieval, foundation inference, and complex case applications in Chinese law.
- [HumanEval](https://github.com/openai/human-eval) - Unit-test-based benchmark for code synthesis and docstring reasoning.
- [MATH](https://github.com/hendrycks/math) - Competition-level math benchmark targeting multi-step symbolic reasoning.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Mostly Basic Programming Problems benchmark for small coding tasks.

### Agent

- [AgentBench](https://github.com/THUDM/AgentBench) - Evaluates LLMs acting as agents across simulated domains like games and coding.
- [GAIA](https://github.com/GAIA-benchmark/GAIA) - Tool-use benchmark requiring grounded reasoning with live web access and planning.
- [MetaTool Tasks](https://github.com/meta-llama/MetaTool) - Tool-calling benchmark and eval harness for agents built around LLaMA models.
- [SuperCLUE-Agent](https://github.com/CLUEbenchmark/SuperCLUE-Agent) - Chinese agent eval covering tool use, planning, long/short-term memory, and APIs.

### Safety

- [AdvBench](https://github.com/centerforaisafety/advbench) - Adversarial prompt benchmark for jailbreak and misuse resistance measurement.
- [BBQ](https://github.com/nyu-mll/BBQ) - Bias-sensitive QA sets measuring stereotype reliance and ambiguous cases.
- [ToxiGen](https://github.com/microsoft/ToxiGen) - Toxic language generation and classification benchmark for robustness checks.
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Measures factuality and hallucination propensity via adversarially written questions.

---

## Leaderboards

- [CompassRank](https://rank.opencompass.org.cn/home) - OpenCompass leaderboard comparing frontier and research models across multi-domain suites.
- [LLM Agents Benchmark Collections](https://llmbench.ai/) - Aggregated leaderboard comparing multi-agent safety and reliability suites.
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - Hugging Face benchmark board with IFEval, MMLU-Pro, GPQA, and more.
- [OpenAI Evals Registry](https://github.com/openai/evals/tree/main/evals/elsuite) - Community suites and scores covering accuracy, safety, and instruction following.
- [Scale SEAL Leaderboard](https://scale.com/leaderboard) - Expert-rated leaderboard covering reasoning, coding, and safety via SEAL evaluations.

---

## Resources

### Guides & Training

- [AI Evals for Engineers & PMs](https://maven.com/parlance-labs/evals?promoCode=FAST25) - Cohort course from Hamel & Shreya with lifetime reader, Discord, AI Eval Assistant, and live office hours.
- [Why AI evals are the hottest new skill](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill) - Lenny's interview covering error analysis, axial coding, eval prompts, and PRD alignment.
- [Error Analysis & Prioritizing Next Steps](https://www.youtube.com/watch?v=bWkQk5_OG8k) - Andrew Ng walkthrough showing how to slice traces and focus eval work via classic ML techniques.

### Examples

- [Arize Phoenix AI Chatbot](https://github.com/Arize-ai/phoenix-ai-chatbot) - Next.js chatbot with Phoenix tracing, dataset replays, and evaluation jobs.
- [Azure LLM Evaluation Samples](https://github.com/Azure-Samples/llm-evaluation) - Prompt Flow and Azure AI Foundry projects demonstrating hosted evals.
- [Deepchecks QA over CSV](https://github.com/deepchecks/qa-over-csv) - Example agent wired to Deepchecks scoring plus tracing dashboards.
- [OpenAI Evals Demo Evals](https://github.com/withmartian/demo-evals) - Templates for extending OpenAI Evals with custom datasets.
- [Promptfoo Examples](https://github.com/promptfoo/promptfoo/tree/main/examples) - Ready-made prompt regression suites for RAG, summarization, and agents.
- [ZenML Projects](https://github.com/zenml-io/zenml-projects) - End-to-end pipelines showing how to weave evaluation steps into LLMOps stacks.

### Related Collections

- [Awesome ChainForge](https://github.com/loloMD/awesome_chainforge) - Ecosystem list centered on ChainForge experiments and extensions.
- [Awesome-LLM-Eval](https://github.com/onejune2018/Awesome-LLM-Eval) - Cross-lingual (Chinese) compendium of eval tooling, papers, datasets, and leaderboards.
- [Awesome LLMOps](https://github.com/tensorchord/awesome-llmops) - Curated tooling for training, deployment, and monitoring of LLM apps.
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) - Language-specific ML resources that often host evaluation building blocks.
- [Awesome RAG](https://github.com/noworneverev/Awesome-RAG) - Broad coverage of retrieval-augmented generation techniques and tools.
- [Awesome Self-Hosted](https://github.com/awesome-selfhosted/awesome-selfhosted) - Massive catalog of self-hostable software, including observability stacks.
- [GenAI Notes](https://github.com/eugeneyan/genai-notes) - Continuously updated notes and resources on GenAI systems, evaluation, and operations.

---

## Licensing

Released under the [CC0 1.0 Universal](LICENSE) license.

---

## Contributing

Contributions are welcome—please read [CONTRIBUTING.md](CONTRIBUTING.md) for scope, entry rules, and the pull-request checklist before submitting updates.

[✌️](https://www.vvkmnn.xyz)
