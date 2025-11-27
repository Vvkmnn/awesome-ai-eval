# Awesome AI Eval [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of tools, methods & platforms for evaluating AI quality in real applications.

<img src="./assets/robot-shades.svg" align="right" width="150" alt="Awesome AI Eval robot logo" />

A curated list of tools, frameworks, benchmarks, and observability platforms for evaluating LLMs, RAG pipelines, and autonomous agents to minimize hallucinations & evaluate practical performance in real production environments.

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

- [**Anthropic Model Evals**](https://github.com/anthropics/evals) ![](https://img.shields.io/github/stars/anthropics/evals?style=social&label=github.com) - Anthropic's evaluation suite for safety, capabilities, and alignment testing of language models.
- [**ColossalEval**](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) ![](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social&label=github.com) - Unified pipeline for classic metrics plus GPT-assisted scoring across public datasets.
- [**DeepEval**](https://github.com/confident-ai/deepeval) ![](https://img.shields.io/github/stars/confident-ai/deepeval?style=social&label=github.com) - Python unit-test style metrics for hallucination, relevance, toxicity, and bias.
- [**Hugging Face lighteval**](https://github.com/huggingface/lighteval) ![](https://img.shields.io/github/stars/huggingface/lighteval?style=social&label=github.com) - Toolkit powering HF leaderboards with 1k+ tasks and pluggable metrics.
- [**Inspect AI**](https://github.com/UKGovernmentBEIS/inspect_ai) ![](https://img.shields.io/github/stars/UKGovernmentBEIS/inspect_ai?style=social&label=github.com) - UK AI Safety Institute framework for scripted eval plans, tool calls, and model-graded rubrics.
- [**MLflow Evaluators**](https://github.com/mlflow/mlflow) ![](https://img.shields.io/github/stars/mlflow/mlflow?style=social&label=github.com) - Eval API that logs LLM scores next to classic experiment tracking runs.
- [**OpenAI Evals**](https://github.com/openai/evals) ![](https://img.shields.io/github/stars/openai/evals?style=social&label=github.com) - Reference harness plus registry spanning reasoning, extraction, and safety evals.
- [**OpenCompass**](https://github.com/open-compass/opencompass) ![](https://img.shields.io/github/stars/open-compass/opencompass?style=social&label=github.com) - Research harness with CascadeEvaluator, CompassRank syncing, and LLM-as-judge utilities.
- [**Prompt Flow**](https://github.com/microsoft/promptflow) ![](https://img.shields.io/github/stars/microsoft/promptflow?style=social&label=github.com) - Flow builder with built-in evaluation DAGs, dataset runners, and CI hooks.
- [**Promptfoo**](https://github.com/promptfoo/promptfoo) ![](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social&label=github.com) - Local-first CLI and dashboard for evaluating prompts, RAG flows, and agents with cost tracking and regression detection.
- [**Ragas**](https://github.com/explodinggradients/ragas) ![](https://img.shields.io/github/stars/explodinggradients/ragas?style=social&label=github.com) - Evaluation library that grades answers, context, and grounding with pluggable scorers.
- [**TruLens**](https://github.com/truera/trulens) ![](https://img.shields.io/github/stars/truera/trulens?style=social&label=github.com) - Feedback function framework for chains and agents with customizable judge models.
- [**W&B Weave Evaluations**](https://wandb.ai/site/evaluations/) ![](https://img.shields.io/badge/wandb.ai-active-blue?style=social) - Managed evaluation orchestrator with dataset versioning and dashboards.
- [**ZenML**](https://github.com/zenml-io/zenml) ![](https://img.shields.io/github/stars/zenml-io/zenml?style=social&label=github.com) - Pipeline framework that bakes evaluation steps and guardrail metrics into LLM workflows.

#### Application and Agent Harnesses

- [**Braintrust**](https://www.braintrust.dev/) ![](https://img.shields.io/badge/braintrust.dev-active-blue?style=social) - Hosted evaluation workspace with CI-style regression tests, agent sandboxes, and token cost tracking.
- [**LangSmith**](https://smith.langchain.com/) ![](https://img.shields.io/badge/smith.langchain.com-active-blue?style=social) - Hosted tracing plus datasets, batched evals, and regression gating for LangChain apps.
- [**W&B Prompt Registry**](https://docs.wandb.ai/weave/guides/core-types/evaluations) ![](https://img.shields.io/badge/docs.wandb.ai-active-blue?style=social) - Prompt evaluation templates with reproducible scoring and reviews.

### RAG and Retrieval

#### RAG Frameworks

- [**EvalScope RAG**](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) ![](https://img.shields.io/badge/evalscope.readthedocs.io-active-blue?style=social) - Guides and templates that extend Ragas-style metrics with domain rubrics.
- [**LlamaIndex Evaluation**](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) ![](https://img.shields.io/badge/docs.llamaindex.ai-active-blue?style=social) - Modules for replaying queries, scoring retrievers, and comparing query engines.
- [**Open RAG Eval**](https://github.com/vectara/open-rag-eval) ![](https://img.shields.io/github/stars/vectara/open-rag-eval?style=social&label=github.com) - Vectara harness with pluggable datasets for comparing retrievers and prompts.
- [**RAGEval**](https://github.com/OpenBMB/RAGEval) ![](https://img.shields.io/github/stars/OpenBMB/RAGEval?style=social&label=github.com) - Framework that auto-generates corpora, questions, and RAG rubrics for completeness.
- [**R-Eval**](https://github.com/THU-KEG/R-Eval) ![](https://img.shields.io/github/stars/THU-KEG/R-Eval?style=social&label=github.com) - Toolkit for robust RAG scoring aligned with the Evaluation of RAG survey taxonomy.

#### Retrieval Benchmarks

- [**BEIR**](https://github.com/beir-cellar/beir) ![](https://img.shields.io/github/stars/beir-cellar/beir?style=social&label=github.com) - Benchmark suite covering dense, sparse, and hybrid retrieval tasks.
- [**ColBERT**](https://github.com/stanford-futuredata/ColBERT) ![](https://img.shields.io/github/stars/stanford-futuredata/ColBERT?style=social&label=github.com) - Late-interaction dense retriever with evaluation scripts for IR datasets.
- [**MTEB**](https://github.com/embeddings-benchmark/mteb) ![](https://img.shields.io/github/stars/embeddings-benchmark/mteb?style=social&label=github.com) - Embeddings benchmark measuring retrieval, reranking, and similarity quality.

#### RAG Datasets and Surveys

- [**Awesome-RAG-Evaluation**](https://github.com/YHPeter/Awesome-RAG-Evaluation) ![](https://img.shields.io/github/stars/YHPeter/Awesome-RAG-Evaluation?style=social&label=github.com) - Curated catalog of RAG evaluation metrics, datasets, and leaderboards.
- [**Comparing LLMs on Real-World Retrieval**](https://www.sh-reya.com/blog/needle-in-the-real-world/) ![](https://img.shields.io/badge/sh--reya.com-active-blue?style=social) - Empirical analysis of how language models perform on practical retrieval tasks.
- [**RAG Evaluation Survey**](https://arxiv.org/abs/2405.07437) ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) - Comprehensive paper covering metrics, judgments, and open problems for RAG.
- [**RAGTruth**](https://github.com/zhengzangw/RAGTruth) ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) - Human-annotated dataset for measuring hallucinations and faithfulness in RAG answers.

### Prompt Evaluation & Safety

- [**AlpacaEval**](https://github.com/tatsu-lab/alpaca_eval) ![](https://img.shields.io/github/stars/tatsu-lab/alpaca_eval?style=social&label=github.com) - Automated instruction-following evaluator with length-controlled LLM judge scoring.
- [**ChainForge**](https://github.com/ianarawjo/ChainForge) ![](https://img.shields.io/github/stars/ianarawjo/ChainForge?style=social&label=github.com) - Visual IDE for comparing prompts, sampling models, and scoring batches with rubrics.
- [**Guardrails AI**](https://github.com/ShreyaR/guardrails) ![](https://img.shields.io/github/stars/ShreyaR/guardrails?style=social&label=github.com) - Declarative validation framework that enforces schemas, correction chains, and judgments.
- [**Lakera Guard**](https://www.lakera.ai/lakera-guard) ![](https://img.shields.io/badge/lakera.ai-active-blue?style=social) - Hosted prompt security platform with red-team datasets for jailbreak and injection testing.
- [**PromptBench**](https://github.com/microsoft/promptbench) ![](https://img.shields.io/github/stars/microsoft/promptbench?style=social&label=github.com) - Benchmark suite for adversarial prompt stress tests across diverse tasks.
- [**Red Teaming Handbook**](https://learn.microsoft.com/en-us/security/) ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) - Microsoft playbook for adversarial prompt testing and mitigation patterns.

### Datasets and Methodology

- [**Deepchecks Evaluation Playbook**](https://www.deepchecks.com/llm-evaluation/best-tools/) ![](https://img.shields.io/badge/deepchecks.com-active-blue?style=social) - Survey of evaluation metrics, failure modes, and platform comparisons.
- [**HELM**](https://crfm.stanford.edu/helm/latest/) ![](https://img.shields.io/badge/crfm.stanford.edu-active-blue?style=social) - Holistic Evaluation of Language Models methodology emphasizing multi-criteria scoring.
- [**Instruction-Following Evaluation (IFEval)**](https://github.com/google-research/google-research/tree/master/instruction_following_eval) ![](https://img.shields.io/github/stars/google-research/google-research?style=social&label=github.com) - Constraint-verification prompts for automatically checking instruction compliance.
- [**OpenAI Cookbook Evals**](https://github.com/openai/openai-cookbook/tree/main/examples/evals) ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) - Practical notebooks showing how to build custom evals.
- [**Safety Evaluation Guides**](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/safety-evaluations-transparency-note) ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) - Cloud vendor recipes for testing quality, safety, and risk.
- [**Who Validates the Validators?**](https://arxiv.org/abs/2404.12272) ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) - EvalGen workflow aligning LLM judges with human rubrics via mixed-initiative criteria design.
- [**ZenML Evaluation Playbook**](https://www.zenml.io/blog/the-evaluation-playbook-making-llms-production-ready) ![](https://img.shields.io/badge/zenml.io-active-blue?style=social) - Playbook for embedding eval gates into pipelines and deployments.

---

## Platforms

### Open Source Platforms

- [**Agenta**](https://github.com/Agenta-AI/agenta) ![](https://img.shields.io/github/stars/Agenta-AI/agenta?style=social&label=github.com) - End-to-end LLM developer platform for prompt engineering, evaluation, and deployment.
- [**Arize Phoenix**](https://github.com/Arize-ai/phoenix) ![](https://img.shields.io/github/stars/Arize-ai/phoenix?style=social&label=github.com) - OpenTelemetry-native observability and evaluation toolkit for RAG, LLMs, and agents.
- [**DocETL**](https://github.com/ucbepic/docetl) ![](https://img.shields.io/github/stars/ucbepic/docetl?style=social&label=github.com) - ETL system for complex document processing with LLMs and built-in quality checks.
- [**Giskard**](https://github.com/Giskard-AI/giskard) ![](https://img.shields.io/github/stars/Giskard-AI/giskard?style=social&label=github.com) - Testing framework for ML models with vulnerability scanning and LLM-specific detectors.
- [**Helicone**](https://github.com/Helicone/helicone) ![](https://img.shields.io/github/stars/Helicone/helicone?style=social&label=github.com) - Open-source LLM observability platform with cost tracking, caching, and evaluation tools.
- [**Langfuse**](https://github.com/langfuse/langfuse) ![](https://img.shields.io/github/stars/langfuse/langfuse?style=social&label=github.com) - Open-source LLM engineering platform providing tracing, eval dashboards, and prompt analytics.
- [**Lilac**](https://github.com/lilacai/lilac) ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) - Data curation tool for exploring and enriching datasets with semantic search and clustering.
- [**LiteLLM**](https://github.com/BerriAI/litellm) ![](https://img.shields.io/github/stars/BerriAI/litellm?style=social&label=github.com) - Unified API for 100+ LLM providers with cost tracking, fallbacks, and load balancing.
- [**Lunary**](https://github.com/lunary-ai/lunary) ![](https://img.shields.io/github/stars/lunary-ai/lunary?style=social&label=github.com) - Production toolkit for LLM apps with tracing, prompt management, and evaluation pipelines.
- [**Mirascope**](https://github.com/mirascope/mirascope) ![](https://img.shields.io/github/stars/mirascope/mirascope?style=social&label=github.com) - Python toolkit for building LLM applications with structured outputs and evaluation utilities.
- [**OpenLIT**](https://github.com/openlit/openlit) ![](https://img.shields.io/github/stars/openlit/openlit?style=social&label=github.com) - Telemetry instrumentation for LLM apps with built-in quality metrics and guardrail hooks.
- [**OpenLLMetry**](https://github.com/traceloop/openllmetry) ![](https://img.shields.io/github/stars/traceloop/openllmetry?style=social&label=github.com) - OpenTelemetry instrumentation for LLM traces that feed any backend or custom eval logic.
- [**Opik**](https://github.com/comet-ml/opik) ![](https://img.shields.io/github/stars/comet-ml/opik?style=social&label=github.com) - Self-hostable evaluation and observability hub with datasets, scoring jobs, and interactive traces.
- [**Rhesis**](https://github.com/rhesis-ai/rhesis) ![](https://img.shields.io/github/stars/rhesis-ai/rhesis?style=social&label=github.com) - Collaborative testing platform with automated test generation and multi-turn conversation simulation for LLM and agentic applications.
- [**traceAI**](https://github.com/future-agi/traceAI) ![](https://img.shields.io/github/stars/future-agi/traceAI?style=social&label=github.com) - Open-source multi-modal tracing and diagnostics framework for LLM, RAG, and agent workflows built on OpenTelemetry.
- [**UpTrain**](https://github.com/uptrain-ai/uptrain) ![](https://img.shields.io/github/stars/uptrain-ai/uptrain?style=social&label=github.com) - OSS/hosted evaluation suite with 20+ checks, RCA tooling, and LlamaIndex integrations.
- [**VoltAgent**](https://github.com/VoltAgent/voltagent) ![](https://img.shields.io/github/stars/VoltAgent/voltagent?style=social&label=github.com) - TypeScript agent framework paired with VoltOps for trace inspection and regression testing.
- [**Zeno**](https://zenoml.com/) ![](https://img.shields.io/badge/zenoml.com-active-blue?style=social) - Data-centric evaluation UI for slicing failures, comparing prompts, and debugging retrieval quality.

  
### Hosted Platforms

- [**ChatIntel**](https://chatintel.ai/) ![](https://img.shields.io/badge/chatintel.ai-active-blue?style=social) - Conversation analytics platform for evaluating chatbot quality, sentiment, and user satisfaction.
- [**Confident AI**](https://www.confident-ai.com/) ![](https://img.shields.io/badge/confident--ai.com-active-blue?style=social) - DeepEval-backed platform for scheduled eval suites, guardrails, and production monitors.
- [**Datadog LLM Observability**](https://www.datadoghq.com/product/llm-observability/) ![](https://img.shields.io/badge/datadoghq.com-active-blue?style=social) - Datadog module capturing LLM traces, metrics, and safety signals.
- [**Deepchecks LLM Evaluation**](https://www.deepchecks.com/solutions/llm-evaluation/) ![](https://img.shields.io/badge/deepchecks.com-active-blue?style=social) - Managed eval suites with dataset versioning, dashboards, and alerting.
- [**Eppo**](https://www.geteppo.com/) ![](https://img.shields.io/badge/geteppo.com-active-blue?style=social) - Experimentation platform with AI-specific evaluation metrics and statistical rigor for LLM A/B testing.
- [**Future AGI**](https://futureagi.com/) ![](https://img.shields.io/badge/futureagi.com-active-blue?style=social) - Multi-modal evaluation, simulation, and optimization platform for reliable AI systems across software and hardware.
- [**Galileo**](https://www.galileo.ai/) ![](https://img.shields.io/badge/galileo.ai-active-blue?style=social) - Evaluation and data-curation studio with labeling, slicing, and issue triage.
- [**HoneyHive**](https://www.honeyhive.ai/) ![](https://img.shields.io/badge/honeyhive.ai-active-blue?style=social) - Evaluation and observability platform with prompt versioning, A/B testing, and fine-tuning workflows.
- [**Humanloop**](https://humanloop.com/) ![](https://img.shields.io/badge/humanloop.com-active-blue?style=social) - Production prompt management with human-in-the-loop evals and annotation queues.
- [**Maxim AI**](https://www.getmaxim.ai/) ![](https://img.shields.io/badge/getmaxim.ai-active-blue?style=social) - Evaluation and observability platform focusing on agent simulations and monitoring.
- [**Orq.ai**](https://orq.ai/) ![](https://img.shields.io/badge/orq.ai-active-blue?style=social) - LLM operations platform with prompt management, evaluation workflows, and deployment pipelines.
- [**PostHog LLM Analytics**](https://posthog.com/llm-analytics) ![](https://img.shields.io/badge/posthog.com-active-blue?style=social) - Product analytics toolkit extended to track custom LLM events and metrics.
- [**PromptLayer**](https://www.promptlayer.com/) ![](https://img.shields.io/badge/promptlayer.com-active-blue?style=social) - Prompt engineering platform with version control, evaluation tracking, and team collaboration.


### Cloud Platforms

- [**Amazon Bedrock Evaluations**](https://aws.amazon.com/bedrock/evaluations/) ![](https://img.shields.io/badge/aws.amazon.com-active-blue?style=social) - Managed service for scoring foundation models and RAG pipelines.
- [**Amazon Bedrock Guardrails**](https://aws.amazon.com/bedrock/guardrails/) ![](https://img.shields.io/badge/aws.amazon.com-active-blue?style=social) - Safety layer that evaluates prompts and responses for policy compliance.
- [**Azure AI Foundry Evaluations**](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluate-generative-ai-app) ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) - Evaluation flows and risk reports wired into Prompt Flow projects.
- [**Vertex AI Generative AI Evaluation**](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) ![](https://img.shields.io/badge/cloud.google.com-active-blue?style=social) - Adaptive rubric-based evaluation for Google and third-party models.

---

## Benchmarks

### General

- [**AGIEval**](https://github.com/ruixiangcui/AGIEval) ![](https://img.shields.io/github/stars/ruixiangcui/AGIEval?style=social&label=github.com) - Human-centric standardized exams spanning entrance tests, legal, and math scenarios.
- [**BIG-bench**](https://github.com/google/BIG-bench) ![](https://img.shields.io/github/stars/google/BIG-bench?style=social&label=github.com) - Collaborative benchmark probing reasoning, commonsense, and long-tail tasks.
- [**CommonGen-Eval**](https://github.com/allenai/CommonGen-Eval) ![](https://img.shields.io/github/stars/allenai/CommonGen-Eval?style=social&label=github.com) - GPT-4 judged CommonGen-lite suite for constrained commonsense text generation.
- [**DyVal**](https://arxiv.org/abs/2309.17167) ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) - Dynamic reasoning benchmark that varies difficulty and graph structure to stress models.
- [**LM Evaluation Harness**](https://github.com/EleutherAI/lm-evaluation-harness) ![](https://img.shields.io/github/stars/EleutherAI/lm-evaluation-harness?style=social&label=github.com) - Standard harness for scoring autoregressive models on dozens of tasks.
- [**LLM-Uncertainty-Bench**](https://github.com/smartyfh/LLM-Uncertainty-Bench) ![](https://img.shields.io/github/stars/smartyfh/LLM-Uncertainty-Bench?style=social&label=github.com) - Adds uncertainty-aware scoring across QA, RC, inference, dialog, and summarization.
- [**LLMBar**](https://github.com/princeton-nlp/LLMBar) ![](https://img.shields.io/github/stars/princeton-nlp/LLMBar?style=social&label=github.com) - Meta-eval testing whether LLM judges can spot instruction-following failures.
- [**LV-Eval**](https://github.com/infinigence/LVEval) ![](https://img.shields.io/github/stars/infinigence/LVEval?style=social&label=github.com) - Long-context suite with five length tiers up to 256K tokens and distraction controls.
- [**MMLU**](https://github.com/hendrycks/test) ![](https://img.shields.io/github/stars/hendrycks/test?style=social&label=github.com) - Massive multitask language understanding benchmark for academic and professional subjects.
- [**MMLU-Pro**](https://github.com/TIGER-AI-Lab/MMLU-Pro) ![](https://img.shields.io/github/stars/TIGER-AI-Lab/MMLU-Pro?style=social&label=github.com) - Harder 10-choice extension focused on reasoning-rich, low-leakage questions.
- [**PertEval**](https://github.com/aigc-apps/PertEval) ![](https://img.shields.io/github/stars/aigc-apps/PertEval?style=social&label=github.com) - Knowledge-invariant perturbations to debias multiple-choice accuracy inflation.

### Domain

- [**FinEval**](https://github.com/SUFE-AIFLM-Lab/FinEval) ![](https://img.shields.io/github/stars/SUFE-AIFLM-Lab/FinEval?style=social&label=github.com) - Chinese financial QA and reasoning benchmark across regulation, accounting, and markets.
- [**LAiW**](https://github.com/Dai-shen/LAiW) ![](https://img.shields.io/github/stars/Dai-shen/LAiW?style=social&label=github.com) - Legal benchmark covering retrieval, foundation inference, and complex case applications in Chinese law.
- [**HumanEval**](https://github.com/openai/human-eval) ![](https://img.shields.io/github/stars/openai/human-eval?style=social&label=github.com) - Unit-test-based benchmark for code synthesis and docstring reasoning.
- [**MATH**](https://github.com/hendrycks/math) ![](https://img.shields.io/github/stars/hendrycks/math?style=social&label=github.com) - Competition-level math benchmark targeting multi-step symbolic reasoning.
- [**MBPP**](https://github.com/google-research/google-research/tree/master/mbpp) ![](https://img.shields.io/github/stars/google-research/google-research?style=social&label=github.com) - Mostly Basic Programming Problems benchmark for small coding tasks.

### Agent

- [**AgentBench**](https://github.com/THUDM/AgentBench) ![](https://img.shields.io/github/stars/THUDM/AgentBench?style=social&label=github.com) - Evaluates LLMs acting as agents across simulated domains like games and coding.
- [**GAIA**](https://huggingface.co/datasets/gaia-benchmark/GAIA) ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) - Tool-use benchmark requiring grounded reasoning with live web access and planning.
- [**MetaTool Tasks**](https://github.com/meta-llama/MetaTool) ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) - Tool-calling benchmark and eval harness for agents built around LLaMA models.
- [**SuperCLUE-Agent**](https://github.com/CLUEbenchmark/SuperCLUE-Agent) ![](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUE-Agent?style=social&label=github.com) - Chinese agent eval covering tool use, planning, long/short-term memory, and APIs.

### Safety

- [**AdvBench**](https://github.com/llm-attacks/llm-attacks) ![](https://img.shields.io/github/stars/llm-attacks/llm-attacks?style=social&label=github.com) - Adversarial prompt benchmark for jailbreak and misuse resistance measurement.
- [**BBQ**](https://github.com/nyu-mll/BBQ) ![](https://img.shields.io/github/stars/nyu-mll/BBQ?style=social&label=github.com) - Bias-sensitive QA sets measuring stereotype reliance and ambiguous cases.
- [**ToxiGen**](https://github.com/microsoft/ToxiGen) ![](https://img.shields.io/github/stars/microsoft/ToxiGen?style=social&label=github.com) - Toxic language generation and classification benchmark for robustness checks.
- [**TruthfulQA**](https://github.com/sylinrl/TruthfulQA) ![](https://img.shields.io/github/stars/sylinrl/TruthfulQA?style=social&label=github.com) - Measures factuality and hallucination propensity via adversarially written questions.

---

## Leaderboards

- [**CompassRank**](https://rank.opencompass.org.cn/home) ![](https://img.shields.io/badge/rank.opencompass.org.cn-active-blue?style=social) - OpenCompass leaderboard comparing frontier and research models across multi-domain suites.
- [**LLM Agents Benchmark Collections**](https://llmbench.ai/) ![](https://img.shields.io/badge/llmbench.ai-active-blue?style=social) - Aggregated leaderboard comparing multi-agent safety and reliability suites.
- [**Open LLM Leaderboard**](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) - Hugging Face benchmark board with IFEval, MMLU-Pro, GPQA, and more.
- [**OpenAI Evals Registry**](https://github.com/openai/evals/tree/main/evals/elsuite) ![](https://img.shields.io/github/stars/openai/evals?style=social&label=github.com) - Community suites and scores covering accuracy, safety, and instruction following.
- [**Scale SEAL Leaderboard**](https://scale.com/leaderboard) ![](https://img.shields.io/badge/scale.com-active-blue?style=social) - Expert-rated leaderboard covering reasoning, coding, and safety via SEAL evaluations.

---

## Resources

### Guides & Training

- [**AI Evals for Engineers & PMs**](https://maven.com/parlance-labs/evals?promoCode=FAST25) ![](https://img.shields.io/badge/maven.com-active-blue?style=social) - Cohort course from Hamel & Shreya with lifetime reader, Discord, AI Eval Assistant, and live office hours.
- [**AlignEval**](https://eugeneyan.com/writing/aligneval/) ![](https://img.shields.io/badge/eugeneyan.com-active-blue?style=social) - Eugene Yan's guide on building LLM judges by following methodical alignment processes.
- [**Applied LLMs**](https://applied-llms.org/) ![](https://img.shields.io/badge/applied--llms.org-active-blue?style=social) - Practical lessons from a year of building with LLMs, emphasizing evaluation as a core practice.
- [**Data Flywheels for LLM Applications**](https://www.sh-reya.com/blog/ai-engineering-flywheel/) ![](https://img.shields.io/badge/sh--reya.com-active-blue?style=social) - Iterative data improvement processes for building better LLM systems.
- [**Error Analysis & Prioritizing Next Steps**](https://www.youtube.com/watch?v=bWkQk5_OG8k) ![](https://img.shields.io/badge/youtube.com-active-blue?style=social) - Andrew Ng walkthrough showing how to slice traces and focus eval work via classic ML techniques.
- [**Error Analysis Before Tests**](https://hamel.dev/notes/llm/officehours/erroranalysis.html) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Office hours notes on why error analysis should precede writing automated tests.
- [**Eval Tools Comparison**](https://hamel.dev/blog/posts/eval-tools/) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Detailed comparison of evaluation tools including Braintrust, LangSmith, and Promptfoo.
- [**Evals for AI Engineers**](https://www.oreilly.com/library/view/evals-for-ai/9798341660717/) ![](https://img.shields.io/badge/oreilly.com-active-blue?style=social) - O'Reilly book by Shreya Shankar & Hamel Husain on systematic error analysis, evaluation pipelines, and LLM-as-a-judge.
- [**Evaluating RAG Systems**](https://hamel.dev/blog/posts/evals-faq/#how-should-i-approach-evaluating-my-rag-system) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Practical guidance on RAG evaluation covering retrieval quality and generation assessment.
- [**Field Guide to Rapidly Improving AI Products**](https://hamel.dev/blog/posts/field-guide/) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Comprehensive guide on error analysis, data viewers, and systematic improvement from 30+ implementations.
- [**Inspect AI Deep Dive**](https://hamel.dev/notes/llm/evals/inspect.html) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Technical deep dive into Inspect AI framework with hands-on examples.
- [**LLM Evals FAQ**](https://hamel.dev/blog/posts/evals-faq/) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Comprehensive FAQ with 45+ articles covering evaluation questions from practitioners.
- [**LLM Evaluators Survey**](https://eugeneyan.com/writing/llm-evaluators/) ![](https://img.shields.io/badge/eugeneyan.com-active-blue?style=social) - Survey of LLM-as-judge use cases and approaches with practical implementation patterns.
- [**LLM-as-a-Judge Guide**](https://hamel.dev/blog/posts/llm-judge/) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - In-depth guide on using LLMs as judges for automated evaluation with calibration tips.
- [**Mastering LLMs Open Course**](https://parlance-labs.com/education/) ![](https://img.shields.io/badge/parlance--labs.com-active-blue?style=social) - Free 40+ hour course covering evals, RAG, and fine-tuning taught by 25+ industry practitioners.
- [**Modern IR Evals For RAG**](https://hamel.dev/notes/llm/rag/p2-evals.html) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Why traditional IR evals are insufficient for RAG, covering BEIR and modern approaches.
- [**Multi-Turn Chat Evals**](https://hamel.dev/notes/llm/officehours/evalmultiturn.html) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Strategies for evaluating multi-turn conversational AI systems.
- [**Open Source LLM Tools Comparison**](https://posthog.com/blog/best-open-source-llm-observability-tools) ![](https://img.shields.io/badge/posthog.com-active-blue?style=social) - PostHog comparison of open-source LLM observability and evaluation tools.
- [**Scoping LLM Evals**](https://hamel.dev/notes/llm/officehours/scoping.html) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Case study on managing evaluation complexity through proper scoping and topic distribution.
- [**Why AI evals are the hottest new skill**](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill) ![](https://img.shields.io/badge/lennysnewsletter.com-active-blue?style=social) - Lenny's interview covering error analysis, axial coding, eval prompts, and PRD alignment.
- [**Your AI Product Needs Evals**](https://hamel.dev/blog/posts/evals/) ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) - Foundational article on why every AI product needs systematic evaluation.

### Examples

- [**Arize Phoenix AI Chatbot**](https://github.com/Arize-ai/phoenix-ai-chatbot) ![](https://img.shields.io/github/stars/Arize-ai/phoenix-ai-chatbot?style=social&label=github.com) - Next.js chatbot with Phoenix tracing, dataset replays, and evaluation jobs.
- [**Azure LLM Evaluation Samples**](https://github.com/Azure-Samples/llm-evaluation) ![](https://img.shields.io/github/stars/Azure-Samples/llm-evaluation?style=social&label=github.com) - Prompt Flow and Azure AI Foundry projects demonstrating hosted evals.
- [**Deepchecks QA over CSV**](https://github.com/deepchecks/qa-over-csv) ![](https://img.shields.io/github/stars/deepchecks/qa-over-csv?style=social&label=github.com) - Example agent wired to Deepchecks scoring plus tracing dashboards.
- [**OpenAI Evals Demo Evals**](https://github.com/withmartian/demo-evals) ![](https://img.shields.io/github/stars/withmartian/demo-evals?style=social&label=github.com) - Templates for extending OpenAI Evals with custom datasets.
- [**Promptfoo Examples**](https://github.com/promptfoo/promptfoo/tree/main/examples) ![](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social&label=github.com) - Ready-made prompt regression suites for RAG, summarization, and agents.
- [**ZenML Projects**](https://github.com/zenml-io/zenml-projects) ![](https://img.shields.io/github/stars/zenml-io/zenml-projects?style=social&label=github.com) - End-to-end pipelines showing how to weave evaluation steps into LLMOps stacks.

### Related Collections

- [**Awesome ChainForge**](https://github.com/loloMD/awesome_chainforge) ![](https://img.shields.io/github/stars/loloMD/awesome_chainforge?style=social&label=github.com) - Ecosystem list centered on ChainForge experiments and extensions.
- [**Awesome-LLM-Eval**](https://github.com/onejune2018/Awesome-LLM-Eval) ![](https://img.shields.io/github/stars/onejune2018/Awesome-LLM-Eval?style=social&label=github.com) - Cross-lingual (Chinese) compendium of eval tooling, papers, datasets, and leaderboards.
- [**Awesome LLMOps**](https://github.com/tensorchord/awesome-llmops) ![](https://img.shields.io/github/stars/tensorchord/awesome-llmops?style=social&label=github.com) - Curated tooling for training, deployment, and monitoring of LLM apps.
- [**Awesome Machine Learning**](https://github.com/josephmisiti/awesome-machine-learning) ![](https://img.shields.io/github/stars/josephmisiti/awesome-machine-learning?style=social&label=github.com) - Language-specific ML resources that often host evaluation building blocks.
- [**Awesome RAG**](https://github.com/noworneverev/Awesome-RAG) ![](https://img.shields.io/github/stars/noworneverev/Awesome-RAG?style=social&label=github.com) - Broad coverage of retrieval-augmented generation techniques and tools.
- [**Awesome Self-Hosted**](https://github.com/awesome-selfhosted/awesome-selfhosted) ![](https://img.shields.io/github/stars/awesome-selfhosted/awesome-selfhosted?style=social&label=github.com) - Massive catalog of self-hostable software, including observability stacks.
- [**GenAI Notes**](https://github.com/eugeneyan/genai-notes) ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) - Continuously updated notes and resources on GenAI systems, evaluation, and operations.

---

## Licensing

Released under the [CC0 1.0 Universal](LICENSE) license.

---

## Contributing

Contributions are welcome—please read [CONTRIBUTING.md](CONTRIBUTING.md) for scope, entry rules, and the pull-request checklist before submitting updates.

<a href="https://www.vvkmnn.xyz"><img src="https://github.githubassets.com/images/icons/emoji/unicode/270c.png" height="24" alt="✌️"></a> 
