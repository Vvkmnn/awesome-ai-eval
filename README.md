# Awesome AI Eval [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of tools, methods & platforms for evaluating AI quality in real applications.

<img src="./assets/robot-shades.svg" align="right" width="150" alt="Awesome AI Eval robot logo" />

Evaluation is how you know if your AI actually works (and not hallucinating). This list covers the frameworks, benchmarks, datasets, and platforms you need to test LLMs, debug RAG pipelines, and monitor autonomous agents in production, organized by what you're trying to measure and how.

## Contents

- [Tools](#tools)
  - [Evaluators and Test Harnesses](#evaluators-and-test-harnesses)
  - [RAG and Retrieval](#rag-and-retrieval)
  - [Prompt Evaluation & Safety](#prompt-evaluation--safety)
  - [Red Teaming & Adversarial Testing](#red-teaming--adversarial-testing)
  - [Datasets and Methodology](#datasets-and-methodology)
- [Platforms](#platforms)
  - [Open Source Platforms](#open-source-platforms)
  - [Hosted Platforms](#hosted-platforms)
  - [Cloud Platforms](#cloud-platforms)
- [Benchmarks](#benchmarks)
  - [General](#general)
  - [Long Context](#long-context)
  - [Domain](#domain)
  - [Agent](#agent)
  - [Reasoning](#reasoning)
  - [Multimodal](#multimodal)
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

- ![](https://img.shields.io/github/stars/Aleph-Alpha-Research/eval-framework?style=social&label=github.com) [**Aleph Alpha Eval Framework**](https://github.com/Aleph-Alpha-Research/eval-framework) - Production-ready evaluation framework with 90+ pre-loaded benchmarks for reasoning, coding, and safety.
- ![](https://img.shields.io/github/stars/anthropics/evals?style=social&label=github.com) [**Anthropic Model Evals**](https://github.com/anthropics/evals) - Anthropic's evaluation suite for safety, capabilities, and alignment testing of language models.
- ![](https://img.shields.io/github/stars/safety-research/bloom?style=social&label=github.com) [**Bloom**](https://github.com/safety-research/bloom) - Anthropic's open-source agentic framework for automated behavioral evaluations of frontier AI models.
- ![](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social&label=github.com) [**ColossalEval**](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) - Unified pipeline for classic metrics plus GPT-assisted scoring across public datasets.
- ![](https://img.shields.io/github/stars/confident-ai/deepeval?style=social&label=github.com) [**DeepEval**](https://github.com/confident-ai/deepeval) - Python unit-test style metrics for hallucination, relevance, toxicity, and bias.
- ![](https://img.shields.io/github/stars/huggingface/lighteval?style=social&label=github.com) [**Hugging Face lighteval**](https://github.com/huggingface/lighteval) - Toolkit powering HF leaderboards with 1k+ tasks and pluggable metrics.
- ![](https://img.shields.io/github/stars/UKGovernmentBEIS/inspect_ai?style=social&label=github.com) [**Inspect AI**](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Safety Institute framework for scripted eval plans, tool calls, and model-graded rubrics.
- ![](https://img.shields.io/github/stars/EvolvingLMMs-Lab/lmms-eval?style=social&label=github.com) [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) - One-for-all multimodal evaluation toolkit supporting 100+ tasks across text, image, video, and audio.
- ![](https://img.shields.io/github/stars/mlflow/mlflow?style=social&label=github.com) [**MLflow Evaluators**](https://github.com/mlflow/mlflow) - Eval API that logs LLM scores next to classic experiment tracking runs.
- ![](https://img.shields.io/github/stars/openai/evals?style=social&label=github.com) [**OpenAI Evals**](https://github.com/openai/evals) - Reference harness plus registry spanning reasoning, extraction, and safety evals.
- ![](https://img.shields.io/github/stars/open-compass/opencompass?style=social&label=github.com) [**OpenCompass**](https://github.com/open-compass/opencompass) - Research harness with CascadeEvaluator, CompassRank syncing, and LLM-as-judge utilities.
- ![](https://img.shields.io/github/stars/microsoft/promptflow?style=social&label=github.com) [**Prompt Flow**](https://github.com/microsoft/promptflow) - Flow builder with built-in evaluation DAGs, dataset runners, and CI hooks.
- ![](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social&label=github.com) [**Promptfoo**](https://github.com/promptfoo/promptfoo) - Local-first CLI and dashboard for evaluating prompts, RAG flows, and agents with cost tracking and regression detection.
- ![](https://img.shields.io/github/stars/explodinggradients/ragas?style=social&label=github.com) [**Ragas**](https://github.com/explodinggradients/ragas) - Evaluation library that grades answers, context, and grounding with pluggable scorers.
- ![](https://img.shields.io/github/stars/truera/trulens?style=social&label=github.com) [**TruLens**](https://github.com/truera/trulens) - Feedback function framework for chains and agents with customizable judge models.
- ![](https://img.shields.io/badge/wandb.ai-active-blue?style=social) [**W&B Weave Evaluations**](https://wandb.ai/site/evaluations/) - Managed evaluation orchestrator with dataset versioning and dashboards.
- ![](https://img.shields.io/github/stars/zenml-io/zenml?style=social&label=github.com) [**ZenML**](https://github.com/zenml-io/zenml) - Pipeline framework that bakes evaluation steps and guardrail metrics into LLM workflows.

#### Application and Agent Harnesses

- ![](https://img.shields.io/badge/athina.ai-active-blue?style=social) [**Athina AI**](https://www.athina.ai/) - SOC-2 compliant LLM evaluation and monitoring platform with 50+ preset evaluations and VPC deployment.
- ![](https://img.shields.io/badge/braintrust.dev-active-blue?style=social) [**Braintrust**](https://www.braintrust.dev/) - Hosted evaluation workspace with CI-style regression tests, agent sandboxes, and token cost tracking.
- ![](https://img.shields.io/badge/smith.langchain.com-active-blue?style=social) [**LangSmith**](https://smith.langchain.com/) - Hosted tracing plus datasets, batched evals, and regression gating for LangChain apps.
- ![](https://img.shields.io/badge/parea.ai-active-blue?style=social) [**Parea AI**](https://www.parea.ai/) - Developer tools for evaluating, testing, and monitoring LLM-powered applications with actionable insights.
- ![](https://img.shields.io/badge/patronus.ai-active-blue?style=social) [**Patronus AI**](https://www.patronus.ai/) - Evaluation platform with multimodal LLM-as-judge, hallucination detection, and industry benchmarks like FinanceBench.
- ![](https://img.shields.io/badge/docs.wandb.ai-active-blue?style=social) [**W&B Prompt Registry**](https://docs.wandb.ai/weave/guides/core-types/evaluations) - Prompt evaluation templates with reproducible scoring and reviews.

### RAG and Retrieval

#### RAG Frameworks

- ![](https://img.shields.io/badge/evalscope.readthedocs.io-active-blue?style=social) [**EvalScope RAG**](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) - Guides and templates that extend Ragas-style metrics with domain rubrics.
- ![](https://img.shields.io/badge/docs.llamaindex.ai-active-blue?style=social) [**LlamaIndex Evaluation**](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) - Modules for replaying queries, scoring retrievers, and comparing query engines.
- ![](https://img.shields.io/github/stars/vectara/open-rag-eval?style=social&label=github.com) [**Open RAG Eval**](https://github.com/vectara/open-rag-eval) - Vectara harness with UMBRELA and AutoNuggetizer metrics that don't require golden answers.
- ![](https://img.shields.io/github/stars/OpenBMB/RAGEval?style=social&label=github.com) [**RAGEval**](https://github.com/OpenBMB/RAGEval) - Framework that auto-generates corpora, questions, and RAG rubrics for completeness.
- ![](https://img.shields.io/github/stars/THU-KEG/R-Eval?style=social&label=github.com) [**R-Eval**](https://github.com/THU-KEG/R-Eval) - Toolkit for robust RAG scoring aligned with the Evaluation of RAG survey taxonomy.
- ![](https://img.shields.io/github/stars/OpenBMB/UltraRAG?style=social&label=github.com) [**UltraRAG**](https://github.com/OpenBMB/UltraRAG) - MCP-based RAG development framework with built-in evaluation workflows and multimodal support.

#### Retrieval Benchmarks

- ![](https://img.shields.io/github/stars/beir-cellar/beir?style=social&label=github.com) [**BEIR**](https://github.com/beir-cellar/beir) - Benchmark suite covering dense, sparse, and hybrid retrieval tasks.
- ![](https://img.shields.io/github/stars/stanford-futuredata/ColBERT?style=social&label=github.com) [**ColBERT**](https://github.com/stanford-futuredata/ColBERT) - Late-interaction dense retriever with evaluation scripts for IR datasets.
- ![](https://img.shields.io/github/stars/embeddings-benchmark/mteb?style=social&label=github.com) [**MTEB**](https://github.com/embeddings-benchmark/mteb) - Embeddings benchmark measuring retrieval, reranking, and similarity quality.

#### RAG Datasets and Surveys

- ![](https://img.shields.io/github/stars/YHPeter/Awesome-RAG-Evaluation?style=social&label=github.com) [**Awesome-RAG-Evaluation**](https://github.com/YHPeter/Awesome-RAG-Evaluation) - Curated catalog of RAG evaluation metrics, datasets, and leaderboards.
- ![](https://img.shields.io/github/stars/DavidZWZ/Awesome-RAG-Reasoning?style=social&label=github.com) [**Awesome-RAG-Reasoning**](https://github.com/DavidZWZ/Awesome-RAG-Reasoning) - EMNLP 2025 collection of RAG + reasoning benchmarks, datasets, and implementations.
- ![](https://img.shields.io/badge/sh--reya.com-active-blue?style=social) [**Comparing LLMs on Real-World Retrieval**](https://www.sh-reya.com/blog/needle-in-the-real-world/) - Empirical analysis of how language models perform on practical retrieval tasks.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**RAG Evaluation Survey**](https://arxiv.org/abs/2405.07437) - Comprehensive paper covering metrics, judgments, and open problems for RAG.
- ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) [**RAGTruth**](https://github.com/zhengzangw/RAGTruth) - Human-annotated dataset for measuring hallucinations and faithfulness in RAG answers.

### Prompt Evaluation & Safety

- ![](https://img.shields.io/github/stars/tatsu-lab/alpaca_eval?style=social&label=github.com) [**AlpacaEval**](https://github.com/tatsu-lab/alpaca_eval) - Automated instruction-following evaluator with length-controlled LLM judge scoring.
- ![](https://img.shields.io/github/stars/ianarawjo/ChainForge?style=social&label=github.com) [**ChainForge**](https://github.com/ianarawjo/ChainForge) - Visual IDE for comparing prompts, sampling models, and scoring batches with rubrics.
- ![](https://img.shields.io/github/stars/ShreyaR/guardrails?style=social&label=github.com) [**Guardrails AI**](https://github.com/ShreyaR/guardrails) - Declarative validation framework that enforces schemas, correction chains, and judgments.
- ![](https://img.shields.io/badge/lakera.ai-active-blue?style=social) [**Lakera Guard**](https://www.lakera.ai/lakera-guard) - Hosted prompt security platform with red-team datasets for jailbreak and injection testing.
- ![](https://img.shields.io/github/stars/microsoft/promptbench?style=social&label=github.com) [**PromptBench**](https://github.com/microsoft/promptbench) - Benchmark suite for adversarial prompt stress tests across diverse tasks.
- ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) [**Red Teaming Handbook**](https://learn.microsoft.com/en-us/security/) - Microsoft playbook for adversarial prompt testing and mitigation patterns.

### Red Teaming & Adversarial Testing

- ![](https://img.shields.io/github/stars/BCG-X-Official/artkit?style=social&label=github.com) [**ARTKIT**](https://github.com/BCG-X-Official/artkit) - Automated multi-turn red teaming framework that simulates attacker-target interactions for jailbreak testing.
- ![](https://img.shields.io/github/stars/confident-ai/deepteam?style=social&label=github.com) [**DeepTeam**](https://github.com/confident-ai/deepteam) - Open-source LLM red teaming framework testing for bias, data exposure, and prompt injection vulnerabilities.
- ![](https://img.shields.io/github/stars/NVIDIA/garak?style=social&label=github.com) [**Garak**](https://github.com/NVIDIA/garak) - NVIDIA's adversarial testing toolkit with 100+ attack modules for prompt injection and data extraction.
- ![](https://img.shields.io/github/stars/Azure/PyRIT?style=social&label=github.com) [**PyRIT**](https://github.com/Azure/PyRIT) - Microsoft's Python Risk Identification Toolkit for orchestrating LLM attack suites and red team automation.

### Datasets and Methodology

- ![](https://img.shields.io/badge/deepchecks.com-active-blue?style=social) [**Deepchecks Evaluation Playbook**](https://www.deepchecks.com/llm-evaluation/best-tools/) - Survey of evaluation metrics, failure modes, and platform comparisons.
- ![](https://img.shields.io/badge/crfm.stanford.edu-active-blue?style=social) [**HELM**](https://crfm.stanford.edu/helm/latest/) - Holistic Evaluation of Language Models methodology emphasizing multi-criteria scoring.
- ![](https://img.shields.io/github/stars/google-research/google-research?style=social&label=github.com) [**Instruction-Following Evaluation (IFEval)**](https://github.com/google-research/google-research/tree/master/instruction_following_eval) - Constraint-verification prompts for automatically checking instruction compliance.
- ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) [**OpenAI Cookbook Evals**](https://github.com/openai/openai-cookbook/tree/main/examples/evals) - Practical notebooks showing how to build custom evals.
- ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) [**Safety Evaluation Guides**](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/safety-evaluations-transparency-note) - Cloud vendor recipes for testing quality, safety, and risk.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**Who Validates the Validators?**](https://arxiv.org/abs/2404.12272) - EvalGen workflow aligning LLM judges with human rubrics via mixed-initiative criteria design.
- ![](https://img.shields.io/badge/zenml.io-active-blue?style=social) [**ZenML Evaluation Playbook**](https://www.zenml.io/blog/the-evaluation-playbook-making-llms-production-ready) - Playbook for embedding eval gates into pipelines and deployments.

---

## Platforms

### Open Source Platforms

- ![](https://img.shields.io/github/stars/Agenta-AI/agenta?style=social&label=github.com) [**Agenta**](https://github.com/Agenta-AI/agenta) - End-to-end LLM developer platform for prompt engineering, evaluation, and deployment.
- ![](https://img.shields.io/github/stars/Arize-ai/phoenix?style=social&label=github.com) [**Arize Phoenix**](https://github.com/Arize-ai/phoenix) - OpenTelemetry-native observability and evaluation toolkit for RAG, LLMs, and agents.
- ![](https://img.shields.io/github/stars/ucbepic/docetl?style=social&label=github.com) [**DocETL**](https://github.com/ucbepic/docetl) - ETL system for complex document processing with LLMs and built-in quality checks.
- ![](https://img.shields.io/github/stars/Giskard-AI/giskard?style=social&label=github.com) [**Giskard**](https://github.com/Giskard-AI/giskard) - Testing framework for ML models with vulnerability scanning and LLM-specific detectors.
- ![](https://img.shields.io/github/stars/Helicone/helicone?style=social&label=github.com) [**Helicone**](https://github.com/Helicone/helicone) - Open-source LLM observability platform with cost tracking, caching, and evaluation tools.
- ![](https://img.shields.io/github/stars/langfuse/langfuse?style=social&label=github.com) [**Langfuse**](https://github.com/langfuse/langfuse) - Open-source LLM engineering platform providing tracing, eval dashboards, and prompt analytics.
- ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) [**Lilac**](https://github.com/lilacai/lilac) - Data curation tool for exploring and enriching datasets with semantic search and clustering.
- ![](https://img.shields.io/github/stars/BerriAI/litellm?style=social&label=github.com) [**LiteLLM**](https://github.com/BerriAI/litellm) - Unified API for 100+ LLM providers with cost tracking, fallbacks, and load balancing.
- ![](https://img.shields.io/github/stars/lunary-ai/lunary?style=social&label=github.com) [**Lunary**](https://github.com/lunary-ai/lunary) - Production toolkit for LLM apps with tracing, prompt management, and evaluation pipelines.
- ![](https://img.shields.io/github/stars/mirascope/mirascope?style=social&label=github.com) [**Mirascope**](https://github.com/mirascope/mirascope) - Python toolkit for building LLM applications with structured outputs and evaluation utilities.
- ![](https://img.shields.io/github/stars/openlit/openlit?style=social&label=github.com) [**OpenLIT**](https://github.com/openlit/openlit) - Telemetry instrumentation for LLM apps with built-in quality metrics and guardrail hooks.
- ![](https://img.shields.io/github/stars/traceloop/openllmetry?style=social&label=github.com) [**OpenLLMetry**](https://github.com/traceloop/openllmetry) - OpenTelemetry instrumentation for LLM traces that feed any backend or custom eval logic.
- ![](https://img.shields.io/github/stars/comet-ml/opik?style=social&label=github.com) [**Opik**](https://github.com/comet-ml/opik) - Self-hostable evaluation and observability hub with datasets, scoring jobs, and interactive traces.
- ![](https://img.shields.io/github/stars/rhesis-ai/rhesis?style=social&label=github.com) [**Rhesis**](https://github.com/rhesis-ai/rhesis) - Collaborative testing platform with automated test generation and multi-turn conversation simulation for LLM and agentic applications.
- ![](https://img.shields.io/github/stars/future-agi/traceAI?style=social&label=github.com) [**traceAI**](https://github.com/future-agi/traceAI) - Open-source multi-modal tracing and diagnostics framework for LLM, RAG, and agent workflows built on OpenTelemetry.
- ![](https://img.shields.io/github/stars/uptrain-ai/uptrain?style=social&label=github.com) [**UpTrain**](https://github.com/uptrain-ai/uptrain) - OSS/hosted evaluation suite with 20+ checks, RCA tooling, and LlamaIndex integrations.
- ![](https://img.shields.io/github/stars/VoltAgent/voltagent?style=social&label=github.com) [**VoltAgent**](https://github.com/VoltAgent/voltagent) - TypeScript agent framework paired with VoltOps for trace inspection and regression testing.
- ![](https://img.shields.io/badge/zenoml.com-active-blue?style=social) [**Zeno**](https://zenoml.com/) - Data-centric evaluation UI for slicing failures, comparing prompts, and debugging retrieval quality.

### Hosted Platforms

- ![](https://img.shields.io/badge/chatintel.ai-active-blue?style=social) [**ChatIntel**](https://chatintel.ai/) - Conversation analytics platform for evaluating chatbot quality, sentiment, and user satisfaction.
- ![](https://img.shields.io/badge/confident--ai.com-active-blue?style=social) [**Confident AI**](https://www.confident-ai.com/) - DeepEval-backed platform for scheduled eval suites, guardrails, and production monitors.
- ![](https://img.shields.io/badge/datadoghq.com-active-blue?style=social) [**Datadog LLM Observability**](https://www.datadoghq.com/product/llm-observability/) - Datadog module capturing LLM traces, metrics, and safety signals.
- ![](https://img.shields.io/badge/deepchecks.com-active-blue?style=social) [**Deepchecks LLM Evaluation**](https://www.deepchecks.com/solutions/llm-evaluation/) - Managed eval suites with dataset versioning, dashboards, and alerting.
- ![](https://img.shields.io/badge/geteppo.com-active-blue?style=social) [**Eppo**](https://www.geteppo.com/) - Experimentation platform with AI-specific evaluation metrics and statistical rigor for LLM A/B testing.
- ![](https://img.shields.io/badge/futureagi.com-active-blue?style=social) [**Future AGI**](https://futureagi.com/) - Multi-modal evaluation, simulation, and optimization platform for reliable AI systems across software and hardware.
- ![](https://img.shields.io/badge/galileo.ai-active-blue?style=social) [**Galileo**](https://www.galileo.ai/) - Evaluation and data-curation studio with labeling, slicing, and issue triage.
- ![](https://img.shields.io/badge/honeyhive.ai-active-blue?style=social) [**HoneyHive**](https://www.honeyhive.ai/) - Evaluation and observability platform with prompt versioning, A/B testing, and fine-tuning workflows.
- ![](https://img.shields.io/badge/humanloop.com-active-blue?style=social) [**Humanloop**](https://humanloop.com/) - Production prompt management with human-in-the-loop evals and annotation queues.
- ![](https://img.shields.io/badge/getmaxim.ai-active-blue?style=social) [**Maxim AI**](https://www.getmaxim.ai/) - Evaluation and observability platform focusing on agent simulations and monitoring.
- ![](https://img.shields.io/badge/orq.ai-active-blue?style=social) [**Orq.ai**](https://orq.ai/) - LLM operations platform with prompt management, evaluation workflows, and deployment pipelines.
- ![](https://img.shields.io/badge/posthog.com-active-blue?style=social) [**PostHog LLM Analytics**](https://posthog.com/llm-analytics) - Product analytics toolkit extended to track custom LLM events and metrics.
- ![](https://img.shields.io/badge/promptlayer.com-active-blue?style=social) [**PromptLayer**](https://www.promptlayer.com/) - Prompt engineering platform with version control, evaluation tracking, and team collaboration.

### Cloud Platforms

- ![](https://img.shields.io/badge/aws.amazon.com-active-blue?style=social) [**Amazon Bedrock Evaluations**](https://aws.amazon.com/bedrock/evaluations/) - Managed service for scoring foundation models and RAG pipelines.
- ![](https://img.shields.io/badge/aws.amazon.com-active-blue?style=social) [**Amazon Bedrock Guardrails**](https://aws.amazon.com/bedrock/guardrails/) - Safety layer that evaluates prompts and responses for policy compliance.
- ![](https://img.shields.io/badge/learn.microsoft.com-active-blue?style=social) [**Azure AI Foundry Evaluations**](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluate-generative-ai-app) - Evaluation flows and risk reports wired into Prompt Flow projects.
- ![](https://img.shields.io/badge/cloud.google.com-active-blue?style=social) [**Vertex AI Generative AI Evaluation**](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) - Adaptive rubric-based evaluation with agent assessment, LangChain/CrewAI support, and test-driven evaluation framework.

---

## Benchmarks

### General

- ![](https://img.shields.io/github/stars/ruixiangcui/AGIEval?style=social&label=github.com) [**AGIEval**](https://github.com/ruixiangcui/AGIEval) - Human-centric standardized exams spanning entrance tests, legal, and math scenarios.
- ![](https://img.shields.io/github/stars/google/BIG-bench?style=social&label=github.com) [**BIG-bench**](https://github.com/google/BIG-bench) - Collaborative benchmark probing reasoning, commonsense, and long-tail tasks.
- ![](https://img.shields.io/github/stars/allenai/CommonGen-Eval?style=social&label=github.com) [**CommonGen-Eval**](https://github.com/allenai/CommonGen-Eval) - GPT-4 judged CommonGen-lite suite for constrained commonsense text generation.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**DyVal**](https://arxiv.org/abs/2309.17167) - Dynamic reasoning benchmark that varies difficulty and graph structure to stress models.
- ![](https://img.shields.io/github/stars/EleutherAI/lm-evaluation-harness?style=social&label=github.com) [**LM Evaluation Harness**](https://github.com/EleutherAI/lm-evaluation-harness) - Standard harness for scoring autoregressive models on dozens of tasks.
- ![](https://img.shields.io/github/stars/smartyfh/LLM-Uncertainty-Bench?style=social&label=github.com) [**LLM-Uncertainty-Bench**](https://github.com/smartyfh/LLM-Uncertainty-Bench) - Adds uncertainty-aware scoring across QA, RC, inference, dialog, and summarization.
- ![](https://img.shields.io/github/stars/princeton-nlp/LLMBar?style=social&label=github.com) [**LLMBar**](https://github.com/princeton-nlp/LLMBar) - Meta-eval testing whether LLM judges can spot instruction-following failures.
- ![](https://img.shields.io/github/stars/hendrycks/test?style=social&label=github.com) [**MMLU**](https://github.com/hendrycks/test) - Massive multitask language understanding benchmark for academic and professional subjects.
- ![](https://img.shields.io/github/stars/TIGER-AI-Lab/MMLU-Pro?style=social&label=github.com) [**MMLU-Pro**](https://github.com/TIGER-AI-Lab/MMLU-Pro) - Harder 10-choice extension focused on reasoning-rich, low-leakage questions.
- ![](https://img.shields.io/github/stars/aigc-apps/PertEval?style=social&label=github.com) [**PertEval**](https://github.com/aigc-apps/PertEval) - Knowledge-invariant perturbations to debias multiple-choice accuracy inflation.
- ![](https://img.shields.io/badge/simplebench.ai-active-blue?style=social) [**SimpleBench**](https://simplebench.ai/) - Fundamental reasoning benchmark where humans (83.7%) significantly outperform best AI models (62.4%).

### Long Context

- ![](https://img.shields.io/github/stars/OpenBMB/InfiniteBench?style=social&label=github.com) [**InfiniteBench**](https://github.com/OpenBMB/InfiniteBench) - First LLM benchmark with average data length surpassing 100K tokens across 12 tasks.
- ![](https://img.shields.io/badge/longbench2.github.io-active-blue?style=social) [**LongBench v2**](https://longbench2.github.io/) - Long-context benchmark with 8k-2M word contexts and 503 challenging questions across six task categories.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**LongGenBench**](https://arxiv.org/abs/2409.02076) - ICLR 2025 benchmark evaluating 16K-32K token long-form text generation quality.
- ![](https://img.shields.io/github/stars/infinigence/LVEval?style=social&label=github.com) [**LV-Eval**](https://github.com/infinigence/LVEval) - Long-context suite with five length tiers up to 256K tokens and distraction controls.
- ![](https://img.shields.io/github/stars/NVIDIA/RULER?style=social&label=github.com) [**RULER**](https://github.com/NVIDIA/RULER) - NVIDIA's synthetic long-context benchmark with configurable sequence length and 13 tasks across 4 categories.

### Domain

- ![](https://img.shields.io/badge/patronus.ai-active-blue?style=social) [**FinanceBench**](https://www.patronus.ai/announcements/patronus-ai-launches-financebench-the-industrys-first-benchmark-for-llm-performance-on-financial-questions) - Industry benchmark for LLM performance on financial questions and reasoning.
- ![](https://img.shields.io/github/stars/SUFE-AIFLM-Lab/FinEval?style=social&label=github.com) [**FinEval**](https://github.com/SUFE-AIFLM-Lab/FinEval) - Chinese financial QA and reasoning benchmark across regulation, accounting, and markets.
- ![](https://img.shields.io/github/stars/openai/human-eval?style=social&label=github.com) [**HumanEval**](https://github.com/openai/human-eval) - Unit-test-based benchmark for code synthesis and docstring reasoning.
- ![](https://img.shields.io/github/stars/Dai-shen/LAiW?style=social&label=github.com) [**LAiW**](https://github.com/Dai-shen/LAiW) - Legal benchmark covering retrieval, foundation inference, and complex case applications in Chinese law.
- ![](https://img.shields.io/github/stars/hendrycks/math?style=social&label=github.com) [**MATH**](https://github.com/hendrycks/math) - Competition-level math benchmark targeting multi-step symbolic reasoning.
- ![](https://img.shields.io/github/stars/google-research/google-research?style=social&label=github.com) [**MBPP**](https://github.com/google-research/google-research/tree/master/mbpp) - Mostly Basic Programming Problems benchmark for small coding tasks.
- ![](https://img.shields.io/badge/crfm.stanford.edu-active-blue?style=social) [**MedHELM**](https://crfm.stanford.edu/helm/medhelm/latest/) - Comprehensive medical LLM benchmark with 121 clinician-validated tasks and LLM-jury evaluation protocol.

### Agent

- ![](https://img.shields.io/github/stars/THUDM/AgentBench?style=social&label=github.com) [**AgentBench**](https://github.com/THUDM/AgentBench) - Evaluates LLMs acting as agents across simulated domains like games and coding.
- ![](https://img.shields.io/badge/allenai.org-active-blue?style=social) [**AstaBench**](https://allenai.org/blog/astabench) - AI2 benchmark for scientific research AI agents covering literature review, experiment replication, and data analysis.
- ![](https://img.shields.io/badge/openai.com-active-blue?style=social) [**BrowseComp**](https://openai.com/index/browsecomp/) - OpenAI benchmark of 1,266 problems measuring AI agents' ability to find entangled information on the web.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**ColBench**](https://arxiv.org/abs/2503.08452) - Multi-turn benchmark evaluating LLMs as collaborative coding agents with simulated human partners.
- ![](https://img.shields.io/badge/letta.com-active-blue?style=social) [**Context-Bench**](https://www.letta.com/blog/context-bench) - Letta's benchmark for evaluating AI agent context management and memory capabilities.
- ![](https://img.shields.io/badge/jetbrains.com-active-blue?style=social) [**DPAI Arena**](https://blog.jetbrains.com/ai/2025/10/dpai-arena/) - JetBrains benchmark evaluating full multi-workflow, multi-language developer agents across the engineering lifecycle.
- ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) [**GAIA**](https://huggingface.co/datasets/gaia-benchmark/GAIA) - Tool-use benchmark requiring grounded reasoning with live web access and planning.
- ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) [**MetaTool Tasks**](https://github.com/meta-llama/MetaTool) - Tool-calling benchmark and eval harness for agents built around LLaMA models.
- ![](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUE-Agent?style=social&label=github.com) [**SuperCLUE-Agent**](https://github.com/CLUEbenchmark/SuperCLUE-Agent) - Chinese agent eval covering tool use, planning, long/short-term memory, and APIs.
- ![](https://img.shields.io/github/stars/SWE-bench/SWE-bench?style=social&label=github.com) [**SWE-bench**](https://github.com/SWE-bench/SWE-bench) - Real-world GitHub issue resolution benchmark for coding agents.
- ![](https://img.shields.io/badge/swe--bench--live.github.io-active-blue?style=social) [**SWE-bench Live**](https://swe-bench-live.github.io/) - Continuously updated benchmark with monthly refreshes for contamination-free evaluation.
- ![](https://img.shields.io/badge/scale.com-active-blue?style=social) [**SWE-bench Pro**](https://scale.com/leaderboard/swe_bench_pro_public) - Enterprise-level coding benchmark with 1,865 problems across 41 repos requiring hours-to-days solutions.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**Terminal-Bench**](https://arxiv.org/abs/2505.09876) - Stanford/Laude benchmark evaluating AI agents operating in sandboxed command-line environments.

### Reasoning

- ![](https://img.shields.io/badge/arcprize.org-active-blue?style=social) [**ARC-AGI-2**](https://arcprize.org/arc-agi/2/) - Next-generation reasoning benchmark where pure LLMs score 0% but humans can solve every task.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**JudgeBench**](https://arxiv.org/abs/2410.12784) - ICLR 2025 benchmark for evaluating LLM-based judges on challenging response pairs across knowledge, reasoning, math, and coding.

### Multimodal

- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**MERLIM**](https://arxiv.org/abs/2312.02394) - 300K+ image-question pairs with focus on detecting cross-modal hallucination and hidden hallucinations.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**MME**](https://arxiv.org/abs/2306.13394) - Comprehensive MLLM evaluation measuring perception and cognition across 14 subtasks.
- ![](https://img.shields.io/badge/mmmu--benchmark.github.io-active-blue?style=social) [**MMMU-Pro**](https://mmmu-benchmark.github.io/) - Harder extension of MMMU benchmark for multimodal understanding with expert-level questions.
- ![](https://img.shields.io/github/stars/OpenGVLab/MMT-Bench?style=social&label=github.com) [**MMT-Bench**](https://github.com/OpenGVLab/MMT-Bench) - 31K+ questions across image, text, video, and point cloud modalities with 162 subtasks.
- ![](https://img.shields.io/badge/video--mme.github.io-active-blue?style=social) [**Video-MME**](https://video-mme.github.io/) - CVPR 2025 benchmark for comprehensive evaluation of multimodal LLMs in video analysis.
- ![](https://img.shields.io/github/stars/showlab/VisualToolBench?style=social&label=github.com) [**VisualToolBench**](https://github.com/showlab/VisualToolBench) - First "think with image" benchmark evaluating MLLMs on tasks requiring active visual interaction.

### Safety

- ![](https://img.shields.io/github/stars/llm-attacks/llm-attacks?style=social&label=github.com) [**AdvBench**](https://github.com/llm-attacks/llm-attacks) - Adversarial prompt benchmark for jailbreak and misuse resistance measurement.
- ![](https://img.shields.io/github/stars/nyu-mll/BBQ?style=social&label=github.com) [**BBQ**](https://github.com/nyu-mll/BBQ) - Bias-sensitive QA sets measuring stereotype reliance and ambiguous cases.
- ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) [**SimpleSafetyTests**](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) - 100-prompt test suite for identifying critical safety risks across five harm areas.
- ![](https://img.shields.io/github/stars/microsoft/ToxiGen?style=social&label=github.com) [**ToxiGen**](https://github.com/microsoft/ToxiGen) - Toxic language generation and classification benchmark for robustness checks.
- ![](https://img.shields.io/github/stars/sylinrl/TruthfulQA?style=social&label=github.com) [**TruthfulQA**](https://github.com/sylinrl/TruthfulQA) - Measures factuality and hallucination propensity via adversarially written questions.

---

## Leaderboards

- ![](https://img.shields.io/badge/arcprize.org-active-blue?style=social) [**ARC Prize Leaderboard**](https://arcprize.org/leaderboard) - AGI reasoning leaderboard tracking ARC-AGI-2 performance across frontier models and open submissions.
- ![](https://img.shields.io/badge/rank.opencompass.org.cn-active-blue?style=social) [**CompassRank**](https://rank.opencompass.org.cn/home) - OpenCompass leaderboard comparing frontier and research models across multi-domain suites.
- ![](https://img.shields.io/badge/llmbench.ai-active-blue?style=social) [**LLM Agents Benchmark Collections**](https://llmbench.ai/) - Aggregated leaderboard comparing multi-agent safety and reliability suites.
- ![](https://img.shields.io/badge/lmarena.ai-active-blue?style=social) [**LMArena**](https://lmarena.ai/) - Crowdsourced LLM comparison platform (formerly LMSYS Chatbot Arena) with 6M+ user votes for Elo ratings.
- ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) [**Open LLM Leaderboard**](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - Hugging Face benchmark board with IFEval, MMLU-Pro, GPQA, and more.
- ![](https://img.shields.io/badge/huggingface.co-active-blue?style=social) [**Open Medical-LLM Leaderboard**](https://huggingface.co/blog/leaderboard-medicalllm) - Hugging Face leaderboard for medical domain LLM performance across healthcare benchmarks.
- ![](https://img.shields.io/github/stars/openai/evals?style=social&label=github.com) [**OpenAI Evals Registry**](https://github.com/openai/evals/tree/main/evals/elsuite) - Community suites and scores covering accuracy, safety, and instruction following.
- ![](https://img.shields.io/badge/scale.com-active-blue?style=social) [**Scale SEAL Leaderboard**](https://scale.com/leaderboard) - Expert-rated leaderboard covering reasoning, coding, and safety via SEAL evaluations.

---

## Resources

### Guides & Training

- ![](https://img.shields.io/badge/maven.com-active-blue?style=social) [**AI Evals for Engineers & PMs**](https://maven.com/parlance-labs/evals?promoCode=FAST25) - Cohort course from Hamel & Shreya with lifetime reader, Discord, AI Eval Assistant, and live office hours.
- ![](https://img.shields.io/badge/eugeneyan.com-active-blue?style=social) [**AlignEval**](https://eugeneyan.com/writing/aligneval/) - Eugene Yan's guide on building LLM judges by following methodical alignment processes.
- ![](https://img.shields.io/badge/applied--llms.org-active-blue?style=social) [**Applied LLMs**](https://applied-llms.org/) - Practical lessons from a year of building with LLMs, emphasizing evaluation as a core practice.
- ![](https://img.shields.io/badge/sh--reya.com-active-blue?style=social) [**Data Flywheels for LLM Applications**](https://www.sh-reya.com/blog/ai-engineering-flywheel/) - Iterative data improvement processes for building better LLM systems.
- ![](https://img.shields.io/badge/youtube.com-active-blue?style=social) [**Error Analysis & Prioritizing Next Steps**](https://www.youtube.com/watch?v=bWkQk5_OG8k) - Andrew Ng walkthrough showing how to slice traces and focus eval work via classic ML techniques.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Error Analysis Before Tests**](https://hamel.dev/notes/llm/officehours/erroranalysis.html) - Office hours notes on why error analysis should precede writing automated tests.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Eval Tools Comparison**](https://hamel.dev/blog/posts/eval-tools/) - Detailed comparison of evaluation tools including Braintrust, LangSmith, and Promptfoo.
- ![](https://img.shields.io/badge/oreilly.com-active-blue?style=social) [**Evals for AI Engineers**](https://www.oreilly.com/library/view/evals-for-ai/9798341660717/) - O'Reilly book by Shreya Shankar & Hamel Husain on systematic error analysis, evaluation pipelines, and LLM-as-a-judge.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Evaluating RAG Systems**](https://hamel.dev/blog/posts/evals-faq/#how-should-i-approach-evaluating-my-rag-system) - Practical guidance on RAG evaluation covering retrieval quality and generation assessment.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Field Guide to Rapidly Improving AI Products**](https://hamel.dev/blog/posts/field-guide/) - Comprehensive guide on error analysis, data viewers, and systematic improvement from 30+ implementations.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Inspect AI Deep Dive**](https://hamel.dev/notes/llm/evals/inspect.html) - Technical deep dive into Inspect AI framework with hands-on examples.
- ![](https://img.shields.io/badge/sap--samples.github.io-active-blue?style=social) [**KDD 2025 Tutorial: Evaluation & Benchmarking of LLM Agents**](https://sap-samples.github.io/llm-agents-eval-tutorial/) - Academic tutorial covering LLM agent evaluation methodology and best practices.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**LLM Evals FAQ**](https://hamel.dev/blog/posts/evals-faq/) - Comprehensive FAQ with 45+ articles covering evaluation questions from practitioners.
- ![](https://img.shields.io/badge/eugeneyan.com-active-blue?style=social) [**LLM Evaluators Survey**](https://eugeneyan.com/writing/llm-evaluators/) - Survey of LLM-as-judge use cases and approaches with practical implementation patterns.
- ![](https://img.shields.io/badge/arxiv.org-active-blue?style=social) [**LLM-as-a-Judge Survey**](https://arxiv.org/abs/2411.15594) - Comprehensive 2025 survey on building reliable LLM-as-a-Judge systems with bias mitigation strategies.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**LLM-as-a-Judge Guide**](https://hamel.dev/blog/posts/llm-judge/) - In-depth guide on using LLMs as judges for automated evaluation with calibration tips.
- ![](https://img.shields.io/badge/parlance--labs.com-active-blue?style=social) [**Mastering LLMs Open Course**](https://parlance-labs.com/education/) - Free 40+ hour course covering evals, RAG, and fine-tuning taught by 25+ industry practitioners.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Modern IR Evals For RAG**](https://hamel.dev/notes/llm/rag/p2-evals.html) - Why traditional IR evals are insufficient for RAG, covering BEIR and modern approaches.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Multi-Turn Chat Evals**](https://hamel.dev/notes/llm/officehours/evalmultiturn.html) - Strategies for evaluating multi-turn conversational AI systems.
- ![](https://img.shields.io/badge/posthog.com-active-blue?style=social) [**Open Source LLM Tools Comparison**](https://posthog.com/blog/best-open-source-llm-observability-tools) - PostHog comparison of open-source LLM observability and evaluation tools.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Scoping LLM Evals**](https://hamel.dev/notes/llm/officehours/scoping.html) - Case study on managing evaluation complexity through proper scoping and topic distribution.
- ![](https://img.shields.io/badge/lennysnewsletter.com-active-blue?style=social) [**Why AI evals are the hottest new skill**](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill) - Lenny's interview covering error analysis, axial coding, eval prompts, and PRD alignment.
- ![](https://img.shields.io/badge/hamel.dev-active-blue?style=social) [**Your AI Product Needs Evals**](https://hamel.dev/blog/posts/evals/) - Foundational article on why every AI product needs systematic evaluation.

### Examples

- ![](https://img.shields.io/github/stars/Arize-ai/phoenix-ai-chatbot?style=social&label=github.com) [**Arize Phoenix AI Chatbot**](https://github.com/Arize-ai/phoenix-ai-chatbot) - Next.js chatbot with Phoenix tracing, dataset replays, and evaluation jobs.
- ![](https://img.shields.io/github/stars/Azure-Samples/llm-evaluation?style=social&label=github.com) [**Azure LLM Evaluation Samples**](https://github.com/Azure-Samples/llm-evaluation) - Prompt Flow and Azure AI Foundry projects demonstrating hosted evals.
- ![](https://img.shields.io/github/stars/deepchecks/qa-over-csv?style=social&label=github.com) [**Deepchecks QA over CSV**](https://github.com/deepchecks/qa-over-csv) - Example agent wired to Deepchecks scoring plus tracing dashboards.
- ![](https://img.shields.io/github/stars/withmartian/demo-evals?style=social&label=github.com) [**OpenAI Evals Demo Evals**](https://github.com/withmartian/demo-evals) - Templates for extending OpenAI Evals with custom datasets.
- ![](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social&label=github.com) [**Promptfoo Examples**](https://github.com/promptfoo/promptfoo/tree/main/examples) - Ready-made prompt regression suites for RAG, summarization, and agents.
- ![](https://img.shields.io/github/stars/zenml-io/zenml-projects?style=social&label=github.com) [**ZenML Projects**](https://github.com/zenml-io/zenml-projects) - End-to-end pipelines showing how to weave evaluation steps into LLMOps stacks.

### Related Collections

- ![](https://img.shields.io/github/stars/loloMD/awesome_chainforge?style=social&label=github.com) [**Awesome ChainForge**](https://github.com/loloMD/awesome_chainforge) - Ecosystem list centered on ChainForge experiments and extensions.
- ![](https://img.shields.io/github/stars/onejune2018/Awesome-LLM-Eval?style=social&label=github.com) [**Awesome-LLM-Eval**](https://github.com/onejune2018/Awesome-LLM-Eval) - Cross-lingual (Chinese) compendium of eval tooling, papers, datasets, and leaderboards.
- ![](https://img.shields.io/github/stars/tensorchord/awesome-llmops?style=social&label=github.com) [**Awesome LLMOps**](https://github.com/tensorchord/awesome-llmops) - Curated tooling for training, deployment, and monitoring of LLM apps.
- ![](https://img.shields.io/github/stars/josephmisiti/awesome-machine-learning?style=social&label=github.com) [**Awesome Machine Learning**](https://github.com/josephmisiti/awesome-machine-learning) - Language-specific ML resources that often host evaluation building blocks.
- ![](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models?style=social&label=github.com) [**Awesome-Multimodal-Large-Language-Models**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) - Latest advances on multimodal LLMs including evaluation benchmarks and surveys.
- ![](https://img.shields.io/github/stars/noworneverev/Awesome-RAG?style=social&label=github.com) [**Awesome RAG**](https://github.com/noworneverev/Awesome-RAG) - Broad coverage of retrieval-augmented generation techniques and tools.
- ![](https://img.shields.io/github/stars/awesome-selfhosted/awesome-selfhosted?style=social&label=github.com) [**Awesome Self-Hosted**](https://github.com/awesome-selfhosted/awesome-selfhosted) - Massive catalog of self-hostable software, including observability stacks.
- ![](https://img.shields.io/badge/github-archived-lightgray?style=social&logo=github) [**GenAI Notes**](https://github.com/eugeneyan/genai-notes) - Continuously updated notes and resources on GenAI systems, evaluation, and operations.

---

## Licensing

Released under the [CC0 1.0 Universal](LICENSE) license.

---

## Contributing

Contributions are welcome—please read [CONTRIBUTING.md](CONTRIBUTING.md) for scope, entry rules, and the pull-request checklist before submitting updates.

<a href="https://www.vvkmnn.xyz"><img src="https://github.githubassets.com/images/icons/emoji/unicode/270c.png" height="24" alt="✌️"></a>
