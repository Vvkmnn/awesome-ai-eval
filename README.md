# Awesome AI Eval [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<a href="https://github.com/Vvkmnn/awesome-ai-eval"><img src="./assets/robot-shades.svg" align="right" width="96" alt="Robot wearing pink shades"></a>

> Assessing the reliability, accuracy, and safety of large language models, RAG pipelines, and AI agents.

## Contents

- [Tools](#tools)
  - [Frameworks](#frameworks)
  - [Retrieval](#retrieval)
  - [Adversarial](#adversarial)
- [Platforms](#platforms)
  - [Self-Hosted](#self-hosted)
  - [Managed](#managed)
  - [Cloud](#cloud)
- [Benchmarks](#benchmarks)
  - [General](#general)
  - [Long-Context](#long-context)
  - [Specialized](#specialized)
  - [Agents](#agents)
  - [Multimodal](#multimodal)
  - [Safety](#safety)
- [Leaderboards](#leaderboards)
- [Resources](#resources)
  - [Papers](#papers)
  - [Guides](#guides)
  - [Lists](#lists)

## Tools

### Frameworks

- [agenttrace](https://github.com/luoyuctl/agenttrace) - Terminal UI for reviewing coding-agent sessions by cost, failures, latency, and diffs.
- [Aleph Alpha Eval Framework](https://github.com/Aleph-Alpha-Research/eval-framework) - Evaluation framework with built-in benchmarks for reasoning, coding, and safety.
- [autoevals](https://github.com/braintrustdata/autoevals) - Scorers for LLM outputs, from string heuristics to model-graded factuality, in Python and TypeScript.
- [Bloom](https://github.com/safety-research/bloom) - Agentic framework that generates behavioral evaluations of frontier models from a described behavior.
- [ChainForge](https://github.com/ianarawjo/ChainForge) - Visual environment for comparing prompts and models and scoring responses.
- [ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) - Pipeline combining classic metrics with GPT-assisted scoring across public datasets.
- [DeepEval](https://github.com/confident-ai/deepeval) - Unit-test style metrics for hallucination, relevance, toxicity, and bias.
- [Evidently](https://github.com/evidentlyai/evidently) - Evaluation and monitoring library for ML and LLM systems.
- [HELM](https://github.com/stanford-crfm/helm) - Stanford framework for evaluating foundation models across scenarios and metrics.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Security Institute framework for scripted evals with tool calls and model-graded scoring.
- [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) - Community-maintained collection of evaluations implemented in Inspect AI.
- [lighteval](https://github.com/huggingface/lighteval) - Hugging Face toolkit for evaluating LLMs across many backends and tasks.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Harness for running language models on hundreds of academic benchmarks.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) - Multimodal evaluation toolkit covering text, image, video, and audio tasks.
- [MLflow](https://github.com/mlflow/mlflow) - Evaluation API that logs judge scores alongside experiment tracking runs.
- [OpenAI Evals](https://github.com/openai/evals) - Framework and registry of evals for LLMs and LLM systems.
- [OpenAI simple-evals](https://github.com/openai/simple-evals) - Reference implementations of standard benchmarks such as MMLU, GPQA, and HumanEval.
- [OpenCompass](https://github.com/open-compass/opencompass) - Evaluation platform with LLM judge support across knowledge, reasoning, and coding datasets.
- [openevals](https://github.com/langchain-ai/openevals) - Evaluators for LLM apps, including LLM judge and structured-output checks.
- [Prompt Flow](https://github.com/microsoft/promptflow) - Flow builder with built-in evaluation flows and CI integration.
- [Promptfoo](https://github.com/promptfoo/promptfoo) - CLI for testing and red teaming prompts, RAG flows, and agents with regression detection.
- [skill-optimizer](https://github.com/fastxyz/skill-optimizer) - Docker workbench and agent skill for running deterministic evals against agent skills.
- [TruLens](https://github.com/truera/trulens) - Feedback functions for evaluating and tracing chains and agents.

### Retrieval

- [BEIR](https://github.com/beir-cellar/beir) - Benchmark suite covering dense, sparse, and hybrid retrieval tasks.
- [LlamaIndex Evaluation](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/) - Modules for scoring retrievers and comparing query engines.
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Embedding benchmark measuring retrieval, reranking, and similarity quality.
- [Open RAG Eval](https://github.com/vectara/open-rag-eval) - Vectara harness with UMBRELA and AutoNuggetizer metrics that need no golden answers.
- [Ragas](https://github.com/vibrantlabsai/ragas) - Metrics for answer quality, context relevance, and faithfulness in RAG pipelines.
- [RAGTruth](https://github.com/ParticleMedia/RAGTruth) - Human-annotated hallucination corpus for RAG answers.
- [UltraRAG](https://github.com/OpenBMB/UltraRAG) - MCP-based RAG framework with built-in evaluation workflows.

### Adversarial

- [AgentDojo](https://github.com/ethz-spylab/agentdojo) - Environment for testing prompt-injection attacks and defenses on tool-using agents.
- [DeepTeam](https://github.com/confident-ai/deepteam) - Red teaming framework for bias, data leakage, and prompt-injection vulnerabilities.
- [Garak](https://github.com/NVIDIA/garak) - NVIDIA vulnerability scanner that probes LLMs for prompt injection, leakage, and jailbreaks.
- [Petri](https://github.com/meridianlabs-ai/inspect_petri) - Auditing agent that explores model behavior through simulated multi-turn scenarios.
- [PyRIT](https://github.com/microsoft/PyRIT) - Microsoft toolkit for automating attack strategies against generative AI systems.

## Platforms

### Self-Hosted

- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - OpenTelemetry-native tracing and evaluation for RAG, LLMs, and agents.
- [Giskard](https://github.com/giskard-ai/giskard-oss) - Testing library that scans LLM apps for vulnerabilities and quality issues.
- [Helicone](https://github.com/Helicone/helicone) - AI gateway and observability platform with cost tracking and evaluation integrations.
- [Langfuse](https://github.com/langfuse/langfuse) - Engineering platform for tracing, evals, datasets, and prompt management.
- [Latitude](https://github.com/latitude-dev/latitude-llm) - Platform for observing agent traces and tracking recurring failures as issues.
- [OpenLIT](https://github.com/openlit/openlit) - OpenTelemetry instrumentation for LLM apps with quality metrics and guardrail hooks.
- [Opik](https://github.com/comet-ml/opik) - Evaluation and observability platform with datasets, scoring jobs, and trace inspection.
- [Rhesis](https://github.com/rhesis-ai/rhesis) - Collaborative testing platform with test generation and multi-turn conversation simulation.
- [Zeno](https://zenoml.com/) - Evaluation UI for slicing failures and comparing models and prompts.

### Managed

- [Athina AI](https://www.athina.ai/) - Evaluation and monitoring platform with preset evals and VPC deployment.
- [Braintrust](https://www.braintrust.dev/) - Evaluation and observability platform with experiments, datasets, and CI regression tests.
- [Confident AI](https://www.confident-ai.com/) - DeepEval-backed platform for scheduled eval suites and production monitoring.
- [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/) - LLM traces, quality metrics, and safety signals inside Datadog.
- [Deepchecks LLM Evaluation](https://www.deepchecks.com/solutions/llm-evaluation/) - Managed eval suites with dataset versioning and alerting.
- [Future AGI](https://futureagi.com/) - Evaluation, simulation, and optimization platform for multimodal AI systems.
- [Galileo](https://www.galileo.ai/) - Observability and evaluation platform with built-in metrics and guardrails.
- [HoneyHive](https://www.honeyhive.ai/) - Agent observability and evaluation platform with prompt versioning.
- [LangSmith](https://smith.langchain.com/) - Tracing, datasets, and batch evals with regression gating.
- [Lunary](https://lunary.ai/) - Observability and prompt management platform for LLM apps.
- [PromptLayer](https://www.promptlayer.com/) - Prompt management with version control and evaluation tracking.
- [W&B Weave](https://wandb.ai/site/evaluations/) - Evaluation toolkit with dataset versioning, scorers, and comparison dashboards.

### Cloud

- [Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/) - Managed scoring of foundation models and RAG pipelines.
- [Google Gen AI Evaluation Service](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview) - Rubric-based evaluation of models and agents on Google Cloud.
- [Microsoft Foundry Evaluations](https://learn.microsoft.com/en-us/azure/foundry/how-to/evaluate-generative-ai-app) - Built-in evaluators and reports for generative AI apps.

## Benchmarks

### General

- [AGIEval](https://github.com/ruixiangcui/AGIEval) - Human-centric exams spanning admission tests, law, and math.
- [ARC-AGI-2](https://arcprize.org/arc-agi/2/) - Abstract reasoning puzzles that are easy for people and hard for models.
- [CommonGen-Eval](https://github.com/allenai/CommonGen-Eval) - GPT-4 judged suite for constrained commonsense text generation.
- [DyVal](https://arxiv.org/abs/2309.17167) - Dynamically generated reasoning tasks with controllable difficulty.
- [Humanity's Last Exam](https://github.com/centerforaisafety/hle) - Expert-written closed-ended questions across academic subjects.
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) - Prompts with verifiable constraints for checking instruction compliance.
- [JudgeBench](https://arxiv.org/abs/2410.12784) - Benchmark for LLM judges on hard response pairs across knowledge, reasoning, math, and coding.
- [LiveBench](https://github.com/LiveBench/LiveBench) - Contamination-limited benchmark with regularly refreshed questions and objective scoring.
- [LLM-Uncertainty-Bench](https://github.com/smartyfh/LLM-Uncertainty-Bench) - Uncertainty-aware scoring across QA, reading comprehension, inference, dialog, and summarization.
- [LLMBar](https://github.com/princeton-nlp/LLMBar) - Meta-evaluation of whether LLM judges can spot instruction-following failures.
- [MMLU](https://github.com/hendrycks/test) - Multiple-choice questions across academic and professional subjects. The standard knowledge benchmark; complete and no longer updated.
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) - Harder ten-option extension of MMLU focused on reasoning.
- [PertEval](https://github.com/aigc-apps/PertEval) - Knowledge-invariant perturbations that expose inflated multiple-choice accuracy.
- [SimpleBench](https://simple-bench.com/) - Everyday reasoning questions where people still outperform frontier models.
- [StructEval](https://github.com/TIGER-AI-Lab/StructEval) - Structured-output generation and conversion across 18 text and renderable formats.

### Long-Context

- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) - Twelve tasks with average context length beyond 100K tokens.
- [LongBench v2](https://longbench2.github.io/) - Questions over contexts from 8K to 2M words in six task categories.
- [LongGenBench](https://arxiv.org/abs/2409.02076) - Long-form generation benchmark for long-context models.
- [LV-Eval](https://github.com/infinigence/LVEval) - Five length tiers up to 256K words with distraction controls.
- [RULER](https://github.com/NVIDIA/RULER) - Synthetic long-context tasks with configurable sequence length.

### Specialized

- [FinanceBench](https://www.patronus.ai/announcements/patronus-ai-launches-financebench-the-industrys-first-benchmark-for-llm-performance-on-financial-questions) - Questions over public company filings.
- [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval) - Chinese financial knowledge and reasoning across regulation, accounting, and markets.
- [HumanEval](https://github.com/openai/human-eval) - Hand-written programming problems checked by unit tests. Complete and no longer updated.
- [LAiW](https://github.com/Dai-shen/LAiW) - Chinese legal benchmark covering retrieval, inference, and case application.
- [MATH](https://github.com/hendrycks/math) - Competition mathematics problems with step-by-step solutions.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Entry-level Python programming problems with tests.
- [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/) - Clinician-validated medical tasks scored with an LLM jury.

### Agents

- [AgentBench](https://github.com/THUDM/AgentBench) - LLMs acting as agents across operating system, database, web, and game environments.
- [AstaBench](https://allenai.org/blog/astabench) - Scientific research tasks covering literature review, replication, and data analysis.
- [BrowseComp](https://openai.com/index/browsecomp/) - Questions that require persistent web browsing to answer.
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Gym environment for web agents with access to WebArena, WorkArena, and other benchmarks.
- [ClawBench](https://github.com/TIGER-AI-Lab/ClawBench) - Browser agents on everyday tasks across live websites, with full action and HTTP traces.
- [ColBench](https://arxiv.org/abs/2503.15478) - Multi-turn collaborative coding and design tasks with a simulated human partner.
- [Context-Bench](https://www.letta.com/blog/context-bench) - Measures how well agents manage context and memory.
- [DPAI Arena](https://dpaia.dev/) - Benchmark for developer agents across languages and engineering workflows.
- [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) - General assistant questions requiring web access, tools, and planning.
- [MetaTool](https://github.com/HowieHwong/MetaTool) - Tests whether models know when to use a tool and which one to pick.
- [MLE-bench](https://github.com/openai/mle-bench) - Kaggle competitions measuring agents on machine learning engineering.
- [OSWorld](https://github.com/xlang-ai/OSWorld) - Computer tasks in real desktop environments for multimodal agents.
- [SuperCLUE-Agent](https://github.com/CLUEbenchmark/SuperCLUE-Agent) - Chinese agent evaluation covering tool use, planning, and memory.
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) - Real GitHub issues that coding agents must resolve.
- [SWE-bench Live](https://swe-bench-live.github.io/) - Continuously refreshed issues to limit contamination.
- [SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro) - Long-horizon coding problems drawn from public, commercial, and held-out repositories.
- [tau2-bench](https://github.com/sierra-research/tau2-bench) - Customer-service simulations testing tool use and policy adherence against a simulated user.
- [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) - Continuously updated benchmark of difficult command-line tasks for agents.
- [WebArena](https://github.com/web-arena-x/webarena) - Self-hosted websites for evaluating autonomous web agents.

### Multimodal

- [MERLIM](https://arxiv.org/abs/2312.02219) - 300K+ image-question pairs targeting cross-modal hallucination.
- [MME](https://arxiv.org/abs/2306.13394) - Perception and cognition across 14 subtasks.
- [MMMU-Pro](https://mmmu-benchmark.github.io/) - Harder version of MMMU with expert-level multimodal questions.
- [MMT-Bench](https://github.com/OpenGVLab/MMT-Bench) - 31K+ questions across image, text, video, and point-cloud inputs.
- [Video-MME](https://video-mme.github.io/) - Video understanding across short, medium, and long clips.

### Safety

- [AdvBench](https://github.com/llm-attacks/llm-attacks) - Harmful behaviors and strings for measuring jailbreak resistance.
- [Anthropic Model-Written Evals](https://github.com/anthropics/evals) - Model-written datasets for testing persona, sycophancy, and AI risk behaviors.
- [BBQ](https://github.com/nyu-mll/BBQ) - Question sets measuring reliance on social stereotypes.
- [HarmBench](https://github.com/centerforaisafety/HarmBench) - Standardized evaluation of automated red teaming methods and model robustness.
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) - Robustness benchmark with a jailbreak artifact repository and leaderboard.
- [SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) - 100 prompts covering five critical harm areas.
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Adversarially written questions that tempt models into common falsehoods.

## Leaderboards

- [ARC Prize Leaderboard](https://arcprize.org/leaderboard) - ARC-AGI scores for frontier models and open submissions.
- [Arena](https://arena.ai/) - Crowdsourced head-to-head model voting, formerly LMArena.
- [CompassRank](https://rank.opencompass.org.cn/home) - OpenCompass rankings across multi-domain suites.
- [HELM Leaderboards](https://crfm.stanford.edu/helm/latest/) - Stanford results across HELM scenarios and metrics.
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - Hugging Face rankings of open models on IFEval, MMLU-Pro, GPQA, and more.
- [Open Medical-LLM Leaderboard](https://huggingface.co/blog/leaderboard-medicalllm) - Hugging Face rankings on medical question answering.
- [Scale SEAL Leaderboard](https://scale.com/leaderboard) - Expert-rated rankings for reasoning, coding, and safety.

## Resources

### Papers

- [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2510.04265) - Bayesian scores with uncertainty (ICLR 2026). [Video](https://mohsenhariri.github.io/assets/videos/bayes-at-n/bayes-at-n.mp4). [Code](https://github.com/mohsenhariri/scorio).
- [LLM-as-a-Judge Survey](https://arxiv.org/abs/2411.15594) - Survey on building reliable LLM judges and mitigating their biases.
- [RAG Evaluation Survey](https://arxiv.org/abs/2405.07437) - Survey of metrics, judgments, and open problems in evaluating RAG.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) - Paper on aligning LLM judges with human criteria through the EvalGen workflow.

### Guides

- [AI Evals for Engineers & PMs](https://maven.com/parlance-labs/evals) - Cohort course by Hamel Husain and Shreya Shankar.
- [AlignEval](https://eugeneyan.com/writing/aligneval/) - Eugene Yan on building and aligning LLM judges.
- [Applied LLMs](https://applied-llms.org/) - Lessons from a year of building with LLMs, with evaluation as a core practice.
- [Comparing LLMs on Real-World Retrieval](https://www.sh-reya.com/blog/needle-in-the-real-world/) - How models perform on practical retrieval tasks.
- [Data Flywheels for LLM Applications](https://www.sh-reya.com/blog/ai-engineering-flywheel/) - Iterative data improvement for LLM systems.
- [Eval Tools Comparison](https://hamel.dev/blog/posts/eval-tools/) - Comparison of Braintrust, LangSmith, and Promptfoo.
- [Evals for AI Engineers](https://www.oreilly.com/library/view/evals-for-ai/9798341660717/) - O'Reilly book on error analysis, evaluation pipelines, and LLM judges.
- [EvalScope RAG](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html) - Tutorial on evaluating RAG applications with Ragas, from the EvalScope documentation.
- [Field Guide to Rapidly Improving AI Products](https://hamel.dev/blog/posts/field-guide/) - Error analysis, data viewers, and systematic improvement from 30+ implementations.
- [KDD 2025 Tutorial: Evaluation & Benchmarking of LLM Agents](https://sap-samples.github.io/llm-agents-eval-tutorial/) - Academic tutorial on agent evaluation methodology.
- [LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) - Answers to common evaluation questions from practitioners.
- [LLM Evaluators Survey](https://eugeneyan.com/writing/llm-evaluators/) - Survey of LLM judge use cases and implementation patterns.
- [LLM-as-a-Judge Guide](https://hamel.dev/blog/posts/llm-judge/) - Using LLMs as judges, with calibration advice.
- [Mastering LLMs Open Course](https://parlance-labs.com/education/) - Free course covering evals, RAG, and fine-tuning.
- [Microsoft Foundry Safety Evaluations Transparency Note](https://learn.microsoft.com/en-us/azure/foundry/concepts/safety-evaluations-transparency-note) - Microsoft documentation of how the safety evaluators work and where they fall short.
- [Open Source LLM Tools Comparison](https://posthog.com/blog/best-open-source-llm-observability-tools) - PostHog comparison of open-source observability and evaluation tools.
- [OpenAI Cookbook Evals](https://github.com/openai/openai-cookbook/tree/main/examples/evals) - Notebooks showing how to build custom evals.
- [Why AI Evals Are the Hottest New Skill](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill) - Interview covering error analysis, axial coding, and eval prompts.
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) - Why every AI product needs systematic evaluation.
- [ZenML Evaluation Playbook](https://www.zenml.io/blog/the-evaluation-playbook-making-llms-production-ready) - Guide to adding evaluation gates to pipelines and deployments.

### Lists

- [awesome-evals](https://github.com/benchflow-ai/awesome-evals) - Resources for building and evaluating AI agents.
- [Awesome-LLM-Eval](https://github.com/onejune2018/Awesome-LLM-Eval) - Evaluation tools, papers, datasets, and leaderboards, in Chinese and English.
- [Awesome-RAG-Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) - RAG evaluation metrics, datasets, and leaderboards.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
