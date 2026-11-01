# Awesome AI Eval [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<a href="https://github.com/Vvkmnn/awesome-ai-eval"><img src="./assets/robot-shades.svg" align="right" width="120" alt="Robot wearing pink shades"></a>

> Assessing the reliability, accuracy, and safety of large language models, RAG pipelines, and AI agents.

## Contents

- [Benchmarks](#benchmarks)
  - [Agents](#agents)
  - [General](#general)
  - [Long-Context](#long-context)
  - [Multimodal](#multimodal)
  - [Safety](#safety)
  - [Specialized](#specialized)
- [Leaderboards](#leaderboards)
- [Platforms](#platforms)
  - [Cloud](#cloud)
  - [Managed](#managed)
  - [Self-Hosted](#self-hosted)
- [Resources](#resources)
  - [Guides](#guides)
  - [Lists](#lists)
  - [Papers](#papers)
- [Tools](#tools)
  - [Adversarial](#adversarial)
  - [Frameworks](#frameworks)
  - [Retrieval](#retrieval)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Benchmarks

### Agents

- [Agent Retry Rodeo](https://github.com/RishabhPande/agent-retry-rodeo) - Local test range that replays lost acknowledgements and partial failures, then checks an agent's final state.
- [AgentBench](https://github.com/THUDM/AgentBench) - LLMs acting as agents across operating system, database, web, and game environments.
- [AppWorld](https://github.com/StonyBrookNLP/appworld) - Controllable world of apps and simulated people for benchmarking function calling and interactive coding agents.
- [AstaBench](https://allenai.org/blog/astabench) - Scientific research tasks covering literature review, replication, and data analysis.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) - Measures how accurately models call functions and tools.
- [BrowseComp](https://openai.com/index/browsecomp/) - Questions that require persistent web browsing to answer.
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Gym environment for web agents with access to WebArena, WorkArena, and other benchmarks.
- [ClawBench](https://github.com/TIGER-AI-Lab/ClawBench) - Browser agents on everyday tasks across live websites, with full action and HTTP traces.
- [ColBench](https://arxiv.org/abs/2503.15478) - Multi-turn collaborative coding and design tasks with a simulated human partner.
- [Context-Bench](https://www.letta.com/blog/context-bench) - Measures how well agents manage context and memory.
- [DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench) - PhD-level research tasks for scoring deep research agents on report quality and citations.
- [DPAI Arena](https://dpaia.dev/) - Benchmark for developer agents across languages and engineering workflows.
- [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) - General assistant questions requiring web access, tools, and planning.
- [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe) - Benchmark and framework for evaluating agents on real Model Context Protocol servers.
- [MetaTool](https://github.com/HowieHwong/MetaTool) - Tests whether models know when to use a tool and which one to pick.
- [MLE-bench](https://github.com/openai/mle-bench) - Kaggle competitions measuring agents on machine learning engineering.
- [OpenAI Frontier Evals](https://github.com/openai/frontier-evals) - PaperBench, SWE-Lancer, and EVMbench: paper replication, freelance coding, and smart contract security tasks.
- [OSWorld](https://github.com/xlang-ai/OSWorld) - Computer tasks in real desktop environments for multimodal agents.
- [ShopPay Audit Benchmark](https://github.com/Dmatut7/shoppay-audit-benchmark) - Spec-grounded business-logic audit tasks for coding agents.
- [SuperCLUE-Agent](https://github.com/CLUEbenchmark/SuperCLUE-Agent) - Chinese agent evaluation covering tool use, planning, and memory.
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) - Real GitHub issues that coding agents must resolve.
- [SWE-bench Live](https://swe-bench-live.github.io/) - Continuously refreshed issues to limit contamination.
- [SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro) - Long-horizon coding problems drawn from public, commercial, and held-out repositories.
- [tau2-bench](https://github.com/sierra-research/tau2-bench) - Customer-service simulations testing tool use and policy adherence against a simulated user.
- [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) - Continuously updated benchmark of difficult command-line tasks for agents.
- [TIB-AK79](https://github.com/andruwka79-cloud/tib-ak79) - Scores what agent actions changed in a mock environment and checks the agent's claims against the result.
- [WebArena](https://github.com/web-arena-x/webarena) - Self-hosted websites for evaluating autonomous web agents.

### General

- [AGIEval](https://github.com/ruixiangcui/AGIEval) - Human-centric exams spanning admission tests, law, and math.
- [ARC-AGI-2](https://arcprize.org/arc-agi/2/) - Abstract reasoning puzzles that are easy for people and hard for models.
- [ARC-AGI-3](https://arcprize.org/arc-agi/3/) - Interactive reasoning benchmark of novel game environments in which agents must discover the goals.
- [CommonGen-Eval](https://github.com/allenai/CommonGen-Eval) - GPT-4 judged suite for constrained commonsense text generation.
- [DyVal](https://arxiv.org/abs/2309.17167) - Dynamically generated reasoning tasks with controllable difficulty.
- [GDPval](https://arxiv.org/abs/2510.04374) - Economically valuable tasks built from the work of professionals in 44 occupations.
- [GPQA](https://github.com/idavidrein/gpqa) - Graduate-level science questions written by experts and designed to resist web search.
- [Humanity's Last Exam](https://github.com/centerforaisafety/hle) - Expert-written closed-ended questions across academic subjects.
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) - Prompts with verifiable constraints for checking instruction compliance.
- [JudgeBench](https://arxiv.org/abs/2410.12784) - Benchmark for LLM judges on hard response pairs across knowledge, reasoning, math, and coding.
- [LiveBench](https://github.com/LiveBench/LiveBench) - Contamination-limited benchmark with regularly refreshed questions and objective scoring.
- [LLM-Uncertainty-Bench](https://github.com/smartyfh/LLM-Uncertainty-Bench) - Uncertainty-aware scoring across QA, reading comprehension, inference, dialog, and summarization.
- [LLMBar](https://github.com/princeton-nlp/LLMBar) - Meta-evaluation of whether LLM judges can spot instruction-following failures.
- [MMLU](https://github.com/hendrycks/test) - Multiple-choice questions across academic and professional subjects. The standard knowledge benchmark; complete and no longer updated.
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) - Harder ten-option extension of MMLU focused on reasoning.
- [PertEval](https://github.com/aigc-apps/PertEval) - Knowledge-invariant perturbations that expose inflated multiple-choice accuracy.
- [RewardBench](https://github.com/allenai/reward-bench) - Benchmark and leaderboard for evaluating reward models.
- [SimpleBench](https://simple-bench.com/) - Everyday reasoning questions where people still outperform frontier models.
- [SimpleQA](https://arxiv.org/abs/2411.04368) - Short fact-seeking questions with a single indisputable answer, for measuring factuality.
- [StructEval](https://github.com/TIGER-AI-Lab/StructEval) - Structured-output generation and conversion across 18 text and renderable formats.

### Long-Context

- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) - Twelve tasks with average context length beyond 100K tokens.
- [LongBench v2](https://longbench2.github.io/) - Questions over contexts from 8K to 2M words in six task categories.
- [LongGenBench](https://arxiv.org/abs/2409.02076) - Long-form generation benchmark for long-context models.
- [LV-Eval](https://github.com/infinigence/LVEval) - Five length tiers up to 256K words with distraction controls.
- [RULER](https://github.com/NVIDIA/RULER) - Synthetic long-context tasks with configurable sequence length.

### Multimodal

- [MERLIM](https://arxiv.org/abs/2312.02219) - 300K+ image-question pairs targeting cross-modal hallucination.
- [MME](https://arxiv.org/abs/2306.13394) - Perception and cognition across 14 subtasks.
- [MMMU-Pro](https://mmmu-benchmark.github.io/) - Harder version of MMMU with expert-level multimodal questions.
- [MMT-Bench](https://github.com/OpenGVLab/MMT-Bench) - 31K+ questions across image, text, video, and point-cloud inputs.
- [PaperEdits Agentic Video Benchmark](https://paperedits.com/benchmarking/gemini-agentic-video-understanding-benchmark) - Exploratory benchmark of agentic and static video processing on long-form editing tasks.
- [Video-MME](https://video-mme.github.io/) - Video understanding across short, medium, and long clips.

### Safety

- [AdvBench](https://github.com/llm-attacks/llm-attacks) - Harmful behaviors and strings for measuring jailbreak resistance.
- [Anthropic Model-Written Evals](https://github.com/anthropics/evals) - Model-written datasets for testing persona, sycophancy, and AI risk behaviors.
- [BBQ](https://github.com/nyu-mll/BBQ) - Question sets measuring reliance on social stereotypes.
- [Cybench](https://github.com/andyzorigin/cybench) - Capture-the-flag tasks for evaluating the cybersecurity capabilities and risks of agents.
- [CyberSecEval](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks) - Meta benchmarks measuring cybersecurity risks and defensive capabilities of language models.
- [HarmBench](https://github.com/centerforaisafety/HarmBench) - Standardized evaluation of automated red teaming methods and model robustness.
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) - Robustness benchmark with a jailbreak artifact repository and leaderboard.
- [SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) - 100 prompts covering five critical harm areas.
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Adversarially written questions that tempt models into common falsehoods.

### Specialized

- [FinanceBench](https://www.patronus.ai/announcements/patronus-ai-launches-financebench-the-industrys-first-benchmark-for-llm-performance-on-financial-questions) - Questions over public company filings.
- [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval) - Chinese financial knowledge and reasoning across regulation, accounting, and markets.
- [FrontierMath](https://epoch.ai/frontiermath) - Epoch AI benchmark of original, exceptionally hard mathematics problems.
- [HealthBench](https://arxiv.org/abs/2505.08775) - Multi-turn health conversations graded against rubrics written by physicians.
- [HumanEval](https://github.com/openai/human-eval) - Hand-written programming problems checked by unit tests. Complete and no longer updated.
- [LAiW](https://github.com/Dai-shen/LAiW) - Chinese legal benchmark covering retrieval, inference, and case application.
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) - Contamination-free code evaluation built from newly published competitive programming problems.
- [MATH](https://github.com/hendrycks/math) - Competition mathematics problems with step-by-step solutions.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Entry-level Python programming problems with tests.
- [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/) - Clinician-validated medical tasks scored with an LLM jury.

## Leaderboards

- [ARC Prize Leaderboard](https://arcprize.org/leaderboard) - ARC-AGI scores for frontier models and open submissions.
- [Arena](https://arena.ai/) - Crowdsourced head-to-head model voting, formerly LMArena.
- [CompassRank](https://rank.opencompass.org.cn/home) - OpenCompass rankings across multi-domain suites.
- [Council of AI GSPC](https://github.com/CSOAI-ORG/councilof-ai) - Public board of signed AI behaviour measurements across governance and safety axes.
- [Epoch AI Benchmarking Hub](https://epoch.ai/benchmarks) - Epoch AI results for leading models on challenging benchmarks.
- [HELM Leaderboards](https://crfm.stanford.edu/helm/latest/) - Stanford results across HELM scenarios and metrics.
- [ImageBench](https://imagebench.ai/) - Text-to-image comparisons with side-by-side results and published methodology.
- [MathArena](https://matharena.ai/) - Evaluation of LLMs on the latest math competitions, run after each contest.
- [ModelBenchmark](https://modelbenchmark.io) - Public benchmark scores, prices, and context windows gathered per model.
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - Hugging Face rankings of open models on IFEval, MMLU-Pro, GPQA, and more.
- [Open Medical-LLM Leaderboard](https://huggingface.co/blog/leaderboard-medicalllm) - Hugging Face rankings on medical question answering.
- [Scale SEAL Leaderboard](https://scale.com/leaderboard) - Expert-rated rankings for reasoning, coding, and safety.
- [Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) - Ranks LLMs by how often they hallucinate when summarizing short documents.

## Platforms

### Cloud

- [Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/) - Managed scoring of foundation models and RAG pipelines.
- [Google Gen AI Evaluation Service](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview) - Rubric-based evaluation of models and agents on Google Cloud.
- [Microsoft Foundry Evaluations](https://learn.microsoft.com/en-us/azure/foundry/how-to/evaluate-generative-ai-app) - Built-in evaluators and reports for generative AI apps.

### Managed

- [Athina AI](https://www.athina.ai/) - Evaluation and monitoring platform with preset evals and VPC deployment.
- [Braintrust](https://www.braintrust.dev/) - Evaluation and observability platform with experiments, datasets, and CI regression tests.
- [ChatIntel](https://chatintel.ai/) - No-code toolkit for evaluating and measuring the consistency of AI output.
- [Confident AI](https://www.confident-ai.com/) - DeepEval-backed platform for scheduled eval suites and production monitoring.
- [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/) - LLM traces, quality metrics, and safety signals inside Datadog.
- [Deepchecks LLM Evaluation](https://www.deepchecks.com/solutions/llm-evaluation/) - Managed eval suites with dataset versioning and alerting.
- [Firedrill](https://docs.firedrill.run/) - Simulates AI agents with optional synthetic tools, semantic scoring, exact checks, and retained evidence in customer-owned runners.
- [Future AGI](https://futureagi.com/) - Evaluation, simulation, and optimization platform for multimodal AI systems.
- [Galileo](https://www.galileo.ai/) - Observability and evaluation platform with built-in metrics and guardrails.
- [HoneyHive](https://www.honeyhive.ai/) - Agent observability and evaluation platform with prompt versioning.
- [Jetty](https://jetty.io/) - Runs agent workloads in sandboxes and scores each run with checks and LLM judges.
- [LangSmith](https://smith.langchain.com/) - Tracing, datasets, and batch evals with regression gating.
- [Lunary](https://lunary.ai/) - Observability and prompt management platform for LLM apps.
- [PromptLayer](https://www.promptlayer.com/) - Prompt management with version control and evaluation tracking.
- [Respan](https://www.respan.ai/) - AI gateway with tracing, monitoring, red teaming, and evaluators that score test sets and production traffic.
- [TestMu AI Agent Testing](https://www.testmuai.com/agent-testing/) - Tests chat, voice, and phone agents through simulated conversations and real calls, including their tool calls.
- [W&B Weave](https://wandb.ai/site/evaluations/) - Evaluation toolkit with dataset versioning, scorers, and comparison dashboards.

### Self-Hosted

- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - OpenTelemetry-native tracing and evaluation for RAG, LLMs, and agents.
- [Bifrost](https://github.com/maximhq/bifrost) - Self-hosted AI gateway with LLM-judge guardrails and evaluation integrations such as Patronus AI.
- [Giskard](https://github.com/giskard-ai/giskard-oss) - Testing library that scans LLM apps for vulnerabilities and quality issues.
- [Helicone](https://github.com/Helicone/helicone) - AI gateway and observability platform with cost tracking and evaluation integrations.
- [Langfuse](https://github.com/langfuse/langfuse) - Engineering platform for tracing, evals, datasets, and prompt management.
- [Latitude](https://github.com/latitude-dev/latitude-llm) - Platform for observing agent traces and tracking recurring failures as issues.
- [OpenLIT](https://github.com/openlit/openlit) - OpenTelemetry instrumentation for LLM apps with quality metrics and guardrail hooks.
- [Opik](https://github.com/comet-ml/opik) - Evaluation and observability platform with datasets, scoring jobs, and trace inspection.
- [Rhesis](https://github.com/rhesis-ai/rhesis) - Collaborative testing platform with test generation and multi-turn conversation simulation.
- [Zeno](https://zenoml.com/) - Evaluation UI for slicing failures and comparing models and prompts.

## Resources

### Guides

- [AI Evals for Engineers & PMs](https://maven.com/parlance-labs/evals) - Cohort course by Hamel Husain and Shreya Shankar.
- [AI Failure Mode Taxonomy](https://github.com/prashibadkur11-creator/ai-failure-mode-taxonomy) - Catalog of AI and LLM product failure modes with detection and mitigation for each.
- [AlignEval](https://eugeneyan.com/writing/aligneval/) - Eugene Yan on building and aligning LLM judges.
- [Applied LLMs](https://applied-llms.org/) - Lessons from a year of building with LLMs, with evaluation as a core practice.
- [Comparing LLMs on Real-World Retrieval](https://www.sh-reya.com/blog/needle-in-the-real-world/) - How models perform on practical retrieval tasks.
- [Data Flywheels for LLM Applications](https://www.sh-reya.com/blog/ai-engineering-flywheel/) - Iterative data improvement for LLM systems.
- [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) - Anthropic guide to grader types, consistency metrics, and building an agent eval suite.
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

- [Awesome Agent Evals](https://github.com/benchflow-ai/awesome-evals) - Annotated papers, posts, talks, tools, and benchmarks for building and evaluating AI agents.
- [Awesome Red Teaming LLMs](https://github.com/dapurv5/awesome-red-teaming-llms) - Papers on red teaming language models, from a survey accepted at TMLR.
- [Awesome-LLM-as-a-judge](https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge) - Papers on using LLMs as judges, from the accompanying survey.
- [Awesome-LLM-Eval](https://github.com/onejune2018/Awesome-LLM-Eval) - Evaluation tools, papers, datasets, and leaderboards, in Chinese and English.
- [Awesome-RAG-Evaluation](https://github.com/YHPeter/Awesome-RAG-Evaluation) - RAG evaluation metrics, datasets, and leaderboards.
- [LLM Evaluation Survey](https://github.com/MLGroupJLU/LLM-eval-survey) - Papers organized by the survey "A Survey on Evaluation of Large Language Models".
- [LLM-Agent-Benchmark-List](https://github.com/zhangxjohn/LLM-Agent-Benchmark-List) - Benchmarks for evaluating LLMs and LLM-based agents.

### Papers

- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) - Statistical methods for analyzing and reporting uncertainty in language model evaluations.
- [Don't Pass@k](https://arxiv.org/abs/2510.04265) - Bayesian framework that reports LLM evaluation results with uncertainty in place of pass@k.
- [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825) - Checklist for avoiding task and reward design flaws in agent benchmarks.
- [LLM-as-a-Judge Survey](https://arxiv.org/abs/2411.15594) - Survey on building reliable LLM judges and mitigating their biases.
- [RAG Evaluation Survey](https://arxiv.org/abs/2405.07437) - Survey of metrics, judgments, and open problems in evaluating RAG.
- [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879) - Analysis of systematic issues that distort Chatbot Arena rankings.
- [What's Missing in Autonomous Research?](https://www.researchgate.net/publication/406952713) - Survey mapping autonomous research systems against their benchmarks and verification methods.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) - Paper on aligning LLM judges with human criteria through the EvalGen workflow.

## Tools

### Adversarial

- [AgentDojo](https://github.com/ethz-spylab/agentdojo) - Environment for testing prompt-injection attacks and defenses on tool-using agents.
- [DeepTeam](https://github.com/confident-ai/deepteam) - Red teaming framework for bias, data leakage, and prompt-injection vulnerabilities.
- [Garak](https://github.com/NVIDIA/garak) - NVIDIA vulnerability scanner that probes LLMs for prompt injection, leakage, and jailbreaks.
- [Petri](https://github.com/meridianlabs-ai/inspect_petri) - Auditing agent that explores model behavior through simulated multi-turn scenarios.
- [PyRIT](https://github.com/microsoft/PyRIT) - Microsoft toolkit for automating attack strategies against generative AI systems.

### Frameworks

- [Agent Stack](https://github.com/MukundaKatta/agent-stack) - Libraries for agent reliability, including snapshot tests for tool-call traces, tool-argument validation, and structured output with retry.
- [agentest](https://github.com/srpadmanabhuni/agentest) - Offline, deterministic checks of agent OpenTelemetry traces, A2A AgentCards, and MCP tools, with no LLM judge.
- [Agentrial](https://github.com/alepot55/agentrial) - Runs agents repeatedly, computes confidence intervals, and flags regressions in CI.
- [agenttrace](https://github.com/luoyuctl/agenttrace) - Terminal UI for reviewing coding-agent sessions by cost, failures, latency, and diffs.
- [AI Eval Forge](https://github.com/MukundaKatta/ai-eval-forge) - Regression harness for LLM and agent outputs using deterministic checks such as regex, token F1, and JSON validity.
- [AI Prompt Quality](https://github.com/pntkrl/ai-prompt-quality) - Static prompt evaluation with a seven-dimension rubric and JSON scoring schema.
- [Aleph Alpha Eval Framework](https://github.com/Aleph-Alpha-Research/eval-framework) - Evaluation framework with built-in benchmarks for reasoning, coding, and safety.
- [align](https://github.com/ggrigo/align) - Human-in-the-loop evals for Claude Code that rate individual claims and archive corrections.
- [autoevals](https://github.com/braintrustdata/autoevals) - Scorers for LLM outputs, from string heuristics to model-graded factuality, in Python and TypeScript.
- [Bloom](https://github.com/safety-research/bloom) - Agentic framework that generates behavioral evaluations of frontier models from a described behavior.
- [ChainForge](https://github.com/ianarawjo/ChainForge) - Visual environment for comparing prompts and models and scoring responses.
- [ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) - Pipeline combining classic metrics with GPT-assisted scoring across public datasets.
- [DeepEval](https://github.com/confident-ai/deepeval) - Unit-test style metrics for hallucination, relevance, toxicity, and bias.
- [digline](https://github.com/digline/digline) - Regression testing for LLM apps against a versioned baseline of approved scores.
- [EvalScope](https://github.com/modelscope/evalscope) - ModelScope framework for evaluating and benchmarking LLMs, vision-language models, and generative models.
- [Evidently](https://github.com/evidentlyai/evidently) - Evaluation and monitoring library for ML and LLM systems.
- [fact-gate](https://github.com/AnonZ7/fact-gate) - Deterministic verification of AI-written text against diffs, measured facts, or prose.
- [Harbor](https://github.com/harbor-framework/harbor) - Framework for evaluating and optimizing agents and models in containers; the official Terminal-Bench 2.0 harness.
- [HELM](https://github.com/stanford-crfm/helm) - Stanford framework for evaluating foundation models across scenarios and metrics.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Security Institute framework for scripted evals with tool calls and model-graded scoring.
- [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) - Community-maintained collection of evaluations implemented in Inspect AI.
- [lighteval](https://github.com/huggingface/lighteval) - Hugging Face toolkit for evaluating LLMs across many backends and tasks.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Harness for running language models on hundreds of academic benchmarks.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) - Multimodal evaluation toolkit covering text, image, video, and audio tasks.
- [MLflow](https://github.com/mlflow/mlflow) - Evaluation API that logs judge scores alongside experiment tracking runs.
- [Narrative Scorer](https://github.com/cittaverse/narrative-scorer) - Scoring library for Chinese autobiographical narratives in reminiscence therapy.
- [OpenAI Evals](https://github.com/openai/evals) - Framework and registry of evals for LLMs and LLM systems.
- [OpenAI simple-evals](https://github.com/openai/simple-evals) - Reference implementations of standard benchmarks such as MMLU, GPQA, and HumanEval.
- [OpenCompass](https://github.com/open-compass/opencompass) - Evaluation platform with LLM judge support across knowledge, reasoning, and coding datasets.
- [openevals](https://github.com/langchain-ai/openevals) - Evaluators for LLM apps, including LLM judge and structured-output checks.
- [OpenJudge](https://github.com/agentscope-ai/OpenJudge) - Framework of graders for evaluating agents and chatbots and producing quality rewards.
- [Prompt Flow](https://github.com/microsoft/promptflow) - Flow builder with built-in evaluation flows and CI integration.
- [Promptfoo](https://github.com/promptfoo/promptfoo) - CLI for testing and red teaming prompts, RAG flows, and agents with regression detection.
- [semantix-ai](https://github.com/labrat-akhona/semantix-ai) - NLI judges, datasets, and audit trails for checking LLM outputs against regulatory clauses.
- [skill-optimizer](https://github.com/fastxyz/skill-optimizer) - Docker workbench and agent skill for running deterministic evals against agent skills.
- [TruLens](https://github.com/truera/trulens) - Feedback functions for evaluating and tracing chains and agents.
- [Voice Eval](https://github.com/rand0wn/voice-eval) - Regression harness for voice agents using scripted multi-turn scenarios.

### Retrieval

- [BEIR](https://github.com/beir-cellar/beir) - Benchmark suite covering dense, sparse, and hybrid retrieval tasks.
- [LlamaIndex Evaluation](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/) - Modules for scoring retrievers and comparing query engines.
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Embedding benchmark measuring retrieval, reranking, and similarity quality.
- [Open RAG Eval](https://github.com/vectara/open-rag-eval) - Vectara harness with UMBRELA and AutoNuggetizer metrics that need no golden answers.
- [Ragas](https://github.com/vibrantlabsai/ragas) - Metrics for answer quality, context relevance, and faithfulness in RAG pipelines.
- [RagCiteCheck](https://github.com/ppon1086/ragcitecheck) - Measures document-level and span-level evidence stability in RAG pipelines across runs.
- [RAGTruth](https://github.com/ParticleMedia/RAGTruth) - Human-annotated hallucination corpus for RAG answers.
- [UltraRAG](https://github.com/OpenBMB/UltraRAG) - MCP-based RAG framework with built-in evaluation workflows.

<p align="center"><img src="./assets/footer.svg" width="480" alt=""></p>
