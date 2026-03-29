# CORE-Bench Hard

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/siegelz/CoreBenchHard)

## Description

**CORE-Bench Hard** is an environment for evaluating computational reproducibility of scientific code. Agents receive a code repository from a published paper and must figure out how to install dependencies, run the code, and answer specific questions about the output — without reproduction instructions or pre-computed results.

## Capabilities

- Code comprehension and dependency installation
- Scientific code execution (Python and R)
- Result extraction and reporting

## Compute Requirements

Agents are given a sandboxed Docker environment. Default sandbox size is 1 CPU and 4 GB RAM. Network access enabled (agents may need to install dependencies). No GPU.

## License

See the [CORE-Bench dataset page](https://huggingface.co/datasets/siegelz/core-bench) for license information.

## Tasks

- **Train split**: 45 capsules (unencrypted)
- **Test split**: 45 capsules (GPG-encrypted with passphrase "reproducibility")
- Fields: Computer Science, Medical Sciences, Social Sciences
- Languages: Python, R

## Reward Structure

Binary reward (matching original CORE-Bench methodology):
- **1.0**: All questions in report.json answered correctly
- **0.0**: Any question wrong or report.json missing/invalid

Answer comparison:
- **Numeric**: Must fall within a 95% prediction interval (computed from multiple gold runs)
- **String**: Case-insensitive, trailing punctuation stripped
- **List**: Exact match

## Data

- **Source**: [siegelz/core-bench](https://huggingface.co/datasets/siegelz/core-bench) on HuggingFace (task metadata)
- **Capsule tarballs**: Pre-downloaded from `corebench.cs.princeton.edu` and staged in `sandbox_data/{capsule_id}/` for bucket mounting (~13GB total)
- **Test encryption**: GPG with passphrase "reproducibility"

## Tools

- **`bash`**: Execute shell commands in the sandbox (install deps, run code, create report.json)
- **`submit`**: Submit report.json for evaluation (terminal action, one attempt)

## Time Horizon

Multi-turn. Agents typically need many bash calls to explore code, install dependencies, run scripts, and create report.json. Expected: 10–50+ tool calls.

## Environment Difficulty

Hard. Agents must independently figure out how to reproduce scientific code without explicit instructions. Requires reading READMEs, installing correct dependencies, and running code end-to-end.

## Safety

Code is executed in an isolated sandbox. Capsule tarballs are from published scientific papers hosted by Princeton.

## Citations

```bibtex
@article{siegel2024corebench,
  title={CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark},
  author={Siegel, Zachary S. and Kapoor, Sayash and Nagdir, Nitya and Stroebl, Benedikt and Narayanan, Arvind},
  journal={arXiv preprint arXiv:2409.11363},
  year={2024}
}

@article{bragg2025astabench,
      title={AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite},
      author={Bragg, Jonathan and D'Arcy, Mike and Balepur, Nishant and Bareket, Dan and Dalvi, Bhavana and Feldman, Sergey and Haddad, Dany and Hwang, Jena D. and Jansen, Peter and Kishore, Varsha and Majumder, Bodhisattwa Prasad and Naik, Aakanksha and Rahamimov, Sigal and Richardson, Kyle and Singh, Amanpreet and Surana, Harshit and Tiktinsky, Aryeh and Vasu, Rosni and Wiener, Guy and Anastasiades, Chloe and Candra, Stefan and Dunkelberger, Jason and Emery, Dan and Evans, Rob and Hamada, Malachi and Huff, Regan and Kinney, Rodney and Latzke, Matt and Lochner, Jaron and Lozano-Aguilera, Ruben and Nguyen, Cecile and Rao, Smita and Tanaka, Amber and Vlahos, Brooke and Clark, Peter and Downey, Doug and Goldberg, Yoav and Sabharwal, Ashish and Weld, Daniel S.},
      journal={arXiv preprint arXiv:2510.21652},
      year={2025},
}
```
