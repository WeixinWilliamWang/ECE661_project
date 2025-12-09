# EDeep Generative Priors for Efficient Exploration in Neural Contextual Bandits
2025Fall Duke ECE661 course project

**Course:** ECE 661 - Fall 2025
**Authors:** Weixin Wang (ww211), Zetong Pan (zp74), Yunkai Gao (yg262)

## Project Overview
This repository contains the code for our final course project, which addresses the challenge of efficient exploration in multi-task contextual bandits. While standard Thompson Sampling relies on simple conjugate priors that fail in complex environments, this project proposes using pre-trained diffusion models as flexible, data-driven priors.

**Note:** This codebase contains extensive experimental work and various implementations explored during the semester. However, the final project report specifically selects and analyzes the two algorithms detailed below.

## Key Algorithms
Based on the project report, this repository focuses on the implementation and evaluation of the following proposed methods:

* **Diffusion Langevin Thompson Sampling (DLTS):** A method that expands the standard TS framework by updating the posterior through a reverse diffusion process. It applies Langevin Monte Carlo at each diffusion step to approximate samples from the conditional posterior.
* **Diffusion Posterior Sampling with Guidance and Multi-step Projection (DPSG-MP):** A variant of Diffusion Posterior Sampling that utilizes a single likelihood drift per level via Tweedie's estimate. To improve robustness against biased guidance, we implemented a Multi-step Projection (MP) mechanism that stabilizes learning by projecting updates back onto the hypersphere.

## Experiments
The code includes setups for evaluating these methods against baselines (such as LinUCB, NeuralTS, and NeuralUCB) in the following environments:

1.  **Synthetic Linear Bandits:** Standard linear reward settings.
2.  **Synthetic Nonlinear Bandits:** Environments where rewards are governed by nonlinear functions (e.g., sigmoid-gated functions).
3.  **Real-World Application:** A high-dimensional task constructed from the **MovieLens-25M** dataset, using SVD embeddings for users and movies.

## References
This project builds upon a manuscript currently under review at ICLR and extends it with refined algorithm designs and real-world experiments.

For full algorithmic details, please refer to the project report included in this repository.
