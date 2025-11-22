# LLada-sft

Welcome to LLada-sft — a nimble, elegant fine-tuning toolkit crafted for those who demand clarity, flexibility, and performance. This repository contains the essential scaffolding to prepare and consume training data for streaming, multi-GPU fine-tuning workflows using the Lightning framework.

---

## ✨ Quick Overview

LLada-sft is designed to be straightforward yet powerful. The core responsibilities you need to know about are:

- Preparing your dataset in the required format
- Customizing special data handling logic (if necessary)
- Adjusting configuration files in the `config/` folder
- Running training with PyTorch Lightning

---

## 🗂 Data — Process

To ensure smooth, efficient streaming across multiple devices, please prepare your data as follows:

- Convert your dataset files to JSON Lines format (`.jsonl`).
- Because the loader uses multi-GPU streaming, you must provide at least as many data files as there are GPU devices (i.e., number of files >= number of GPUs).
- Split your dataset into multiple `.jsonl` files yourself (for example, by sharding into N files where N is the number of GPUs or more).

Notes:
- Each line in a `.jsonl` file should be a valid JSON object representing one training example.
- Sharding well helps prevent IO contention and ensures balanced distribution across devices.

---

## 🧩 Custom Data Handling

If your dataset requires special preprocessing or non-standard parsing, override the data processing hook:

- Implement your custom logic by rewriting `utils/process_data`.
- Keep the contract / input-output expectations consistent so downstream components can consume the processed examples seamlessly.

This hook gives you full control over tokenization, field mapping, filtering, augmentation, or any bespoke data transformation your experiments demand.

---

## ⚙️ Configuration

All runtime and training hyperparameters live under the `config/` directory. Use those files to:

- Tune training parameters (batch size, learning rate, scheduler, etc.)
- Configure data paths and behavior for multi-GPU streaming
- Enable or disable special modules or callbacks

Treat `config/` as the source of truth for reproducible experiments.

---

## 🚀 Framework

This project leverages PyTorch Lightning to provide:

- Clean separation between model, training loop, and engineering concerns
- Built-in, battle-tested multi-GPU support
- A concise, extensible training pipeline

If you're familiar with Lightning, you should feel right at home. If not, consult the Lightning docs for patterns and best practices.

---

## 📝 Contributing & Next Steps

Contributions are welcome! If you want me to, I can:
- Polish or expand any section of this README,
- Create example scripts for data sharding or a starter config,
- Open a branch and push the updated README to the repository.

---

Thank you for using LLada-sft — may your experiments be fast, stable, and reproducible!
