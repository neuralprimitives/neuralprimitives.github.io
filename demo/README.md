
# Neural Primitives 🧩

This repository provides the **demo, inference code, and training pipeline** for **Neural Primitives**, a framework for learning parametric 3D reconstruction via neural primitive prediction and assembly.

---

## 🚀 Quick Start: Demo

You can run the demo in **two ways**:

* **Google Colab (Recommended)** – No local installation, quick testing
* **Local Linux** – GPU-accelerated inference and full pipeline

---

## ☁️ Google Colab (Recommended)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32"/>](https://colab.research.google.com/github/neuralprimitives/neuralprimitives.github.io/blob/main/demo/demo.ipynb)

👉 Click the badge above

**Notes**

* No installation required
* Follow the notebook instructions cell-by-cell
* Initial setup takes ~10 minutes

---

## 🖥️ Local Linux (GPU Inference)

> **Requirement**
> A CUDA-enabled GPU is required for local inference.

---

### 1️⃣ Clone the Repository

```bash
git clone git@github.com:neuralprimitives/neuralprimitives.github.io.git
cd neuralprimitives.github.io && git lfs pull && cd demo
```

---

### 2️⃣ Set Up the Inference Environment

Create the conda environment and install all dependencies:

```bash
. install.sh
```

> ⏱️ Takes ~5 minutes depending on network speed

---

### 3️⃣ Run Neural Primitive Inference

Choose a transformation mode:

* `sim3`
* `translation`
* `scale`
* `rotation`

```bash
conda activate neural_primitive \
&& NEURALPRIMITIVE_TEST_ID=bag_0518100000337695 \
&& NEURALPRIMITIVE_AUG_MODE=sim3 \
python inference.py
```

📁 Outputs will be saved to:

```
evaluation/vg_sim3/
```

---

### 4️⃣ Set Up Primitive Assembly Environment

Primitive assembly is handled by **ComPoD**.

```bash
cd compod
. install.sh
```

---

### 5️⃣ Run Primitive Assembly

Convert predicted `.vg` primitives into assembled meshes:

```bash
python example/single_vg_to_obj.py \
  --input_file ../evaluation/vg_sim3/bag_0518100000337695.vg \
  --output_file ../evaluation/result_sim3/bag_0518100000337695.obj
```

📁 Output meshes:

```
evaluation/result_sim3/
```

---

## 🏃 Full Pipeline: Training & Evaluation

If you want to **train Neural Primitives end-to-end**, follow the steps below.

---

### ⚙️ Extra Dependencies

Install Chamfer Distance extension:

```bash
cd extensions/chamfer_dist
python setup.py install
```

---

### 📂 Dataset

We use the **[Building-PCC benchmark dataset](https://github.com/tudelft3d/Building-PCC-Building-Point-Cloud-Completion-Benchmarks)**:



### 🏋️ Training (Distributed)

Neural Primitives supports **Distributed Data Parallel (DDP)** training.

#### General Command

```bash
bash ./scripts/dist_train.sh <NUM_GPU> <PORT> \
    --config <config> \
    --exp_name <experiment_name> \
    [--resume] \
    [--start_ckpts <path>]
```

#### Example

```bash
bash ./scripts/dist_train.sh 2 12345 \
    --config cfgs/BuildingNL_models/neural_primitive_multi.yaml \
    --exp_name neural_primitive
```



## 📌 Notes

**Neural Primitives** builds on a series of prior and complementary research efforts.
For readers interested in **modular or standalone components**, please refer to the following repositories:

* **[abspy](https://github.com/chenzhaiyu/abspy)**
* **[PolyGNN](https://github.com/chenzhaiyu/polygnn)**
* **[PaCo](https://github.com/complete3d/paco)**
* **[SIMECO](https://github.com/complete3d/simeco)**
* **[UniCo](https://github.com/complete3d/unico)**