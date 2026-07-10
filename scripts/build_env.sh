#!/usr/bin/env bash
# Build the pinned GPU conda env for PINNs-MPF (CUDA 12.x-capable GPU + recent driver).
# TensorFlow GPU runtime libs come from pip nvidia-* wheels (no system CUDA toolkit needed).
set -euo pipefail

ENV=pinns-mpf-gpu
PY=3.11
# Source conda from wherever it is installed (conda/mamba must be on PATH).
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== [1/5] create env $ENV (python $PY) ==="
mamba create -n "$ENV" "python=${PY}" -y

conda activate "$ENV"
python -m pip install --upgrade pip wheel

echo "=== [2/5] install GPU TensorFlow (bundled CUDA 12 runtime) ==="
# TF 2.16.2 + matching tf-keras (Keras 2) + TFP 0.24.0 is a known-good GPU trio.
pip install "tensorflow[and-cuda]==2.16.2"

echo "=== [3/5] install tf-keras + tensorflow-probability (no [tf] extra: keep the CUDA TF) ==="
pip install "tf-keras==2.16.*" "tensorflow-probability==0.24.0"

echo "=== [4/5] scientific stack ==="
pip install "numpy<2.1" scipy matplotlib seaborn pyDOE psutil h5py imageio

echo "=== [5/5] verify TF / TFP / GPU / float64 ==="
export TF_CPP_MIN_LOG_LEVEL=2
python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
try:
    import tensorflow_probability as tfp
    print("TFP:", tfp.__version__)
    from tensorflow_probability.python.optimizer import lbfgs_minimize  # noqa
    print("TFP lbfgs_minimize: OK")
except Exception as e:
    print("TFP import issue:", e)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
if gpus:
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception as e: print("mem-growth:", e)
    with tf.device('/GPU:0'):
        a = tf.random.normal([512, 512], dtype=tf.float64)
        b = tf.linalg.matmul(a, a)
        x = tf.Variable(tf.ones([8], dtype=tf.float64))
        with tf.GradientTape() as t:
            y = tf.reduce_sum(x * x)
        g = t.gradient(y, x)
    print("float64 matmul OK, dtype:", b.dtype, "device:", b.device)
    print("float64 grad OK:", g.numpy()[:3])
else:
    print("WARNING: no GPU visible to TensorFlow")
print("ENV_BUILD_DONE")
PY
