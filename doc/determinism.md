# Running DeepMD in full deterministic mode

With the default settings DeepMD does not guarantee that two successive trainings using the same data will return the same model weights. The results will also depend on the processing units GPU vs CPU. Variations might also be observed between different families of GPUs. This document explains how to set up DeepMD to obtain reproducible results for a given set of training data and hardware architecture. It only applies to the forces and stress calculations during the training and inference phases.

DeepMD will use the deterministic versions of the forces and stress kernels when the environment variable `DEEPMD_ENABLE_DETERMINISM` is defined at runtime. It will also set TensorFlow to run in deterministic mode.  More information about running TensorFlow in deterministic mode and what it implies, can be found [here](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism). The `OMP_NUM_THREADS` variable seems to have less or no impact when the GPU version of DeepMD is used.

Adding this line of code in the run scripts is enough to get reproducible results on the same hardware. This will also set tensorflow deterministic mode

```[sh]
export DEEPMD_ENABLE_DETERMINISM=1
```


