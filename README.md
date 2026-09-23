# MedTriFlow: Efficient Resolution-Agnostic 3D Medical Image Generation with Implicit Triplane Representation

Chenfan Xu, Yulong Dou, Tao Luo, Qian Wang, and Zhiming Cui.

MICCAI 2026.

## 📦 Released

- TriplaneVQVAE training and inference code
- TriFlow training and sampling code
- TriFlow-ControlNet training and inference code
- Pretrained TriplaneVQVAE and TriFlow checkpoints

## 📋 Overview

This package provides three stages:

1. Prepare preprocessed volumes.
2. Train or load TriplaneVQVAE.
3. Generate latents, then train and run TriFlow or TriFlow-ControlNet.

This README focuses on commands, paths, and checks. It does not describe network internals.

Use /path/to/project as the project root in every example. Replace it with the real project path before running a command.

### Process overview

~~~mermaid
flowchart LR
    accTitle: TriFlow release workflow
    accDescr: Data preparation flows into VQVAE latent generation, followed by TriFlow or ControlNet training and inference.

    raw_volume([Raw NIfTI]) --> preprocess[Prepare 256 cube]
    preprocess --> vqvae_train[Train VQVAE]
    vqvae_train --> latent_infer[Generate latents]
    latent_infer --> triflow_train[Train TriFlow]
    triflow_train --> triflow_infer[Sample volumes]
    latent_infer --> condition_infer[Generate SVR conditions]
    condition_infer --> control_train[Train ControlNet]
    control_train --> control_infer[Conditional sampling]

    classDef input_style fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process_style fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef output_style fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class raw_volume input_style
    class preprocess,vqvae_train,latent_infer,triflow_train,condition_infer,control_train process_style
    class triflow_infer,control_infer output_style
~~~

## 🧰 Environment and project layout

### Runtime requirements

Run the .job files on Linux or on the L20 cluster. The files use Bash.

| Item | Current setting |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.6.0 with the CUDA 12.4 build |
| PyTorch Lightning | 2.0.0 |
| Main data format | NIfTI .nii.gz |
| Model volume size | 256 x 256 x 256 |
| Inference output spacing | 1.00 mm |
| Default GPUs | 4 x A100 80G |
| Environment | The cluster Gen environment |

The required packages are listed in [requirements.txt](requirements.txt).

Activate the existing Gen environment first. Use the environment path for your cluster account.

~~~bash
export PROJECT_ROOT=/path/to/project
source /path/to/Gen/bin/activate
cd "$PROJECT_ROOT"

python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python -m pip install -r requirements.txt
~~~

If the Gen environment already has the matching CUDA build of PyTorch, keep it. Do not replace it with a CPU build.

### Project layout

~~~text
/path/to/project/
├── ckpts/
│   ├── TriplaneVQVAE.ckpt
│   └── TriFlow.ckpt
├── data/
│   ├── split-All.npy
│   ├── CTChestAbdomen/256^256^256/*.nii.gz
│   ├── CTHeadNeck/256^256^256/*.nii.gz
│   ├── CTLegs/256^256^256/*.nii.gz
│   ├── MRBody/256^256^256/*.nii.gz
│   └── MRBrain/{T1,T2}_256^256^256/*.nii.gz
├── exp/
├── codes/
│   ├── configs/
│   ├── scripts/
│   ├── infer_triflow.py
│   └── run.py
└── requirements.txt
~~~

Use codes/configs as the canonical config directory. Use codes/scripts for cluster jobs.

### Data split

data/split-All.npy stores train, val, and test arrays. Each entry uses this form:

~~~text
data/<dataset>/<resolution>/<case>.nii.gz
~~~

The data loader resolves relative entries from the project root. The same split works when a job starts from codes.

The configured class IDs are:

| Dataset | Class ID |
| --- | ---: |
| CTChestAbdomen | 1 |
| CTHeadNeck | 2 |
| CTLegs | 3 |
| MRBody | 4 |
| MRBrain/T1 | 5 |
| MRBrain/T2 | 6 |

## 🧪 Raw volume preparation

The release loader expects preprocessed volumes. It does not perform the full raw-data preprocessing pipeline.

Apply these steps before adding a volume to data:

1. Crop the foreground region.
2. Resample to the target spacing for the dataset FOV.
3. Use CenterCrop or CenterPad to obtain 256³ voxels.
4. Compute robust lower and upper intensity limits.
5. Clip to these limits and normalize to [0, 1].

For CT-like data, use the lower and upper percentiles:

~~~python
low = volume.quantile(0.05)
high = volume.quantile(0.95)
volume = ((volume - low) / (high - low)).clip(0.0, 1.0)
~~~

Use the spacing selected for each dataset FOV. Do not use one spacing for every raw dataset without checking physical coverage.

Save the final volume as NIfTI. The loader clips the array to [0, 1] and maps it to [-1, 1] during model input preparation.

## ⚙️ Configs and scripts

codes/run.py is the shared training and test entry point. It loads a YAML config and applies command-line overrides.

| Task | Config | Script | Output |
| --- | --- | --- | --- |
| VQVAE training | TriplaneVQVAE.yaml | TriplaneVQVAE_train.job | VQVAE experiment directory |
| VQVAE volume test | TriplaneVQVAE.yaml | TriplaneVQVAE_infer_volume.job | exp/test_TriplaneVQVAE_infer/infer_volume/ |
| VQVAE latent test | TriplaneVQVAE.yaml | TriplaneVQVAE_infer_latent.job | exp/test_TriplaneVQVAE_infer/infer_latent/ |
| SVR condition latent test | TriplaneVQVAE.yaml | TriplaneVQVAE_infer_svr.job | exp/test_TriplaneVQVAE_infer/infer_SVR_latent/ |
| TriFlow training | TriFlow.yaml | TriFlow_train.job | TriFlow experiment directory |
| TriFlow sampling | TriFlow.yaml | infer_triflow.py | User-selected output directory |
| ControlNet training | TriFlow-ControlNet.yaml | TriFlow-ControlNet_train.job | exp/TriFlow-ControlNet-SVR/ |

The configs use /path/to/project/... placeholders. The job files use PROJECT_ROOT.

## 🔹 TriplaneVQVAE training and inference

### Run VQVAE training

~~~bash
export PROJECT_ROOT=/path/to/project
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_train.job"
~~~

The job uses the batch size and loss start steps from the base config. The current values are batch size 1, validation batch size 1, LPIPS start step 30000, and discriminator start step 60000.

Optional fine-tuning from the released VQVAE checkpoint:

~~~bash
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python run.py \
  -b configs/TriplaneVQVAE.yaml \
  -l "$PROJECT_ROOT/exp" \
  -n TriplaneVQVAE-finetune \
  --finetune_from "$PROJECT_ROOT/ckpts/TriplaneVQVAE.ckpt" \
  data.params.split_path="$PROJECT_ROOT/data/split-All.npy"
~~~

`--finetune_from` loads the checkpoint weights and starts a new training run. It does not restore the optimizer or scheduler state. Do not combine it with `--resume_from`.

### Reconstruct a volume

~~~bash
export PROJECT_ROOT=/path/to/project
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_infer_volume.job"
~~~

The output directory is:

~~~text
/path/to/project/exp/test_TriplaneVQVAE_infer/infer_volume/
~~~

### Generate triplane latents

~~~bash
export PROJECT_ROOT=/path/to/project
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_infer_latent.job"
~~~

The job sets:

~~~text
data.params.gen_latent=True
data.params.fixed_aug=3
model.params.resolution=-1
~~~

fixed_aug is important for TriFlow. A value of 0 disables fixed augmentation. A value of 3 creates three indexed samples per source path in the current implementation. The original sample uses aug_idx=0. The extra samples use suffixes such as -1 and -2.

TriFlow must read the same augmented latent set. This reduces repeated exposure to one unaugmented latent and helps reduce overfitting.

The output directory is:

~~~text
/path/to/project/exp/test_TriplaneVQVAE_infer/infer_latent/
~~~

Check the original and augmented latent files before TriFlow training. Their stems must match the split entries. Their suffixes must match fixed_aug.

### Generate SVR condition latents

~~~bash
export PROJECT_ROOT=/path/to/project
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_infer_svr.job"
~~~

The job uses:

~~~text
data.params.gen_latent=True
data.params.task_path=/path/to/project/data/SVR_data/FDK/
data.params.path_include=['CTChestAbdomen','CTHeadNeck','CTLegs']
model.params.resolution=-1
~~~

The output directory is:

~~~text
/path/to/project/exp/test_TriplaneVQVAE_infer/infer_SVR_latent/
~~~

## 🌊 TriFlow training and inference

### Prepare the latent input

TriFlow needs the VQVAE checkpoint and triplane latent files. Generate the latent files first.

The release job writes latents to:

~~~text
/path/to/project/exp/test_TriplaneVQVAE_infer/infer_latent/
~~~

The TriFlow job passes this directory as data.params.latent_path. The current TriFlow config uses fixed_aug: 3. Keep this value aligned with the latent generation job.

### Train TriFlow

For the default four-GPU A100 80G layout:

~~~bash
export PROJECT_ROOT=/path/to/project
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$PROJECT_ROOT/codes/scripts/TriFlow_train.job"
~~~

The physical GPUs are mapped to logical devices [0,1,2,3] by Lightning.

The job starts a new run by default. Set RESUME_FROM to resume an existing run.

Optional fine-tuning from the released TriFlow checkpoint:

~~~bash
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python run.py \
  -b configs/TriFlow.yaml \
  -l "$PROJECT_ROOT/exp" \
  -n TriFlow-finetune \
  --finetune_from "$PROJECT_ROOT/ckpts/TriFlow.ckpt" \
  data.params.latent_path="$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_latent/" \
  data.params.split_path="$PROJECT_ROOT/data/split-All.npy"
~~~

This loads the released TriFlow weights and starts a new optimizer state. Use `--resume_from` instead when you need to continue an existing TriFlow experiment.

The same run can be started manually:

~~~bash
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python run.py \
  -b configs/TriFlow.yaml \
  -l "$PROJECT_ROOT/exp" \
  -n TriFlow \
  data.params.split_path="$PROJECT_ROOT/data/split-All.npy" \
  data.params.latent_path="$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_latent/"
~~~

Resume with:

~~~bash
python run.py \
  -b configs/TriFlow.yaml \
  -l "$PROJECT_ROOT/exp" \
  -n TriFlow \
  --resume_from "$PROJECT_ROOT/exp/TriFlow/ckpts/last.ckpt"
~~~

### Run TriFlow sampling

~~~bash
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python infer_triflow.py \
  --config "$PROJECT_ROOT/codes/configs/TriFlow.yaml" \
  --flow-ckpt "$PROJECT_ROOT/ckpts/TriFlow.ckpt" \
  --first-stage-ckpt "$PROJECT_ROOT/ckpts/TriplaneVQVAE.ckpt" \
  --output "$PROJECT_ROOT/exp/test_TriFlow_infer" \
  --num-samples 1 \
  --label 1 \
  --volume-res 256
~~~

The script writes latent .npy, volume .npy, and 1.00 mm NIfTI .nii.gz files. The label value uses the class IDs in the config.

The TriFlow config uses use_ema: True. Keep it aligned with the released TriFlow checkpoint.

## 🎛️ TriFlow-ControlNet training and inference

### Prepare the condition data

Generate SVR condition latents before ControlNet training. Use the same split and class filter as the ControlNet config.

~~~text
GT latent: /path/to/project/exp/test_TriplaneVQVAE_infer/infer_latent/
SVR cond:  /path/to/project/exp/test_TriplaneVQVAE_infer/infer_SVR_latent/
~~~

The relative case names must match between GT latent and condition files.

### Train ControlNet

The current job uses four visible GPUs. Lightning sees them as logical devices [0,1,2,3].

~~~bash
export PROJECT_ROOT=/path/to/project
bash "$PROJECT_ROOT/codes/scripts/TriFlow-ControlNet_train.job"
~~~

The config sets model.params.stage=2 and batch size 64. It loads the pretrained TriFlow checkpoint from `pretrained_ldm_ckpt`. The job overrides `num_workers` to 4 and `val_num_workers` to 0.

The job starts a new run by default. Set RESUME_FROM to resume an existing run.

The current ControlNet config keeps fixed_aug at 0. Keep this setting unless both the GT latent files and the condition files are regenerated with the same augmentation count.

Resume with:

~~~bash
export PROJECT_ROOT=/path/to/project
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python run.py \
  -b configs/TriFlow-ControlNet.yaml \
  -l "$PROJECT_ROOT/exp" \
  -n TriFlow-ControlNet-SVR \
  --resume_from "$PROJECT_ROOT/exp/TriFlow-ControlNet-SVR/ckpts/last.ckpt" \
  data.params.latent_path="$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_latent/" \
  data.params.cond_path="$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_SVR_latent/" \
  data.params.split_path="$PROJECT_ROOT/data/split-All.npy" \
  data.params.num_workers=4 \
  data.params.val_num_workers=0
~~~

### Run conditional inference

Use the same entry point with the ControlNet config and checkpoint:

~~~bash
cd "$PROJECT_ROOT/codes"
export PYTHONPATH="$PROJECT_ROOT/codes"
python infer_triflow.py \
  --config "$PROJECT_ROOT/codes/configs/TriFlow-ControlNet.yaml" \
  --flow-ckpt "$PROJECT_ROOT/exp/TriFlow-ControlNet-SVR/ckpts/last.ckpt" \
  --first-stage-ckpt "$PROJECT_ROOT/ckpts/TriplaneVQVAE.ckpt" \
  --output "$PROJECT_ROOT/exp/test_TriFlow_ControlNet_infer" \
  --num-samples 1 \
  --label 1 \
  --volume-res 256
~~~

The current standalone inference entry point sends hint=None. It supports unconditional sampling. A condition-guided run needs a caller that passes the SVR latent as hint.

## ✅ Verification checks

Run these checks before a long job:

~~~bash
export PROJECT_ROOT=/path/to/project
test -f "$PROJECT_ROOT/data/split-All.npy"
test -f "$PROJECT_ROOT/ckpts/TriplaneVQVAE.ckpt"
test -f "$PROJECT_ROOT/ckpts/TriFlow.ckpt"
python -m py_compile "$PROJECT_ROOT/codes/run.py" "$PROJECT_ROOT/codes/infer_triflow.py"
~~~

Check latent files:

~~~bash
find "$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_latent" -name '*.npy' | head
find "$PROJECT_ROOT/exp/test_TriplaneVQVAE_infer/infer_SVR_latent" -name '*.npy' | head
~~~

Check the first training log for:

~~~text
Running on GPUs ...
Number of training cases: ...
Setting learning rate to ...
~~~

Check the inference directory for .nii.gz output after sampling.

## 🔧 Troubleshooting

### A volume or latent file is missing

Check PROJECT_ROOT, split-All.npy, and the latent output directory. Check that fixed_aug matches the generated filename suffixes.

### Checkpoint keys do not match

Use the VQVAE checkpoint with TriplaneVQVAE.yaml. Use the TriFlow checkpoint with TriFlow.yaml. Use the ControlNet checkpoint with TriFlow-ControlNet.yaml.

### CUDA out of memory

Reduce batch_size. Reduce val_batch_size first if validation fails. Keep the logical device list equal to the visible GPU count.

### Distributed initialization fails

Check CUDA_VISIBLE_DEVICES. It defines physical GPU order. Lightning devices use logical indexes after that mapping.

### Inference produces no NIfTI file

Check that --no-nifti is not set. Check that SimpleITK is installed. Check the final output directory in the inference log.

## 🔗 Key files

- [VQVAE config](codes/configs/TriplaneVQVAE.yaml)
- [TriFlow config](codes/configs/TriFlow.yaml)
- [ControlNet config](codes/configs/TriFlow-ControlNet.yaml)
- [Shared runner](codes/run.py)
- [TriFlow inference](codes/infer_triflow.py)
- [VQVAE latent job](codes/scripts/TriplaneVQVAE_infer_latent.job)
- [TriFlow job](codes/scripts/TriFlow_train.job)
- [ControlNet job](codes/scripts/TriFlow-ControlNet_train.job)

<details>
<summary><strong>📋 Quick reference</strong></summary>

| Action | Command |
| --- | --- |
| Set root | export PROJECT_ROOT=/path/to/project |
| Generate augmented latent | bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_infer_latent.job" |
| Generate SVR condition latent | bash "$PROJECT_ROOT/codes/scripts/TriplaneVQVAE_infer_svr.job" |
| Train TriFlow | bash "$PROJECT_ROOT/codes/scripts/TriFlow_train.job" |
| Train ControlNet | bash "$PROJECT_ROOT/codes/scripts/TriFlow-ControlNet_train.job" |
| Run TriFlow inference | python "$PROJECT_ROOT/codes/infer_triflow.py" |

</details>

---
