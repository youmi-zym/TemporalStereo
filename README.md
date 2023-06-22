<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network</h1>
  <p align="center">
    <a href="https://youmi-zym.github.io"><strong>Youmin Zhang</strong></a>
    ·
    <a href="https://mattpoggi.github.io/"><strong>Matteo Poggi</strong></a>
    ·
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html"><strong>Stefano Mattoccia</strong></a>
  </p>
  <!--h2 align="center">Arxiv</h2 -->
  <h3 align="center"><a href="https://arxiv.org/">Arxiv</a> | <a href="https://github.com/youmi-zym/TemporalStereo.git">Project Page</a> | <a href="https://www.youtube.com/watch?v=faSgN2THhEM">Youtube Video</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=faSgN2THhEM">
    <img src="./media/architecture.png" alt="Logo" width="98%">
  </a>
</p>
<p align="center">
<strong>TemporalStereo Architecture</strong>, the first supervised stereo network based on video.
</p>

## Code is coming soon...

## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda create -n temporalstereo python=3.8
conda activate temporalstereo
```
We ran our experiments with PyTorch 1.10.1, CUDA 11.3, Python 3.8 and Ubuntu 20.04.

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts. -->

#### NVIDIA Apex

We used NVIDIA Apex (commit @ 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a) for multi-GPU training.

Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```



## 💾 Datasets
We used three datasets for training and evaluation.

#### Flyingthings3D

The [Flyingthings3D/SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) can be downloaded here.

After that, you will get a data structure as follows:

```
FlyingThings3D
├── disparity
│    ├── TEST
│    ├── TRAIN
└── frames_finalpass
│    ├── TEST
│    ├── TRAIN
```

#### KITTI 2012/2015

KITTI 2012/2015 dataset is available at the [KITTI Stereo Website](https://www.cvlibs.net/datasets/kitti/eval_stereo.php).

KITTI Raw Sequences after preprocessed can be downloaded [here]().


## ⏳ Training

Note: batch size is set for each GPU

```bash
$ cd THIS_PROJECT_ROOT/

```


During the training, tensorboard logs are saved under the experiments directory. To run the tensorboard:

```bash
$ cd THIS_PROJECT_ROOT/
$ tensorboard --logdir=. --bind_all
```

Then you can access the tensorboard via http://YOUR_SERVER_IP:6006

## 📊 Testing

```bash
$ cd THIS_PROJECT_ROOT/

```


## 👩‍⚖️ Acknowledgement
Thanks the authors for their works:

[AcfNet](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark)

[CoEx](https://github.com/antabangun/coex)

