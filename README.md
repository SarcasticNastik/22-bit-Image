# File structure
```bash
.
├── graphx-conv
│  ├── install_lib.sh                                                   # Ada installer script
│  ├── src                                                              # Source files
│  │  ├── __pycache__
│  │  │  ├── data_loader.cpython-37.pyc
│  │  │  ├── networks.cpython-37.pyc
│  │  │  └── ops.cpython-37.pyc
│  │  ├── configs                                                       # NeuralNet Configuration files
│  │  │  ├── common.gin
│  │  │  ├── fc-final.gin
│  │  │  ├── graphx-up-final.gin
│  │  │  └── lowrankgraphx-up-final.gin
│  │  ├── data_loader.py                                                
│  │  ├── networks.py
│  │  ├── ops.py
│  │  ├── results                                                       # Results and trained models
│  │  │  ├── ICCV-lowrankgraphx-conv-up-final                           # Contains trained models for our project
│  │  │  │  ├── network.txt
│  │  │  │  ├── plots                                                   # Training Chamfer Loss Graph
│  │  │  │  ├── scheduler.txt
│  │  │  │  ├── training-45666.pt
│  │  │  │  ├── training-60888.pt
│  │  │  │  ├── training-76110.pt
│  │  │  │  ├── training-91332.pt
│  │  │  │  └── training-106554.pt
│  │  │  └── test-and-demo                                              # Contains various plots b/w ground truth and predicted pointclouds
│  │  │     ├── plots
│  │  │     └── pointcloud_batches                                      # Contains predicted 3D-np representations for demo purposes
│  │  ├── test.py
│  │  ├── visualizer.py                                                 # Visualize given 3D point cloud interactively
│  │  └── train.py
│  ├── test.sh
│  └── train.sh
├── README.md
├── results
│  ├── find_test_chamfer_loss_avg.sh
│  ├── test.txt
│  └── train-chamfer.jpg
├── test_output.txt
└── training_output.txt

```

