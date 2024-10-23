# Solving Occlusion
This repository contains the official implementation for the paper "Temporal Enhanced Floating Car Observers," presented at IEEE IV 2024. The paper demonstrates that the concept of Floating Car Observers—vehicles equipped with sensors to detect other traffic participants for traffic state estimation—can be enhanced by incorporating detections from previous time steps. 

By leveraging previously seen but currently undetectable traffic participants, this approach utilizes a data-driven method and spatio-temporal deep learning architectures to enrich traffic state information.

## Installation

### Pre-requisites

- [Anaconda or Miniconda](https://www.anaconda.com/distribution/)

### Cloning the repository


```bash
git clone git@github.com:jegerner/solving_occlusion.git
cd solving-occlusion
```

### Setting up the Conda environment

Use the following command to create a Conda environment based on the `environment.yaml` file:

```bash
conda env create -f environment.yaml --name solving-occlusion
conda activate solving-occlusion
```

## Structure

colving_occlusion/  
  ├── configs/ # dir containing config files  
  ├── content/ # dir containing visualizations  
  ├── data/ # dir containing data  
  ├── sub_models/ # dir containing models for the temporal data enhancement  
  ├── tools/  
  │   ├── analyzation/ # dir containing tools for analyzation  
  │   ├── dataset_creation/ # dir containing tools for dataset creation  
  ├── trained_models/ # dir containing trained models further seperated by model type  
  ├── utils/ # dir containing different utility functions  
  ├── models.py # file for model handling  
  ├── train.py # main file for training the temporal data enhancement  
  ├── test.py # main file for testing the temporal data enhancement  
  ├── environment.yaml # Conda environment file      

## Usage

The foundation of this repository is a dataset containing information about traffic participants, detected or undetected, based on a specific penetration rate within an area of interest in the investigated traffic network. These datasets can be generated using the [SUMO_FCO repository](https://github.com/urbanAIthi/SUMO_FCO).

To train the temporal enhancement model, follow these steps:

1. **Generate Prediction Targets**: Use `tools/dataset_creation/create_fco_target.py` to generate prediction targets for a specific sequence length.
2. **Train the Model**: Train the spatio-temporal enhancement model using `train.py`, which utilizes configurations defined in `configs/config.py`.
3. **Test the Model**: After successful training, test the models using `test.py`.

Additionally, we provide tools for analysis and visualization under `tools/analyzation`.

(If you are interested in the datasets and trained parameters used in the paper, please contact jeremias.gerner@thi.de)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10588538,
  author={Gerner, Jeremias and Bogenberger, Klaus and Schmidtner, Stefanie},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Temporal Enhanced Floating Car Observers}, 
  year={2024},
  volume={},
  number={},
  pages={1035-1040},
  keywords={Codes;Three-dimensional displays;Microscopy;Observers;Traffic control;Stability analysis;Spatiotemporal phenomena},
  doi={10.1109/IV55156.2024.10588538}
}
```

The ViT implementation is based on the following repository: https://github.com/lucidrains/vit-pytorch.git
The base dataset for this work is generated with: https://github.com/urbanAIthi/SUMO_FCO

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.