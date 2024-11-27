<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables.
*** This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<a name="readme-top"></a>

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="readme_files/logo.png" alt="Logo" width="80">

<h3 align="center">Flora Vision Image Classifier</h3>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#main-features">Main Features</a></li>
    <li><a href="#installation-steps">Installation Steps</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#udacity-certificate">Udacity Certificate</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol> 
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<img src="readme_files/cover.png" alt="Cover Image">

- **Project Name:** Flora Vision Image Classifier
- **Version:** v1.0.0
- **Purpose:** To classify images of flowers into their respective categories using a deep learning model.
- **HTML Notebook Preview:** The project contains an HTML version of the notebook for quick access, available at `docs/Image Classifier Project.html` inside the repository.

### Description

Flora Vision Image Classifier is a deep learning-based project designed to identify and classify images of flowers.
Initially, the project was developed as a Jupyter Notebook for experimentation and analysis. Later, it was refactored
into a modularized command-line application for practical usage and easier deployment. Users can now train models,
validate performance, and make predictions directly from the command line by running the `train.py` and `predict.py`
scripts.

The application is configurable via a `config.yaml` file, which simplifies hyperparameter tuning and dataset management.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Main Features

- **Notebook to CLI Transformation:** Developed initially in a Jupyter Notebook and transformed into a fully modular
  command-line application.
- **Training Module:** Train the classifier using a pre-trained model with fine-tuned parameters.
- **Validation Module:** Validate the trained model to assess its performance and accuracy.
- **Testing Module:** Test the model on unseen images and measure its classification accuracy.
- **Prediction Module:** Predict the top flower categories for a given image.
- **Config-Driven Architecture:** Configurable hyperparameters like learning rate, batch size, and number of epochs
  via `config.yaml`.
- **Modular Design:** Organized into reusable modules and packages for scalability.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation Steps

Follow these steps to set up your environment and install the necessary packages:

1. **Create a Conda Environment**  
   To isolate the project dependencies, create a Conda environment using Python 3.10:  
   `conda create --name pytorch python=3.10`

   Confirm the installation of required packages when prompted. The environment will be created at a location like `C:\Users\Mohamed\anaconda3\envs\pytorch`.

2. **Activate the Conda Environment**  
   Activate the newly created environment to ensure all commands are executed within it:  
   `conda activate pytorch`

3. **Install Essential Python Packages**  
   Install the following Python packages for visualization, computation, and integration with Jupyter:  
   `conda install ipykernel notebook matplotlib numpy pillow pyyaml`

4. **Register the Kernel with Jupyter**  
   To use the `pytorch` environment in Jupyter notebooks, register it as a kernel:  
   `python -m ipykernel install --user --name=pytorch`

   You can verify the installation by listing all available Jupyter kernels:  
   `jupyter kernelspec list`

5. **Verify CUDA Installation**  
   Check if the NVIDIA CUDA toolkit is installed and accessible. This ensures PyTorch can leverage GPU acceleration:  
   `nvcc --version`

   The output should show a version like:

   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2024 NVIDIA Corporation
   Built on Thu_Sep_12_02:55:00_Pacific_Daylight_Time_2024
   Cuda compilation tools, release 12.6, V12.6.77
   Build cuda_12.6.r12.6/compiler.34841621_0
   ```

6. **Install PyTorch with CUDA Support**  
   Install PyTorch and its related libraries (`torchvision` and `torchaudio`) with CUDA support for GPU acceleration:  
   `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
   
   or
   
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  `

7. **Verify PyTorch Installation**  
   Run the following Python commands to confirm the installation:
   - `import torch`
   - `print(torch.__version__)`  # Should print the PyTorch version
   - `print(torch.cuda.is_available())`  # Should return True if GPU is available
   - `print(torch.version.cuda)`  # Should print the CUDA version being used by PyTorch
   - `print(torch.backends.cudnn.enabled)`  # Should return True if cuDNN is enabled

8. **Ready to Use**  
   Once all steps are completed, the environment is ready for training and running your deep learning models.


## Built With

- **Python** - Programming Language
- **PyTorch** - Deep Learning Framework
- **TorchVision** - Utilities for Computer Vision
- **Matplotlib** - Visualization
- **Pillow** - Image Processing
- **JSON** - Category Mapping

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Screenshots

### Training and Validation Losses Per Epoch

<img src="readme_files/training_and_validation_loss_per_epoch.png" alt="Training and Validation Losses Per Epoch">

### Validation Accuracy Per Epoch

<img src="readme_files/validation_accuracy_perc_per_epoch.png" alt="Training and Validation Losses Per Epoch">

### Training in Progress

<img src="readme_files/training_progress.jpg" alt="Training Progress">

### Validation Results

<img src="readme_files/validation_results.jpg" alt="Validation Results">

### Top Predictions

<img src="readme_files/cover.png" alt="Predictions">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Udacity Certificate

This project was the graduation project for the Udacity Nanodegree **AI Programming with Python**.  
Below is the certificate for the program:

<img src="readme_files/ai_programming_with_python_certificate.jpg" alt="Udacity Certificate">

The certificate can be verified using the following [verification link](https://www.udacity.com/certificate/e/ea925be2-fb92-11eb-bc2d-3f0a8c114be8).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Mohamed AbdelGawad Ibrahim - [@m-abdelgawad](https://www.linkedin.com/in/m-abdelgawad/) - <a href="tel:+201069052620">
+201069052620</a> - muhammadabdelgawwad@gmail.com

GitHub Profile Link: [https://github.com/m-abdelgawad](https://github.com/m-abdelgawad)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/m-abdelgawad/
