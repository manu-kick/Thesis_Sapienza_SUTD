# EMANUELE RUCCI - Thesis Project (Temp Readme) 
Thesis developed in Singapore university of technology and design, for the course of Artificial Intelligence and Robotics held in Sapienza, University of Rome.

![Universities Logos](assets/other_contents/Uni_Logos.jpg)

## Overview
This repository contains the materials and code for my Master's thesis project. The project focuses the integration of explicit gaussian splatting optimization method with an implicit (Deep neural network).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
The primary objective of my thesis is volumetric video rendering. The task involves generating high-quality novel views by leveraging Gaussian-based scene representations.

This study focuses on evaluating one of the implicit methods for predicting Gaussian parameters, as proposed in the MV-Splat paper, which serves as the foundation for this research. My contribution bridges the gap between explicit and implicit approaches by integrating the basic Gaussian splatting algorithm with implicit training. This integration enables a better initialization of the parameters, which are subsequently refined through a Gaussian splatting optimization procedure. This cascading approach enhances the performance of the implicit method while significantly reducing the computational time compared to running the Gaussian splatting algorithm alone.

The second contribution of this study focuses on knowledge distillation. Given the optimal parameters obtained from the Gaussian splatting process, we explore the possibility of training the implicit model to predict such high-quality parameters in a single forward pass. This would ultimately eliminate the need for the refinement step, allowing the implicit model to achieve high-quality rendering independently. Such an approach would significantly enhance inference speed, making the method more practical for real-time applications.

Feel free to reach out if you have any questions or require further clarification.


## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Navigate to the project directory:
    ```sh
    cd your-repo-name
    ```
3. Install the required dependencies:
    ```sh
    [Provide installation commands, e.g., pip install -r requirements.txt]
    ```

## Usage
[Provide instructions on how to use the code or run the experiments. Include examples if possible.]

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or feedback, please contact [your email address].
