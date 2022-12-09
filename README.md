George Mason University (GMU) - DAEN 690 Capstone Project Fall 2022
===================================================================
- Faculty Advisor: [Dr. F. Brett Berlin](https://volgenau.gmu.edu/profiles/fberlin), Term Faculty, GMU Data Analytics Engineering M.S. Program.
- Partner/Customer: [Dr. Lance Sherry](https://volgenau.gmu.edu/profiles/lsherry), Associate Professor in the GMU Systems Engineering and Operations Research Department and Director of the Center for Air Transportation Systems Research at the GMU Volgenau School of Engineering.
- Team Name: Black Swan Safety Crew

Modern systems, like aircraft or ships, are comprised of numerous complex sub-systems.  Such a system of systems is difficult to architect, develop, and test within planned funding and time constraints.  Moreover, these systems can be hard to upgrade or maintain due to multiple configuration baselines over time.  These baselines arise from a variety of reasons to include end of life hardware, operating system or other infrastructure changes, or new capabilities.  This translates into numerous configuration baselines with multiple combinations of software and hardware.

The goal of this project is to investigate Deep Learning (DL) for system testing and validation. What is the goal of this testing?  At a minimum, the team seeks to verify all desired functionality.  To fully test and validate an individual baseline requires the ship or aircraft to exercise all sub-systems and system functions in a variety of conditions in multiple scenarios.  The objective is to use Deep Learning to help identify the initial/terminal conditions and optimal subset of test cases that capture all unsafe outcomes while minimizing time and cost.

System Requirements
-------------------
The team developed and trained hand-coded models using the following libraries:
- Python
- PyTorch
- TensorFlow
- Keras

In addition, the team leveraged the Momentum AI low-code/no-code toolkit hosted on Amazon Web Services (AWS).  According to the [Accure website](https://accure.ai/momentum-ai/), Momentum democratizes AI by providing no-coding toolkits to rapidly train and deploy ML models in production, thus increasing the overall productivity of the data science team.  No specialized skill is needed to work with Momentum.

GitHub Folder Structure
-----------------------
- Data: contains the simulated data produced from data generation scripts in Python (see Data_generation folder below)
- Data_generation: contains Python programs to generate simulated data for modeling
- EDA: exploratory data analysis - contains scratch code and visualizations used in weekly progress presentations
- Heatmap_data: contains our top 10 model results for use in making a heatmap (see visualizations folder)
- Model_results: contains model parameters, raw results, and results summary from running 800 case holdout programs
- Models: contains single saved PyTorch models that can be reloaded to run quickly, used in hyperparameter tuning
- PyTorch: contains Python scripts to train deep learning neural networks with the PyTorch packages
- R: contains R scripts to create visualizations for presentations
- TensorFlow: contains Python scripts to train deep learning neural networks with the TensorFlow packages
- Visualizations: contains resulting visualizations from R and Python programs

Data
----
The datset for this project was provided by Dr. Lance Sherry.  It is comprised of 4 columns by 800 rows.  It is simulated data produced by Visual Basic for Applications (VBA) code.  This code was converted to Python ( [NormalizedDataGeneration.py](https://github.com/shan825/blackswan_safetycrew/blob/main/scripts/NormalizedDataGeneration.py) ).  This allowed the team to confirm that the input and target cases were correct, that the data was accurate, before building our NN model.

### Dataset Variables:
- xTemp: initial X position (representing temperature), range 0-9
- yVolume: initial Y position (representing volume), range 0-9
- direction: the initial direction the particle is moving, range 0-7 where 0 is North, 1 is NE, 2 is East, etc
- target: labeled variable neural network is trying to learn

Code
----
This project was completed across five sprints.  The initial sprint focused on problem definition and project ramp up, which involved development environment setup and completing required training.  This includes training to use the high performance computing clusters managed by the GMU Office of Research Computing and the Momentum AI low-code/no-code toolkit.  In addition, the team successfully reproduced the results of the [previous team](https://github.com/oelkassa/DAEN690digitaltwin/) that worked this project using the dataset described above.  The team focused on building Deep Learning classification models to identify the desired target classes using PyTorch and Keras/TensorFlow.  The goal of sprint two and three was to develop proofs of concept with increasing accuracy results ( [TenByTenDLNN_model_refactor.py](https://github.com/shan825/blackswan_safetycrew/blob/main/scripts/TenByTenDLNN_model_refactor.py) ) and ( [TF_MulticlassClassificationDLNN_BaselineModel1stAttempt.py](https://github.com/shan825/blackswan_safetycrew/blob/main/scripts/TF_MulticlassClassificationDLNN_BaselineModel1stAttempt.py) ).  In sprint five, the team delivered our minimum viable product (MVP) for each model implmentation.

### Final Model Programs:
Below are the final Python programs that we used to train PyTorch and TensorFlow models:
- PyTorch - [PyTorch_DLNN_model.py](https://github.com/shan825/blackswan_safetycrew/blob/main/pytorch/PyTorch_DLNN_model.py)
- TensorFlow - [TensorFlow_DLNN_model.py](https://github.com/shan825/blackswan_safetycrew/blob/main/tensorflow/TensorFlow_DLNN_model.py)

Each of the above programs runs all 800 cases like so:
1) Take out 1 case (holdout)
2) Split remaining 799 into training and testing datasets (typically 80/20)
3) Train a neural network model with respective package
4) Model attempts to predict holdout case
5) Record model results - predicted holdout correctly, testing accuracy
6) Repeat for next case

Executing Code
--------------
To execute the codebase, each of the team developers used a different integrated developement enviornment (IDE) or text editor to include Sublime Text, PyCharm, Jupyter Notebook, and Visual Studio Code.  Any editor/IDE should work with the code in this repository.

Change Log
----------
12-09-2022: Reorganized folders to split programs into those for PyTorch and TensorFlow models, those for data generation, and R scripts for creating visualizations
Added code explanation to PyTorch and TensorFlow model programs so it is more apparent what each program is doing

11-22-2022: Added R scripts to make heatmap and multiple bar plots comparing 4 vs 5 layer models and PyTorch vs TensorFlow models.

11-21-2022: Added Visio diagram of 4 and 5 Layer DLNN's.

11-21-2022: Added Confidence interval RMD script and results for top 10 TF/PT trials.

11-21-2022: Added a heatmap and dot plot to visualizations folder. They show model performance for the top 10 models results.

11-12-2022: Added Tensorflow and PyTorch results for SGD with 5 Layers and 0.09 momentum. Added correct PyTorch results for SGD with 5 Layers (no momentum).

11-11-2022: Added Tensorflow and PyTorch results for SGD with 5 Layers.

11-11-2022: Added 3 TensorFlow results files.

11-9-2022: Added 2 TensorFlow results files.

11-8-2022: Added 2 TensorFlow results files.

11-7-2022: Added 5 TensorFlow results into the Model_results folder.

11-4-2022: Added a TensorFlow_loopHoldout script that runs and mirrors PyTorch NN model script.

11-3-2022: Added a TensorFlowRefactor script. Still needs work.

11-1-2022: Added a second attempt at TF hold out script. Still needs work.

10-24-2022: Code for Visualizations of HP's using TensorBoard in Section 3.3 of Report

10-18-2022: Added a fourth attempt for hyperparameter tuning for TensorFlow model.

10-7-2022: Edited second script for functionality with hopper

10-7-2022: Added a third attempt for HP tuning TF training model

10-5-2022: Added a second attempt for hyperparameter tuning for TensorFlow model.

10-4-2022: Added results file for PyTorch model.

10-3-2022: Added results file for PyTorch model.

10-3-2022: Added a first attempt for hyperparameter tuning for TensorFlow model.

10-1-2022: Added 3d_scatter_animated.py and 3d_scatter_interactive.html for interactive version of 3D holdout scatterplot (in HTML file).

9-27-2022: Added visualizations folder and an examples subfolder to store visualizations and associated code

9-20-2022: Updated data generation file listed in Data section, added dataset variable ranges

9-19-2022: Added Script for Visualization Creation Utilizing 800 Cases with 100% Accuracy for Some Testing Sets

9-15-2022: Added model_results/ folder for storing "800 cases with 1 holdout" program results.

9-13-2022: Added models/ folder for storing trained PyTorch models for quick load and runs.

9-12-2022: Multiple updates to the readme file.

9-12-2022: Committed another iteration of a multi-class classification TensorFlow model.

9-06-2022: Committed Sprint22TestCode.py.  Used previous team's code with our dataset and produced similar accuracy results.

8-30-2022: The Github has been set up and the team is at work currently deciding how to tackle this problem, and the best methods and packages to use.
