George Mason University (GMU) - DAEN 690 Capstone Project Fall 2022
===================================================================
- Faculty Advisor: [Dr. F. Brett Berlin][https://volgenau.gmu.edu/profiles/fberlin], Term Faculty, GMU Data Analytics Engineering M.S. Program.
- Partner/Customer: [Dr. Lance Sherry][https://volgenau.gmu.edu/profiles/lsherry], Associate Professor in the GMU Systems Engineering and Operations Research Department and Director of the Center for Air Transportation Systems Research at the GMU Volgenau School of Engineering.
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

In addition, the team leveraged the Momentum AI low-code/no-code toolkit hosted on Amazon Web Services (AWS).  According to the [Accure website][https://accure.ai/momentum-ai/], Momentum democratizes AI by providing no-coding toolkits to rapidly train and deploy ML models in production, thus increasing the overall productivity of the data science team.  No specialized skill is needed to work with Momentum.



GitHub Folder Structure
-----------------------
- Data: contains the data either simulated or collected.
- Documents: contains the documents relevant to the project, either academic research papers, online warticles etc.
- Scripts: contains current scripts that are used for testing and analysis.
- Test: contains deprecated scripts that are no longer being used as well as an explanation either in the code or the readme.md for why the script was abandoned for use.



Data
----
The datset for this project was provided by Dr. Lance Sherry.  It is comprised of 4 columns by 800 rows.  It is simulated data produced by Visual Basic for Applications (VBA) code.  This code was converted to Python ( [DataGeneration.py](https://github.com/shan825/blackswan_safetycrew/blob/main/scripts/DataGeneration.py) ).  This allowed the team to confirm that the input and target cases were correct, that the data was accurate, before building our NN model.

### Dataset Variables:
- xTemp: initial X position (representing temperature)
- yVolume: initial Y position (representing volume)
- direction: the initial direction the particle is moving
- target: labeled variable neural network is trying to learn



Code
----
This project was completed across five sprints:

### Sprint 1: Problem Definition & Project Ramp Up
The focus of this sprint was to set up our development environment and to complete required training.  This includes training to use the high performance computing clusters managed by the GMU Office of Research Computing and the Momentum AI low-code/no-code toolkit.  In addition, the team successfully reproduced the results of the [previous team](https://github.com/oelkassa/DAEN690digitaltwin/) that worked this project using the dataset described above.  The team also attempted to 

### Sprint 2: Dataset

### Sprint 3: Algorithms & Analytics

### Sprint 4: Visualizations & ML Model

### Sprint 5: Refine & Final Presentation



Change Log
----------
9-12-2022: Multiple updates to the readme file.
8-30-2022: The Github has been set up and the team is at work currently deciding how to tackle this problem, and the best methods and packages to use.
