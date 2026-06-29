# Academic & Research Profile

## About Me

I am a Master's student specializing in Robotics and Embodied AI, with a background in Electromechanical Engineering.

My research focuses on integrating Vision-Language Models (VLMs) with autonomous robotic systems, aiming to build embodied agents capable of perception, reasoning, planning, and execution in real-world environments.

My long-term goal is to contribute to the development of embodied intelligence systems that combine semantic understanding, memory, reasoning, and autonomous navigation.

---

# Research Interests

## Core Areas

* Embodied AI
* Robotics
* Vision-Language Models (VLMs)
* Autonomous Navigation
* Semantic Mapping
* Robot Memory Systems
* Function Calling Agents
* Multi-Agent Systems
* Human-Robot Interaction

## Future Research Directions

* Embodied Foundation Models
* Persistent Semantic Memory
* Multi-Robot Collaboration
* Active Perception
* Semantic World Models
* Cloud-Edge Embodied Intelligence
* Long-Horizon Robot Planning
* General-Purpose Robot Agents

---

# Current Research

## VLM-Driven Autonomous Robot Navigation

I am developing a complete end-to-end navigation framework that replaces traditional navigation planners with a Vision-Language Model.

### Goal

Enable robots to navigate unknown indoor environments without relying on:

* Pre-built maps
* Traditional SLAM pipelines
* Hand-crafted planning rules

Instead, the robot directly reasons from visual observations and generates navigation actions.

### System Pipeline

RGB Image
→ Gemini Vision-Language Model
→ Navigation Planning
→ Path Generation
→ Robot Controller
→ Autonomous Motion

The system forms a complete closed loop:

Perception → Reasoning → Planning → Control

---

# Major Research Contributions

## 1. Function Calling Robot Agent

Designed a Function Calling architecture that transforms a VLM from a passive predictor into an active robotic agent.

The agent can autonomously invoke robot tools such as:

* get_robot_pose()
* get_front_image()
* get_obstacle_distance()
* publish_path()
* semantic labeling tools

This allows the model to:

* Query environmental information
* Access robot states
* Perform spatial reasoning
* Generate navigation decisions

instead of relying solely on a single image input.

---

## 2. Skill-Based YAML Framework

Designed a modular prompt-engineering framework based on YAML skill files.

Examples include:

* Navigation Skill
* Exploration Skill
* Inspection Skill

New robot behaviors can be introduced without modifying source code.

The framework supports:

* Rapid experimentation
* Behavior switching
* Prompt version control
* Agent specialization

---

## 3. Progressive Cognition Architecture

Proposed a hierarchical embodied cognition framework:

Scout
→ Inspector
→ Navigator

### Scout

Performs coarse-grained environmental exploration.

### Inspector

Identifies objects and semantic landmarks.

### Navigator

Uses accumulated semantic knowledge for precise navigation.

This architecture aims to mimic how humans progressively understand environments.

---

## 4. Semantic Memory-Based Navigation

Exploring the use of multi-turn memory to improve navigation performance.

Key idea:

Previous observations and semantic summaries are stored and reused in future planning cycles.

Potential benefits:

* Long-horizon reasoning
* Escaping local navigation failures
* Environment understanding
* Semantic map construction

---

# Technical Skills

## Robotics

* ROS2 Humble
* TF2
* Launch Systems
* Topic / Service / Action Communication
* MultiThreadedExecutor
* QoS Configuration
* Robot System Integration

## Computer Vision

* OpenCV
* RGB-D Processing
* Pinhole Camera Model
* Depth Reprojection
* Pixel-to-World Transformation
* Semantic Scene Understanding

## Sensors

Hands-on experience with:

* Orbbec Aurora
* Intel RealSense D435
* Orbbec Gemini 2L
* Orbbec Gemini 336

Practical experience in:

* Camera calibration
* Depth sensing
* Sensor evaluation
* Real-world perception issues

---

## Robot Control

Implemented and tuned:

* Pure Pursuit Controller
* PID Controller
* Reactive Replanning
* Obstacle Avoidance

Real-world debugging experience:

* Oscillation suppression
* Path tracking optimization
* Motion smoothness tuning

---

## Embodied AI & LLM Systems

Experience with:

* Gemini 2.5 Flash
* Gemini Robotics
* Prompt Engineering
* Function Calling
* Agent Architectures
* Multi-Turn Reasoning
* Semantic Memory Design

---

## Deployment

* Ubuntu 22.04
* ROS2 Humble
* WSL2 Development
* NVIDIA Jetson Orin Nano
* ONNX
* TensorRT Deployment Pipeline

---

# Research Experience

## Multi-Robot Pose Estimation

### Objective

Develop pose alignment methods for unmanned aerial vehicles.

### Method

* Motion capture systems
* Image-based pose alignment
* Graph Neural Networks

### Contributions

* CAN bus testing
* ROS2 workspace debugging
* System validation

---

## Motion Capture Based Robot Navigation Demo

### Objective

Demonstrate the advantages of motion-capture-assisted robot navigation.

### Contributions

* Robot navigation implementation
* Motion capture integration
* Path alignment algorithms

---

## Cloud-Edge Intelligent Robot

### Objective

Develop a cloud-based embodied robot using large multimodal models.

### Contributions

* Simulation environment construction
* Camera-to-Gemini integration
* VLM-based navigation pipeline

---

## VLM-Based Real Robot Navigation

### Objective

Deploy Vision-Language Models on physical robots for real-world navigation.

### Contributions

* Full ROS2 architecture design
* Function Calling Agent implementation
* Semantic navigation framework
* Real robot deployment and testing

Hardware Platform:

* Jetson Orin Nano
* Mecanum Chassis
* Depth Camera
* IMU

---

# Research Strengths

## Strong Areas

* ROS2 Development
* Robotics System Integration
* Embodied AI
* Vision-Language Models
* Function Calling Agents
* Prompt Engineering for Robotics
* Real Robot Deployment

## Developing Areas

* Semantic Mapping
* Multi-Agent Systems
* Navigation Algorithms
* Robot Cognition Architectures

## Areas for Further Growth

* SLAM
* Reinforcement Learning
* State Estimation
* Sensor Fusion
* Computer Vision Theory
* Academic Writing
* Large-Scale Experimental Design

---

# Career Goal

I aim to become a researcher or engineer working at the intersection of:

* Embodied AI
* Robotics
* Vision-Language Models
* Autonomous Navigation
* Multi-Agent Systems

Potential future paths include:

* PhD in Embodied AI / Robotics
* Robotics Research Scientist
* Embodied AI Engineer
* Autonomous Navigation Engineer
* Multi-Modal AI Researcher

---

# One-Sentence Summary

I am a Master's student working on Vision-Language-Model-driven robot navigation, focusing on Function Calling agents, semantic memory, embodied intelligence, and real-world deployment of autonomous robotic systems.
