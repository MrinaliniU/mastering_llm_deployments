Mastering LLM Deployment
Prepared for Intuit Last Updated: Jan 23, 2025
Course Overview
Mastering LLM Deployment is designed for software engineers and data scientists who aim to deploy large language models (LLMs) both efficiently and cost-effectively.
Objectives and Content
• Participants will learn essential techniques for optimizing LLMs, including model distillation, quantization, and pruning.
• The course provides hands-on experience deploying these models into AWS ECS using Docker.
• It includes strategic insights into cost-saving measures.
• By the end of the course, participants will possess the skills necessary to deploy optimized LLMs in a production environment, ensuring efficient resource usage and cost optimization.
Duration and Audience
• Course Duration: 2 days.
• Audience: Software engineers and data scientists with a basic familiarity with TensorFlow, Keras, and AWS. Essential prerequisites include an understanding of NLP, Deep Learning, and familiarity with Python.
Productivity Objectives
Upon completion of this course, you should be able to:
• Distill, quantize, and prune large language models.
• Analyze and optimize the resource requirements for LLM deployment.
• Deploy optimized LLMs into AWS ECS using Docker.
• Implement TensorFlow Serving and Flask API for LLM deployment.
• Understand and apply cost-saving strategies for LLM deployment.

--------------------------------------------------------------------------------
Course Outline
I. Course Introduction and Case Study
• Overview of LLM Deployment Challenges and Objectives
    ◦ Introduction to the course structure, objectives, and key challenges in LLM deployment.
• Case Study: Successful LLM Deployment
    ◦ Detailed analysis of a real-world LLM deployment case, highlighting challenges, solutions, and outcomes.
II. Quick Recap of TensorFlow and Keras
• Overview of TensorFlow and Keras
    ◦ Brief refresher on TensorFlow and Keras functionalities relevant to LLMs.
• Lab: Basic Keras and TensorFlow Exercises
    ◦ Hands-on exercises to familiarize participants with essential TensorFlow and Keras functions.
III. Model Distillation
• Introduction to Model Distillation
    ◦ Overview of model distillation and its benefits for LLMs.
• Lab: Distilling a Pre-trained LLM using TensorFlow
    ◦ Hands-on exercise to distill a given LLM, using the SQuAD dataset.
    ◦ Participants will learn to reduce the model size and improve inference speed.
IV. Model Quantization
• Understanding Model Quantization
    ◦ Introduction to quantization techniques and their benefits.
• Lab: Quantizing an LLM using TensorFlow
    ◦ Practical lab to quantize a pre-trained LLM, using the IMDB dataset for sentiment analysis.
    ◦ Participants will convert the model to lower precision to save memory and improve performance.
V. Model Pruning
• Fundamentals of Model Pruning
    ◦ Explanation of pruning methods and their benefits.
• Lab: Pruning an LLM using TensorFlow
    ◦ Hands-on exercise to prune an LLM, using the SST-2 dataset for sentiment analysis.
    ◦ Participants will learn to remove redundant neurons and weights to optimize the model.
VI. Deployment and Cost Optimization
• Preparing for Deployment
    ◦ Introduction to TensorFlow Serving and Flask API
        ▪ Overview of serving models using TensorFlow Serving and Flask.
    ◦ Lab: Setting Up Docker for Deployment
        ▪ Hands-on lab to create Docker containers for LLM deployment.
        ▪ Participants will learn to package the optimized LLMs into Docker containers.
• Deploying to AWS ECS
    ◦ Overview of AWS ECS and Deployment Strategies
        ▪ Introduction to AWS ECS services and deployment options.
    ◦ Lab: Deploying LLMs with TensorFlow Serving on AWS ECS
        ▪ Practical exercise to deploy an LLM using TensorFlow Serving on AWS ECS.
        ▪ Participants will learn to set up ECS tasks and services.
    ◦ Lab: Deploying LLMs with Flask API on AWS ECS
        ▪ Hands-on lab to deploy an LLM using Flask API on AWS ECS.
        ▪ Participants will implement and test REST API endpoints for model inference.
VII. Final Hackathon (3 hours)
• Project: Text Summarization using CNN/DailyMail Dataset
    ◦ Participants will work individually to deploy a fine-tuned LLM for text summarization using the CNN/DailyMail dataset.
    ◦ They will apply distillation, quantization, and pruning techniques, and deploy the model using Docker and AWS ECS.