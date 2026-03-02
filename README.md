# SHL Assessment Recommendation Engine
Submitted for SHL Technical Assessment – 2026

This project recommends SHL assessments based on user skills or job roles.

## Approach

The system uses TF-IDF vectorization and cosine similarity to match skills with assessments.

## Tech Stack

Python  
FastAPI  
Scikit-learn  
Pandas  

## Features

- Skill based recommendations
- Role based recommendations
- FastAPI REST API
- TF-IDF similarity engine

## Project Structure

data/
    assessment.csv
    roles.csv

src/
    api.py
    recommender.py

test_system.py

## Installation

pip install -r requirements.txt

## Run API

uvicorn src.api:app --reload

Open:

http://127.0.0.1:8000/docs

## Example Query

/recommend/skills?skills=python machine learning

/recommend/role?role=data scientist
