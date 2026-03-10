# Movie Recommender System – System Map

## Purpose of the System

This system provides movie recommendations for users through a REST API.
It uses a recommendation algorithm and movie ratings stored in MongoDB.
The API is deployed in AWS using Docker containers running on ECS Fargate.

## High-Level Architecture

User
↓
api.reelradarhq.com (Domain)
↓
AWS Application Load Balancer
↓
ECS Service (Fargate)
↓
Docker Container
↓
FastAPI Application
↓
Recommender Service
↓
Recommendation Algorithm
↓
MongoDB Atlas

## Components

User  
Client that sends request to the API.

Domain  
DNS name pointing to AWS infrastructure.

ALB  
Receives HTTP request and forwards it to ECS.

ECS  
Runs Docker containers.

Docker Container  
Contains the FastAPI application.

FastAPI  
Handles API requests.

Recommender Service  
Coordinates recommendation logic.

Recommendation Algorithm  
Computes movie recommendations.

MongoDB Atlas  
Stores ratings and movie data.
