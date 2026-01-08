Nutrition-Bot

Flask + MongoDB + RAG-based Nutrition and Symptom Tracking System

Overview

Nutrition-Bot is a full-stack web application designed to help users track food intake, monitor nutritional values, log health symptoms, manage personal health goals, and interact with an AI-powered nutrition assistant.
The system combines structured nutritional data with a Retrieval-Augmented Generation (RAG) pipeline to deliver context-aware dietary guidance.

Key Features
User Authentication

User registration and login

Password hashing using bcrypt

Session-based authentication

Profile management (username, date of birth, gender)

Food Intake Logging

Log meals by quantity and unit (grams or servings)

Automatic nutrient calculation

Indian food nutrition dataset support

Fuzzy matching for food name suggestions

Nutrition Dashboard

Daily, weekly, and monthly aggregation

Tracked nutrients:

Energy (kcal)

Carbohydrates

Protein

Fat

Calcium

Iron

Vitamin C

Symptom Tracking and Analytics

Log symptoms with severity levels

Support for custom symptoms

Symptom resolution tracking

Trend analytics API suitable for Chart.js visualizations

Goals Management

Create, update, and delete personal health goals

Completion tracking

RESTful API support

AI Chat Assistant

Context-aware responses using RAG

Inputs include:

Recent food intake

Active symptoms

Conversation history

Multilingual support using Google Translator

Persistent chat session management

Technology Stack
Backend

Python (Flask)

Flask-Session

Flask-Bcrypt

MongoDB (PyMongo)

AI and NLP

Custom RAG pipeline

Vector database (FAISS or Chroma)

LLM integration (Groq or OpenAI-compatible)

Data Processing

Pandas

Indian food nutrition dataset (CSV)

Frontend

Jinja2 templates

Chart.js for analytics visualization
