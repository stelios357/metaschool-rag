# RAG-Based Course Recommendation System

## Overview

This repository contains a RAG (Retrieval-Augmented Generation) based course recommendation system. The system takes user queries related to learning topics and provides course recommendations with relevant snippets, learning outcomes, and target audience details.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Install Dependencies

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python rag_system.py
```

Then, input your learning topic, and the system will generate relevant course recommendations.

## Example Inputs and Outputs

### Example 1

**Input:**
```
Ask: I want to learn smart contract security
```

**Output:**
```
--- Recommendations for: "I want to learn smart contract security" ---

* Recommendation #1: Start Building on Core *
Snippet Preview: # Writing Secure Smart Contracts Welcome back, buddy! We love how you are being so great and...

- This course specifically focuses on smart contract security
- Key learning outcomes: understanding common vulnerabilities, avoiding them, implementing best practices
- Recommended for intermediate developers with smart contract knowledge
--------------------------------------------------
... (other recommendations)
```

### Example 2

**Input:**
```
Ask: Recommend courses for NFT development
```

**Output:**
```
--- Recommendations for: "Recommend courses for NFT development" ---

* Recommendation #1: Core C3 Launching a 10K NFT collection on Core *
Snippet Preview: Here's a simplified roadmap for creating your 10,000 NFT collection...
--------------------------------------------------
... (other recommendations)
```

### Example 3

**Input:**
```
Ask: quit
```

**Output:**
```
Exiting...
```

## Features

- Accepts natural language queries
- Retrieves and presents course recommendations
- Provides snippets, learning outcomes, and audience suitability

## Future Improvements

- Enhance recommendation accuracy with advanced retrieval models
- Add filtering options based on user expertise level
- Integrate with external APIs for real-time course updates
