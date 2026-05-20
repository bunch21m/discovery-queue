# Discovery Queue

**Discovery Queue** is an end-to-end Machine Learning Recommender System designed to provide personalized video game recommendations. Users are presented with a slate of games, interact with them to indicate preference, and receive continuously updated recommendations.

This personal project was built to explore the complexities of modern recommendation systems. Through researching industry standards, I implemented a two-stage architecture: a Two-Tower model for deep candidate generation and a Learning-to-Rank (LTR) model for downstream reranking, supported by vector search.

*Note on Development: Generative AI was used throughout this project as a pair-programmer and tutor. It was instrumental in helping me research architectural concepts, optimize code, and understand concepts such as the practical trade-offs between relevance and diversity in recommendation systems. My focus was on deeply understanding the generated implementations to maximize my learning experience.*

## Key Architectural Components

![Prediction Pipeline Architecture](docs/design/DQ%20Prediction%20Pipeline.drawio.png)

- **Two-Tower Neural Network**: A PyTorch-based model responsible for large-scale candidate generation. It produces dense embeddings for both users and game items, allowing for rapid candidate retrieval.
- **Vector Database**: Utilizes `PostgreSQL` with the `pgvector` extension for efficient Approximate Nearest Neighbor (ANN) search over a catalog of 42,000+ items.
- **Advanced Reranking Pipeline**: Implements a Learning-to-Rank (LTR) stage using `LightGBM LambdaMART`, paired with a Maximal Marginal Relevance (MMR) policy framework. This balances the raw relevance of recommendations with catalog diversity.
- **Automated ML Pipeline**: Fully containerized via Docker Compose. A single command handles database initialization, embedding generation, synthetic training pair generation, model training, and metrics evaluation, before serving the frontend.

## Performance Metrics

The system's performance was evaluated by simulating various multi-faceted user personas (e.g., *ActionFanatic*, *StrategyPro*).

- **Candidate Generation (Two-Tower)**:
  - Effectively narrows down the candidate pool, consistently retrieving target items ranked within the **Top 7.2%** of the 42,000+ catalog.
- **Final Slate Generation (LambdaMART + MMR)**:
  - **Overall Session Hit Rate**: 66.7% (binary success rate for sessions).
  - **Diversity Improvement**: Achieved a **+548.9% MMR Diversity Gain**.
  - Retained strong Precision@10 metrics (as high as 86.7% for specific personas) while significantly diversifying the final recommendations presented to the user.

## Technology Stack

- **Machine Learning**: PyTorch, LightGBM, Scikit-Learn
- **Data & Vector Store**: PostgreSQL, `pgvector`
- **Backend & Serving**: Python, Flask, Pandas, NumPy
- **Infrastructure**: Docker, Docker Compose

## Running the Project

### Prerequisites

Before starting the pipeline, there are a few files intentionally excluded from version control that you must set up locally:

1. **Dataset Download**: This project relies on the [Steam Games Dataset from Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset). Please download the dataset and place the extracted file into the `data/` directory (e.g., as `data/games.json`).
2. **Database Secrets**: A `.secrets/` folder must be created in the root directory (alongside `data` and `src`) containing three files: `postgres_db`, `postgres_password`, and `postgres_user`. These text files should contain your desired credentials for setting up the PostgreSQL database.
3. **Environment Variables**: An optional `.env` file can be placed in the project root if you need to override any specific local configurations.

### Startup

For first-time startup, run:
```powershell
docker compose up --build
```
in the `discovery-queue` folder.

The startup sequence automatically handles the entire pipeline: Database Setup -> Embedding Indexing -> Two-Tower Model Training -> LTR Reranker Training -> Evaluation Metrics Generation -> Web Server Initialization.

**Important Note**: Because the initial startup includes full data ingestion, embedding indexing, and training multiple machine learning models from scratch, the first run can take a significant amount of time. Please be patient and allow the Docker container to complete the automated ML pipeline.

Once the application is running, navigate to `http://127.0.0.1:5000/` to test the recommender system.

## Content Disclaimer

This project utilizes real-world dataset metadata to train the recommendation models and populate the frontend. As a result, some of the generated game recommendations, cover art, and descriptions may contain mature themes or content that is inappropriate for all audiences.
