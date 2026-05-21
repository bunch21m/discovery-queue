# Discovery Queue

![Discovery Queue in Action](docs/design/Discovery%20Queue.png)

**Discovery Queue** is an end-to-end Machine Learning Recommender System designed to provide personalized video game recommendations. Users are presented with a set of games, interact with them to indicate preference, and receive continuously updated recommendations.

This personal project was built to explore the complexities of modern recommendation systems. Through researching industry standards, I implemented a two-stage architecture: a Two-Tower model for deep candidate generation and a Learning-to-Rank (LTR) model for downstream reranking. Built as a learning project with GenAI-assisted development and some help from my friend [@DirusLupito](https://github.com/DirusLupito) (see Development Notes at the bottom for details).

## Key Architectural Components

![Prediction Pipeline Architecture](docs/design/DQ%20Prediction%20Pipeline.drawio.png)

- **Candidate Generation (Two-Tower NN)**: A PyTorch-based model responsible for large-scale candidate retrieval. It produces dense embeddings for both users and game items, narrowing the 42,000+ game catalog down to ~5,000 candidates.
- **Indexed Game Embeddings**: Utilizes a `PostgreSQL` vector database with the `pgvector` extension for efficient Approximate Nearest Neighbor (ANN) search to feed the candidate generation stage.
- **Scoring (LambdaMART)**: A Learning-to-Rank (LTR) stage using `LightGBM LambdaMART` and a Game Feature Store to score and filter the ~5,000 candidates down to a highly relevant top 500.
- **Re-ranking (MMR)**: A Maximal Marginal Relevance (MMR) policy framework that processes the scored candidates. It utilizes a custom 50/50 penalty blend of Embedding Cosine Similarity and Genre Jaccard Similarity to mathematically enforce catalog diversity for the final 10 recommendations without destroying raw relevance.
- **Automated ML Pipeline**: Fully containerized via Docker Compose. A single command handles database initialization, embedding generation, synthetic training pair generation, model training, and then serves the frontend.

## Preliminary Performance Metrics

To establish a baseline understanding of the pipeline's effectiveness, the system's performance was evaluated by simulating various multi-faceted user personas (e.g., *ActionFanatic*, *StrategyPro*) to generate synthetic interactions. 

*Note: Something I learned while building this project is that standard hold-out evaluation can undercount relevant recommendations in domains with high item substitutability (e.g., many video games are very similar to each other). Because of this, I adjusted my evaluation method to emphasize genre-level precision and session-level success. These baselines use synthetic persona interactions; performance is expected to improve with real user data.*

While these represent small-scale offline metrics, they provide positive initial signals for the architecture:

- **Candidate Generation (Two-Tower)**:
  - Demonstrated the ability to narrow down the candidate pool, retrieving target items ranked within the **Top 7.2%** of the 42,000+ catalog on average.
  - **Overall Candidate Hit Rate**: 66.7% (Evaluated using a temporal leave-5-out validation. The query generated successfully retrieved at least 1 of the 5 hidden 'future' games within the top 250 candidates in two-thirds of sessions).
- **Final Recommendation Generation (LambdaMART + MMR)**:
  - **Diversity Improvement**: Showed a **+548.9% MMR Diversity Gain**, indicating that the MMR policy successfully mitigates filter bubbles.
  - Showed strong **Genre Precision@10** metrics during evaluation (averaging 66.7% across all personas, peaking at 86.7%), suggesting the model successfully captures core interests while MMR diversifies the final slate.

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

The startup sequence automatically handles the entire pipeline: Database Setup -> Embedding Initialization -> Training Pair Generation -> Two-Tower Model Training -> Embedding Indexing -> LTR Reranker Training -> Web Server Initialization.

**Important Note**: Because the initial startup includes full data ingestion, embedding indexing, and training multiple machine learning models from scratch, the first run can take a significant amount of time. Please be patient and allow the Docker container to complete the automated ML pipeline.

Once the application is running, navigate to `http://127.0.0.1:5000/` to test the recommender system.

## Content Disclaimer

This project utilizes real-world dataset metadata to train the recommendation models and populate the frontend. As a result, some of the generated game recommendations, cover art, and descriptions may contain mature themes or content that is inappropriate for all audiences.

## Development Notes

Generative AI was used throughout this project as a pair-programmer and tutor. It was instrumental in helping me research architectural concepts, optimize code, and understand concepts such as the practical trade-offs between relevance and diversity in recommendation systems. My focus was on deeply understanding the generated implementations to maximize my learning experience.

Assistance from my friend [@DirusLupito](https://github.com/DirusLupito) was also utilized, in the form of helping with the frontend, an automated wishlist/skipping tool, and in exploring simple alternative recommendation methods.
