# discovery-queue
AI/ML Project.

## Running with Docker

For first time startup, run
```powershell
docker compose up --build
```
in the `discovery-queue` folder.

Note that a `.secrets` folder must be present alongside the `data` and `src`
folders with a `postgres_db`, `postgres_password`, and `postgres_user` file present
within it for proper functionality.

Once the application is running, navigate to `http://127.0.0.1:5000/` to see the webpage.
