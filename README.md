# discovery-queue
AI/ML Project.

## Running with Docker

For first time startup, run
```powershell
docker compose up --build
```
Note that a `.secrets` folder must be present alongside the `data` and `src`
folders with a `postgres_db`, `postgres_password`, and `postgres_user` file present
within it for proper functionality.

To run the application interactively (required for user input), run:

```powershell
docker-compose run --rm --service-ports web
```
