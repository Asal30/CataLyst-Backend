# CataLyst-Backend

This is the backend API for the CataLyst application - an intelligent cataract screening system using mobile images.

## Overview

The CataLyst backend is built with FastAPI and provides endpoints for:
- Image upload and preprocessing
- AI-powered cataract prediction using CBM (Concept Bottleneck Models)
- Visual explanations with Grad-CAM
- Prediction logging and analysis

## Quick Start with Docker 🐳

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Development Environment
```bash
# Clone the repository
git clone <repository-url>
cd CataLyst-Backend

# Start development environment
docker-compose up --build

# Or use the build script (Windows)
.\build.ps1 -Command dev
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Production Environment
```bash
# Start production environment with nginx
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

# Or use the build script (Windows)
.\build.ps1 -Command prod
```

## Manual Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

#### Development Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Mode (with Gunicorn)
```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120 app.main:app
```

## Docker Commands

### Build Script (Windows)
```powershell
# Build the image
.\build.ps1 -Command build

# Start development
.\build.ps1 -Command dev

# Start production
.\build.ps1 -Command prod

# Stop all containers
.\build.ps1 -Command stop

# Clean up resources
.\build.ps1 -Command clean

# View logs
.\build.ps1 -Command logs

# Open shell in container
.\build.ps1 -Command shell
```

### Manual Docker Commands
```bash
# Build image
docker build -t catalyst-backend:latest .

# Run with Docker Compose
docker-compose up --build

# Run production with nginx
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Open shell in running container
docker-compose exec catalyst-api bash
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
CataLyst-Backend/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── api/
│   │   └── routes.py      # API endpoints
│   ├── models/            # ML models
│   ├── services/          # Business logic
│   └── utils/             # Utilities
├── uploads/               # Uploaded images (mounted volume)
├── outputs/               # Generated outputs (mounted volume)
├── logs/                  # Application logs (mounted volume)
├── nginx/                 # Nginx configuration for production
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Development environment
├── docker-compose.prod.yml # Production environment
├── docker-compose.override.yml # Development overrides
├── .dockerignore         # Docker ignore file
├── build.ps1            # Windows build script
└── requirements.txt      # Python dependencies
```

## Environment Configuration

The application uses hardcoded configuration for simplicity. Key settings:

- **Port**: 8000
- **Upload Directory**: `uploads/`
- **Output Directory**: `outputs/`
- **Log Directory**: `logs/`
- **CORS**: Allows all origins (`*`)

## Health Checks

The application includes health check endpoints:
- `GET /health` - Basic health check
- Container health checks are configured in Docker Compose

## Volumes

The following directories are mounted as Docker volumes:
- `uploads/` - For storing uploaded images
- `outputs/` - For storing generated heatmaps and overlays
- `logs/` - For storing application logs

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 8000
   netstat -ano | findstr :8000
   # Kill the process or change the port in docker-compose.yml
   ```

2. **Permission issues with volumes**
   ```bash
   # Ensure directories exist and have proper permissions
   mkdir uploads outputs logs
   ```

3. **Build fails**
   ```bash
   # Clear Docker cache
   docker system prune -f
   # Rebuild
   docker-compose build --no-cache
   ```

### Logs

```bash
# View application logs
docker-compose logs catalyst-api

# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f
```

## Contributing

1. Make changes to the code
2. Test with Docker: `docker-compose up --build`
3. Ensure all tests pass
4. Submit a pull request

