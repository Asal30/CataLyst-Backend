# CataLyst Backend - Deployment Guide

## Prerequisites
- Docker installed
- Docker Compose (optional, for local testing)

## Building the Docker Image

### Local Build
```bash
docker build -t catalyst-api:latest .
```

### Build with specific tag (for registry)
```bash
docker build -t your-registry/catalyst-api:1.0 .
```

## Running Locally

### Using Docker Compose (Recommended for testing)
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Using Docker directly
```bash
docker run -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  catalyst-api:latest
```

## Deployment Options

### 1. **AWS ECR + ECS/Fargate**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag catalyst-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/catalyst-api:latest

# Push to ECR
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/catalyst-api:latest
```

### 2. **Google Cloud Run**
```bash
# Build with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT/catalyst-api

# Deploy
gcloud run deploy catalyst-api --image gcr.io/YOUR_PROJECT/catalyst-api --platform managed --region us-central1 --memory 4Gi --timeout 3600
```

### 3. **Docker Hub**
```bash
# Tag image
docker tag catalyst-api:latest YOUR_USERNAME/catalyst-api:latest

# Push to Docker Hub
docker push YOUR_USERNAME/catalyst-api:latest
```

### 4. **Self-hosted (Docker Swarm / Kubernetes)**
```bash
# For Docker Swarm
docker service create --name catalyst-api \
  -p 8000:8000 \
  --mount type=volume,source=uploads,target=/app/uploads \
  --mount type=volume,source=outputs,target=/app/outputs \
  catalyst-api:latest
```

## Environment Variables

Add environment variables by modifying docker-compose.yml or passing them to `docker run`:

```bash
docker run -p 8000:8000 \
  -e DEBUG=false \
  catalyst-api:latest
```

## Persistent Storage

The container uses volumes for:
- `/app/uploads` - Uploaded images
- `/app/outputs` - Generated outputs

Ensure these are either:
- Mounted from host volumes
- Using Docker volumes
- Using cloud storage solutions

## Performance Considerations

- **Workers**: Currently set to 2 (adjust based on CPU cores)
- **Timeout**: Set to 120s (for model inference)
- **Memory**: Recommend 4GB+ due to PyTorch models
- **GPU Support**: Add `--gpus all` to docker run for CUDA support

## Monitoring

Check container health:
```bash
docker ps  # View running containers
docker logs catalyst-api  # View logs
docker exec catalyst-api curl http://localhost:8000/health  # Health check
```

## Production Deployment Checklist

- [ ] Update CORS origins from "*" to specific domains
- [ ] Set up environment variables (.env file)
- [ ] Configure persistent storage for uploads/outputs
- [ ] Set up monitoring and logging (CloudWatch, Datadog, etc.)
- [ ] Configure SSL/TLS at load balancer level
- [ ] Set resource limits (memory, CPU)
- [ ] Test image upload and model inference
- [ ] Set up auto-scaling if needed
- [ ] Configure backup strategy for model files
