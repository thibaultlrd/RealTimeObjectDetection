# Run RealTimeObjectDetection on Kubernetes (minikube)

## Build images inside minikube
```bash
minikube start --cpus=4 --memory=8192
minikube addons enable metrics-server
minikube addons enable ingress

# Use minikube's docker daemon
eval $(minikube -p minikube docker-env)

# Build images (tags must match manifests)
DOCKER_BUILDKIT=1 docker build -t rtod-backend:latest -f Dockerfile.backend ..
DOCKER_BUILDKIT=1 docker build -t rtod-frontend:latest -f Dockerfile.frontend ..
```

## Deploy manifests
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

## Access
```bash
echo "$(minikube ip) rtod.local" | sudo tee -a /etc/hosts
xdg-open http://rtod.local/
# API
curl http://rtod.local/api/health
```
