@echo off
echo 🔐 Logging into ECR...
aws ecr get-login-password --region us-east-1 --profile slalom_IsbUsersPS-701274126193 | docker login --username AWS --password-stdin 701274126193.dkr.ecr.us-east-1.amazonaws.com

echo 🔨 Building Docker image...
docker build --platform linux/amd64 -f Dockerfile.lambda -t insurance-helper-backend:latest .

echo 📦 Tagging image...
docker tag insurance-helper-backend:latest 701274126193.dkr.ecr.us-east-1.amazonaws.com/insurance-helper-backend:latest

echo 🚀 Pushing to ECR...
docker push 701274126193.dkr.ecr.us-east-1.amazonaws.com/insurance-helper-backend:latest

echo ⚡ Updating Lambda...
aws lambda update-function-code --function-name insurance-helper-backend --image-uri 701274126193.dkr.ecr.us-east-1.amazonaws.com/insurance-helper-backend:latest --region us-east-1 --profile slalom_IsbUsersPS-701274126193

echo ✅ Done! Lambda updated successfully.