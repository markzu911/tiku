#!/bin/bash
# 部署更新到远程服务器
# 用法: bash deploy-update.sh

set -e

SERVER="root@218.244.140.226"
PROJECT_DIR="/root/exam-bank"

echo "=== 1. 打包项目文件 ==="
cd "$(dirname "$0")"
tar czf /tmp/exam-bank-update.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='debug' \
    --exclude='.venv' \
    --exclude='runtime' \
    --exclude='exam_bank.db' \
    --exclude='.env' \
    app/ static/ Dockerfile docker-compose.yml requirements.txt .dockerignore

echo "=== 2. 传输到服务器 ==="
scp /tmp/exam-bank-update.tar.gz $SERVER:/tmp/exam-bank-update.tar.gz

echo "=== 3. 服务器端部署 ==="
ssh $SERVER << 'REMOTE_SCRIPT'
set -e
PROJECT_DIR="/root/exam-bank"
cd $PROJECT_DIR

# 备份旧代码
echo "备份旧代码..."
cp -r app /tmp/app-backup-$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# 解压新代码
echo "解压更新..."
tar xzf /tmp/exam-bank-update.tar.gz -C $PROJECT_DIR
rm /tmp/exam-bank-update.tar.gz

# 重新构建并启动
echo "=== 4. 重新构建 Docker 镜像 ==="
docker compose --env-file runtime/docker-stack.env --project-name exam-bank build app

echo "=== 5. 重启应用容器 ==="
docker compose --env-file runtime/docker-stack.env --project-name exam-bank up -d app

echo "=== 6. 等待健康检查 ==="
sleep 5
docker ps --filter "name=exam-bank-app" --format "{{.Names}} {{.Status}}"

echo ""
echo "=== 部署完成 ==="
echo "访问地址: http://218.244.140.226:8000/"
REMOTE_SCRIPT
