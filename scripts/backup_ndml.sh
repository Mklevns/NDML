#!/bin/bash
# NDML Backup Script

set -e

# Configuration
NDML_HOME=${NDML_HOME:-/opt/ndml}
BACKUP_DIR=${BACKUP_DIR:-/backup/ndml}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ndml_backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

echo "Starting NDML backup..."

# Backup configuration
echo "Backing up configuration..."
cp -r "${NDML_HOME}/config" "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup checkpoints
echo "Backing up checkpoints..."
cp -r "${NDML_HOME}/data/checkpoints" "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup Redis data
echo "Backing up Redis..."
redis-cli BGSAVE
sleep 5  # Wait for background save
cp /var/lib/redis/dump.rdb "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup logs (last 7 days)
echo "Backing up logs..."
find "${NDML_HOME}/logs" -name "*.log" -mtime -7 -exec cp {} "${BACKUP_DIR}/${BACKUP_NAME}/logs/" \;

# Create metadata file
cat > "${BACKUP_DIR}/${BACKUP_NAME}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "ndml_version": "$(cat ${NDML_HOME}/VERSION 2>/dev/null || echo 'unknown')",
    "backup_type": "full",
    "components": ["config", "checkpoints", "redis", "logs"]
}
EOF

# Compress backup
echo "Compressing backup..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

# Clean old backups (keep last 7)
echo "Cleaning old backups..."
ls -t "${BACKUP_DIR}"/ndml_backup_*.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}