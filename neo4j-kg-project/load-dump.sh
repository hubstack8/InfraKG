#!/bin/bash
echo "Unzipping dump..."
gzip -d ./dump/your-kg.dump.gz

echo "Starting Neo4j container..."
docker-compose up -d

echo "Waiting 10 seconds for Neo4j to start..."
sleep 10

echo "Loading dump into Neo4j..."
docker exec -it kg-neo4j bash -c "neo4j-admin load --from=/import/../dump/your-kg.dump --database=neo4j --force"

echo "Done. Restarting Neo4j..."
docker-compose restart
