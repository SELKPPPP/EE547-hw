#!/bin/bash

./build.sh
./run.sh

echo ""
echo "Testing all queries..."
for i in {1..10}; do
    docker-compose run --rm app python queries.py --query Q$i --dbname transit --format json
done

docker-compose down