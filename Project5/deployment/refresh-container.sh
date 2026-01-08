#!/bin/bash

CONTAINER_NAME="site"

IMAGE_NAME="26piyush/aws-outage-site:v2"

sudo docker stop $CONTAINER_NAME
sudo docker rm $CONTAINER_NAME

echo "Pulling latest Docker image $IMAGE_NAME..."
sudo docker pull $IMAGE_NAME

echo "Starting new container $CONTAINER_NAME..."
sudo docker run -d --name $CONTAINER_NAME --restart unless-stopped -p 8080:80 $IMAGE_NAME