# Project 5

This repository contains the implementation of a continuous deployment pipeline for a Dockerized web application using DockerHub webhooks and an AWS EC2 application server.

The project demonstrates how a running production container can be automatically refreshed when a new Docker image is pushed, without manual intervention.

**Overview:**
This repository contains all configuration, infrastructure, CI, and CD components for deploying an automated Docker-based web application using AWS EC2, DockerHub, GitHub Actions, and a custom webhook listener.

- `AGGARWAL-lb-cf.yml` - AWS Cloud Formation template from Project 3 which I used on the new AWS account. 
- `README-CI.md` - README file for Project 4 where it goes over Cotinuous integration this includes details about GitHub actions, tags, and auto pushing docker images. 
- `README-CD.md`- README file for Project 5 which goes over Continous Deployment. Inlcudes info about Docker webhooks, bash scripts, and webhook service. 

- `deployment/hooks.json` - Webhok defintion file that refreshes the site by running refresh-container.sh script when there is a valid webhook request is recieved. Only authenticated triggers can refresh the the web container. 
- `deployment/refresh-container.sh` - This script stops any previously running containers and pulls the latest version of the image from DockerHub and starts a new container. 
- `deployment/webhook.service` - Starts the webhook listening on boot and restart if it fails. 
- `web-content/` - Static web application files served by the container
- Web Application: `http://54.165.215.255:8080`


