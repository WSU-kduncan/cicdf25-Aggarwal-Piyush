# Overview 

In this part of the project, I extended my CI pipeline so that it doesn’t just push a single “latest” tag. Instead, it now supports semantic versioning, which allows me to keep older builds, roll back if needed, and follow a clean release structure based on the version tag I push to GitHub.

The main idea is simple:
When I push a git tag such as v1.2.0, GitHub Actions automatically:

- Builds a new Docker image of my website

- Pushes multiple tags to DockerHub:

- latest

- 1

- 1.2


# Git Tags

Before configuring the workflow, I learned how to view, create, and push tags.

See existing git tags

`git tag`

Create a new tag

`git tag -a v1.0.0 -m "Initial version"`

Push the tag to GitHub

`git push origin v1.0.0`


The moment the tag is pushed, my GitHub Action runs automatically.


# Semantic Versioning

My workflow now triggers on tag pushes only, not on branch commits.

This prevents accidental builds and ensures only intentional releases create a new Docker image.

The workflow uses:

- docker/metadata-action → automatically generates semantic tags

- docker/setup-buildx-action → builds the image

- docker/login-action → authenticates to DockerHub

- docker/build-push-action → builds and pushes all tags

The metadata action reads my git tag (ex. v1.2.3) and automatically builds these tags:

`latest`

`1`

`1.2`

`1.2.3`

This solves everything: versioning, rollback, and cleaner traceability.

# Workflow File

The workflow runs only when I push a Git tag.

It checks out my repository so it can access my Dockerfile and web-content folder.

It sets up Buildx (Docker’s advanced builder).

It logs in to DockerHub using my GitHub Secrets.

It uses the metadata action to automatically create semantic version tags.

It builds and pushes the image with all generated tags.

Everything happens automatically from a single command:

`git push origin v1.0.0`