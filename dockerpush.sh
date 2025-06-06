docker login
docker tag ml-deploy-flask:latest <dockerhub-username>/ml-deploy-flask:latest
docker push <dockerhub-username>/<repository-name>:<tag>
