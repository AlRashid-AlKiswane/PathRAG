


docker run -it --rm \
  --network host \
  --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/app \
  ncec-path-rag