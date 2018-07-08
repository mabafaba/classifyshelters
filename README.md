

# Overview

## Image Segmentation: refugee shelter identification

### Docker setup

1. Clone my github repo: https://github.com/mawall/docker_img (take the CPU container to work on your laptop: tensorflow-keras)
2. ./build_image.sh
3. You have to set some environment variables so that the container can access the repo and the google drive data folder. These are my paths, you’ll have to replace them with your own:
```
# export NOTEBOOK_DIR=/Users/Marcus/Documents/CS/projects/notebooks/
# export SHELTER_REPO_PATH=/Users/Marcus/Documents/CS/projects/classify_shelters/
# export SHELTER_DATA_PATH=/Users/Marcus/googledrive/shelterdata/
```
4. ./run.sh
This will start the container and the notebook server. You can cmd-doubleclick on the link to start the jupyter interface.

5. The jupyter interface has a terminal build in, but it’s not ideal. I recomment to open up a separate shell to work in the cotnainer directly if necessary: `docker exec -it <<CONTAINERNAME>> /bin/bash` will start a terminal inside the container.
- The repo can be found under `/repo`
-  The data folder unter `/media/data`
-  The jupyer notebooks are under `/notebooks`

GitHub mawall/docker_img (docker_img - CV-ML docker images)

Using a terminal to get inside the container isn’t always necessary. Depending on what you want to do, I recommend to use notebooks to test stuff out in general. If you want to work on code, you can do that as you’re used to outside of the container in your preferred IDE, since everything is just linked in.
To dynamically reload the package code in a notebook, use the autoreload magic:
```
%load_ext autoreload
%autoreload 2
```

## Notebook for current workflow

notebook_template/train_model_template.ipynb

## Models / Architectures

shelter/designs/



