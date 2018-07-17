








# on google cloud start instance
click on instance name
click drop down next to "ssh"
gcloud command -> copy&and paste and run in terminal
something like:`gcloud compute --project "shelter-209614" ssh --zone "us-east1-b" "cpuonly")
if sucessfull you are now in the google drive instance virtual machine command line

# now install some basics:
## git:
sudo apt-get install git

## docker:
sudo curl -sSL https://get.docker.com/ | sh


# now download our stuff:
## to get docker image build files from marcus repo run:
not run this: git clone https://github.com/mawall/docker_img.git

instead get m's fork that shows you the true path(s):
https://github.com/mabafaba/docker_img


## get the model
git clone https://github.com/mabafaba/classifyshelters.git


# now build the image:
cd docker_img
sh build_image.sh

wait this takes 5-30 minutes
you now have the docker image ready to rumble on the VM, how sweet is that?

# mount bucket
the data is on google cloud storage that we need to "mount" to the computing instance. Then to the system It will look like a plugged in usb drive or something.
*in the bucket I only put the *.npy files, not the original images!* So you can't run create_training_data() or create_testing_data() from here but you don't have to yay.

# mounting bucket failed so copy the data from the bucket instead:
gsutil -m cp -R gs://shelterdata media

if that fails too youre doomed. (as far as this doc goes)

install gcsfuse

https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md


Add the gcsfuse distribution URL as a package source and import its public key:

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

Update the list of packages available and install gcsfuse.

sudo apt-get update
sudo apt-get install gcsfuse

(Ubuntu before wily only) Add yourself to the fuse group, then log out and back in:

sudo usermod -a -G fuse $USER
exit

Future updates to gcsfuse can be installed in the usual way: sudo apt-get update && sudo apt-get upgrade.

# MOUNT ME PLEASE UHHH
```
mkdir media
gcsfuse shelterdata ./media
```
.... to make notebook accessible:

in run.sh:
jupyter notebook --ip=0.0.0.0
then terminal: ifconfig


# known issues & solution
### when starting container
`Error response from daemon: Conflict. The container name "/shelter_container" is already in use by container`  ..... You have to remove (or rename) that container to be able to reuse that name.
docker rm .....
then try again






