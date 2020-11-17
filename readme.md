# mo445 first task

# to run
docker run -it --rm \
  --env DISPLAY=$DISPLAY --env NUMIMAGES=1 --env NUMFILTERS=1 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) \
  adnrv/opencv \
  /bin/bash -c \
  "\
  python3 main.py
  "

If desired, you can increase the variable NUMIMAGES up to 200 (which is the entirety of the image dataset) and the NUMFILTERS variable to whichever integer you'd like (this increases the number of kernels in the KernelBank).