
# Convert video format from mp4 to wav to perform audio recognition module

import os
from glob import glob

cmd = "ffmpeg -i {} -f wav -ar 16000 {}"
pattern = "audios/*/*.mp4"

for vp in glob(pattern):
    _cmd = "ffmpeg -i {} -f wav -ar 16000 {}".format(vp,vp.replace("mp4","wav"))
    os.system(_cmd)
