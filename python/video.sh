ffmpeg -start_number 0 -y -framerate 50 -i ./stitched/%d.jpg -vcodec libx264 -profile:v high444 -refs 5 -crf 0 ./video.mp4
