import time, os, tqdm, argparse, glob, subprocess, warnings, cv2, pickle
from shutil import rmtree
from talkNetRun import (
    scene_detect,
    inference_video,
    track_shot,
    crop_video,
    evaluate_network,
    visualization,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument("--videoName", type=str, default="001", help="Demo video name")
parser.add_argument(
    "--videoFolder", type=str, default="demo", help="Path for inputs, tmps and outputs"
)
parser.add_argument(
    "--pretrainModel",
    type=str,
    default="pretrain_TalkSet.model",
    help="Path for the pretrained TalkNet model",
)

parser.add_argument(
    "--nDataLoaderThread", type=int, default=10, help="Number of workers"
)
parser.add_argument(
    "--facedetScale",
    type=float,
    default=0.25,
    help="Scale factor for face detection, the frames will be scale to 0.25 orig",
)
parser.add_argument(
    "--minTrack", type=int, default=10, help="Number of min frames for each shot"
)
parser.add_argument(
    "--numFailedDet",
    type=int,
    default=10,
    help="Number of missed detections allowed before tracking is stopped",
)
parser.add_argument(
    "--minFaceSize", type=int, default=1, help="Minimum face size in pixels"
)
parser.add_argument("--cropScale", type=float, default=0.40, help="Scale bounding box")

parser.add_argument("--start", type=int, default=0, help="The start time of the video")
parser.add_argument(
    "--duration",
    type=int,
    default=0,
    help="The duration of the video, when set as 0, will extract the whole video",
)

parser.add_argument(
    "--evalCol",
    dest="evalCol",
    action="store_true",
    help="Evaluate on Columnbia dataset",
)
parser.add_argument(
    "--colSavePath",
    type=str,
    default="/data08/col",
    help="Path for inputs, tmps and outputs",
)

args = parser.parse_args()

if os.path.isfile(args.pretrainModel) == False:  # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s" % (Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

if args.evalCol == True:
    # The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
    # 	              2. extract audio, extract video frames
    #                 3. scend detection, face detection and face tracking
    #                 4. active speaker detection for the detected face clips
    #                 5. use iou to find the identity of each face clips, compute the F1 results
    # The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
    # The step 4 and 5 need less than 10 minutes
    # Need about 20G space finally
    # ```
    args.videoName = "col"
    args.videoFolder = args.colSavePath
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.videoPath = os.path.join(args.videoFolder, args.videoName + ".mp4")
    args.duration = 0
    if os.path.isfile(args.videoPath) == False:  # Download video
        link = "https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s"
        cmd = "youtube-dl -f best -o %s '%s'" % (args.videoPath, link)
        output = subprocess.call(cmd, shell=True, stdout=None)
    if os.path.isdir(args.videoFolder + "/col_labels") == False:  # Download label
        link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
        cmd = "gdown --id %s -O %s" % (link, args.videoFolder + "/col_labels.tar.gz")
        subprocess.call(cmd, shell=True, stdout=None)
        cmd = "tar -xzvf %s -C %s" % (
            args.videoFolder + "/col_labels.tar.gz",
            args.videoFolder,
        )
        subprocess.call(cmd, shell=True, stdout=None)
        os.remove(args.videoFolder + "/col_labels.tar.gz")
else:
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + ".*"))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)


# Main function
def main():
    start_all = time.time()
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization
    args.pyaviPath = os.path.join(args.savePath, "pyavi")
    args.pyframesPath = os.path.join(args.savePath, "pyframes")
    args.pyworkPath = os.path.join(args.savePath, "pywork")
    args.pycropPath = os.path.join(args.savePath, "pycrop")
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(
        args.pyaviPath, exist_ok=True
    )  # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames
    os.makedirs(
        args.pyworkPath, exist_ok=True
    )  # Save the results in this process by the pckl method
    os.makedirs(
        args.pycropPath, exist_ok=True
    )  # Save the detected face clips (audio+video) in this process

    # Extract video
    start_extract = time.time()
    args.videoFilePath = os.path.join(args.pyaviPath, "video.avi")
    if args.duration == 0:
        command = (
            "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic"
            % (args.videoPath, args.nDataLoaderThread, args.videoFilePath)
        )
    else:
        command = (
            "ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic"
            % (
                args.videoPath,
                args.nDataLoaderThread,
                args.start,
                args.start + args.duration,
                args.videoFilePath,
            )
        )
    subprocess.call(command, shell=True, stdout=None)
    end_extract = time.time()
    print("Extract time: %.2f" % (end_extract - start_extract))

    # Framerate
    video = cv2.VideoCapture(args.videoFilePath)
    args.frameRate = int(video.get(cv2.CAP_PROP_FPS))
    video.release()

    # Extract audio
    extract_audio_start = time.time()
    args.audioFilePath = os.path.join(args.pyaviPath, "audio.wav")
    command = (
        "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
        % (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath)
    )
    subprocess.call(command, shell=True, stdout=None)
    extract_audio_end = time.time()
    print("Extract audio time: %.2f" % (extract_audio_end - extract_audio_start))

    # Extract the video frames
    extract_video_frames_start = time.time()
    command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % (
        args.videoFilePath,
        args.nDataLoaderThread,
        os.path.join(args.pyframesPath, "%06d.jpg"),
    )
    subprocess.call(command, shell=True, stdout=None)
    extract_video_frames_end = time.time()
    print(
        "Extract video frames time: %.2f"
        % (extract_video_frames_end - extract_video_frames_start)
    )

    # Scene detection for the video frames
    scene_detect_start = time.time()
    scene = scene_detect(args)
    scene_detect_end = time.time()
    print("Scene detection time: %.2f" % (scene_detect_end - scene_detect_start))

    # Face detection for the video frames
    face_detect_start = time.time()
    faces = inference_video(args)
    face_detect_end = time.time()
    print("Face detection time: %.2f" % (face_detect_end - face_detect_start))

    # Face tracking
    face_tract_start = time.time()
    allTracks, vidTracks = [], []
    for shot in scene:
        if (
            shot[1].frame_num - shot[0].frame_num >= args.minTrack
        ):  # Discard the shot frames less than minTrack frames
            allTracks.extend(
                track_shot(args, faces[shot[0].frame_num : shot[1].frame_num])
            )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    face_tract_end = time.time()
    print("Face tracking time: %.2f" % (face_tract_end - face_tract_start))

    # Face clips cropping
    face_crop_start = time.time()
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(
            crop_video(args, track, os.path.join(args.pycropPath, "%05d" % ii))
        )
    savePath = os.path.join(args.pyworkPath, "tracks.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(vidTracks, fil)
    face_crop_end = time.time()
    print("Face clips cropping time: %.2f" % (face_crop_end - face_crop_start))

    # Active Speaker Detection by TalkNet
    speaker_detect_start = time.time()
    files = glob.glob("%s/*.avi" % args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, "scores.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(scores, fil)
    speaker_detect_end = time.time()
    print(
        "Active speaker detection time: %.2f"
        % (speaker_detect_end - speaker_detect_start)
    )

    # Visualization, save the result as the new video
    visualize_start = time.time()
    visualization(vidTracks, scores, args)
    visualize_end = time.time()
    print("Visualization time: %.2f" % (visualize_end - visualize_start))

    end_all = time.time()
    print("Total time: %.2f" % (end_all - start_all))


if __name__ == "__main__":
    main()
