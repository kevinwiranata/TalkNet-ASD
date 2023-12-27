import sys, os, tqdm, torch, glob, subprocess, cv2, pickle, numpy, math, python_speech_features
import concurrent.futures

from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy import signal

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from talkNet import talkNet

import numpy as np

modelFile = "model/DNN/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "model/DNN/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, "scene.pckl")
    if sceneList == []:
        sceneList = [
            (videoManager.get_base_timecode(), videoManager.get_current_timecode())
        ]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write(
            "%s - scenes detected %d\n" % (args.videoFilePath, len(sceneList))
        )
    return sceneList


# Function to process an image, can be used with multiprocessing if needed
def process_image(fname):
    image = cv2.imread(fname)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    return blob, h, w


def inference_video(args):
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))
    flist.sort()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(process_image, flist))

    dets = []
    for fidx, (blob, h, w) in enumerate(processed_images):
        net.setInput(blob)
        faces3 = net.forward()
        dets.append([])
        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([w, h, w, h])
                dets[-1].append(
                    {"frame": fidx, "bbox": box.tolist(), "conf": confidence}
                )
        sys.stderr.write(
            "%s-%05d; %d dets\r" % (args.videoFilePath, fidx, len(dets[-1]))
        )

    savePath = os.path.join(args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if (
                max(
                    numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > args.minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def read_and_pad_image(image_path, bsi, cs, dets, det_idx):
    image = cv2.imread(image_path)
    frame = np.pad(
        image,
        ((bsi, bsi), (bsi, bsi), (0, 0)),
        "constant",
        constant_values=(110, 110),
    )
    my = dets["y"][det_idx] + bsi  # BBox center Y
    mx = dets["x"][det_idx] + bsi  # BBox center X
    bs = dets["s"][det_idx]  # Detection box size
    face = frame[
        int(my - bs) : int(my + bs * (1 + 2 * cs)),
        int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
    ]
    return cv2.resize(face, (224, 224))


def crop_video(args, track, cropFile):
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    cs = args.cropScale
    bsi = int(max(dets["s"]) * (1 + 2 * cs))

    frame_to_det_index = {frame: i for i, frame in enumerate(track["frame"])}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        faces = list(
            executor.map(
                lambda f: read_and_pad_image(
                    flist[f], bsi, cs, dets, frame_to_det_index[f]
                ),
                track["frame"],
            )
        )

    for face in faces:
        vOut.write(face)

    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)
    )
    subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
        % (cropFile, audioTmp, args.nDataLoaderThread, cropFile)
    )  # Combine audio and video file
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")

    return {"track": track, "proc_track": dets}


def evaluate_network(files, args):
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + ".wav"))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret is True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use TalkNet
        duration = 3  # 2 seconds
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = (
                    torch.FloatTensor(
                        audioFeature[i * duration * 100 : (i + 1) * duration * 100, :]
                    )
                    .unsqueeze(0)
                    .cpu()
                )
                inputV = (
                    torch.FloatTensor(
                        videoFeature[i * duration * 25 : (i + 1) * duration * 25, :, :]
                    )
                    .unsqueeze(0)
                    .cpu()
                )
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)
                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                score = s.lossAV.forward(out, labels=None)
                scores.extend(score)
        allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(
            float
        )
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
    command = (
        "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic"
        % (
            os.path.join(args.pyaviPath, "video_only.avi"),
            os.path.join(args.pyaviPath, "audio.wav"),
            args.nDataLoaderThread,
            os.path.join(args.pyaviPath, "video_out.avi"),
        )
    )
    subprocess.call(command, shell=True, stdout=None)
