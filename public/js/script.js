const video = document.createElement('video');
document.body.appendChild(video);

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models') //heavier/accurate version of tiny face detector
]).then(start)

async function start() {
    document.body.append('Models Loaded')
    
    // Access webcam
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        video.srcObject = stream;
    } catch (err) {
        console.error(err);
    }
    
    video.addEventListener('loadeddata', async () => {
        console.log('Video loaded');
        video.play(); // Start playing the webcam stream
        recognizeFaces();
    });
}

async function recognizeFaces() {
    const labeledDescriptors = await loadLabeledImages();
    console.log(labeledDescriptors);
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.7);

    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        const results = resizedDetections.map((d) => {
            return faceMatcher.findBestMatch(d.descriptor);
        });

        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
            drawBox.draw(canvas);
        });
    }, 100);
}

async function loadLabeledImages() {
    const labels = ['Vanshika Srivastava']; // for Webcam
    return Promise.all(
        labels.map(async (label) => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
                // Here, you can capture images from the webcam and use them for recognition
                // For simplicity, you can prompt the user to capture images manually
                // Or you can capture images programmatically using a library like `canvas.captureStream()`
                // Then, you can pass the captured images to face-api.js for face detection and recognition
                // For example:
                // const img = await captureImageFromWebcam();
                // const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                // descriptions.push(detections.descriptor);
            }
            document.body.append(label + ' Faces Loaded | ');
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}
