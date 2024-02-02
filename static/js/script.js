document.getElementById('startButton').addEventListener('click', function() {
    var video = document.getElementById('videoElement');
    video.style.display = 'block';
    this.style.display = 'none';
    document.getElementById('stopButton').style.display = 'inline';

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch(function (error) {
                console.log("No se pudo acceder a la cámara: ", error);
            });
    }
});

document.getElementById('stopButton').addEventListener('click', function() {
    var video = document.getElementById('videoElement');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    video.style.display = 'none';
    this.style.display = 'none';
    document.getElementById('startButton').style.display='inline';
});