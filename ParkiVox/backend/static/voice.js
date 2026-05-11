let mediaRecorder;
let audioChunks = [];
let timerInterval;
let seconds = 0;

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const audioPreview = document.getElementById("audioPreview");
const recordedFile = document.getElementById("recordedFile");
const timer = document.getElementById("timer");

recordBtn.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;

  // Reset timer
  seconds = 0;
  timer.textContent = "00:00";
  timerInterval = setInterval(() => {
    seconds++;
    let mins = String(Math.floor(seconds / 60)).padStart(2, '0');
    let secs = String(seconds % 60).padStart(2, '0');
    timer.textContent = `${mins}:${secs}`;
    if (seconds >= 30) { stopBtn.click(); } // auto-stop at 30 sec
  }, 1000);

  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = () => {
    clearInterval(timerInterval);
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    audioPreview.src = audioUrl;
    audioPreview.style.display = "block";

    // Convert blob to Base64 for hidden input
    const reader = new FileReader();
    reader.onloadend = () => {
      recordedFile.value = reader.result; // Base64 string
    };
    reader.readAsDataURL(audioBlob);
  };
});

stopBtn.addEventListener("click", () => {
  mediaRecorder.stop();
  recordBtn.disabled = false;
  stopBtn.disabled = true;
});
