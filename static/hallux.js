let stream = null;
let previewData = null;
let fileName = '';
let isProcessing = false;
let pauseIoU = false;
let iouLeft = 0, iouRight = 0;
let txtiouLeft = "0", txtiouRight = "0";
let leftFootBox, rightFootBox;
let alpha = 0, beta = 0, gamma = 0;
let uploadSource = null;
window.hasGyroListener = false;

function updateGyroData(event) {
    alpha = event.alpha.toFixed(2);
    beta = event.beta.toFixed(2);
    gamma = event.gamma.toFixed(2);
}

async function requestGyroPermission() {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
            const permission = await DeviceMotionEvent.requestPermission();
            if (permission === 'granted') {
                window.addEventListener('deviceorientation', updateGyroData);
                window.hasGyroListener = true;
            } else {
                alert('未授予陀螺儀訪問權限，請允許後再試');
            }
        } catch (error) {
            console.error('請求陀螺儀許可時出錯:', error);
            alert('無法請求陀螺儀訪問權限');
        }
    } else {
        window.addEventListener('deviceorientation', updateGyroData);
        window.hasGyroListener = true;
    }
}

async function openCamera() {
    if (stream) return;
    iouLeft = 0;
    iouRight = 0;
    txtiouLeft = "0";
    txtiouRight = "0";
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        const videoElement = document.getElementById('camera-view');
        videoElement.srcObject = stream;
        document.getElementById('camera-container').style.display = 'block';
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            drawOverlay();
            requestGyroPermission();
            pauseIoU = false;
            updateIoU();
        };
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("無法存取相機，請確認已授權相機權限");
    }
    //pauseIoU = false;
    //updateIoU();
}

function drawOverlay() {
    const video = document.getElementById('camera-view');
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    const overlayImg1 = new Image();
    overlayImg1.src = '/static/asset/leftfoot_outline.png';
    const overlayImg2 = new Image();
    overlayImg2.src = '/static/asset/rightfoot_outline.png';

    function resizeCanvas() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    function updateCaptureButtonVisibility() {
        const captureButton = document.getElementById("captureButton");
        if (iouLeft > 0.5 && iouRight > 0.5) {
            captureButton.style.visibility = "visible";
            captureButton.style.opacity = "1";
        } else {
            captureButton.style.opacity = "0";
            setTimeout(() => { captureButton.style.visibility = "hidden"; }, 300);
        }
    }

    function renderFrame() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = 'white';
        ctx.font = '45px Arial'; 
        ctx.textAlign = 'center';
        ctx.fillText('請將螢幕朝上保持水平讓', canvas.width / 2, 40);  
        ctx.fillText('X 介於 -20° ~ 5°', canvas.width / 2, 80);        
        ctx.fillText('Y 介於 -5° ~ 5°', canvas.width / 2, 120);     
    
        if (overlayImg1.complete) {
            const scale1 = Math.max(canvas.width / overlayImg1.width, canvas.height / overlayImg1.height);
            const drawWidth1 = overlayImg1.width * scale1;
            const drawHeight1 = overlayImg1.height * scale1;
            const offsetX1 = (canvas.width - drawWidth1) / 2;
            const offsetY1 = (canvas.height - drawHeight1) / 2;
            ctx.drawImage(overlayImg1, offsetX1, offsetY1, drawWidth1, drawHeight1);
            leftFootBox = [
                drawWidth1 / 10 - 70,
                drawHeight1 / 3 + 50,
                (drawWidth1 / 10) * 4 - 100,
                (drawHeight1 / 3) * 2.5 + 50
            ];
        }
        if (overlayImg2.complete) {
            const scale2 = Math.max(canvas.width / overlayImg2.width, canvas.height / overlayImg2.height);
            const drawWidth2 = overlayImg2.width * scale2;
            const drawHeight2 = overlayImg2.height * scale2;
            const offsetX2 = (canvas.width - drawWidth2) / 2;
            const offsetY2 = (canvas.height - drawHeight2) / 2;
            ctx.drawImage(overlayImg2, offsetX2 - 50, offsetY2, drawWidth2, drawHeight2);
            rightFootBox = [
                drawWidth2 / 2 - 100,
                drawHeight2 / 3 + 50,
                (drawWidth2 / 10) * 7 + 20,
                (drawHeight2 / 3) * 2.5 + 50
            ];
        }
    
        // 顯示 IoU 值
        ctx.fillStyle = 'blue';
        ctx.font = '60px Arial';
        ctx.textAlign = 'left'; // 恢復左對齊
        ctx.fillText(`IoU L: ${txtiouLeft}`, 10, 400);
        ctx.fillText(`IoU R: ${txtiouRight}`, 10, 450);
    
        // 顯示角度數值
        ctx.fillStyle = 'red';
        ctx.font = '60px Arial';
        ctx.fillText(`X: ${beta} 度`, 10, 200);
        ctx.fillText(`Y: ${gamma} 度`, 10, 300);
        updateCaptureButtonVisibility();
        requestAnimationFrame(renderFrame);
    }
    
    video.addEventListener('loadeddata', () => { resizeCanvas(); renderFrame(); updateIoU();});
    resizeCanvas();
}

function updateIoU() {
    if (pauseIoU || isProcessing) return;
    isProcessing = true;
    const video = document.getElementById('camera-view');
    const canvasCapture = document.createElement('canvas');
    canvasCapture.width = video.videoWidth;
    canvasCapture.height = video.videoHeight;
    const ctxCapture = canvasCapture.getContext('2d');
    ctxCapture.drawImage(video, 0, 0, canvasCapture.width, canvasCapture.height);
    canvasCapture.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'camera-capture.jpg');
        formData.append('left_bbox', JSON.stringify(leftFootBox || [0, 0, canvasCapture.width / 2, canvasCapture.height]));
        formData.append('right_bbox', JSON.stringify(rightFootBox || [canvasCapture.width / 2, 0, canvasCapture.width, canvasCapture.height]));
        try {
            const response = await fetch('/iou', {
                method: 'POST',
                headers: { 'X-Requested-With': 'XMLHttpRequest' },
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                iouLeft = parseFloat(data.iou_left);
                iouRight = parseFloat(data.iou_right);
                txtiouLeft = iouLeft.toFixed(4);
                txtiouRight = iouRight.toFixed(4);
            } else {
                iouLeft = 0;
                iouRight = 0;
                txtiouLeft = "0";
                txtiouRight = "0";
            }
        } catch (error) {
            console.error('IoU 請求錯誤:', error);
            iouLeft = 0;
            iouRight = 0;
            txtiouLeft = "0";
            txtiouRight = "0";
        }
        isProcessing = false;
        if (!pauseIoU) setTimeout(updateIoU, 200);
    }, 'image/jpeg', 0.95);
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    document.getElementById('camera-container').style.display = 'none';
    if (window.hasGyroListener) {
        window.removeEventListener('deviceorientation', updateGyroData);
        window.hasGyroListener = false;
    }
    pauseIoU = true;
}

function showLoading() {
    document.getElementById('loading-indicator').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-indicator').style.display = 'none';
}

async function capturePhoto() {
    pauseIoU = true;
    uploadSource = 'camera'; 
    const video = document.getElementById('camera-view');
    const canvas = document.getElementById('canvas');
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'camera-capture.jpg');
            formData.append('stage', 'removebg'); // 標記為去背階段
            try {
                showLoading();
                const response = await fetch('/hallux', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    previewData = data;
                    showRemoveBgPreview(data); // 顯示去背預覽
                    fileName = data.image_url;
                } else {
                    alert(data.error || '處理圖片時發生錯誤');
                }
            } catch (error) {
                console.error('Error:', error);
                alert(error.message || '上傳圖片時發生錯誤，請檢查網路連線或伺服器設定');
            } finally {
                hideLoading();
            }
        }, 'image/jpeg', 0.95);
    }
}

function showRemoveBgPreview(data) {
    document.getElementById('back-of-feet').src = data.image_url;
    document.getElementById('preview-container').style.display = 'block';
    document.getElementById('confirm-photo-btn').style.display = 'block';
    document.getElementById('retake-photo-btn').style.display = 'block';
}

async function confirmPhoto() {
    //deleteImage(fileName, "hallux");
    document.getElementById('preview-container').style.display = 'none';
    closeCamera();
    const formData = new FormData();
    formData.append('stage', 'analyze'); // 標記為分析階段
    formData.append('bg_removed_filename', previewData.image_url.split('/').pop()); // 取得文件名
    try {
        showLoading();
        const response = await fetch('/hallux', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            previewData = data;
            updateResults(data); // 顯示最終結果
        } else {
            alert(data.error || '分析圖片時發生錯誤');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('分析圖片時發生錯誤，請檢查網路連線或伺服器設定');
    } finally {
        hideLoading();
    }
}

function retakePhoto() {
    if (fileName) {
        deleteImage(fileName, "hallux"); // 刪除臨時檔案
    }
    document.getElementById('preview-container').style.display = 'none';

    if (uploadSource === 'camera') {
        stream = null;
        openCamera();
    } else if (uploadSource === 'album') {
        const fileInput = document.getElementById('fileInput');
        fileInput.value = ''; 
        uploadSource = null;
        fileInput.click();
    }
}

function deleteImage(filename, filetype) {
    fetch(`/delete-image?filename=${filename}&type=${filetype}`, {
        method: 'DELETE'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log("檔案已刪除");
            } else {
                console.log("刪除失敗:", data.error);
            }
        })
        .catch(error => console.error("錯誤:", error));
}

function updateResults(data) {
    const leftAngle = data.result.left_hva_angle !== null ? data.result.left_hva_angle : '-';
    const rightAngle = data.result.right_hva_angle !== null ? data.result.right_hva_angle : '-';
    const leftSeverity = data.result.left_severity || '-';
    const rightSeverity = data.result.right_severity || '-';
    const leftColor = data.result.left_color || 'gray';
    const rightColor = data.result.right_color || 'gray';

    if ((leftAngle !== '-' && leftAngle > 60) || (rightAngle !== '-' && rightAngle > 60)) {
        let message = "偵測到的 HVA 角度過大（左腳: " + leftAngle + "°, 右腳: " + rightAngle + "°），可能模型偵測失敗。\n";
        message += "建議您：\n";
        message += "1. 重新拍攝照片，確保腳部清晰可見。\n";
        message += "2. 注意拍攝時的光源，避免過暗或過曝。\n";
        message += "3. 確保背景乾淨，避免雜物干擾偵測。\n";
        alert(message);
        return; // 提前結束，不顯示結果
    }

    document.getElementById('left-hallux-angle').textContent = leftAngle;
    document.getElementById('right-hallux-angle').textContent = rightAngle;
    document.getElementById('left-severity').textContent = leftSeverity;
    document.getElementById('right-severity').textContent = rightSeverity;
    document.getElementById('left-hallux-color').style.color = leftColor;
    document.getElementById('right-hallux-color').style.color = rightColor;

    const severityRanges = [
        { max: 15, index: 0 },
        { min: 15, max: 20, index: 1 },
        { min: 20, max: 35, index: 2 },
        { min: 35, index: 3 }
    ];

    document.querySelectorAll('.results-table td').forEach(cell => {
        cell.classList.remove('green', 'yellow', 'orange', 'red', 'gray');
    });

    if (leftAngle !== '-') {
        const leftRange = severityRanges.find(range => {
            if (range.max === undefined) return leftAngle > range.min;
            if (range.min === undefined) return leftAngle < range.max;
            return leftAngle >= range.min && leftAngle <= range.max;
        });
        if (leftRange) {
            const leftCell = document.getElementById(`severity-left-${leftRange.index}`);
            leftCell.classList.add(leftColor);
        }
    }
    if (rightAngle !== '-') {
        const rightRange = severityRanges.find(range => {
            if (range.max === undefined) return rightAngle > range.min;
            if (range.min === undefined) return rightAngle < range.max;
            return rightAngle >= range.min && rightAngle <= range.max;
        });
        if (rightRange) {
            const rightCell = document.getElementById(`severity-right-${rightRange.index}`);
            rightCell.classList.add(rightColor);
        }
    }
    document.getElementById('result-image').src = data.image_url;
    document.getElementById('result-section').style.display = 'block';
}

document.getElementById('save-result').addEventListener('click', function () {
    const resultImage = document.getElementById('result-image');
    const imageUrl = resultImage.src;
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `hallux_result_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    uploadSource = 'album'; 
    const fileInput = this.querySelector('input[type="file"]');
    const file = fileInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('stage', 'removebg');

    showLoading();
    try {
        const response = await fetch('/hallux', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        hideLoading();
        if (data.image_url) {
            previewData = data;
            showRemoveBgPreview(data); // 顯示去背預覽
        } else {
            alert(data.error || '處理圖片時發生錯誤');
        }
    } catch (err) {
        hideLoading();
        console.error('Error:', err);
        alert('上傳圖片時發生錯誤');
    }
});