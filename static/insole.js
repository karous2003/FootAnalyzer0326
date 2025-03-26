let stream = null;
let selectedPoints = [];
let alpha = 0, beta = 0, gamma = 0;
let boxPosition = { x: 0, y: 0, width: 200, height: 283 };
let isDragging = false;
let isResizing = false;
let dragStart = { x: 0, y: 0 };
let currentImageDataURL = null;

window.addEventListener('deviceorientation', function (event) {
    alpha = event.alpha ? event.alpha.toFixed(2) : 0;
    beta = event.beta ? event.beta.toFixed(2) : 0;
    gamma = event.gamma ? event.gamma.toFixed(2) : 0;
});

async function openCamera() {
    try {
        if (navigator.permissions) {
            const permission = await navigator.permissions.query({ name: 'camera' });
            if (permission.state === 'denied') {
                alert('相機權限已被拒絕，請到瀏覽器設定中開啟。');
                return;
            }
        }
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        const videoElement = document.getElementById('camera-view');
        videoElement.srcObject = stream;
        document.getElementById('camera-container').style.display = 'block';
        videoElement.onloadedmetadata = function () {
            videoElement.play();
            drawOverlay();
        };
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('無法存取相機，請確認已授予相機權限。');
    }
}

function drawOverlay() {
    const container = document.getElementById('camera-container');
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    const captureButton = document.getElementById('captureButton');

    function resizeCanvas() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    }

    function renderFrame() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 繪製半透明黑色背景
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 計算 A4 比例的框線
        const a4Aspect = 210 / 297;
        let rectWidth = canvas.width * 0.9;
        let rectHeight = rectWidth / a4Aspect;
        if (rectHeight > canvas.height * 0.9) {
            rectHeight = canvas.height * 0.9;
            rectWidth = rectHeight * a4Aspect;
        }
        const rectX = (canvas.width - rectWidth) / 2;
        const rectY = (canvas.height - rectHeight) / 2;

        // 挖空矩形區域
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'rgba(0, 0, 0, 1)';
        ctx.fillRect(rectX, rectY, rectWidth, rectHeight);
        ctx.globalCompositeOperation = 'source-over';

        // 繪製藍色虛線框
        ctx.strokeStyle = 'rgba(0, 0, 255, 0.5)';
        ctx.lineWidth = 4;
        ctx.setLineDash([10, 10]);
        ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);

        // 繪製對角線
        ctx.beginPath(); 
        ctx.moveTo(rectX, rectY);
        ctx.lineTo(rectX + rectWidth, rectY + rectHeight); //左上到右下
        ctx.stroke(); 
        ctx.beginPath();
        ctx.moveTo(rectX + rectWidth, rectY); 
        ctx.lineTo(rectX, rectY + rectHeight);  //右上到左下
        ctx.stroke(); 
        ctx.setLineDash([]); 

        // 儲存框線資訊
        window.blueRect = {
            rectX: rectX,
            rectY: rectY,
            rectWidth: rectWidth,
            rectHeight: rectHeight,
            canvasWidth: canvas.width,
            canvasHeight: canvas.height
        };

        // 添加提示文字在畫面頂部
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('請將螢幕朝上保持水平讓', canvas.width / 2, 20);
        ctx.fillText('X 介於 -2° ~ 2°', canvas.width / 2, 40);
        ctx.fillText('Y 介於 -2° ~ 2°', canvas.width / 2, 60);

        // 添加「請將鞋墊放置於對角線上」
        ctx.fillStyle = 'gray'; 
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        const centerX = rectX + rectWidth / 2;
        const centerY = rectY + rectHeight / 2;
        ctx.fillText('請將鞋墊放置於對角線上', centerX, centerY);

        // 檢查角度是否在 ±2 度範圍內
        const betaAbs = Math.abs(beta);
        const gammaAbs = Math.abs(gamma);
        const threshold = 2;
        const canCapture = betaAbs <= threshold && gammaAbs <= threshold;

        // 更新按鈕透明度
        if (captureButton) {
            captureButton.style.opacity = canCapture ? '1' : '0';
        }

        // 顯示角度狀態訊息
        const message = canCapture ? "已調整至合適角度" : "請調整相機角度";
        ctx.fillStyle = canCapture ? 'green' : 'red';
        ctx.font = '15px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, 80); // 調整位置以避免與提示文字重疊

        // 顯示角度數值
        ctx.fillStyle = 'white';
        ctx.font = '15px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`X: ${beta}`, 10, 25);
        ctx.fillText(`Y: ${gamma}`, 10, 45);

        

        requestAnimationFrame(renderFrame);
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    renderFrame();
}

function closeModal() {
    document.getElementById('point-selection-modal').style.display = 'none';
    selectedPoints = [];
    document.getElementById('confirm-points').disabled = true;
}

document.getElementById('close-modal')?.addEventListener('click', function () {
    const modal = document.getElementById('box-selection-modal');
    const box = document.getElementById('resizable-box');
    const resizeHandle = document.getElementById('resize-handle');

    const handlers = JSON.parse(box.dataset.eventHandlers || '{}');
    if (handlers) {
        box.removeEventListener('mousedown', handlers.startDragging);
        resizeHandle.removeEventListener('mousedown', handlers.startResizing);
        document.removeEventListener('mousemove', handlers.move);
        document.removeEventListener('mouseup', handlers.stop);

        box.removeEventListener('touchstart', handlers.startDragging);
        resizeHandle.removeEventListener('touchstart', handlers.startResizing);
        document.removeEventListener('touchmove', handlers.move);
        document.removeEventListener('touchend', handlers.stop);
    }

    modal.style.display = 'none';
    selectedPoints = [];
    document.body.style.overflow = 'auto';
});

document.addEventListener('click', function (e) {
    const modal = document.getElementById('point-selection-modal');
    const modalContent = modal.querySelector('.modal-content');
    if (modal.style.display === 'flex' && !modalContent?.contains(e.target)) {
        closeModal();
    }
});

function showModal() {
    document.getElementById('point-selection-modal').style.display = 'flex';
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    document.getElementById('camera-container').style.display = 'none';
}

function showLoading() {
    document.getElementById('loading-indicator').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-indicator').style.display = 'none';
}

function checkResults(data) {
    const issues = [];

    if (data.result_data.length_cm === 0) {
        issues.push("鞋墊關鍵點");
    }
    if (data.result_data.forefoot_width_cm === 0) {
        issues.push("前掌關鍵點");
    }
    if (data.result_data.midfoot_width_cm === 0) {
        issues.push("中足關鍵點");
    }
    if (data.result_data.heel_width_cm === 0) {
        issues.push("後跟關鍵點");
    }

    if (issues.length > 0) {
        let message = "以下關鍵點未抓到：\n";
        message += issues.join("、") + "\n";
        message += "請重新上傳或拍攝圖片。";
        alert(message);
        return false;
    }
    return true;
}

function updateResults(data) {
    if (!checkResults(data)) {
        return;}
        
    document.getElementById('length').textContent = data.result_data.insole_length_cm;
    document.getElementById('forefoot-width').textContent = data.result_data.forefoot_width_cm;
    document.getElementById('midfoot-width').textContent = data.result_data.midfoot_width_cm;
    document.getElementById('heel-width').textContent = data.result_data.heel_width_cm;
    document.getElementById('processing-time').textContent = data.result_data.processing_time;
    document.getElementById('result-image').src = `${data.image_url}`;
    document.getElementById('result-section').style.display = 'block';
    document.getElementById('save-result').addEventListener('click', function () {
        const resultImage = document.getElementById('result-image');
        const imageUrl = resultImage.src;
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = `insole_result_${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

function dataURLtoBlob(dataURL) {
    var arr = dataURL.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]),
        n = bstr.length,
        u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}

async function capturePhoto() {
    const video = document.getElementById('camera-view');
    const canvas = document.getElementById('canvas');
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg', 1);
        console.log("Photo captured, dataURL length:", dataURL.length);
        currentImageDataURL = dataURL;

        if (window.blueRect) {
            const blueRect = window.blueRect;
            const scaleX = video.videoWidth / blueRect.canvasWidth;
            const scaleY = video.videoHeight / blueRect.canvasHeight;
            selectedPoints = [
                { x: blueRect.rectX * scaleX, y: blueRect.rectY * scaleY },
                { x: (blueRect.rectX + blueRect.rectWidth) * scaleX, y: blueRect.rectY * scaleY },
                { x: (blueRect.rectX + blueRect.rectWidth) * scaleX, y: (blueRect.rectY + blueRect.rectHeight) * scaleY },
                { x: blueRect.rectX * scaleX, y: (blueRect.rectY + blueRect.rectHeight) * scaleY }
            ];
        } else {
            alert('無法取得框線資訊，請重新拍攝！');
            return;
        }
        closeCamera();
        uploadImageWithPoints();
    } else {
        alert('請等待相機完全啟動');
    }
}

document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
    console.log('表單已提交，正在處理檔案...');
    const fileInput = this.querySelector('input[type="file"]');
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const imageSrc = e.target.result;
        currentImageDataURL = imageSrc;
        const img = new Image();
        img.onload = function () {
            const uploadedImage = document.getElementById('uploaded-image');
            uploadedImage.src = imageSrc;
            showBoxModal(img);
        };
        img.src = imageSrc;
    };
    reader.readAsDataURL(file);
});

document.getElementById('selection-canvas')?.addEventListener('click', function (e) {
    const canvas = this;
    const rect = canvas.getBoundingClientRect();
    const scale = window.imageScale;

    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    if (selectedPoints.length < 4) {
        selectedPoints.push({
            x: x / scale,
            y: y / scale
        });

        const ctx = canvas.getContext('2d');
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = "16px Arial";
        ctx.fillText(selectedPoints.length, x - 5, y - 5);

        if (selectedPoints.length === 4) {
            document.getElementById('confirm-points').disabled = false;
        }
    }
});

function showBoxModal(img) {
    const modal = document.getElementById('box-selection-modal');
    const uploadedImage = document.getElementById('uploaded-image');

    const maxWidth = window.innerWidth * 0.4;
    const maxHeight = window.innerHeight * 0.6;
    const scaleFactor = 0.5;
    let displayWidth = img.width * scaleFactor;

    if (displayWidth > maxWidth) {
        displayWidth = maxWidth;
    }
    const aspectRatio = img.height / img.width;
    if (displayWidth * aspectRatio > maxHeight) {
        displayWidth = maxHeight / aspectRatio;
    }

    uploadedImage.style.width = `${displayWidth}px`;
    uploadedImage.style.height = `${displayWidth * aspectRatio}px`;
    uploadedImage.style.margin = '0'; 
    uploadedImage.style.display = 'block';
    uploadedImage.style.position = 'relative'; 


    const modalContent = modal.querySelector('.modal-content');
    if (modalContent) {
        modalContent.style.textAlign = 'left'; 
    }

    document.body.style.overflow = 'hidden';
    modal.style.display = 'flex';
    initializeBox(displayWidth);
}

function initializeBox(displayWidth) {
    const box = document.getElementById('resizable-box');
    const img = document.getElementById('uploaded-image');
    const displayHeight = img.clientHeight; 

    // 根據 A4 比例 (210mm x 297mm) 計算框線的初始尺寸
    const a4Aspect = 210 / 297;
    let initialWidth = displayWidth * 0.9; 
    let initialHeight = initialWidth / a4Aspect;

    // 確保框線高度不超過圖片高度
    if (initialHeight > displayHeight * 0.9) {
        initialHeight = displayHeight * 0.9;
        initialWidth = initialHeight * a4Aspect;
    }

    // 將框線置中於圖片
    const initialX = (displayWidth - initialWidth) / 2;
    const initialY = (displayHeight - initialHeight) / 2;

    box.style.top = `${initialY}px`;
    box.style.left = `${initialX}px`;
    box.style.width = `${initialWidth}px`;
    box.style.height = `${initialHeight}px`;
    boxPosition = { x: initialX, y: initialY, width: initialWidth, height: initialHeight };

    let resizeHandle = document.getElementById('resize-handle');
    if (!resizeHandle) {
        resizeHandle = document.createElement('div');
        resizeHandle.id = 'resize-handle';
        resizeHandle.style.cssText = `
            position: absolute;
            width: 10px;
            height: 10px;
            background: blue;
            bottom: -5px;
            right: -5px;
            cursor: se-resize;
        `;
        box.appendChild(resizeHandle);
    }

    const startDraggingHandler = (e) => startDragging(e);
    const startResizingHandler = (e) => startResizing(e);
    const moveHandlerBound = (e) => moveHandler(e);
    const stopHandlerBound = (e) => stopHandler(e);

    box.addEventListener('mousedown', startDraggingHandler);
    resizeHandle.addEventListener('mousedown', startResizingHandler);
    document.addEventListener('mousemove', moveHandlerBound);
    document.addEventListener('mouseup', stopHandlerBound);

    box.addEventListener('touchstart', startDraggingHandler);
    resizeHandle.addEventListener('touchstart', startResizingHandler);
    document.addEventListener('touchmove', moveHandlerBound, { passive: false });
    document.addEventListener('touchend', stopHandlerBound);

    box.dataset.eventHandlers = JSON.stringify({
        startDragging: startDraggingHandler,
        startResizing: startResizingHandler,
        move: moveHandlerBound,
        stop: stopHandlerBound
    });
}

function startDragging(e) {
    if (e.target.id !== 'resize-handle') {
        isDragging = true;
        if (e.type === 'touchstart') {
            dragStart.x = e.touches[0].clientX - boxPosition.x;
            dragStart.y = e.touches[0].clientY - boxPosition.y;
            e.preventDefault();
        } else {
            dragStart.x = e.clientX - boxPosition.x;
            dragStart.y = e.clientY - boxPosition.y;
            e.preventDefault();
        }
    }
}

function startResizing(e) {
    isResizing = true;
    if (e.type === 'touchstart') {
        dragStart.x = e.touches[0].clientX;
        dragStart.y = e.touches[0].clientY;
        e.preventDefault();
    } else {
        dragStart.x = e.clientX;
        dragStart.y = e.clientY;
        e.preventDefault();
    }
}

function moveHandler(e) {
    const box = document.getElementById('resizable-box');
    const img = document.getElementById('uploaded-image');
    const modal = document.getElementById('box-selection-modal');
    let clientX, clientY;

    if (modal.style.display !== 'flex') return;

    if (e.type === 'touchmove') {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
        e.preventDefault();
    } else if (e.type === 'mousemove') {
        clientX = e.clientX;
        clientY = e.clientY;
    } else {
        return;
    }

    if (isDragging) {
        boxPosition.x = clientX - dragStart.x;
        boxPosition.y = clientY - dragStart.y;
        box.style.left = `${boxPosition.x}px`;
        box.style.top = `${boxPosition.y}px`;
    } else if (isResizing) {
        const dx = clientX - dragStart.x;
        const newWidth = Math.max(50, boxPosition.width + dx);
        const newHeight = newWidth * (297 / 210);
        const maxWidth = img.width;
        if (newWidth <= maxWidth) {
            boxPosition.width = newWidth;
            boxPosition.height = newHeight;
            box.style.width = `${newWidth}px`;
            box.style.height = `${newHeight}px`;
        }
        dragStart.x = clientX;
        dragStart.y = clientY;
    }
}

function stopHandler() {
    isDragging = false;
    isResizing = false;
}

document.getElementById('confirm-box')?.addEventListener('click', function () {
    const modal = document.getElementById('box-selection-modal');
    const box = document.getElementById('resizable-box');
    const img = document.getElementById('uploaded-image');
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width; 
    const scaleY = img.naturalHeight / rect.height; 

    // 計算框線相對於圖像的偏移
    const offsetX = rect.left; 
    const offsetY = rect.top; 

    // 將框線坐標從顯示坐標轉換為原始圖像坐標
    const boxX = (boxPosition.x + offsetX - rect.left) * scaleX;
    const boxY = (boxPosition.y + offsetY - rect.top) * scaleY;
    const boxWidth = boxPosition.width * scaleX;
    const boxHeight = boxPosition.height * scaleY;

    const points = [
        { x: boxX, y: boxY }, // 左上角
        { x: boxX + boxWidth, y: boxY }, // 右上角
        { x: boxX + boxWidth, y: boxY + boxHeight }, // 右下角
        { x: boxX, y: boxY + boxHeight } // 左下角
    ];

    // 移除事件監聽器並關閉模態框
    const handlers = JSON.parse(box.dataset.eventHandlers || '{}');
    if (handlers) {
        box.removeEventListener('mousedown', handlers.startDragging);
        document.getElementById('resize-handle').removeEventListener('mousedown', handlers.startResizing);
        document.removeEventListener('mousemove', handlers.move);
        document.removeEventListener('mouseup', handlers.stop);
        box.removeEventListener('touchstart', handlers.startDragging);
        document.getElementById('resize-handle').removeEventListener('touchstart', handlers.startResizing);
        document.removeEventListener('touchmove', handlers.move);
        document.removeEventListener('touchend', handlers.stop);
    }

    selectedPoints = points;
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
    uploadImageWithPoints();
});

document.getElementById('reset-box')?.addEventListener('click', function () {
    const img = document.getElementById('uploaded-image');
    initializeBox(img.width);
});

function uploadImageWithPoints() {
    showLoading();
    const formData = new FormData();

    if (currentImageDataURL) {
        const blob = dataURLtoBlob(currentImageDataURL);
        formData.append('file', blob, 'image.jpg');
    } else {
        const fileInput = document.querySelector('#uploadForm input[type="file"]');
        if (fileInput && fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        } else {
            alert('沒有可用的圖片文件');
            hideLoading();
            return;
        }
    }

    formData.append('points', JSON.stringify(selectedPoints));

    fetch('/insole', {
        method: 'POST',
        body: formData,
        headers: { 'X-Requested-With': 'XMLHttpRequest' }
    })
        .then(response => {
            console.log('伺服器回應狀態碼：', response.status);
            if (!response.ok) {
                throw new Error(`伺服器回應錯誤，狀態碼：${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('收到後端資料：', data);
            hideLoading();
            if (data.success) {
                updateResults(data);
            } else {
                alert(data.error || '處理圖片時發生錯誤');
            }
        })
        .catch(err => {
            hideLoading();
            console.error('上傳錯誤:', err);
            alert('上傳圖片時發生錯誤：' + err.message);
        });
}

document.getElementById('confirm-points')?.addEventListener('click', function () {
    document.getElementById('point-selection-modal').style.display = 'none';
    uploadImageWithPoints();
});

document.getElementById('reset-points')?.addEventListener('click', function () {
    if (!currentImageDataURL) {
        alert('請先上傳或拍攝圖片');
        return;
    }
    selectedPoints = [];
    const canvas = document.getElementById('selection-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        canvas.width = img.width * imageScale;
        canvas.height = img.height * imageScale;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = currentImageDataURL;
    document.getElementById('confirm-points').disabled = true;
    document.getElementById('point-selection-modal').style.display = 'flex';
});

window.onload = function () {
    const imageSrc = "{{ image_url | default('/static/default_image.jpg') }}";
    initializeCanvas(imageSrc);
};

function initializeCanvas(imageSrc) {
    console.log('初始化畫布，圖片來源：', imageSrc);
}