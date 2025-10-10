let currentMode = 'upload';
let isStreaming = false;
let uploadedFile = null;
let mediaStream = null;
let videoElement = null;
let canvas = null;
let context = null;
let detectionInterval = null;

const elements = {
    modeTabs: document.querySelectorAll('.mode-tab'),
    modeContents: document.querySelectorAll('.mode-content'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadPreview: document.getElementById('upload-preview'),
    previewImage: document.getElementById('preview-image'),
    analyzeBtn: document.getElementById('analyze-btn'),
    clearBtn: document.getElementById('clear-btn'),
    cameraView: document.getElementById('camera-view'),
    cameraPlaceholder: document.getElementById('camera-placeholder'),
    videoStream: document.getElementById('video-stream'),
    startCameraBtn: document.getElementById('start-camera'),
    stopCameraBtn: document.getElementById('stop-camera'),
    cameraStatus: document.getElementById('camera-status'),
    resultsPanel: document.getElementById('results-panel'),
    resultsContent: document.getElementById('results-content'),
    closeResultsBtn: document.getElementById('close-results'),
    loadingOverlay: document.getElementById('loading-overlay')
};

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeNavigation();
    initializeCamera();
    checkSystemStatus();
});

function initializeEventListeners() {
    elements.modeTabs.forEach(tab => {
        tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });
    
    if (elements.uploadArea) {
        elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
        elements.uploadArea.addEventListener('dragover', handleDragOver);
        elements.uploadArea.addEventListener('dragleave', handleDragLeave);
        elements.uploadArea.addEventListener('drop', handleFileDrop);
    }
    
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', analyzeImage);
    }
    
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', clearUpload);
    }
    
    if (elements.startCameraBtn) {
        elements.startCameraBtn.addEventListener('click', startCamera);
    }
    
    if (elements.stopCameraBtn) {
        elements.stopCameraBtn.addEventListener('click', stopCamera);
    }
    
    if (elements.closeResultsBtn) {
        elements.closeResultsBtn.addEventListener('click', closeResults);
    }
}

function initializeNavigation() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

function initializeCamera() {
    videoElement = elements.videoStream;
    
    canvas = document.createElement('canvas');
    context = canvas.getContext('2d');
    
    if (videoElement) {
        videoElement.setAttribute('playsinline', 'true');
        videoElement.setAttribute('webkit-playsinline', 'true');
        videoElement.muted = true;
    }
}

async function checkSystemStatus() {
    try {
        const response = await fetch('/model_status');
        const status = await response.json();
        
        if (!status.model_loaded || !status.cascade_loaded) {
            showNotification('System initialization in progress...', 'warning');
        }
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

function switchMode(mode) {
    currentMode = mode;
    
    elements.modeTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    
    elements.modeContents.forEach(content => {
        content.classList.toggle('active', content.id === `${mode}-mode`);
    });
    
    if (mode === 'upload') {
        stopCamera();
        clearUpload();
    } else if (mode === 'camera') {
        clearUpload();
        stopCamera();
    }
    
    closeResults();
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'error');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image size must be less than 10MB', 'error');
        return;
    }
    
    uploadedFile = file;
    displayImagePreview(file);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadPreview.style.display = 'block';
        elements.uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    uploadedFile = null;
    elements.fileInput.value = '';
    elements.uploadPreview.style.display = 'none';
    elements.uploadArea.style.display = 'block';
    closeResults();
}

async function analyzeImage() {
    if (!uploadedFile) {
        showNotification('Please select an image first', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result.image, result.detections);
            showNotification('Analysis complete!', 'success');
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function startCamera() {
    try {
        updateCameraStatus('Starting camera...', 'loading');
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported by this browser. Please use a modern browser with HTTPS.');
        }
        
        const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
        if (!isSecureContext) {
            throw new Error('Camera access requires HTTPS. Please use a secure connection.');
        }
        
        const constraints = {
            video: {
                width: { ideal: 640, max: 1280 },
                height: { ideal: 480, max: 720 },
                facingMode: 'user'
            },
            audio: false
        };
        
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (constraintError) {
            const basicConstraints = {
                video: true,
                audio: false
            };
            
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia(basicConstraints);
            } catch (basicError) {
                throw basicError;
            }
        }
        
        videoElement.srcObject = mediaStream;
        
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            
            canvas.width = videoElement.videoWidth || 640;
            canvas.height = videoElement.videoHeight || 480;
            
            elements.videoStream.style.display = 'block';
            elements.cameraPlaceholder.style.display = 'none';
            elements.startCameraBtn.style.display = 'none';
            elements.stopCameraBtn.style.display = 'inline-block';
            
            elements.cameraView.classList.add('streaming');
            
            const helpElement = document.getElementById('camera-help');
            if (helpElement) {
                helpElement.style.display = 'none';
            }
            
            isStreaming = true;
            updateCameraStatus('Live streaming...', 'active');
            
            startRealTimeDetection();
        };
        
        videoElement.onerror = (error) => {
            throw new Error('Failed to start video stream');
        };
        
    } catch (error) {
        let errorMessage = 'Camera access failed';
        let helpText = '';
        
        if (error.name === 'NotAllowedError') {
            errorMessage = 'Camera permission denied';
            helpText = 'Please click the camera icon in your browser\'s address bar and allow camera access, then try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No camera found on this device';
            helpText = 'Please ensure your device has a camera and it\'s not being used by another application.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage = 'Camera not supported by this browser';
            helpText = 'Please try using Chrome, Firefox, Safari, or Edge with the latest version.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Camera is already in use';
            helpText = 'Please close other applications that might be using your camera and try again.';
        } else if (error.name === 'OverconstrainedError') {
            errorMessage = 'Camera constraints cannot be satisfied';
            helpText = 'Your camera doesn\'t support the required video settings. This should not happen with our fallback constraints.';
        } else if (!navigator.mediaDevices) {
            errorMessage = 'MediaDevices API not supported';
            helpText = 'Please use HTTPS or localhost. Camera access requires a secure connection.';
        } else {
            errorMessage = `Camera error: ${error.message}`;
            helpText = 'Please ensure you\'re using HTTPS and have granted camera permissions.';
        }
        
        updateCameraStatus('Camera failed to start', 'error');
        showNotification(`${errorMessage}. ${helpText}`, 'error');
        
        stopCamera();
    }
}

function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    if (videoElement) {
        videoElement.srcObject = null;
        videoElement.style.display = 'none';
    }
    
    if (elements.cameraView) {
        elements.cameraView.classList.remove('streaming');
        
        const existingBoxes = elements.cameraView.querySelectorAll('.detection-box');
        existingBoxes.forEach(box => box.remove());
    }
    
    elements.cameraPlaceholder.style.display = 'flex';
    elements.startCameraBtn.style.display = 'inline-block';
    elements.stopCameraBtn.style.display = 'none';
    
    const helpElement = document.getElementById('camera-help');
    if (helpElement) {
        helpElement.style.display = 'block';
    }
    
    isStreaming = false;
    updateCameraStatus('Ready', 'ready');
    
    closeResults();
}

function startRealTimeDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    
    detectionInterval = setInterval(async () => {
        if (isStreaming && videoElement && videoElement.readyState === 4) {
            await processVideoFrame();
        }
    }, 1500);
}

async function processVideoFrame() {
    try {
        const tempCanvas = document.createElement('canvas');
        const tempContext = tempCanvas.getContext('2d');
        
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        
        tempContext.imageSmoothingEnabled = true;
        tempContext.imageSmoothingQuality = 'high';
        tempContext.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        tempCanvas.toBlob(async (blob) => {
            if (blob) {
                const formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');
                
                try {
                    const response = await fetch('/process_frame', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success && result.detections) {
                        drawDetectionBoxes(result.detections);
                    }
                } catch (error) {
                    console.error('Frame processing error:', error);
                }
            }
        }, 'image/jpeg', 0.95);
        
    } catch (error) {
        console.error('Video frame processing error:', error);
    }
}

function drawDetectionBoxes(detections) {
    const videoRect = videoElement.getBoundingClientRect();
    const cameraRect = elements.cameraView.getBoundingClientRect();
    
    const existingBoxes = elements.cameraView.querySelectorAll('.detection-box');
    existingBoxes.forEach(box => box.remove());
    
    detections.forEach((detection) => {
        const box = document.createElement('div');
        box.className = 'detection-box';
        
        const scaleX = videoRect.width / canvas.width;
        const scaleY = videoRect.height / canvas.height;
        
        const x = detection.bbox[0] * scaleX;
        const y = detection.bbox[1] * scaleY;
        const width = detection.bbox[2] * scaleX;
        const height = detection.bbox[3] * scaleY;
        
        const borderColor = detection.label === 'Masked' ? '#00ff00' : '#ff0000';
        
        box.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: ${width}px;
            height: ${height}px;
            border: 3px solid ${borderColor};
            border-radius: 4px;
            pointer-events: none;
            z-index: 5;
        `;
        
        const label = document.createElement('div');
        label.className = 'detection-label-box';
        label.textContent = `${detection.label}: ${detection.confidence.toFixed(1)}%`;
        label.style.cssText = `
            position: absolute;
            top: -30px;
            left: 0;
            background: ${borderColor};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            white-space: nowrap;
        `;
        
        box.appendChild(label);
        elements.cameraView.appendChild(box);
    });
}

function updateCameraStatus(text, state = 'ready') {
    const statusElement = elements.cameraStatus;
    if (!statusElement) return;
    
    statusElement.className = `camera-status ${state}`;
    statusElement.querySelector('.status-text').textContent = text;
}

function displayResults(processedImage, detections) {
    let resultsHTML = '';
    
    if (detections && detections.length > 0) {
        resultsHTML = `
            <div class="result-item">
                <img src="${processedImage}" alt="Processed image" class="result-image">
                <div class="result-info">
                    <div class="detection-cards">
                        ${detections.map((detection, index) => `
                            <div class="detection-card ${detection.label.toLowerCase().replace(' ', '-')}">
                                <div>
                                    <div class="detection-label ${detection.label.toLowerCase().replace(' ', '-')}">
                                        ${detection.label}
                                    </div>
                                    <div class="detection-bbox">Face ${index + 1}</div>
                                </div>
                                <div class="confidence-score">${detection.confidence.toFixed(1)}%</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    } else {
        resultsHTML = `
            <div class="no-detections">
                <p>No faces detected in the image</p>
                <p class="text-muted">Make sure the image contains visible faces</p>
            </div>
        `;
    }
    
    elements.resultsContent.innerHTML = resultsHTML;
    elements.resultsPanel.style.display = 'block';
    
    setTimeout(() => {
        elements.resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function closeResults() {
    if (elements.resultsPanel) {
        elements.resultsPanel.style.display = 'none';
    }
}

function showLoading(show) {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            ${message}
        </div>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: var(--text-primary);
        backdrop-filter: var(--glass-backdrop);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        max-width: 300px;
    `;
    
    if (type === 'error') {
        notification.style.borderColor = 'var(--danger)';
        notification.style.background = 'rgba(255, 71, 87, 0.1)';
    } else if (type === 'success') {
        notification.style.borderColor = 'var(--success)';
        notification.style.background = 'rgba(0, 255, 136, 0.1)';
    } else if (type === 'warning') {
        notification.style.borderColor = 'var(--warning)';
        notification.style.background = 'rgba(255, 167, 38, 0.1)';
    }
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeResults();
    }
    
    if (e.key === ' ' && currentMode === 'camera' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        if (isStreaming) {
            stopCamera();
        } else {
            startCamera();
        }
    }
});

const animationStyles = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = animationStyles;
document.head.appendChild(styleSheet);