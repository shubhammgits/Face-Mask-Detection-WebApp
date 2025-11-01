let currentMode = 'capture';
let uploadedFile = null;
let capturedImageData = null;
let mediaStream = null;
let videoElement = null;
let captureCanvas = null;
let captureContext = null;

const elements = {
    modeTabs: document.querySelectorAll('.mode-tab'),
    modeContents: document.querySelectorAll('.mode-content'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadPreview: document.getElementById('upload-preview'),
    previewImage: document.getElementById('preview-image'),
    analyzeBtn: document.getElementById('analyze-btn'),
    clearBtn: document.getElementById('clear-btn'),
    captureView: document.getElementById('capture-view'),
    capturePlaceholder: document.getElementById('capture-placeholder'),
    videoStream: document.getElementById('video-stream'),
    captureCanvas: document.getElementById('capture-canvas'),
    capturePreview: document.getElementById('capture-preview'),
    capturedImage: document.getElementById('captured-image'),
    startCaptureBtn: document.getElementById('start-capture'),
    captureBtn: document.getElementById('capture-btn'),
    stopCaptureBtn: document.getElementById('stop-capture'),
    analyzeCaptureBtn: document.getElementById('analyze-capture-btn'),
    retakeBtn: document.getElementById('retake-btn'),
    resultsPanel: document.getElementById('results-panel'),
    resultsContent: document.getElementById('results-content'),
    closeResultsBtn: document.getElementById('close-results'),
    loadingOverlay: document.getElementById('loading-overlay')
};

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeNavigation();
    initializeCapture();
    checkSystemStatus();
});

function initializeEventListeners() {
    elements.modeTabs.forEach(tab => {
        tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });
    
    // Upload events
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
        elements.analyzeBtn.addEventListener('click', analyzeUploadedImage);
    }
    
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', clearUpload);
    }
    
    // Capture events
    if (elements.startCaptureBtn) {
        elements.startCaptureBtn.addEventListener('click', startCapture);
    }
    
    if (elements.captureBtn) {
        elements.captureBtn.addEventListener('click', captureImage);
    }
    
    if (elements.stopCaptureBtn) {
        elements.stopCaptureBtn.addEventListener('click', stopCapture);
    }
    
    if (elements.analyzeCaptureBtn) {
        elements.analyzeCaptureBtn.addEventListener('click', analyzeCapturedImage);
    }
    
    if (elements.retakeBtn) {
        elements.retakeBtn.addEventListener('click', retakeImage);
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

function initializeCapture() {
    videoElement = elements.videoStream;
    captureCanvas = elements.captureCanvas;
    captureContext = captureCanvas.getContext('2d');
    
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
        stopCapture();
        clearUpload();
    } else if (mode === 'capture') {
        clearUpload();
        stopCapture();
    }
    
    closeResults();
}

// Upload functions
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

// Capture functions
async function startCapture() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported by this browser. Please use a modern browser with HTTPS.');
        }
        
        const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
        if (!isSecureContext) {
            throw new Error('Camera access requires HTTPS. Please use a secure connection.');
        }
        
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        };
        
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        videoElement.srcObject = mediaStream;
        
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            
            elements.videoStream.style.display = 'block';
            elements.capturePlaceholder.style.display = 'none';
            elements.startCaptureBtn.style.display = 'none';
            elements.captureBtn.style.display = 'inline-block';
            elements.stopCaptureBtn.style.display = 'inline-block';
            
            const helpElement = document.getElementById('capture-help');
            if (helpElement) {
                helpElement.style.display = 'none';
            }
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
            helpText = 'Your camera doesn\'t support the required video settings.';
        } else if (!navigator.mediaDevices) {
            errorMessage = 'MediaDevices API not supported';
            helpText = 'Please use HTTPS or localhost. Camera access requires a secure connection.';
        } else {
            errorMessage = `Camera error: ${error.message}`;
            helpText = 'Please ensure you\'re using HTTPS and have granted camera permissions.';
        }
        
        showNotification(`${errorMessage}. ${helpText}`, 'error');
        stopCapture();
    }
}

function stopCapture() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    if (videoElement) {
        videoElement.srcObject = null;
        videoElement.style.display = 'none';
    }
    
    elements.capturePlaceholder.style.display = 'flex';
    elements.startCaptureBtn.style.display = 'inline-block';
    elements.captureBtn.style.display = 'none';
    elements.stopCaptureBtn.style.display = 'none';
    
    const helpElement = document.getElementById('capture-help');
    if (helpElement) {
        helpElement.style.display = 'block';
    }
    
    closeResults();
}

function captureImage() {
    if (!videoElement || videoElement.readyState < 2) {
        showNotification('Video stream not ready', 'error');
        return;
    }
    
    // Set canvas dimensions to match video
    captureCanvas.width = videoElement.videoWidth;
    captureCanvas.height = videoElement.videoHeight;
    
    // Draw current video frame to canvas
    captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Convert to data URL
    const dataURL = captureCanvas.toDataURL('image/jpeg', 0.9);
    
    // Display captured image
    elements.capturedImage.src = dataURL;
    elements.capturePreview.style.display = 'block';
    elements.captureView.style.display = 'none';
    
    // Store image data for analysis
    capturedImageData = dataURL;
}

function retakeImage() {
    elements.capturePreview.style.display = 'none';
    elements.captureView.style.display = 'block';
    capturedImageData = null;
    closeResults();
}

// Analysis functions
async function analyzeUploadedImage() {
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

async function analyzeCapturedImage() {
    if (!capturedImageData) {
        showNotification('No captured image to analyze', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        // Convert data URL to Blob
        const blob = await fetch(capturedImageData).then(res => res.blob());
        
        const formData = new FormData();
        formData.append('file', blob, 'captured_image.jpg');
        
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

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeResults();
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