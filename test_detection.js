// Test script to verify detection box rendering
console.log("Testing detection box rendering...");

// Mock detection data
const mockDetections = [
    {
        bbox: [100, 100, 150, 150],
        label: "Masked",
        confidence: 95.5
    },
    {
        bbox: [300, 120, 140, 140],
        label: "No Mask",
        confidence: 87.2
    }
];

// Mock frame dimensions
const frameWidth = 640;
const frameHeight = 480;

// Function to test drawing detection boxes
function testDrawDetectionBoxes() {
    console.log("Testing drawDetectionBoxes function...");
    
    // Create a mock camera view
    const cameraView = document.createElement('div');
    cameraView.id = 'camera-view';
    cameraView.style.width = '640px';
    cameraView.style.height = '480px';
    cameraView.style.position = 'relative';
    cameraView.style.backgroundColor = '#333';
    
    // Create a mock video element
    const videoElement = document.createElement('video');
    videoElement.id = 'video-stream';
    videoElement.style.width = '100%';
    videoElement.style.height = '100%';
    
    cameraView.appendChild(videoElement);
    document.body.appendChild(cameraView);
    
    // Mock the elements object
    const elements = {
        cameraView: cameraView,
        videoStream: videoElement
    };
    
    // Mock the clearDetectionBoxes function
    function clearDetectionBoxes() {
        const existingBoxes = cameraView.querySelectorAll('.detection-box');
        existingBoxes.forEach(box => box.remove());
    }
    
    // Mock the drawDetectionBoxes function
    function drawDetectionBoxes(detections, frameWidth, frameHeight) {
        console.log("Drawing detection boxes:", detections);
        console.log("Frame dimensions:", frameWidth, "x", frameHeight);
        
        // Clear existing boxes
        clearDetectionBoxes();
        
        // If no detections, exit early
        if (!detections || detections.length === 0) {
            console.log("No detections to draw");
            return;
        }
        
        // Get the camera view and video elements
        const cameraView = elements.cameraView;
        const videoElement = elements.videoStream;
        
        // Get dimensions
        const videoRect = videoElement.getBoundingClientRect();
        const cameraRect = cameraView.getBoundingClientRect();
        
        console.log("Video rect:", videoRect);
        console.log("Camera rect:", cameraRect);
        
        // Calculate scaling factors
        const scaleX = videoRect.width / frameWidth;
        const scaleY = videoRect.height / frameHeight;
        
        console.log("Scale factors:", scaleX, scaleY);
        
        detections.forEach((detection) => {
            const box = document.createElement('div');
            box.className = 'detection-box';
            
            // Scale the bounding box coordinates
            const x = detection.bbox[0] * scaleX;
            const y = detection.bbox[1] * scaleY;
            const width = detection.bbox[2] * scaleX;
            const height = detection.bbox[3] * scaleY;
            
            console.log("Detection box:", detection, "Scaled:", x, y, width, height);
            
            // Set box styles
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
                z-index: 1000;
                box-sizing: border-box;
            `;
            
            // Create label
            const label = document.createElement('div');
            label.className = 'detection-label-box';
            label.textContent = `${detection.label}: ${detection.confidence.toFixed(1)}%`;
            label.style.cssText = `
                position: absolute;
                top: -25px;
                left: 0;
                background: ${borderColor};
                color: white;
                padding: 3px 6px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                white-space: nowrap;
                font-family: 'Inter', sans-serif;
                box-sizing: border-box;
            `;
            
            box.appendChild(label);
            cameraView.appendChild(box);
        });
    }
    
    // Test the function
    drawDetectionBoxes(mockDetections, frameWidth, frameHeight);
    
    // Check if boxes were created
    const boxes = document.querySelectorAll('.detection-box');
    console.log(`Created ${boxes.length} detection boxes`);
    
    if (boxes.length === mockDetections.length) {
        console.log("✓ Detection box rendering test passed!");
    } else {
        console.log("✗ Detection box rendering test failed!");
    }
    
    // Clean up
    document.body.removeChild(cameraView);
}

// Run the test
testDrawDetectionBoxes();