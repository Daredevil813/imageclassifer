# TJSF Model Image Tester

A web application for testing images using the TJSF TensorFlow.js model.

## Features

- Modern, responsive UI
- Drag and drop image upload
- Real-time model inference
- Visual feedback during processing

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload an image by either:
   - Dragging and dropping an image onto the upload area
   - Clicking the upload area to select a file

2. Preview your image and click "Process Image"

3. View the model's predictions in the results section

## Technical Details

- Frontend: HTML5, CSS3, JavaScript
- Backend: Flask
- Model: TensorFlow.js
- Image Processing: Browser-based using TensorFlow.js 