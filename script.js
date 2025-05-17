let model = null;
// Italian labels with English translations and fun facts
let categories = [
    {
        italian: 'cavallo',
        english: 'horse',
        funFact: 'Horses can sleep both lying down and standing up, and they can run shortly after birth!'
    },
    {
        italian: 'pecora',
        english: 'sheep',
        funFact: 'Sheep have rectangular pupils that give them amazing peripheral vision - they can see behind themselves without turning their heads!'
    },
    {
        italian: 'elefante',
        english: 'elephant',
        funFact: 'Elephants are the only mammals that can\'t jump, but they use their trunks like a snorkel and can swim for up to 6 hours!'
    },
    {
        italian: 'gatto',
        english: 'cat',
        funFact: 'A cat\'s purr vibrates at a frequency of 25 to 150 Hz, which can promote healing and bone density!'
    },
    {
        italian: 'scoiattolo',
        english: 'squirrel',
        funFact: 'Squirrels can find food buried beneath a foot of snow and can turn their ankles 180 degrees when climbing down trees!'
    },
    {
        italian: 'gallina',
        english: 'chicken',
        funFact: 'Chickens have better color vision than humans and can remember over 100 different faces of people or animals!'
    },
    {
        italian: 'ragno',
        english: 'spider',
        funFact: 'Spiders can\'t fly, but they can balloon through the air on strands of silk for hundreds of miles!'
    },
    {
        italian: 'mucca',
        english: 'cow',
        funFact: 'Cows have best friends and get stressed when separated. They also produce more milk when listening to calming music!'
    },
    {
        italian: 'cane',
        english: 'dog',
        funFact: 'A dog\'s sense of smell is so powerful that they can detect diseases and know when a storm is coming!'
    },
    {
        italian: 'farfalla',
        english: 'butterfly',
        funFact: 'Butterflies taste with their feet and can see ultraviolet light that\'s invisible to humans!'
    }
];

// Load the model when the page loads
async function loadModel() {
    try {
        // Use loadGraphModel() for TensorFlow SavedModel converted to TFJS
        model = await tf.loadGraphModel('./tfjs_model/model.json');
        console.log('Model loaded successfully');
        
        // Get input shape from the model
        const inputShape = model.inputs[0].shape;
        console.log('Model input shape:', inputShape);
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading the model. Please check the console for details.');
    }
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    loadModel();
    setupImageUpload();
});

function setupImageUpload() {
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.preview-container');
    const clearButton = document.getElementById('clearButton');
    const processButton = document.getElementById('processButton');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const closeButton = document.querySelector('.close-button');

    // Add close button handler
    closeButton.addEventListener('click', () => {
        results.hidden = true;
    });

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2d7dd2';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#4a90e2';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#4a90e2';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        }
    });

    // Handle click to upload
    dropZone.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            handleImageFile(file);
        }
    });

    // Handle clear button
    clearButton.addEventListener('click', clearImage);

    // Handle process button
    processButton.addEventListener('click', processImage);

    function handleImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.hidden = false;
            processButton.disabled = false;
            results.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        imagePreview.src = '';
        previewContainer.hidden = true;
        processButton.disabled = true;
        results.hidden = true;
        imageInput.value = '';
    }
}

async function processImage() {
    if (!model) {
        alert('Model is not loaded yet. Please wait and try again.');
        return;
    }

    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    const imagePreview = document.getElementById('imagePreview');
    const processButton = document.getElementById('processButton');

    try {
        // Show loading state
        loading.hidden = false;
        processButton.disabled = true;
        results.hidden = true;

        // Convert image to tensor
        const tensor = await preprocessImage(imagePreview);
        
        // Log input tensor values for debugging
        console.log('Model input tensor:', tensor);
        tensor.data().then(data => {
            console.log('Input tensor sample values:', data.slice(0, 20));
        });
        
        // Run inference using execute() with the correct input name
        const predictions = await model.execute(
            {"keras_tensor": tensor}
        );
        
        // Log raw predictions for debugging
        console.log('Raw predictions tensor:', predictions);
        const probabilities = await predictions.data();
        console.log('Raw probabilities:', probabilities);
        
        // Process predictions
        const processedResults = formatResults(probabilities);

        // Display results
        resultContent.innerHTML = processedResults;
        results.hidden = false;

        // Cleanup
        tensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('Error processing image:', error);
        alert('Error processing image. Please check the console for details.');
    } finally {
        loading.hidden = true;
        processButton.disabled = false;
    }
}

async function preprocessImage(imgElement) {
    return tf.tidy(() => {
        // Convert the image to a tensor
        let img = tf.browser.fromPixels(imgElement);
        console.log('Original image shape:', img.shape);
        
        // Ensure RGB format
        if (img.shape[2] === 4) {
            img = img.slice([0, 0, 0], [-1, -1, 3]); // Remove alpha channel if present
        }
        
        // Resize to 128x128 as per training
        const resized = tf.image.resizeBilinear(img, [128, 128]);
        console.log('Resized shape:', resized.shape);
        
        // Normalize to [0,1] range
        const normalized = tf.div(resized, 255.0);
        
        // Log a sample of pixel values to verify normalization
        normalized.data().then(data => {
            console.log('Sample of normalized pixels:', data.slice(0, 10));
        });
        
        // Add batch dimension [1, 128, 128, 3]
        const batched = tf.expandDims(normalized, 0);
        console.log('Final input shape:', batched.shape);
        
        return batched;
    });
}

function formatResults(probabilities) {
    // Convert probabilities to array of {category, probability} objects
    const results = categories.map((category, i) => ({
        italian: category.italian,
        english: category.english,
        funFact: category.funFact,
        probability: probabilities[i]
    }));
    
    // Sort by probability descending
    results.sort((a, b) => b.probability - a.probability);
    
    // Get the top prediction
    const topPrediction = results[0];
    const confidence = (topPrediction.probability * 100).toFixed(1);
    
    // Format as HTML
    const resultsHtml = results.map((result, index) => `
        <div class="prediction-item ${index === 0 ? 'top-prediction' : ''}">
            <div class="prediction-header">
                <span class="category">
                    <span class="animal-name">${result.italian}</span>
                    <span class="translation">${result.english}</span>
                </span>
                <span class="probability-value">${(result.probability * 100).toFixed(1)}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${(result.probability * 100).toFixed(1)}%"></div>
            </div>
            ${index === 0 ? `
                <div class="fun-fact">
                    <span class="fun-fact-icon">🎯</span>
                    <p>I'm ${confidence}% sure this is a ${result.english}!</p>
                </div>
                <div class="fun-fact">
                    <span class="fun-fact-icon">✨</span>
                    <p>${result.funFact}</p>
                </div>
            ` : ''}
        </div>
    `).join('');
    
    return `
        <div class="prediction-results">
            <h3>Analysis Results</h3>
            ${resultsHtml}
        </div>
    `;
} 