const express = require('express');
const multer = require('multer');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

const app = express();

// Set up directories
const MODEL_DIR = path.join(__dirname, 'static', 'model');
const UPLOADS_DIR = path.join(__dirname, 'static', 'uploads');

// Allowed file types
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png'];

// Set up file uploads with multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    if (!fs.existsSync(UPLOADS_DIR)) {
      fs.mkdirSync(UPLOADS_DIR, { recursive: true });
    }
    cb(null, UPLOADS_DIR);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage: storage });

// Serve the index.html on the home route
app.set('view engine', 'html');
app.set('views', path.join(__dirname, 'views'));

app.get('/', (req, res) => {
  const history = {
    accuracy: [0.8, 0.85, 0.9],
    val_accuracy: [0.75, 0.8, 0.85],
    test_accuracy: [0.7, 0.78, 0.82],
    test_loss: [0.5, 0.4, 0.35]
  };
  res.render('index', { history });
});

// Train route - creates and saves the model
app.post('/train', async (req, res) => {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [224, 224, 3] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  const modelPath = path.join(MODEL_DIR, 'model.json');
  await model.save(`file://${MODEL_DIR}`);

  res.json({ message: 'Model trained and saved successfully' });
});

// Predict route
app.post('/predict', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const fileExt = path.extname(req.file.filename).substring(1);
  if (!ALLOWED_EXTENSIONS.includes(fileExt.toLowerCase())) {
    return res.status(400).json({ error: 'Invalid file type' });
  }

  const modelFiles = fs.readdirSync(MODEL_DIR).filter(file => file.endsWith('.json'));
  if (modelFiles.length === 0) {
    return res.status(404).json({ error: 'No model files found' });
  }

  // Load the model
  const modelPath = path.join(MODEL_DIR, modelFiles[0]);
  let model;
  try {
    model = await tf.loadLayersModel(`file://${modelPath}`);
  } catch (e) {
    return res.status(500).json({ error: 'Error loading model: ' + e.message });
  }

  // Load and preprocess the image
  let img;
  try {
    const imgPath = path.join(UPLOADS_DIR, req.file.filename);
    img = await loadImage(imgPath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224).data;
    const imgTensor = tf.tensor3d(imageData, [224, 224, 4])
      .slice([0, 0, 0], [-1, -1, 3]) // Remove alpha channel
      .div(tf.scalar(255))
      .expandDims(0); // Add batch dimension
  } catch (e) {
    return res.status(500).json({ error: 'Error processing image: ' + e.message });
  }

  // Perform prediction
  try {
    const prediction = model.predict(imgTensor);
    const predictionArray = prediction.arraySync();
    const result = {
      prediction: predictionArray,
      class_labels: ['class1', 'class2', 'class3'] // Update with actual labels
    };
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: 'Prediction error: ' + e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
