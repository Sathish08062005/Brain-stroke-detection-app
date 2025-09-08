from tensorflow.keras.models import load_model
import numpy as np

try:
    # Load the model from file
    model = load_model('stroke_model.h5')
    print("✅ Model loaded successfully!")

    # Create dummy input with correct shape (example: 224x224 RGB image)
    dummy_input = np.random.rand(1, 224, 224, 3)

    # Run a prediction
    prediction = model.predict(dummy_input)
    print("✅ Model prediction successful! Output:", prediction)

except Exception as e:
    print("❌ Failed to load model or predict:", e)
