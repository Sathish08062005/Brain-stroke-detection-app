from tensorflow.keras.models import load_model
import numpy as np

try:
    # Load your updated/converted model
    model = load_model('stroke_model_compatible.keras')  # or your updated .h5

    print("✅ Model loaded successfully!")

    # Create a dummy input with the correct shape
    dummy_input = np.random.rand(1, 224, 224, 3)

    # Run a prediction
    prediction = model.predict(dummy_input)
    print("✅ Model prediction successful! Output:", prediction)

except Exception as e:
    print("❌ Failed to load model or predict:", e)


