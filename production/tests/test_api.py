import requests
import json

def test_api_prediction():
    """Test the prediction endpoint with sample data"""
    # Test data
    test_car = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2018,
        "mileage": 45000,
        "fuel_type": "Petrol"
    }
    
    # Make request to API
    response = requests.post(
        "http://localhost:8000/predict/",
        data=json.dumps(test_car),
        headers={"Content-Type": "application/json"}
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert "predicted_price" in result
    print(f"Predicted price: {result['predicted_price']}")

if __name__ == "__main__":
    test_api_prediction()
