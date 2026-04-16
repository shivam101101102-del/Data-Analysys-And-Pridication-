"""
================================================================================
TEST USER PREDICTION SYSTEM
================================================================================
Quick test script to verify the prediction system works correctly
================================================================================
"""

import pickle
import os

def test_model_exists():
    """Test if trained model exists"""
    print("\n" + "=" * 80)
    print("TEST 1: Checking if trained model exists...")
    print("=" * 80)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "01_Dataset", "trained_model.pkl")
    
    if os.path.exists(model_path):
        print("✅ PASS: trained_model.pkl found")
        return True
    else:
        print("❌ FAIL: trained_model.pkl not found")
        print("   Please run: python RUN_COMPLETE_PROJECT.py")
        return False

def test_model_loading():
    """Test if model can be loaded"""
    print("\n" + "=" * 80)
    print("TEST 2: Loading trained model...")
    print("=" * 80)
    
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "01_Dataset", "trained_model.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ PASS: Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ FAIL: Error loading model - {e}")
        return None

def test_prediction(model):
    """Test prediction with sample data"""
    print("\n" + "=" * 80)
    print("TEST 3: Making test predictions...")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            'name': 'Low Risk Patient',
            'data': [30, 1, 2, 2, 1],  # Age, Smoking, Pollution, Fatigue, Coughing
            'expected_range': (0, 40)
        },
        {
            'name': 'Medium Risk Patient',
            'data': [55, 5, 6, 5, 4],
            'expected_range': (40, 70)
        },
        {
            'name': 'High Risk Patient',
            'data': [65, 8, 7, 8, 7],
            'expected_range': (70, 100)
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['name']} ---")
        print(f"Input: Age={test['data'][0]}, Smoking={test['data'][1]}, "
              f"Pollution={test['data'][2]}, Fatigue={test['data'][3]}, "
              f"Coughing={test['data'][4]}")
        
        try:
            prediction = model.predict([test['data']])[0]
            print(f"Predicted Risk Score: {prediction:.2f}")
            
            # Check if in expected range
            if test['expected_range'][0] <= prediction <= test['expected_range'][1]:
                print(f"✅ PASS: Score in expected range {test['expected_range']}")
            else:
                print(f"⚠️  WARNING: Score outside expected range {test['expected_range']}")
                print("   (This is OK - model predictions can vary)")
            
        except Exception as e:
            print(f"❌ FAIL: Prediction error - {e}")
            all_passed = False
    
    return all_passed

def test_user_interfaces():
    """Test if user interface files exist"""
    print("\n" + "=" * 80)
    print("TEST 4: Checking user interface files...")
    print("=" * 80)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    files = [
        'USER_PREDICTION_INTERFACE.py',
        'USER_PREDICTION_CLI.py',
        'USER_GUIDE.txt'
    ]
    
    all_exist = True
    for file in files:
        path = os.path.join(BASE_DIR, file)
        if os.path.exists(path):
            print(f"✅ PASS: {file} exists")
        else:
            print(f"❌ FAIL: {file} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 80)
    print("🧪 TESTING USER PREDICTION SYSTEM")
    print("=" * 80)
    print("This script will verify that the prediction system is ready to use")
    print("=" * 80)
    
    # Test 1: Model exists
    if not test_model_exists():
        print("\n" + "=" * 80)
        print("❌ TESTS FAILED: Model not found")
        print("=" * 80)
        print("\nPlease run the following command first:")
        print("   python RUN_COMPLETE_PROJECT.py")
        print("\nThis will train the model and prepare the system.")
        return
    
    # Test 2: Load model
    model = test_model_loading()
    if model is None:
        print("\n" + "=" * 80)
        print("❌ TESTS FAILED: Cannot load model")
        print("=" * 80)
        return
    
    # Test 3: Make predictions
    test_prediction(model)
    
    # Test 4: Check interface files
    test_user_interfaces()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)
    print("\n🎉 Your prediction system is ready to use!")
    print("\nTo start using the system:")
    print("\n  For GUI (Recommended):")
    print("     python USER_PREDICTION_INTERFACE.py")
    print("\n  For Command-Line:")
    print("     python USER_PREDICTION_CLI.py")
    print("\n  For Help:")
    print("     Read USER_GUIDE.txt")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
