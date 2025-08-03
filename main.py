import json
import time
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the corrected NumberPredictionAI class
try:
    from models.predictor import NumberPredictionAI
except ImportError:
    print("❌ Could not import NumberPredictionAI. Make sure predictor.py is in the same directory.")
    sys.exit(1)

def load_config():
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Validate required configuration keys
        required_keys = [
            'api_token', 'api_base_url', 'operator_id', 'partner_id', 
            'game_id', 'table_id', 'provider_id', 'model_dir'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"❌ Missing required configuration keys: {missing_keys}")
            raise ValueError(f"Missing configuration keys: {missing_keys}")
        
        # Set default values for optional keys
        config.setdefault('seq_length', 5)
        config.setdefault('initial_balance', 1000)
        config.setdefault('risk_factor', 0.3)
        config.setdefault('update_interval', 30)
        config.setdefault('initial_data_count', 50)
        config.setdefault('retrain_interval', 50)
        
        print("✅ Configuration loaded successfully")
        return config
        
    except FileNotFoundError:
        print("❌ config.json not found. Please create config.json with the required settings.")
        print("Required settings: api_token, api_base_url, operator_id, partner_id, game_id, table_id, provider_id, model_dir")
        raise
    except json.JSONDecodeError:
        print("❌ Invalid JSON in config.json. Please check the file format.")
        raise
    except Exception as e:
        print(f"❌ Error loading configuration: {str(e)}")
        raise

def test_api_connection(ai):
    """Test basic API connectivity before starting the main loop."""
    print("🔍 Testing API connection...")
    
    # Test market fetch
    markets = ai.fetch_markets()
    if not markets:
        print("❌ Failed to fetch markets. Check your API credentials and network connection.")
        return False
    
    # Test results fetch
    results = ai.fetch_results(5)
    if results.empty:
        print("❌ Failed to fetch results. Check your API credentials and game configuration.")
        return False
    
    print(f"✅ API connection successful. Found {len(results)} recent results.")
    return True

def main():
    """Main function to initialize and run the prediction system."""
    print("🚀 Starting Number Prediction AI System")
    print("=" * 50)
    
    ai = None
    try:
        # Load configuration
        config = load_config()
        
        # Initialize the AI system
        print("⚙️ Initializing AI system...")
        ai = NumberPredictionAI(config)
        
        # Test API connection
        if not test_api_connection(ai):
            print("❌ API connection test failed. Please check your configuration.")
            return
        
        print("⏳ Initializing with historical data...")
        print("This may take a few minutes for the first run...")
        
        # Initial model training
        if not ai.update_model(config['initial_data_count']):
            print("❌ Initial training failed - check API connection and data availability")
            print("Common issues:")
            print("- Invalid API credentials")
            print("- Incorrect game/table IDs")
            print("- Network connectivity problems")
            print("- Insufficient historical data")
            return
        
        print("✅ Initial training completed successfully!")
        print("🚀 Starting real-time prediction service...")
        print(f"📊 Update interval: {config['update_interval']} seconds")
        print("📝 Press Ctrl+C to stop the service")
        print("=" * 50)
        
        # Start real-time prediction
        ai.start_real_time_prediction()
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Received exit signal...")
            
    except KeyboardInterrupt:
        print("\n🛑 Received exit signal...")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        print("Stack trace:")
        traceback.print_exc()
    finally:
        # Cleanup
        if ai is not None:
            try:
                ai.stop_real_time_prediction()
                print("✅ Prediction service stopped gracefully")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {str(e)}")
        
        print("👋 Application terminated")

if __name__ == "__main__":
    main()