"""
NeuroMarket - Complete Demo
Connects products with predictions
"""

import numpy as np
import random
from data_loader import SEEDLoader
from feature_extractor import FeatureExtractor
from predict import NeuroMarketPredictor

class NeuroMarketDemo:
    def __init__(self):
        # Load model and data
        self.predictor = NeuroMarketPredictor()
        self.extractor = FeatureExtractor()
        
        # Product catalog (30 products)
        self.products = [
            # Nike
            {"id": 1, "name": "Nike Air Max", "brand": "Nike", "category": "Shoes", "price": 150},
            {"id": 2, "name": "Nike Dri-FIT Tee", "brand": "Nike", "category": "Apparel", "price": 35},
            {"id": 3, "name": "Nike Pro Shorts", "brand": "Nike", "category": "Apparel", "price": 40},
            {"id": 4, "name": "Nike Vaporfly", "brand": "Nike", "category": "Shoes", "price": 250},
            
            # Adidas
            {"id": 5, "name": "Adidas Ultraboost", "brand": "Adidas", "category": "Shoes", "price": 180},
            {"id": 6, "name": "Adidas Originals Hoodie", "brand": "Adidas", "category": "Apparel", "price": 70},
            {"id": 7, "name": "Adidas Soccer Cleats", "brand": "Adidas", "category": "Shoes", "price": 120},
            {"id": 8, "name": "Adidas Track Pants", "brand": "Adidas", "category": "Apparel", "price": 55},
            
            # Apple
            {"id": 9, "name": "iPhone 15", "brand": "Apple", "category": "Phone", "price": 999},
            {"id": 10, "name": "MacBook Pro", "brand": "Apple", "category": "Laptop", "price": 1299},
            {"id": 11, "name": "AirPods Pro", "brand": "Apple", "category": "Audio", "price": 249},
            {"id": 12, "name": "Apple Watch", "brand": "Apple", "category": "Wearable", "price": 399},
            
            # Samsung
            {"id": 13, "name": "Galaxy S24", "brand": "Samsung", "category": "Phone", "price": 899},
            {"id": 14, "name": "Galaxy Tab", "brand": "Samsung", "category": "Tablet", "price": 649},
            {"id": 15, "name": "Galaxy Buds", "brand": "Samsung", "category": "Audio", "price": 149},
            {"id": 16, "name": "Galaxy Watch", "brand": "Samsung", "category": "Wearable", "price": 299},
            
            # Beverages
            {"id": 17, "name": "Coca-Cola", "brand": "Coke", "category": "Beverage", "price": 2},
            {"id": 18, "name": "Pepsi", "brand": "Pepsi", "category": "Beverage", "price": 2},
            {"id": 19, "name": "Sprite", "brand": "Coke", "category": "Beverage", "price": 2},
            {"id": 20, "name": "Mountain Dew", "brand": "Pepsi", "category": "Beverage", "price": 2},
            
            # Fast Food
            {"id": 21, "name": "McDonald's Burger", "brand": "McDonalds", "category": "Food", "price": 5},
            {"id": 22, "name": "Burger King Whopper", "brand": "Burger King", "category": "Food", "price": 5},
            {"id": 23, "name": "KFC Chicken", "brand": "KFC", "category": "Food", "price": 6},
            {"id": 24, "name": "Taco Bell Tacos", "brand": "Taco Bell", "category": "Food", "price": 3},
            
            # Cars
            {"id": 25, "name": "Tesla Model 3", "brand": "Tesla", "category": "Car", "price": 45000},
            {"id": 26, "name": "BMW 3 Series", "brand": "BMW", "category": "Car", "price": 44000},
            {"id": 27, "name": "Mercedes C-Class", "brand": "Mercedes", "category": "Car", "price": 46000},
            {"id": 28, "name": "Audi A4", "brand": "Audi", "category": "Car", "price": 42000},
            
            # Coffee
            {"id": 29, "name": "Starbucks Latte", "brand": "Starbucks", "category": "Coffee", "price": 5},
            {"id": 30, "name": "Dunkin Donuts Coffee", "brand": "Dunkin", "category": "Coffee", "price": 3},
        ]
        
        # Load SEED data to use as brain responses
        loader = SEEDLoader()
        self.trials = loader.load_trials()
        print(f"\n✅ Demo ready with {len(self.trials)} brain responses")
    
    def get_brain_response_for_product(self, product):
        """Simulate brain response for a product"""
        # In real system, this would be EEG recording
        # Here we just pick a random trial from our dataset
        trial = random.choice(self.trials)
        features = self.extractor.extract_features(trial['eeg'])
        return features, trial['purchase_intent']
    
    def run_demo(self, n_trials=10):
        """Run interactive demo"""
        print("\n" + "="*70)
        print("🧠 NEUROMARKET - Product Testing Demo")
        print("="*70)
        
        results = []
        
        for i in range(n_trials):
            # Pick random product
            product = random.choice(self.products)
            
            print(f"\n▶️ Trial {i+1}: {product['name']} (${product['price']})")
            print(f"   Brand: {product['brand']} | Category: {product['category']}")
            
            # Get "brain response"
            features, actual = self.get_brain_response_for_product(product)
            
            # Predict
            prediction = self.predictor.predict(features)
            explanation = self.predictor.explain_prediction(features, top_n=3)
            
            print(f"\n   🔮 PREDICTION: {prediction['prediction']}")
            print(f"   Confidence: {prediction['confidence']*100:.1f}%")
            print(f"\n   📊 WHY:")
            for feat in explanation['top_features'][:3]:
                print(f"      {feat['interpretation']}")
            
            # In demo, we know the "truth" from our data
            actual_text = "BUY" if actual == 1 else "NOT BUY"
            correct = (prediction['prediction'] == "BUY" and actual == 1) or \
                     (prediction['prediction'] == "NOT BUY" and actual == 0)
            
            print(f"\n   ✅ Actual preference: {actual_text}")
            print(f"   {'✓ CORRECT' if correct else '✗ INCORRECT'}")
            
            results.append(correct)
            
            input("\n   Press Enter for next product...")
        
        # Summary
        accuracy = sum(results) / len(results) * 100
        print("\n" + "="*70)
        print(f"📊 DEMO COMPLETE - Accuracy: {accuracy:.1f}%")
        print("="*70)


if __name__ == "__main__":
    demo = NeuroMarketDemo()
    demo.run_demo(n_trials=5)