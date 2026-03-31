"""
Main script for brain tumor prediction and analysis
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import BrainTumorPredictor
from utils.report_generator import TumorReport
from database.db_manager import TumorDatabase
from config import REPORT_DIR

def analyze_scan(image_path, patient_info=None):
    """
    Complete analysis pipeline for a brain scan
    
    Args:
        image_path: Path to brain MRI image
        patient_info: Dictionary with patient information
    """
    print("\n" + "="*60)
    print("BRAIN TUMOR ANALYSIS SYSTEM")
    print("="*60)
    
    # Initialize components
    predictor = BrainTumorPredictor()
    report_gen = TumorReport()
    db = TumorDatabase()
    
    # Load models
    print("\nLoading AI models...")
    predictor.load_models()
    
    # Run prediction
    print(f"\nAnalyzing image: {image_path}")
    results = predictor.predict(image_path)
    
    # Display results
    print("\n" + predictor.get_prediction_summary(results))
    
    # Store in database
    if patient_info:
        print("\nStoring results in database...")
        
        # Add patient
        patient_id = db.add_patient(
            name=patient_info.get('name', 'Unknown'),
            age=patient_info.get('age'),
            gender=patient_info.get('gender'),
            contact=patient_info.get('contact')
        )
        
        # Add scan
        scan_id = db.add_scan(patient_id, image_path)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{patient_info.get('name', 'patient')}_{timestamp}.pdf"
        report_path = os.path.join(REPORT_DIR, report_filename)
        
        report_gen.generate_report(patient_info, results, report_path)
        
        # Add prediction
        tumor_location = results.get('tumor_location', {})
        tumor_area = results.get('tumor_metrics', {}).get('tumor_area', 0) if results.get('tumor_metrics') else 0
        
        db.add_prediction(
            scan_id=scan_id,
            has_tumor=results['has_tumor'],
            tumor_type=results.get('tumor_type'),
            confidence=results.get('tumor_type_confidence', 0),
            tumor_location=tumor_location,
            tumor_area=tumor_area,
            report_path=report_path
        )
        
        print(f"✓ Report saved: {report_path}")
        print(f"✓ Data saved to database (Patient ID: {patient_id})")
    
    return results

def batch_analysis(image_folder):
    """Analyze multiple images in a folder"""
    import glob
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    predictor = BrainTumorPredictor()
    predictor.load_models()
    
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        result = predictor.predict(image_path)
        results.append(result)
        
        # Quick summary
        if result['has_tumor']:
            print(f"  → TUMOR DETECTED ({result['tumor_type']})")
        else:
            print(f"  → NO TUMOR")
    
    return results

def view_statistics():
    """View database statistics"""
    db = TumorDatabase()
    stats = db.get_statistics()
    
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    print(f"\nTotal Patients: {stats['total_patients']}")
    print(f"Total Scans: {stats['total_scans']}")
    print(f"Tumor Detection Rate: {stats['tumor_detection_rate']:.2f}%")
    
    if stats['tumor_types']:
        print("\nTumor Type Distribution:")
        for tumor_type, count in stats['tumor_types'].items():
            print(f"  - {tumor_type}: {count}")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*60)
        print("BRAIN TUMOR DETECTION SYSTEM - MAIN MENU")
        print("="*60)
        print("\n1. Analyze Single Scan")
        print("2. Batch Analysis")
        print("3. View Patient History")
        print("4. View Statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip()
            
            if not os.path.exists(image_path):
                print("❌ Image not found!")
                continue
            
            # Get patient info
            print("\nPatient Information (press Enter to skip):")
            name = input("Name: ").strip() or "Unknown"
            age = input("Age: ").strip()
            age = int(age) if age.isdigit() else None
            gender = input("Gender: ").strip() or None
            contact = input("Contact: ").strip() or None
            
            patient_info = {
                'name': name,
                'age': age,
                'gender': gender,
                'contact': contact,
                'scan_date': datetime.now().strftime("%Y-%m-%d")
            }
            
            analyze_scan(image_path, patient_info)
        
        elif choice == '2':
            folder_path = input("\nEnter folder path: ").strip()
            
            if not os.path.exists(folder_path):
                print("❌ Folder not found!")
                continue
            
            batch_analysis(folder_path)
        
        elif choice == '3':
            db = TumorDatabase()
            patients = db.get_all_patients()
            
            if not patients:
                print("\n❌ No patients in database")
                continue
            
            print("\nPatients:")
            for p in patients[:10]:  # Show first 10
                print(f"  ID: {p[0]}, Name: {p[1]}, Age: {p[2]}")
            
            patient_id = input("\nEnter Patient ID: ").strip()
            
            if patient_id.isdigit():
                history = db.get_patient_history(int(patient_id))
                
                if history:
                    print(f"\nHistory for Patient ID {patient_id}:")
                    for record in history:
                        print(f"\nScan Date: {record[1]}")
                        print(f"Has Tumor: {'Yes' if record[3] else 'No'}")
                        if record[4]:
                            print(f"Tumor Type: {record[4]}")
                else:
                    print("No history found")
        
        elif choice == '4':
            view_statistics()
        
        elif choice == '5':
            print("\nThank you for using Brain Tumor Detection System!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
