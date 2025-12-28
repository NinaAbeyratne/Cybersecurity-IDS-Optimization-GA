"""
Setup Verification Script
Run this to verify all dependencies are correctly installed
and CICIDS2017 data is in place before running main.py

Usage: python check_setup.py
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def check_python_version():
    """Check Python version"""
    print("\n📍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ⚠️  WARNING: Python 3.10+ recommended")
        return False
    else:
        print("   ✅ Python version OK")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📍 Checking required packages...")
    
    packages = {
        'numpy': 'Core numerical computing',
        'pandas': 'Data manipulation',
        'sklearn': 'Machine learning (scikit-learn)',
        'deap': 'Genetic algorithm framework',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'imblearn': 'Imbalanced learning (optional)'
    }
    
    all_ok = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package:15s} - {description}")
        except ImportError:
            print(f"   ❌ {package:15s} - MISSING - {description}")
            all_ok = False
    
    if not all_ok:
        print("\n   ⚠️  Missing packages detected!")
        print("   Run: pip install -r requirements.txt")
    else:
        print("\n   ✅ All required packages installed")
    
    return all_ok

def check_directory_structure():
    """Check if project directories exist"""
    print("\n📍 Checking directory structure...")
    
    required_dirs = [
        'data',
        'src',
        'reports/figures',
        'reports/results',
        'models',
        'notebooks'
    ]
    
    all_ok = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory:20s} - exists")
        else:
            print(f"   ❌ {directory:20s} - MISSING")
            all_ok = False
    
    if not all_ok:
        print("\n   ⚠️  Missing directories detected!")
        print("   Run: python create_structure.py")
    else:
        print("\n   ✅ Directory structure OK")
    
    return all_ok

def check_source_files():
    """Check if source code files exist"""
    print("\n📍 Checking source code files...")
    
    required_files = [
        'main.py',
        'src/__init__.py',
        'src/preprocessing.py',
        'src/models.py',
        'src/ga.py',
        'src/evaluate.py'
    ]
    
    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   ✅ {filepath:30s} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {filepath:30s} - MISSING")
            all_ok = False
    
    if not all_ok:
        print("\n   ⚠️  Missing source files detected!")
    else:
        print("\n   ✅ All source files present")
    
    return all_ok

def check_data_files():
    """Check if CICIDS2017 CSV files are present"""
    print("\n📍 Checking CICIDS2017 dataset...")
    
    expected_files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]
    
    data_dir = Path('data')
    if not data_dir.exists():
        print("   ❌ data/ directory does not exist")
        return False
    
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("   ❌ No CSV files found in data/ directory")
        print("\n   📥 Please download CICIDS2017 dataset:")
        print("      1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("      2. Download MachineLearningCSV.zip")
        print("      3. Extract all CSV files to data/ directory")
        return False
    
    print(f"   Found {len(csv_files)} CSV file(s) in data/:")
    
    found_expected = 0
    total_size_mb = 0
    
    for csv_file in csv_files:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        total_size_mb += size_mb
        
        if csv_file.name in expected_files:
            print(f"   ✅ {csv_file.name:60s} ({size_mb:.1f} MB)")
            found_expected += 1
        else:
            print(f"   ⚠️  {csv_file.name:60s} ({size_mb:.1f} MB) - unexpected file")
    
    print(f"\n   Total dataset size: {total_size_mb:.1f} MB")
    print(f"   Expected files found: {found_expected}/{len(expected_files)}")
    
    if found_expected == len(expected_files):
        print("\n   ✅ All CICIDS2017 files present")
        return True
    elif found_expected > 0:
        print(f"\n   ⚠️  Only {found_expected}/{len(expected_files)} expected files found")
        print("      You can proceed, but results may be incomplete")
        return True
    else:
        print("\n   ❌ CICIDS2017 dataset not properly installed")
        return False

def check_imports():
    """Try importing key modules"""
    print("\n📍 Testing module imports...")
    
    modules = [
        ('src.preprocessing', 'CICIDS2017Preprocessor'),
        ('src.models', 'BaselineModels'),
        ('src.ga', 'GeneticFeatureSelector'),
        ('src.evaluate', 'ModelEvaluator')
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name:25s} - {class_name}")
        except Exception as e:
            print(f"   ❌ {module_name:25s} - ERROR: {str(e)}")
            all_ok = False
    
    if all_ok:
        print("\n   ✅ All modules import successfully")
    else:
        print("\n   ⚠️  Some modules failed to import")
        print("      Check for syntax errors or missing dependencies")
    
    return all_ok

def estimate_runtime():
    """Estimate execution time based on data size"""
    print("\n📍 Estimating runtime...")
    
    data_dir = Path('data')
    if not data_dir.exists():
        print("   ⚠️  Cannot estimate (no data directory)")
        return
    
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print("   ⚠️  Cannot estimate (no CSV files)")
        return
    
    total_size_gb = sum(f.stat().st_size for f in csv_files) / (1024**3)
    
    # Rough estimates (will vary by hardware)
    preprocessing_time = total_size_gb * 2  # ~2 min per GB
    baseline_time = 3  # ~3 min
    ga_time = 30  # ~30 min for 30 generations
    final_time = 5  # ~5 min
    
    total_time = preprocessing_time + baseline_time + ga_time + final_time
    
    print(f"   Dataset size: {total_size_gb:.2f} GB")
    print(f"   Estimated time breakdown:")
    print(f"      - Preprocessing:  ~{preprocessing_time:.0f} min")
    print(f"      - Baseline model: ~{baseline_time:.0f} min")
    print(f"      - GA evolution:   ~{ga_time:.0f} min")
    print(f"      - Final model:    ~{final_time:.0f} min")
    print(f"   Total estimated:   ~{total_time:.0f} min ({total_time/60:.1f} hours)")
    print("\n   💡 TIP: For faster testing, edit main.py to use max_rows_per_file=50000")

def main():
    """Run all checks"""
    print_header("SETUP VERIFICATION FOR CYBERSECURITY IDS PROJECT")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Source Files", check_source_files),
        ("Dataset Files", check_data_files),
        ("Module Imports", check_imports)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Estimate runtime
    estimate_runtime()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(results.values())
    
    print("\nCheck Results:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status:10s} {name}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ Your environment is ready to run the project")
        print("\nNext steps:")
        print("   1. Run: python main.py")
        print("   2. Wait for completion (~30-60 minutes)")
        print("   3. Check reports/ directory for results")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n❌ Please fix the issues above before running main.py")
        print("\nCommon fixes:")
        print("   - Missing packages: pip install -r requirements.txt")
        print("   - Missing directories: python create_structure.py")
        print("   - Missing data: Download CICIDS2017 and extract to data/")
        print("\n" + "=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())