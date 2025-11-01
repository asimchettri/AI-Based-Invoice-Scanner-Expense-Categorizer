import os
from pathlib import Path
from ocr import OCRUtility

def main():
    print("\n==============================")
    print("🧪 OCR Utility Test Script")
    print("==============================")

    # Initialize OCR utility
    ocr = OCRUtility()

    # Folder containing test files
    test_folder = Path("sample_files")

    if not test_folder.exists():
        print("❌ Folder 'sample_files/' not found. Please create it and add test files.")
        return

    # Loop through all files in the folder
    for file_path in test_folder.iterdir():
        if not file_path.is_file():
            continue
        
        print(f"\n📄 Testing file: {file_path.name}")
        result = ocr.extract_from_file(file_path)

        # Print key results
        print(f"File Type: {result.get('file_type')}")
        print(f"Status: {result.get('status')}")
        print(f"Characters Extracted: {result.get('char_count')}")

        if result.get('status') == 'success':
            print(f"✅ Extracted Text (first 200 chars):\n{result['text'][:200]}")
        else:
            print(f"❌ Error: {result.get('error')}")

    print("\n🎯 Test completed!\n")

if __name__ == "__main__":
    main()
