import cv2
import os
import json
import numpy as np
import urllib.request

class FaceRecognizerSystem:
    def __init__(self, 
                 cascade_path="haarcascade_frontalface_default.xml", 
                 model_path="trained_model.yml", 
                 metadata_path="metadata.json",
                 face_size=(200, 200)):
        """
        Initializes the Face Recognition System.
        
        Args:
            cascade_path (str): Path to Haar Cascade XML file for face detection.
            model_path (str): Path to save/load the trained LBPH model.
            metadata_path (str): Path to save/load the face label-to-metadata mapping JSON.
            face_size (tuple): Width and height to resize all face crops to.
        """
        self.cascade_path = cascade_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.face_size = face_size
        
        # Download Haar Cascade if it doesn't exist
        if not os.path.exists(self.cascade_path):
            print(f"Haar cascade XML not found. Downloading from OpenCV official repository...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, self.cascade_path)
                print("Cascade downloaded successfully.")
            except Exception as e:
                print(f"Error downloading cascade: {e}")
                
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade from {self.cascade_path}")

        # Initialize CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def load_metadata(self):
        """Loads metadata mapping label_id (as string) to person info."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {e}. Starting fresh.")
        return {}

    def save_metadata(self, metadata):
        """Saves metadata mapping to JSON file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def preprocess_face(self, gray_frame, x, y, w, h):
        """
        Crops, resizes, and normalizes a face image.
        
        Applying standard size and CLAHE (contrast enhancement) helps significantly 
        improve accuracy under varying lighting conditions.
        """
        # Crop the face region
        face_crop = gray_frame[y:y+h, x:x+w]
        
        # Resize to a standardized dimension
        face_resized = cv2.resize(face_crop, self.face_size, interpolation=cv2.INTER_AREA)
        
        # Apply CLAHE to equalize local contrast and compensate for lighting changes
        face_normalized = self.clahe.apply(face_resized)
        
        # Optional: apply a slight Gaussian blur to reduce high-frequency noise/pixelation
        face_normalized = cv2.GaussianBlur(face_normalized, (3, 3), 0)
        
        return face_normalized

    def detect_faces(self, gray):
        """
        Detects faces and merges overlapping duplicate boxes (non-max suppression)
        so one real face yields exactly one box, while genuinely separate faces
        each get their own box.
        """
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return []

        boxes = faces.astype(float)
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(y2)
        keep = []
        while order.size > 0:
            i = order[-1]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            overlap = (w * h) / area[order[:-1]]
            order = order[np.where(overlap <= 0.3)[0]]

        return [(int(faces[k, 0]), int(faces[k, 1]), int(faces[k, 2]), int(faces[k, 3]))
                for k in keep]

    def _close_windows(self):
        # On macOS, destroyAllWindows() alone leaves the Cocoa window open
        # (and camera light on) unless the close event is flushed with waitKey.
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def capture_face_data(self, name, additional_info=None, target_count=60):
        """
        Captures face images from the default webcam, processes them, and saves them.
        
        Args:
            name (str): The folder name and display name of the target.
            additional_info (dict): A dictionary containing role, department, notes, etc.
            target_count (int): Number of images to capture.
        """
        # Create user-friendly directories
        safe_name = name.replace(" ", "_").lower()
        save_dir = f"faces/{safe_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return False
            
        count = 0
        print(f"Capturing face data for: {name}")
        print(f"Please move your head slightly. Looking straight, left, right, up, down.")
        print("Press 'q' to abort capture early.")

        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
                
            # Keep a gray version for detection and preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces (with duplicate-box merging)
            faces = self.detect_faces(gray)
            
            for (x, y, w, h) in faces:
                # Preprocess face crop (Resize + CLAHE normalization)
                processed_face = self.preprocess_face(gray, x, y, w, h)
                
                # Save processed crop
                face_filepath = f"{save_dir}/{count}.jpg"
                cv2.imwrite(face_filepath, processed_face)
                count += 1
                
                # Visual indicators on screen
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{target_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # break inner loop to process one face per frame to avoid duplicate crops in case of multi-face detection
                break
                
            cv2.imshow("Capture - Face Recognition System", frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print("Capture aborted by user.")
                break
                
        cap.release()
        self._close_windows()
        
        if count > 0:
            # Load existing metadata to append/update
            metadata = self.load_metadata()
            
            # Store target information
            target_data = {
                "name": name,
                "folder": safe_name,
                "info": additional_info or {}
            }
            
            # Use safe_name as the unique folder lookup.
            # We will map directory names to actual integer IDs during training.
            metadata[safe_name] = target_data
            self.save_metadata(metadata)
            
            print(f"Successfully saved {count} preprocessed face images to {save_dir}")
            print(f"Saved metadata for '{name}' to {self.metadata_path}")
            return True
        else:
            print("No images were captured.")
            return False

    def train_model(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        """
        Reads captured face folders, trains an LBPH Face Recognizer, and saves it.
        
        Args:
            radius, neighbors, grid_x, grid_y: LBPH hyperparameters.
        """
        if not os.path.exists("faces"):
            print("Error: No 'faces' directory exists. Capture some faces first.")
            return False
            
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=radius,
            neighbors=neighbors,
            grid_x=grid_x,
            grid_y=grid_y
        )
        
        faces_list = []
        labels_list = []
        
        # Load metadata to map folder names to label IDs
        metadata = self.load_metadata()
        
        # We will create a fresh clean mapping of label_id (string) -> target_data
        # that will be used by the recognizer during inference.
        new_label_map = {}
        current_label_id = 0
        
        # Scan face directories sorted to ensure consistency
        person_folders = sorted([d for d in os.listdir("faces") if os.path.isdir(f"faces/{d}")])
        
        if not person_folders:
            print("Error: No face data directories found in 'faces/'.")
            return False
            
        for folder_name in person_folders:
            person_dir = f"faces/{folder_name}"
            img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not img_files:
                continue
                
            # Find the corresponding info in our metadata
            original_meta = metadata.get(folder_name, {
                "name": folder_name.replace("_", " ").title(),
                "folder": folder_name,
                "info": {}
            })
            
            # Map this integer label to the target metadata
            new_label_map[str(current_label_id)] = original_meta
            print(f"Mapping Label {current_label_id} -> {original_meta['name']}")
            
            for img_file in img_files:
                img_path = f"{person_dir}/{img_file}"
                # Read as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # In case images were not preprocessed earlier, resize them to standard size
                    if img.shape != self.face_size:
                        img = cv2.resize(img, self.face_size, interpolation=cv2.INTER_AREA)
                    faces_list.append(img)
                    labels_list.append(current_label_id)
            
            current_label_id += 1
            
        if not faces_list:
            print("Error: No training images could be loaded.")
            return False
            
        print(f"Training LBPH recognizer on {len(faces_list)} face samples across {current_label_id} classes...")
        recognizer.train(faces_list, np.array(labels_list))
        recognizer.save(self.model_path)
        
        # Save updated mapping label_map inside the metadata under a special key "label_map"
        full_config = {
            "label_map": new_label_map
        }
        self.save_metadata(full_config)
        
        print(f"Training complete. Model saved to '{self.model_path}'")
        return True

    def recognize_faces(self, confidence_threshold=70):
        """
        Runs real-time webcam face recognition, displaying labels and additional metadata.
        
        Args:
            confidence_threshold (float): LBPH distance threshold. 
                                         Low values mean stricter matching (lower Chi-Square distance).
                                         Normally values < 70-80 are good matches.
        """
        if not os.path.exists(self.model_path):
            print(f"Error: Trained model '{self.model_path}' not found. Please train the model first.")
            return
            
        # Load configuration containing label map
        config = self.load_metadata()
        label_map = config.get("label_map", {})
        
        if not label_map:
            print("Error: Label map is empty in metadata.json. Please retrain the model.")
            return
            
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.model_path)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        print("Real-time Face Recognition running.")
        print(f"Confidence threshold is set to {confidence_threshold} (lower is stricter).")
        print("Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces (with duplicate-box merging)
            faces = self.detect_faces(gray)
            
            for (x, y, w, h) in faces:
                # Preprocess the crop in exactly the same way as during capture (Resize + CLAHE)
                processed_face = self.preprocess_face(gray, x, y, w, h)
                
                # Predict
                label_id, confidence = recognizer.predict(processed_face)
                
                # Look up metadata
                label_str = str(label_id)
                if label_str in label_map and confidence < confidence_threshold:
                    target_data = label_map[label_str]
                    name = target_data["name"]
                    info = target_data.get("info", {})
                    
                    # Choose a success color (green)
                    color = (0, 255, 0)
                    
                    # Create secondary text from info dict (role, dept, etc.)
                    info_lines = []
                    for k, v in info.items():
                        if v:
                            info_lines.append(f"{k.capitalize()}: {v}")
                else:
                    name = "Unknown"
                    info_lines = []
                    color = (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Render label and confidence score
                # Display name and distance (confidence)
                label_text = f"{name} ({confidence:.0f})"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw extra metadata info lines stacked below the face box or next to it
                y_offset = y + h + 20
                for line in info_lines:
                    cv2.putText(frame, line, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 18
                    
            cv2.imshow("Real-Time Recognition - Press 'q' to Quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        self._close_windows()

if __name__ == "__main__":
    # Interactive CLI menu when run directly
    print("=== FACE RECOGNITION SYSTEM CLI ===")
    system = FaceRecognizerSystem()
    
    while True:
        print("\nOptions:")
        print("1. Capture target face & add information")
        print("2. Train model")
        print("3. Start real-time recognition")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter the person's full name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
                
            print("Add some info:")
            role = input("Role/Designation (optional): ").strip()
            dept = input("Department (optional): ").strip()
            notes = input("Other notes (optional): ").strip()
            
            info = {}
            if role: info["role"] = role
            if dept: info["dept"] = dept
            if notes: info["notes"] = notes
            
            target_str = input("How many frames to capture? (default: 60): ").strip()
            target_count = int(target_str) if target_str.isdigit() else 60
            
            system.capture_face_data(name, additional_info=info, target_count=target_count)
            
        elif choice == '2':
            # Ask if they want default or custom training parameters
            print("Training the recognizer model...")
            system.train_model()
            
        elif choice == '3':
            threshold_str = input("Enter confidence threshold (default 70, lower = stricter): ").strip()
            threshold = int(threshold_str) if threshold_str.isdigit() else 70
            system.recognize_faces(confidence_threshold=threshold)
            
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please try again.")
