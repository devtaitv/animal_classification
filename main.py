import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import time
from PIL import Image, ImageTk


class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Conv Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AnimalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)

        return image, label


class AnimalClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân lớp ảnh động vật")

        # Data storage
        self.image_data = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = 0

        # Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.svm_model = SVC(kernel='rbf', probability=True)
        self.cnn_model = None

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create left and right frames for 2-column layout
        left_frame = ttk.Frame(main_container, padding="5")
        left_frame.grid(row=0, column=0, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.Frame(main_container, padding="5")
        right_frame.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        main_container.columnconfigure(0, weight=3)  # Left column takes 3/5 of space
        main_container.columnconfigure(1, weight=2)  # Right column takes 2/5 of space

        # ===== LEFT FRAME - Training Controls =====
        # Data loading section
        data_frame = ttk.LabelFrame(left_frame, text="Dữ liệu huấn luyện", padding="5")
        data_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        data_frame.columnconfigure(1, weight=1)

        ttk.Button(data_frame, text="Tải thư mục ảnh", command=self.load_data).grid(row=0, column=0, padx=5)
        self.info_label = ttk.Label(data_frame, text="Chưa có dữ liệu")
        self.info_label.grid(row=0, column=1, padx=5, sticky=(tk.W))

        # Training parameters
        params_frame = ttk.LabelFrame(left_frame, text="Tham số huấn luyện", padding="5")
        params_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)

        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, padx=5)
        self.epochs_var = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, sticky=(tk.W))

        # Batch size
        ttk.Label(params_frame, text="Batch size:").grid(row=0, column=2, padx=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, sticky=(tk.W))

        # Algorithm selection
        algo_frame = ttk.LabelFrame(left_frame, text="Lựa chọn thuật toán", padding="5")
        algo_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.algo_var = tk.StringVar(value="cnn")
        algorithms = [("KNN", "knn"), ("SVM", "svm"), ("CNN", "cnn")]
        for i, (text, value) in enumerate(algorithms):
            ttk.Radiobutton(algo_frame, text=text, value=value,
                            variable=self.algo_var).grid(row=0, column=i, padx=15)

        # Training button with progress bar
        train_frame = ttk.Frame(left_frame)
        train_frame.grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(train_frame, text="Huấn luyện và đánh giá",
                   command=self.train_and_evaluate).grid(row=0, column=0, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, length=200,
                                            mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E))
        train_frame.columnconfigure(1, weight=1)

        # Training results
        results_frame = ttk.LabelFrame(left_frame, text="Kết quả huấn luyện", padding="5")
        results_frame.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Add scrollbar to results text
        results_scroll = ttk.Scrollbar(results_frame)
        results_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.results_text = tk.Text(results_frame, height=10, width=50, wrap=tk.WORD,
                                    yscrollcommand=results_scroll.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scroll.config(command=self.results_text.yview)

        # ===== RIGHT FRAME - Prediction Controls =====
        # Prediction section
        predict_frame = ttk.LabelFrame(right_frame, text="Dự đoán ảnh mới", padding="5")
        predict_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        predict_frame.columnconfigure(0, weight=1)

        # Image loading button
        ttk.Button(predict_frame, text="Tải ảnh để dự đoán",
                   command=self.load_prediction_image).grid(row=0, column=0, pady=5)

        # Image display frame
        image_display_frame = ttk.LabelFrame(predict_frame, text="Ảnh đang xử lý", padding="5")
        image_display_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.image_label = ttk.Label(image_display_frame)
        self.image_label.grid(row=0, column=0, pady=5)

        # Prediction results frame
        prediction_results_frame = ttk.LabelFrame(predict_frame, text="Kết quả dự đoán", padding="5")
        prediction_results_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.prediction_label = ttk.Label(prediction_results_frame, text="", justify=tk.LEFT)
        self.prediction_label.grid(row=0, column=0, pady=5, sticky=(tk.W))

        # Configure weight for frames
        left_frame.rowconfigure(4, weight=1)  # Make results expand vertically
        right_frame.rowconfigure(0, weight=1)  # Make prediction frame expand vertically

    def load_prediction_image(self):
        """Load and predict a single image"""
        if not hasattr(self, 'idx_to_label') or not self.idx_to_label:
            messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình trước")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])

        if not file_path:
            return

        try:
            # Load and preprocess image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display image
            display_size = (300, 300)  # Larger display size
            display_img = cv2.resize(img, display_size)
            photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Prepare image for prediction
            img = cv2.resize(img, (64, 64))

            algorithm = self.algo_var.get()

            if algorithm == "knn":
                if not hasattr(self, 'knn_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình KNN trước")
                    return

                img_flat = img.reshape(1, -1)
                prediction = self.knn_model.predict(img_flat)[0]
                probabilities = self.knn_model.predict_proba(img_flat)[0]

            elif algorithm == "svm":
                if not hasattr(self, 'svm_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình SVM trước")
                    return

                img_flat = img.reshape(1, -1)
                prediction = self.svm_model.predict(img_flat)[0]
                probabilities = self.svm_model.predict_proba(img_flat)[0]

            elif algorithm == "cnn":
                if not hasattr(self, 'cnn_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình CNN trước")
                    return

                self.cnn_model.eval()
                img_tensor = self.transform(Image.fromarray(img)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.cnn_model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    prediction = int(torch.argmax(outputs).item())

            # Display results with improved formatting
            predicted_class = self.idx_to_label[prediction]
            top_3_idx = np.argsort(probabilities)[-3:][::-1]

            result_text = f"Kết quả dự đoán:\n"
            result_text += f"\nLớp dự đoán: {predicted_class}\n"
            result_text += f"\nXác suất các lớp cao nhất:\n"

            for idx in top_3_idx:
                result_text += f"{self.idx_to_label[idx]}: {probabilities[idx]:.2%}\n"

            self.prediction_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")
    def load_data(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        self.image_data = []
        self.labels = []

        class_names = sorted([d for d in os.listdir(folder_path)
                              if os.path.isdir(os.path.join(folder_path, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(class_names)

        for class_name in class_names:
            class_path = os.path.join(folder_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (64, 64))

                    self.image_data.append(img)
                    self.labels.append(self.label_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
                    continue

        self.image_data = np.array(self.image_data)
        self.labels = np.array(self.labels)

        self.info_label.config(
            text=f"Đã tải {len(self.image_data)} ảnh từ {self.num_classes} lớp")

    def train_cnn(self, X_train, X_test, y_train, y_test):
        self.cnn_model = AnimalCNN(self.num_classes).to(self.device)

        train_dataset = AnimalDataset(X_train, y_train, transform=self.transform)
        test_dataset = AnimalDataset(X_test, y_test, transform=self.transform)

        batch_size = int(self.batch_size_var.get())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        num_epochs = int(self.epochs_var.get())
        best_acc = 0

        for epoch in range(num_epochs):
            # Training phase
            self.cnn_model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.cnn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            epoch_loss = running_loss / len(train_loader)

            # Validation phase
            self.cnn_model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.cnn_model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            val_loss = val_loss / len(test_loader)

            # Update scheduler
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.cnn_model.state_dict(), 'best_model.pth')

            self.results_text.insert(tk.END,
                                     f"Epoch {epoch + 1}/{num_epochs}\n"
                                     f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2%}\n"
                                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}\n"
                                     f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")
            self.results_text.see(tk.END)
            self.root.update()

        # Load best model
        self.cnn_model.load_state_dict(torch.load('best_model.pth'))
        return best_acc

    def train_and_evaluate(self):
        if len(self.image_data) == 0:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            self.image_data, self.labels, test_size=0.2, random_state=42
        )

        algorithm = self.algo_var.get()
        self.results_text.delete(1.0, tk.END)

        if algorithm == "knn":
            start_time = time.time()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

            self.knn_model.fit(X_train_flat, y_train)
            y_pred = self.knn_model.predict(X_test_flat)
            probs = self.knn_model.predict_proba(X_test_flat)
            knn_time = time.time() - start_time
            knn_acc = np.mean(y_pred == y_test)

            self.results_text.insert(tk.END,
                                     f"KNN:\n- Độ chính xác: {knn_acc:.2%}\n"
                                     f"- Thời gian: {knn_time:.2f}s\n\n"
                                     "Chi tiết dự đoán:\n")

            # Show detailed predictions for each class
            for i in range(self.num_classes):
                class_mask = y_test == i
                class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
                self.results_text.insert(tk.END,
                                         f"- Lớp {self.idx_to_label[i]}: {class_acc:.2%}\n")

        elif algorithm == "svm":
            start_time = time.time()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

            self.svm_model.fit(X_train_flat, y_train)
            y_pred = self.svm_model.predict(X_test_flat)
            probs = self.svm_model.predict_proba(X_test_flat)
            svm_time = time.time() - start_time
            svm_acc = np.mean(y_pred == y_test)

            self.results_text.insert(tk.END,
                                     f"SVM:\n- Độ chính xác: {svm_acc:.2%}\n"
                                     f"- Thời gian: {svm_time:.2f}s\n\n"
                                     "Chi tiết dự đoán:\n")

            # Show detailed predictions for each class
            for i in range(self.num_classes):
                class_mask = y_test == i
                class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
                self.results_text.insert(tk.END,
                                         f"- Lớp {self.idx_to_label[i]}: {class_acc:.2%}\n")

        elif algorithm == "cnn":
            start_time = time.time()
            cnn_acc = self.train_cnn(X_train, X_test, y_train, y_test)
            cnn_time = time.time() - start_time

            # Evaluate per-class accuracy
            self.cnn_model.eval()
            test_dataset = AnimalDataset(X_test, y_test, transform=self.transform)
            test_loader = DataLoader(test_dataset, batch_size=32)

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.cnn_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            self.results_text.insert(tk.END,
                                     f"\nKết quả cuối cùng:\nCNN:\n"
                                     f"- Độ chính xác tổng thể: {cnn_acc:.2%}\n"
                                     f"- Thời gian: {cnn_time:.2f}s\n\n"
                                     "Chi tiết dự đoán:\n")

            for i in range(self.num_classes):
                class_mask = all_labels == i
                class_acc = np.mean(all_preds[class_mask] == all_labels[class_mask])
                self.results_text.insert(tk.END,
                                         f"- Lớp {self.idx_to_label[i]}: {class_acc:.2%}\n")

    def load_prediction_image(self):
        """Load and predict a single image"""
        if not hasattr(self, 'idx_to_label') or not self.idx_to_label:
            messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình trước")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])

        if not file_path:
            return

        try:
            # Load and preprocess image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display image
            display_img = cv2.resize(img, (200, 200))
            photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Prepare image for prediction
            img = cv2.resize(img, (64, 64))

            algorithm = self.algo_var.get()

            if algorithm == "knn":
                if not hasattr(self, 'knn_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình KNN trước")
                    return

                img_flat = img.reshape(1, -1)
                prediction = self.knn_model.predict(img_flat)[0]
                probabilities = self.knn_model.predict_proba(img_flat)[0]

            elif algorithm == "svm":
                if not hasattr(self, 'svm_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình SVM trước")
                    return

                img_flat = img.reshape(1, -1)
                prediction = self.svm_model.predict(img_flat)[0]
                probabilities = self.svm_model.predict_proba(img_flat)[0]

            elif algorithm == "cnn":
                if not hasattr(self, 'cnn_model'):
                    messagebox.showerror("Lỗi", "Vui lòng huấn luyện mô hình CNN trước")
                    return

                self.cnn_model.eval()
                img_tensor = self.transform(Image.fromarray(img)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.cnn_model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    prediction = int(torch.argmax(outputs).item())

            # Display results
            predicted_class = self.idx_to_label[prediction]
            top_3_idx = np.argsort(probabilities)[-3:][::-1]

            result_text = f"Dự đoán: {predicted_class}\n\nXác suất top 3:\n"
            for idx in top_3_idx:
                result_text += f"{self.idx_to_label[idx]}: {probabilities[idx]:.2%}\n"

            self.prediction_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalClassificationApp(root)
    root.mainloop()