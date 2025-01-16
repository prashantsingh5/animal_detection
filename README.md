# animal_detection

# Animal Detection and Classification System 🐾

A web-based application for real-time animal detection and classification using YOLOv8 and Flask. The system can process both images and videos, distinguishing between pet animals, wild animals, and farm animals.

## 🌟 Features

- **Real-time Detection**: Process images and videos for animal detection
- **Multi-class Classification**: Identifies various animal types:
  - Pet Animals (dogs, cats, horses)
  - Wild Animals (birds, elephants, bears, zebras, giraffes)
  - Farm Animals (sheep, cows)
- **Queue System**: Manage multiple file uploads with a processing queue
- **Training Interface**: Built-in documentation for training custom models
- **Interactive UI**: User-friendly interface with drag-and-drop functionality
- **Visual Feedback**: Color-coded bounding boxes for different animal types

## 🔧 Technical Stack

- **Backend**: Python, Flask
- **ML Framework**: YOLOv8 (Ultralytics)
- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: OpenCV
- **Video Processing**: FFmpeg (optional)

## 📋 Prerequisites

```bash
- Python 3.8+
- pip
- FFmpeg (optional, for enhanced video processing)
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/animal-detection.git
cd animal-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up the project structure:
```
animal-detection/
├── app.py
├── static/
│   └── images/
│       └── dog.png
├── templates/
│   └── index.html
└── README.md
```

## 💻 Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select file type (image/video) and upload your file
4. Use the queue system to process multiple files
5. View results with color-coded classifications

## 🎯 Supported Animals

| Category | Animals |
|----------|---------|
| Pet | Dogs, Cats, Horses |
| Wild | Birds, Elephants, Bears, Zebras, Giraffes |
| Farm | Sheep, Cows |

## 🔄 Custom Model Training

1. Access the training documentation through the "Train" button
2. Follow the step-by-step guide for:
   - Dataset preparation
   - CVAT annotation
   - Model training
   - Integration

## 🎨 Color Coding

- 🟢 Green: Pet Animals
- 🔴 Red: Wild Animals
- 🟠 Orange: Farm Animals
- 🟣 Purple: People

## ⚙️ Configuration

Modify the `SUPPORTED_ANIMALS` dictionary in `app.py` to add or remove animal classes:

```python
SUPPORTED_ANIMALS = {
    'person': 'person',
    'bird': 'wild',
    'cat': 'pet',
    # Add more animals here
}
```

## ⚙️ Advanced Features

- **Confidence Threshold**: Adjustable detection confidence (default: 0.3)
- **Queue Management**: Process multiple files sequentially
- **Error Handling**: Robust error management for file processing
- **Responsive Design**: Works on different screen sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Flask community
- OpenCV contributors
- CVAT for annotation tools

## 📧 Contact

Your Name - [prashantsingha96@gmail.com](mailto:prashantsingha96@gmail.com)
Project Link: [https://github.com/yourusername/animal-detection](https://github.com/yourusername/animal-detection)
