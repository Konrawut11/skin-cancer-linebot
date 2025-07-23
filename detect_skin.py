import os
import logging
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageStat, ExifTags
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    predicted_class: str
    confidence: float
    confidence_level: ConfidenceLevel
    raw_probabilities: Dict[str, float]
    image_quality_score: float
    recommendations: list

@dataclass
class ImageQuality:
    """Data class for image quality metrics"""
    brightness: float
    contrast: float
    sharpness: float
    resolution: Tuple[int, int]
    is_acceptable: bool
    issues: list

class SkinDiseaseDetector:
    """
    Enhanced skin disease detection system with improved error handling,
    model management, and result interpretation.
    """
    
    def __init__(self, model_path: str = "models/best.pt", config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Default configuration
        self.config = {
            "input_size": (224, 224),
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "min_resolution": (100, 100),
            "confidence_thresholds": {
                "high": 0.8,
                "medium": 0.5,
                "low": 0.0
            },
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        }
        
        # Load custom config if provided
        if self.config_path and self.config_path.exists():
            self._load_config()
        
        # Class definitions with detailed information
        self.class_info = {
            "melanoma": {
                "name": "Melanoma",
                "thai_name": "เมลาโนมา",
                "severity": "high",
                "description": "มะเร็งผิวหนังชนิดร้ายแรง",
                "urgent": True
            },
            "nevus": {
                "name": "Nevus (Mole)",
                "thai_name": "ไฝ",
                "severity": "low",
                "description": "ไฝธรรมดา",
                "urgent": False
            },
            "keratosis": {
                "name": "Keratosis",
                "thai_name": "เคราโตซิส",
                "severity": "medium",
                "description": "ผิวหนังหนาตัวผิดปกติ",
                "urgent": False
            },
            "normal": {
                "name": "Normal Skin",
                "thai_name": "ผิวหนังปกติ",
                "severity": "none",
                "description": "ผิวหนังปกติ",
                "urgent": False
            }
        }
        
        # Model components
        self._model = None
        self._device = None
        self._transform = None
        self._is_loaded = False
        
    def _load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
    
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        return device
    
    def _create_transforms(self) -> transforms.Compose:
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(self.config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config["normalization"]["mean"],
                std=self.config["normalization"]["std"]
            )
        ])
    
    def load_model(self) -> bool:
        """Load the PyTorch model with comprehensive error handling"""
        try:
            # Validate model file
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Setup device and transforms
            self._device = self._setup_device()
            self._transform = self._create_transforms()
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self._device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self._model = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Assuming you need to recreate the model architecture
                # This would need to be customized based on your model
                logger.error("State dict format detected but model architecture not provided")
                return False
            else:
                self._model = checkpoint
            
            # Set model to evaluation mode
            self._model.eval()
            self._model.to(self._device)
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._is_loaded = False
            return False
    
    def _assess_image_quality(self, image: Image.Image) -> ImageQuality:
        """Comprehensive image quality assessment"""
        width, height = image.size
        img_array = np.array(image)
        
        # Calculate quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Calculate sharpness using Laplacian variance
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        laplacian_var = np.var(np.gradient(gray))
        sharpness = laplacian_var
        
        # Assess quality issues
        issues = []
        min_width, min_height = self.config["min_resolution"]
        
        if width < min_width or height < min_height:
            issues.append(f"รูปภาพมีขนาดเล็กเกินไป ({width}x{height})")
        
        if brightness < 50:
            issues.append("ภาพมืดเกินไป")
        elif brightness > 200:
            issues.append("ภาพสว่างเกินไป")
        
        if contrast < 20:
            issues.append("คอนทราสต์ต่ำ")
        
        if sharpness < 100:
            issues.append("ภาพไม่คมชัด")
        
        # Overall quality score (0-1)
        quality_score = min(1.0, (
            (min(brightness, 150) / 150) * 0.3 +
            (min(contrast, 100) / 100) * 0.3 +
            (min(sharpness, 500) / 500) * 0.4
        ))
        
        is_acceptable = len(issues) == 0 and quality_score > 0.5
        
        return ImageQuality(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            resolution=(width, height),
            is_acceptable=is_acceptable,
            issues=issues
        )
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on threshold"""
        thresholds = self.config["confidence_thresholds"]
        if confidence >= thresholds["high"]:
            return ConfidenceLevel.HIGH
        elif confidence >= thresholds["medium"]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_recommendations(self, prediction: str, confidence: float, image_quality: ImageQuality) -> list:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Quality-based recommendations
        if not image_quality.is_acceptable:
            recommendations.append("📸 ถ่ายรูปใหม่ในสภาพแสงที่ดีกว่า เพื่อผลการตรวจที่แม่นยำ")
        
        # Prediction-based recommendations
        class_key = prediction.lower().replace(" ", "_")
        if class_key in self.class_info:
            class_data = self.class_info[class_key]
            
            if class_data["urgent"]:
                recommendations.append("🚨 ควรปรึกษาแพทย์ผิวหนังโดยด่วน")
            elif class_data["severity"] == "medium":
                recommendations.append("👨‍⚕️ แนะนำให้ปรึกษาแพทย์ผิวหนัง")
            
        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("🔍 ผลการตรวจมีความไม่แน่นอน ควรปรึกษาผู้เชี่ยวชาญ")
        
        # General recommendations
        recommendations.extend([
            "🧴 ใช้ครีมกันแดด SPF 30+ ทุกวัน",
            "👀 ตรวจสอบผิวหนังด้วยตนเองเป็นประจำ",
            "🏥 ตรวจสุขภาพประจำปีกับแพทย์"
        ])
        
        return recommendations
    
    def predict(self, image: Image.Image) -> Optional[PredictionResult]:
        """Make prediction with comprehensive error handling"""
        if not self._is_loaded:
            if not self.load_model():
                return None
        
        try:
            # Preprocess image
            input_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self._model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get predictions for all classes
            raw_probs = {}
            for i, (key, info) in enumerate(self.class_info.items()):
                if i < len(probabilities):
                    raw_probs[info["thai_name"]] = float(probabilities[i])
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Map to class name
            class_keys = list(self.class_info.keys())
            if predicted_idx.item() < len(class_keys):
                predicted_class = self.class_info[class_keys[predicted_idx.item()]]["thai_name"]
            else:
                predicted_class = f"Unknown Class {predicted_idx.item()}"
            
            # Assess image quality
            image_quality = self._assess_image_quality(image)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                predicted_class, confidence.item(), image_quality
            )
            
            return PredictionResult(
                predicted_class=predicted_class,
                confidence=float(confidence),
                confidence_level=self._get_confidence_level(float(confidence)),
                raw_probabilities=raw_probs,
                image_quality_score=image_quality.is_acceptable,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def detect_skin_disease(self, image_path: str) -> str:
        """
        Main detection function with enhanced Thai language support
        """
        try:
            # Validate file
            image_path = Path(image_path)
            if not image_path.exists():
                return "❌ ไม่พบไฟล์รูปภาพที่ระบุ"
            
            # Check file format
            if image_path.suffix.lower() not in self.config["supported_formats"]:
                return f"❌ รูปแบบไฟล์ไม่รองรับ กรุณาใช้: {', '.join(self.config['supported_formats'])}"
            
            # Open and process image
            with Image.open(image_path) as image:
                logger.info(f"Processing image: {image.size}, mode: {image.mode}")
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Assess image quality
                image_quality = self._assess_image_quality(image)
                
                # Make prediction
                result = self.predict(image)
                
                # Generate comprehensive report
                return self._generate_report(result, image_quality)
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return self._generate_error_report(str(e))
    
    def _generate_report(self, result: Optional[PredictionResult], image_quality: ImageQuality) -> str:
        """Generate comprehensive Thai language report"""
        report = "🔍 ผลการตรวจสอบด้วย AI:\n\n"
        
        if result:
            # Prediction results
            confidence_emoji = {"high": "🎯", "medium": "⚠️", "low": "❓"}
            emoji = confidence_emoji.get(result.confidence_level.value, "🤔")
            
            report += f"""{emoji} การวินิจฉัยเบื้องต้น:
• ผลตรวจ: {result.predicted_class}
• ความเชื่อมั่น: {result.confidence*100:.1f}% ({result.confidence_level.value.upper()})

"""
            
            # Detailed probabilities
            if result.raw_probabilities:
                report += "📊 ความน่าจะเป็นทั้งหมด:\n"
                for class_name, prob in sorted(result.raw_probabilities.items(), 
                                             key=lambda x: x[1], reverse=True):
                    report += f"• {class_name}: {prob*100:.1f}%\n"
                report += "\n"
            
            # Recommendations
            if result.recommendations:
                report += "💡 คำแนะนำ:\n"
                for rec in result.recommendations:
                    report += f"• {rec}\n"
                report += "\n"
        else:
            report += "❌ ไม่สามารถประมวลผลด้วย AI ได้\n\n"
        
        # Image quality assessment
        report += "📊 การประเมินคุณภาพภาพ:\n"
        if image_quality.is_acceptable:
            report += "• คุณภาพโดยรวม: ดี ✅\n"
        else:
            report += "• คุณภาพโดยรวม: ต้องปรับปรุง ⚠️\n"
            for issue in image_quality.issues:
                report += f"• {issue}\n"
        
        report += f"• ความละเอียด: {image_quality.resolution[0]}x{image_quality.resolution[1]} พิกเซล\n"
        report += f"• ความสว่าง: {image_quality.brightness:.0f}/255\n"
        report += f"• คอนทราสต์: {image_quality.contrast:.0f}\n\n"
        
        # Disclaimer
        report += """⚠️ ข้อจำกัดสำคัญ:
• การตรวจสอบนี้เป็นเพียงเครื่องมือช่วยการตรวจเบื้องต้น
• ไม่สามารถททนคำวินิจฉัยของแพทย์ได้
• หากมีความกังวล กรุณาปรึกษาแพทย์ผิวหนังโดยตรง
• การรักษาใดๆ ควรอยู่ภายใต้การดูแลของผู้เชี่ยวชาญ"""
        
        return report
    
    def _generate_error_report(self, error_msg: str) -> str:
        """Generate error report in Thai"""
        return f"""❌ เกิดข้อผิดพลาดในการประมวลผล

รายละเอียด: {error_msg}

กรุณาตรวจสอบ:
• ไฟล์เป็นรูปภาพที่ถูกต้อง ({', '.join(self.config['supported_formats'])})
• ขนาดไฟล์ไม่เกิน 10 MB
• รูปภาพไม่เสียหาย
• ไฟล์ model อยู่ในตำแหน่งที่ถูกต้อง

หากปัญหายังคงอยู่ กรุณาติดต่อผู้ดูแลระบบ"""

# Convenience functions for backward compatibility
def load_model(model_path: str = "models/best.pt") -> bool:
    """Load model using the enhanced detector"""
    global _detector
    _detector = SkinDiseaseDetector(model_path)
    return _detector.load_model()

def detect_skin_disease(image_path: str) -> str:
    """Detect skin disease using the enhanced detector"""
    global _detector
    if '_detector' not in globals():
        _detector = SkinDiseaseDetector()
    return _detector.detect_skin_disease(image_path)

def initialize_model(model_path: str = "models/best.pt") -> bool:
    """Initialize the model at application startup"""
    return load_model(model_path)

# Global detector instance for backward compatibility
_detector = None
