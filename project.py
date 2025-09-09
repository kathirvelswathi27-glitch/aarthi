import pytesseract
import cv2
# AI Medical Prescription Verification System
# Leveraging IBM Watson and Hugging Face Models

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import re

# IBM Watson imports
from ibm_watson import NaturalLanguageUnderstandingV1, DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, pipeline,
    BertTokenizer, BertForSequenceClassification
)
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Additional libraries
import spacy
import requests
from PIL import Image
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrescriptionVerificationSystem:
    """
    AI-powered prescription verification system using IBM Watson and Hugging Face models
    """
    
    def __init__(self, config: Dict):
        """Initialize the verification system with configuration"""
        self.config = config
        self.setup_watson_services()
        self.setup_huggingface_models()
        self.drug_database = self.load_drug_database()
        self.interaction_database = self.load_interaction_database()
        
    def setup_watson_services(self):
        """Setup IBM Watson services"""
        try:
            # Watson Natural Language Understanding
            nlu_authenticator = IAMAuthenticator(self.config['watson']['nlu_api_key'])
            self.nlu = NaturalLanguageUnderstandingV1(
                version='2022-04-07',
                authenticator=nlu_authenticator
            )
            self.nlu.set_service_url(self.config['watson']['nlu_url'])
            
            # Watson Discovery (for drug knowledge base)
            discovery_authenticator = IAMAuthenticator(self.config['watson']['discovery_api_key'])
            self.discovery = DiscoveryV2(
                version='2020-08-30',
                authenticator=discovery_authenticator
            )
            self.discovery.set_service_url(self.config['watson']['discovery_url'])
            
            logger.info("Watson services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Watson services: {str(e)}")
    
    def setup_huggingface_models(self):
        """Setup Hugging Face models for various tasks"""
        try:
            # Medical NER model for entity extraction
            self.medical_ner = pipeline(
                "token-classification",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            
            # Medical text classification model
            self.medical_classifier = pipeline(
                "text-classification",
                model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            )
            
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Drug interaction classification model
            self.interaction_model = pipeline(
                "text-classification",
                model="emilyalsentzer/Bio_ClinicalBERT"
            )
            
            # OCR pipeline for prescription image processing
            self.ocr_model = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-printed"
            )
            
            logger.info("Hugging Face models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Hugging Face models: {str(e)}")
    
    def load_drug_database(self) -> pd.DataFrame:
        """Load comprehensive drug database"""
        # In production, this would connect to a real pharmaceutical database
        sample_drugs = {
            'drug_name': ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Lisinopril', 'Metformin'],
            'generic_name': ['Acetylsalicylic acid', 'Ibuprofen', 'Paracetamol', 'Lisinopril', 'Metformin'],
            'drug_class': ['NSAID', 'NSAID', 'Analgesic', 'ACE Inhibitor', 'Biguanide'],
            'max_daily_dose': [4000, 3200, 4000, 40, 2000],
            'contraindications': [
                'Bleeding disorders,Peptic ulcer',
                'Kidney disease,Heart failure',
                'Liver disease',
                'Pregnancy,Kidney disease',
                'Kidney disease,Liver disease'
            ],
            'common_interactions': [
                'Warfarin,Heparin',
                'ACE inhibitors,Diuretics',
                'Warfarin,Alcohol',
                'NSAIDs,Diuretics',
                'Contrast agents,Diuretics'
            ]
        }
        return pd.DataFrame(sample_drugs)
    
    def load_interaction_database(self) -> pd.DataFrame:
        """Load drug interaction database"""
        sample_interactions = {
            'drug1': ['Aspirin', 'Warfarin', 'Metformin', 'Lisinopril'],
            'drug2': ['Warfarin', 'Aspirin', 'Contrast agent', 'Ibuprofen'],
            'interaction_type': ['Major', 'Major', 'Moderate', 'Moderate'],
            'severity_score': [9, 9, 6, 5],
            'description': [
                'Increased bleeding risk',
                'Increased bleeding risk',
                'Risk of lactic acidosis',
                'Reduced antihypertensive effect'
            ]
        }
        return pd.DataFrame(sample_interactions)
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from prescription image using OCR"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR results
            denoised = cv2.fastNlMeansDenoising(gray)
            _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(threshold)
            
            # Also use Hugging Face TrOCR model
            pil_image = Image.open(image_path)
            hf_text = self.ocr_model(pil_image)[0]['generated_text']
            
            # Combine results (simple concatenation, could be improved)
            combined_text = f"{extracted_text}\n{hf_text}"
            
            logger.info("Text extracted successfully from image")
            return combined_text
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def extract_medical_entities(self, text: str) -> Dict:
        """Extract medical entities using both Watson NLU and Hugging Face NER"""
        entities = {
            'drugs': [],
            'dosages': [],
            'frequencies': [],
            'conditions': [],
            'doctors': [],
            'patients': []
        }
        
        try:
            # Watson NLU entity extraction
            watson_response = self.nlu.analyze(
                text=text,
                features=Features(
                    entities=EntitiesOptions(emotion=False, sentiment=False, limit=20),
                    keywords=KeywordsOptions(emotion=False, sentiment=False, limit=20),
                    concepts=ConceptsOptions(limit=10)
                )
            ).get_result()
            
            # Process Watson entities
            for entity in watson_response.get('entities', []):
                entity_type = entity['type'].lower()
                if 'drug' in entity_type or 'medication' in entity_type:
                    entities['drugs'].append({
                        'text': entity['text'],
                        'confidence': entity['confidence'],
                        'source': 'watson'
                    })
            
            # Hugging Face medical NER
            hf_entities = self.medical_ner(text)
            
            for entity in hf_entities:
                label = entity['entity_group'].lower()
                if 'drug' in label or 'chemical' in label:
                    entities['drugs'].append({
                        'text': entity['word'],
                        'confidence': entity['score'],
                        'source': 'huggingface'
                    })
                elif 'disease' in label or 'symptom' in label:
                    entities['conditions'].append({
                        'text': entity['word'],
                        'confidence': entity['score'],
                        'source': 'huggingface'
                    })
            
            # Extract dosages and frequencies using regex patterns
            dosage_patterns = [
                r'\d+\s*mg',
                r'\d+\s*g',
                r'\d+\s*ml',
                r'\d+\s*units?',
                r'\d+\s*tablets?'
            ]
            
            frequency_patterns = [
                r'once\s+daily',
                r'twice\s+daily',
                r'three\s+times?\s+daily',
                r'every\s+\d+\s+hours?',
                r'as\s+needed',
                r'PRN'
            ]
            
            for pattern in dosage_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['dosages'].append({
                        'text': match.group(),
                        'confidence': 0.8,
                        'source': 'regex'
                    })
            
            for pattern in frequency_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['frequencies'].append({
                        'text': match.group(),
                        'confidence': 0.8,
                        'source': 'regex'
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {str(e)}")
            return entities
    
    def verify_drug_authenticity(self, drug_name: str) -> Dict:
        """Verify if a drug exists in the database and get its information"""
        try:
            # Search in local drug database
            drug_matches = self.drug_database[
                self.drug_database['drug_name'].str.contains(drug_name, case=False, na=False) |
                self.drug_database['generic_name'].str.contains(drug_name, case=False, na=False)
            ]
            
            if not drug_matches.empty:
                drug_info = drug_matches.iloc[0].to_dict()
                return {
                    'verified': True,
                    'drug_info': drug_info,
                    'confidence': 0.95,
                    'source': 'local_database'
                }
            
            # If not found locally, search using Watson Discovery
            discovery_query = f"drug_name:{drug_name} OR generic_name:{drug_name}"
            discovery_response = self.discovery.query(
                project_id=self.config['watson']['discovery_project_id'],
                query=discovery_query
            ).get_result()
            
            if discovery_response['matching_results'] > 0:
                return {
                    'verified': True,
                    'drug_info': discovery_response['results'][0],
                    'confidence': 0.85,
                    'source': 'watson_discovery'
                }
            
            return {
                'verified': False,
                'drug_info': None,
                'confidence': 0.0,
                'source': 'not_found'
            }
            
        except Exception as e:
            logger.error(f"Error verifying drug authenticity: {str(e)}")
            return {
                'verified': False,
                'drug_info': None,
                'confidence': 0.0,
                'source': 'error'
            }
    
    def check_drug_interactions(self, drugs: List[str]) -> List[Dict]:
        """Check for potential drug interactions"""
        interactions = []
        
        try:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    drug1, drug2 = drugs[i], drugs[j]
                    
                    # Check in interaction database
                    interaction_match = self.interaction_database[
                        ((self.interaction_database['drug1'].str.contains(drug1, case=False)) &
                         (self.interaction_database['drug2'].str.contains(drug2, case=False))) |
                        ((self.interaction_database['drug1'].str.contains(drug2, case=False)) &
                         (self.interaction_database['drug2'].str.contains(drug1, case=False)))
                    ]
                    
                    if not interaction_match.empty:
                        interaction_info = interaction_match.iloc[0]
                        interactions.append({
                            'drug1': drug1,
                            'drug2': drug2,
                            'interaction_type': interaction_info['interaction_type'],
                            'severity_score': interaction_info['severity_score'],
                            'description': interaction_info['description'],
                            'source': 'database'
                        })
                    else:
                        # Use Hugging Face model to predict interaction
                        interaction_text = f"Drug interaction between {drug1} and {drug2}"
                        prediction = self.interaction_model(interaction_text)
                        
                        if prediction[0]['score'] > 0.7:  # High confidence threshold
                            interactions.append({
                                'drug1': drug1,
                                'drug2': drug2,
                                'interaction_type': 'Predicted',
                                'severity_score': prediction[0]['score'] * 10,
                                'description': f"Potential interaction predicted by AI model",
                                'source': 'ai_prediction'
                            })
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error checking drug interactions: {str(e)}")
            return interactions
    
    def validate_dosage(self, drug_name: str, dosage: str, frequency: str) -> Dict:
        """Validate if the prescribed dosage is within safe limits"""
        try:
            # Extract numeric dosage
            dosage_match = re.search(r'(\d+)', dosage)
            if not dosage_match:
                return {
                    'valid': False,
                    'reason': 'Cannot extract numeric dosage',
                    'confidence': 0.9
                }
            
            prescribed_dose = float(dosage_match.group(1))
            
            # Get drug information
            drug_info = self.verify_drug_authenticity(drug_name)
            if not drug_info['verified']:
                return {
                    'valid': False,
                    'reason': 'Drug not found in database',
                    'confidence': 0.8
                }
            
            max_daily_dose = drug_info['drug_info']['max_daily_dose']
            
            # Calculate daily dose based on frequency
            frequency_multipliers = {
                'once daily': 1,
                'twice daily': 2,
                'three times daily': 3,
                'four times daily': 4,
                'every 6 hours': 4,
                'every 8 hours': 3,
                'every 12 hours': 2
            }
            
            multiplier = 1
            for freq_pattern, mult in frequency_multipliers.items():
                if freq_pattern.lower() in frequency.lower():
                    multiplier = mult
                    break
            
            daily_dose = prescribed_dose * multiplier
            
            if daily_dose <= max_daily_dose:
                return {
                    'valid': True,
                    'daily_dose': daily_dose,
                    'max_daily_dose': max_daily_dose,
                    'safety_margin': (max_daily_dose - daily_dose) / max_daily_dose,
                    'confidence': 0.9
                }
            else:
                return {
                    'valid': False,
                    'reason': f'Daily dose ({daily_dose}mg) exceeds maximum safe dose ({max_daily_dose}mg)',
                    'daily_dose': daily_dose,
                    'max_daily_dose': max_daily_dose,
                    'confidence': 0.95
                }
            
        except Exception as e:
            logger.error(f"Error validating dosage: {str(e)}")
            return {
                'valid': False,
                'reason': f'Error in dosage validation: {str(e)}',
                'confidence': 0.0
            }
    
    def analyze_prescription_legitimacy(self, prescription_text: str) -> Dict:
        """Analyze overall prescription legitimacy using multiple AI models"""
        try:
            # Use Watson NLU to analyze sentiment and tone
            watson_analysis = self.nlu.analyze(
                text=prescription_text,
                features=Features(
                    entities=EntitiesOptions(),
                    keywords=KeywordsOptions(),
                    concepts=ConceptsOptions()
                )
            ).get_result()
            
            # Use Hugging Face for medical text classification
            hf_classification = self.medical_classifier(prescription_text)
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'rush|urgent|emergency',
                r'cash only|no insurance',
                r'do not contact doctor',
                r'multiple refills'
            ]
            
            suspicion_score = 0
            suspicious_flags = []
            
            for pattern in suspicious_patterns:
                if re.search(pattern, prescription_text, re.IGNORECASE):
                    suspicion_score += 1
                    suspicious_flags.append(pattern)
            
            # Calculate legitimacy score
            base_legitimacy = 0.8  # Base score for properly formatted prescription
            
            # Adjust based on Watson analysis
            if watson_analysis.get('sentiment', {}).get('score', 0) < -0.5:
                base_legitimacy -= 0.1
            
            # Adjust based on Hugging Face classification
            if hf_classification[0]['score'] < 0.7:
                base_legitimacy -= 0.1
            
            # Adjust based on suspicious patterns
            base_legitimacy -= (suspicion_score * 0.1)
            
            legitimacy_score = max(0.0, min(1.0, base_legitimacy))
            
            return {
                'legitimacy_score': legitimacy_score,
                'is_legitimate': legitimacy_score > 0.6,
                'suspicious_flags': suspicious_flags,
                'watson_analysis': watson_analysis,
                'hf_classification': hf_classification,
                'confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prescription legitimacy: {str(e)}")
            return {
                'legitimacy_score': 0.0,
                'is_legitimate': False,
                'suspicious_flags': ['Analysis error'],
                'confidence': 0.0
            }
    
    def generate_verification_report(self, prescription_data: Dict) -> Dict:
        """Generate comprehensive verification report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'prescription_id': prescription_data.get('id', 'unknown'),
            'verification_results': {
                'overall_status': 'PENDING',
                'legitimacy_score': 0.0,
                'drug_verifications': [],
                'dosage_validations': [],
                'interaction_warnings': [],
                'recommendations': []
            },
            'risk_assessment': {
                'risk_level': 'LOW',
                'risk_factors': [],
                'safety_score': 1.0
            }
        }
        
        try:
            # Extract entities from prescription text
            entities = self.extract_medical_entities(prescription_data['text'])
            
            # Verify each drug
            verified_drugs = []
            for drug_entity in entities['drugs']:
                drug_verification = self.verify_drug_authenticity(drug_entity['text'])
                report['verification_results']['drug_verifications'].append({
                    'drug_name': drug_entity['text'],
                    'verified': drug_verification['verified'],
                    'confidence': drug_verification['confidence']
                })
                
                if drug_verification['verified']:
                    verified_drugs.append(drug_entity['text'])
            
            # Check drug interactions
            if len(verified_drugs) > 1:
                interactions = self.check_drug_interactions(verified_drugs)
                report['verification_results']['interaction_warnings'] = interactions
                
                # Adjust risk level based on interactions
                major_interactions = [i for i in interactions if i['interaction_type'] == 'Major']
                if major_interactions:
                    report['risk_assessment']['risk_level'] = 'HIGH'
                    report['risk_assessment']['risk_factors'].append('Major drug interactions detected')
            
            # Validate dosages
            for i, drug_entity in enumerate(entities['drugs']):
                if i < len(entities['dosages']) and i < len(entities['frequencies']):
                    dosage_validation = self.validate_dosage(
                        drug_entity['text'],
                        entities['dosages'][i]['text'],
                        entities['frequencies'][i]['text']
                    )
                    report['verification_results']['dosage_validations'].append({
                        'drug': drug_entity['text'],
                        'prescribed_dosage': entities['dosages'][i]['text'],
                        'frequency': entities['frequencies'][i]['text'],
                        'valid': dosage_validation['valid'],
                        'reason': dosage_validation.get('reason', '')
                    })
                    
                    if not dosage_validation['valid']:
                        report['risk_assessment']['risk_level'] = 'HIGH'
                        report['risk_assessment']['risk_factors'].append(f"Invalid dosage for {drug_entity['text']}")
            
            # Analyze prescription legitimacy
            legitimacy_analysis = self.analyze_prescription_legitimacy(prescription_data['text'])
            report['verification_results']['legitimacy_score'] = legitimacy_analysis['legitimacy_score']
            
            if not legitimacy_analysis['is_legitimate']:
                report['risk_assessment']['risk_level'] = 'HIGH'
                report['risk_assessment']['risk_factors'].extend(legitimacy_analysis['suspicious_flags'])
            
            # Calculate overall safety score
            safety_factors = [
                legitimacy_analysis['legitimacy_score'],
                len([d for d in report['verification_results']['drug_verifications'] if d['verified']]) / max(len(entities['drugs']), 1),
                len([d for d in report['verification_results']['dosage_validations'] if d['valid']]) / max(len(entities['drugs']), 1)
            ]
            
            report['risk_assessment']['safety_score'] = np.mean(safety_factors)
            
            # Determine overall status
            if report['risk_assessment']['safety_score'] > 0.8 and report['risk_assessment']['risk_level'] != 'HIGH':
                report['verification_results']['overall_status'] = 'VERIFIED'
            elif report['risk_assessment']['safety_score'] > 0.6:
                report['verification_results']['overall_status'] = 'REQUIRES_REVIEW'
            else:
                report['verification_results']['overall_status'] = 'REJECTED'
            
            # Generate recommendations
            if report['verification_results']['interaction_warnings']:
                report['verification_results']['recommendations'].append("Consult with prescribing physician about potential drug interactions")
            
            if any(not d['verified'] for d in report['verification_results']['drug_verifications']):
                report['verification_results']['recommendations'].append("Verify unrecognized medications with pharmacy database")
            
            if report['verification_results']['legitimacy_score'] < 0.7:
                report['verification_results']['recommendations'].append("Manual review recommended due to legitimacy concerns")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating verification report: {str(e)}")
            report['verification_results']['overall_status'] = 'ERROR'
            report['verification_results']['recommendations'].append(f"System error: {str(e)}")
            return report

# Configuration class
class PrescriptionConfig:
    """Configuration management for the prescription verification system"""
    
    @staticmethod
    def get_config():
        return {
            'watson': {
                'nlu_api_key': os.getenv('WATSON_NLU_API_KEY', 'your_nlu_api_key'),
                'nlu_url': os.getenv('WATSON_NLU_URL', 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com'),
                'discovery_api_key': os.getenv('WATSON_DISCOVERY_API_KEY', 'your_discovery_api_key'),
                'discovery_url': os.getenv('WATSON_DISCOVERY_URL', 'https://api.us-south.discovery.watson.cloud.ibm.com'),
                'discovery_project_id': os.getenv('WATSON_DISCOVERY_PROJECT_ID', 'your_project_id')
            },
            'huggingface': {
                'api_token': os.getenv('HUGGINGFACE_API_TOKEN', 'your_hf_token')
            },
            'system': {
                'confidence_threshold': 0.7,
                'max_image_size_mb': 10,
                'supported_formats': ['jpg', 'jpeg', 'png', 'pdf']
            }
        }

# Usage example and testing
def main():
    """Main function to demonstrate the prescription verification system"""
    
    # Initialize configuration
    config = PrescriptionConfig.get_config()
    
    # Initialize the verification system
    verifier = PrescriptionVerificationSystem(config)
    
    # Sample prescription text (in real scenario, this would come from OCR)
    sample_prescription = """
    Dr. John Smith, MD
    Medical Center Hospital
    
    Patient: Jane Doe
    DOB: 01/15/1985
    Date: 12/01/2024
    
    Rx:
    1. Lisinopril 10mg - Take once daily for high blood pressure
    2. Metformin 500mg - Take twice daily with meals for diabetes
    3. Aspirin 81mg - Take once daily for cardiovascular protection
    
    Refills: 2
    
    Dr. John Smith
    License #: MD12345
    """
    
    # Sample prescription data
    prescription_data = {
        'id': 'RX_20241201_001',
        'text': sample_prescription,
        'image_path': None,  # Would contain path to prescription image
        'patient_id': 'PAT_001',
        'doctor_id': 'DOC_001'
    }
    
    print("=== AI Medical Prescription Verification System ===\n")
    
    # Generate verification report
    print("Generating verification report...")
    report = verifier.generate_verification_report(prescription_data)
    
    # Display results
    print("\n=== VERIFICATION REPORT ===")
    print(f"Prescription ID: {report['prescription_id']}")
    print(f"Overall Status: {report['verification_results']['overall_status']}")
    print(f"Legitimacy Score: {report['verification_results']['legitimacy_score']:.2f}")
    print(f"Safety Score: {report['risk_assessment']['safety_score']:.2f}")
    print(f"Risk Level: {report['risk_assessment']['risk_level']}")
    
    print("\n=== DRUG VERIFICATIONS ===")
    for drug_verification in report['verification_results']['drug_verifications']:
        status = "✓ VERIFIED" if drug_verification['verified'] else "✗ NOT VERIFIED"
        print(f"{drug_verification['drug_name']}: {status} (Confidence: {drug_verification['confidence']:.2f})")
    
    print("\n=== INTERACTION WARNINGS ===")
    if report['verification_results']['interaction_warnings']:
        for interaction in report['verification_results']['interaction_warnings']:
            print(f"⚠️  {interaction['drug1']} + {interaction['drug2']}: {interaction['description']} (Severity: {interaction['severity_score']}/10)")
    else:
        print("No significant drug interactions detected.")
    
    print("\n=== RECOMMENDATIONS ===")
    for recommendation in report['verification_results']['recommendations']:
        print(f"• {recommendation}")
    
    if not report['verification_results']['recommendations']:
        print("No specific recommendations at this time.")
    
    # Save report to file
    report_filename = f"verification_report_{prescription_data['id']}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_filename}")

if __name__ == "__main__":
    main()

